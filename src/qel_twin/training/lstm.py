from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PARAM_NAMES = ["gamma_x", "gamma_y", "gamma_z"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_run_id(seed: int, tag: str | None = None) -> str:
    parts = [now_string(), f"seed{seed}"]

    if tag:
        parts.append(safe_name(tag))

    return "_".join(parts)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def infer_dataset_id(dataset_path: str | Path, metadata: dict[str, Any]) -> str:
    if metadata.get("dataset_id"):
        return safe_name(str(metadata["dataset_id"]))

    if metadata.get("dataset_name"):
        return safe_name(str(metadata["dataset_name"]))

    return safe_name(Path(dataset_path).stem)


def load_p3_npz(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    if "X" not in data.files:
        raise KeyError(f"Dataset must contain key 'X'. Found keys: {data.files}")

    if "y_log" in data.files:
        y_log = data["y_log"].astype(np.float32)
        target_key = "y_log"
    elif "theta" in data.files:
        y_log = data["theta"].astype(np.float32)
        target_key = "theta"
    else:
        raise KeyError(f"Dataset must contain key 'y_log' or 'theta'. Found keys: {data.files}")

    X = data["X"].astype(np.float32)

    if X.ndim != 4:
        raise ValueError(f"Expected X shape (N, C, L, T), got {X.shape}")

    if y_log.ndim != 2 or y_log.shape[1] != 3:
        raise ValueError(f"Expected y_log shape (N, 3), got {y_log.shape}")

    metadata: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "dataset_file": dataset_path.name,
        "dataset_stem": dataset_path.stem,
        "keys": list(data.files),
        "target_key": target_key,
        "X_shape": list(X.shape),
        "y_log_shape": list(y_log.shape),
        "num_samples": int(X.shape[0]),
        "num_channels": int(X.shape[1]),
        "num_sites": int(X.shape[2]),
        "num_times": int(X.shape[3]),
    }

    if "dataset_id" in data.files:
        metadata["dataset_id"] = str(data["dataset_id"].item())

    if "dataset_name" in data.files:
        metadata["dataset_name"] = str(data["dataset_name"].item())

    if "created_at" in data.files:
        metadata["dataset_created_at"] = str(data["created_at"].item())

    if "times" in data.files:
        times = data["times"]
        metadata["times_shape"] = list(times.shape)
        metadata["time_min"] = float(np.min(times))
        metadata["time_max"] = float(np.max(times))
        if len(times) > 1:
            metadata["time_step_mean"] = float(np.mean(np.diff(times)))

    if "gamma" in data.files:
        gamma = data["gamma"]
        metadata["gamma_shape"] = list(gamma.shape)
        metadata["gamma_min"] = float(np.min(gamma))
        metadata["gamma_max"] = float(np.max(gamma))

    if "param_names" in data.files:
        metadata["param_names"] = [str(x) for x in data["param_names"].tolist()]
    else:
        metadata["param_names"] = PARAM_NAMES

    if "observable_channels" in data.files:
        metadata["observable_channels"] = [
            str(x) for x in data["observable_channels"].tolist()
        ]

    if "config_json" in data.files:
        try:
            metadata["config"] = json.loads(str(data["config_json"].item()))
        except Exception:
            metadata["config_json_raw"] = str(data["config_json"].item())

    return X, y_log, metadata


def convert_x_to_lstm_sequence(X: np.ndarray) -> np.ndarray:
    """
    Input:
        X: (N, C, L, T)

    Output:
        sequence: (N, T, C*L)

    For current P3 dataset:
        (N, 3, 5, 101) -> (N, 101, 15)
    """
    if X.ndim != 4:
        raise ValueError(f"Expected X shape (N, C, L, T), got {X.shape}")

    # (N, C, L, T) -> (N, T, C, L) -> (N, T, C*L)
    X_seq = np.transpose(X, (0, 3, 1, 2))
    X_seq = X_seq.reshape(X.shape[0], X.shape[3], X.shape[1] * X.shape[2])
    return X_seq.astype(np.float32)


class SequenceNormalizer:
    """
    Standardizes each feature dimension across samples and time.

    fit input:
        X_train: (N_train, T, F)
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SequenceNormalizer":
        self.mean_ = X.mean(axis=(0, 1), keepdims=True)
        self.std_ = X.std(axis=(0, 1), keepdims=True)
        self.std_ = np.maximum(self.std_, 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer has not been fitted.")
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def state_dict(self) -> dict[str, list]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer has not been fitted.")

        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }


class P3SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_log: np.ndarray) -> None:
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X_seq.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X_seq[idx], self.y_log[idx]


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        output_size: int = 3,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_size = output_size

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        directions = 2 if bidirectional else 1
        head_input = hidden_size * directions

        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, head_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (batch, time, features)
        """
        output, _ = self.lstm(x)

        # Last time step representation
        last = output[:, -1, :]

        return self.head(last)


def evaluate_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float]:
    y_true_gamma = 10.0**y_true_log
    y_pred_gamma = 10.0**y_pred_log

    metrics: dict[str, float] = {}

    metrics["mae_log10_mean"] = float(mean_absolute_error(y_true_log, y_pred_log))
    metrics["rmse_log10_mean"] = float(
        np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    )
    metrics["r2_log10_mean"] = float(r2_score(y_true_log, y_pred_log))

    relative_error = np.abs(y_pred_gamma - y_true_gamma) / np.maximum(
        np.abs(y_true_gamma),
        1e-12,
    )

    metrics["median_relative_error_gamma"] = float(np.median(relative_error))
    metrics["mean_relative_error_gamma"] = float(np.mean(relative_error))

    for i, name in enumerate(PARAM_NAMES):
        metrics[f"{name}_mae_log10"] = float(
            mean_absolute_error(y_true_log[:, i], y_pred_log[:, i])
        )
        metrics[f"{name}_rmse_log10"] = float(
            np.sqrt(mean_squared_error(y_true_log[:, i], y_pred_log[:, i]))
        )
        metrics[f"{name}_r2_log10"] = float(
            r2_score(y_true_log[:, i], y_pred_log[:, i])
        )
        metrics[f"{name}_median_relative_error_gamma"] = float(
            np.median(relative_error[:, i])
        )
        metrics[f"{name}_mean_relative_error_gamma"] = float(
            np.mean(relative_error[:, i])
        )

    return metrics


def save_predictions_csv(
    output_path: str | Path,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    indices: np.ndarray | None = None,
) -> None:
    output_path = Path(output_path)

    rows: dict[str, np.ndarray] = {}

    if indices is not None:
        rows["dataset_index"] = indices.astype(int)

    for i, name in enumerate(PARAM_NAMES):
        true_log = y_true_log[:, i]
        pred_log = y_pred_log[:, i]

        true_gamma = 10.0**true_log
        pred_gamma = 10.0**pred_log

        rows[f"{name}_true_log10"] = true_log
        rows[f"{name}_pred_log10"] = pred_log
        rows[f"{name}_true"] = true_gamma
        rows[f"{name}_pred"] = pred_gamma
        rows[f"{name}_abs_error_log10"] = np.abs(pred_log - true_log)
        rows[f"{name}_relative_error"] = np.abs(pred_gamma - true_gamma) / np.maximum(
            true_gamma,
            1e-12,
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true_all = []
    y_pred_all = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)

        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    return np.vstack(y_true_all), np.vstack(y_pred_all)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> float:
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if is_train:
                loss.backward()

                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

        batch_size = xb.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(1, total_count)


def create_run_dir(
    output_root: str | Path,
    dataset_id: str,
    model_name: str,
    run_id: str,
) -> Path:
    run_dir = Path(output_root) / dataset_id / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_lstm_model(
    dataset_path: str | Path,
    output_dir: str | Path = "outputs/ml_runs",
    model_name: str = "lstm",
    seed: int = 1234,
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 64,
    epochs: int = 200,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    bidirectional: bool = False,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    grad_clip: float = 1.0,
    run_tag: str | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    set_seed(seed)

    dataset_path = Path(dataset_path)

    X_raw, y_log, dataset_metadata = load_p3_npz(dataset_path)
    dataset_id = infer_dataset_id(dataset_path, dataset_metadata)
    run_id = make_run_id(seed=seed, tag=run_tag)

    run_dir = create_run_dir(
        output_root=output_dir,
        dataset_id=dataset_id,
        model_name=model_name,
        run_id=run_id,
    )

    X_seq = convert_x_to_lstm_sequence(X_raw)

    indices = np.arange(X_seq.shape[0])

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    val_fraction = val_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    X_train = X_seq[train_idx]
    y_train = y_log[train_idx]

    X_val = X_seq[val_idx]
    y_val = y_log[val_idx]

    X_test = X_seq[test_idx]
    y_test = y_log[test_idx]

    normalizer = SequenceNormalizer().fit(X_train)

    X_train = normalizer.transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

    train_ds = P3SequenceDataset(X_train, y_train)
    val_ds = P3SequenceDataset(X_val, y_val)
    test_ds = P3SequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train.shape[-1]

    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        output_size=3,
    ).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, patience // 4),
    )

    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "model_name": model_name,
        "model_family": "neural_network",
        "architecture": "lstm",
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "grad_clip": grad_clip,
        "device": str(device),
        "input_shape_raw": list(X_raw.shape),
        "input_shape_sequence": list(X_seq.shape),
        "input_size": int(input_size),
        "target_shape": list(y_log.shape),
        "dataset_metadata": dataset_metadata,
    }

    write_json(run_dir / "run_metadata.json", run_metadata)

    print("=" * 88)
    print("QEL Twin LSTM Training")
    print("=" * 88)
    print(f"Dataset ID:       {dataset_id}")
    print(f"Dataset path:     {dataset_path}")
    print(f"Model:            {model_name}")
    print(f"Run ID:           {run_id}")
    print(f"Run dir:          {run_dir}")
    print(f"Raw X shape:      {X_raw.shape}")
    print(f"LSTM X shape:     {X_seq.shape}")
    print(f"Target shape:     {y_log.shape}")
    print(f"Input size:       {input_size}")
    print(f"Train/val/test:   {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"Device:           {device}")
    print("=" * 88)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    best_model_path = run_dir / "model.pt"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=grad_clip,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            loss_fn=loss_fn,
            device=device,
            grad_clip=None,
        )

        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_start

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "metadata": run_metadata,
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "bidirectional": bidirectional,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "epoch_seconds": epoch_seconds,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
        )

        pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)

        print(
            f"epoch={epoch:04d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_seconds:.2f}s"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    training_seconds = time.perf_counter() - training_start

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_train_true, y_train_pred = predict(model, train_loader, device)
    y_val_true, y_val_pred = predict(model, val_loader, device)
    y_test_true, y_test_pred = predict(model, test_loader, device)

    train_metrics = evaluate_predictions(y_train_true, y_train_pred)
    val_metrics = evaluate_predictions(y_val_true, y_val_pred)
    test_metrics = evaluate_predictions(y_test_true, y_test_pred)

    save_predictions_csv(
        run_dir / "train_predictions.csv",
        y_train_true,
        y_train_pred,
        indices=train_idx,
    )
    save_predictions_csv(
        run_dir / "val_predictions.csv",
        y_val_true,
        y_val_pred,
        indices=val_idx,
    )
    save_predictions_csv(
        run_dir / "test_predictions.csv",
        y_test_true,
        y_test_pred,
        indices=test_idx,
    )

    np.savez(
        run_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "metadata": dataset_metadata,
        "model_name": model_name,
        "model_family": "neural_network",
        "architecture": "lstm",
        "raw_X_shape": list(X_raw.shape),
        "sequence_X_shape": list(X_seq.shape),
        "y_log_shape": list(y_log.shape),
        "input_size": int(input_size),
        "seed": seed,
        "timing": {
            "training_seconds": float(training_seconds),
        },
        "requested_params": {
            "test_size": test_size,
            "val_size": val_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "grad_clip": grad_clip,
        },
        "effective_params": {
            "best_epoch": int(best_epoch),
            "epochs_completed": int(len(history_rows)),
            "best_val_loss": float(best_val_loss),
            "device": str(device),
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    write_json(run_dir / "metrics.json", result)

    print("\nTrain metrics:")
    print(json.dumps(train_metrics, indent=2))

    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    print(f"\nSaved LSTM run to: {run_dir}")

    return result