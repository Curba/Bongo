from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from qel_twin.training.lstm import (
    LSTMRegressor,
    P3SequenceDataset,
    SequenceNormalizer,
    create_run_dir,
    infer_dataset_id,
    make_run_id,
    predict,
    run_epoch,
    set_seed,
    write_json,
)
from qel_twin.training.noise_dataset import (
    evaluate_predictions,
    load_noise_dataset,
    save_predictions_csv,
    split_dataset_indices,
)


def convert_noise_to_lstm_sequence(X: np.ndarray) -> np.ndarray:
    """Convert qel-ml ``(N,O,T)`` trajectories to LSTM ``(N,T,O)`` sequences."""
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected dynamics shape (N, O, T), got {X.shape}")

    return np.transpose(X, (0, 2, 1)).astype(np.float32)


def train_lstm_noise_model(
    dataset_path: str | Path,
    output_dir: str | Path = "outputs/noise_ml_runs",
    model_name: str = "lstm",
    seed: int = 1234,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    batch_size: int = 64,
    epochs: int = 200,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    bidirectional: bool | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    grad_clip: float = 1.0,
    run_tag: str | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Train LSTM/BiLSTM directly on qel-ml ``NoiseDataset`` files."""
    normalized_model_name = model_name.lower()
    if normalized_model_name not in {"lstm", "bilstm"}:
        raise ValueError("model_name must be 'lstm' or 'bilstm'.")

    if bidirectional is None:
        bidirectional = normalized_model_name == "bilstm"

    set_seed(seed)
    dataset_path = Path(dataset_path)

    dataset = load_noise_dataset(dataset_path)
    X_raw = dataset.dynamics
    y_log = dataset.log10_gamma
    parameter_names = dataset.parameter_names
    num_targets = y_log.shape[1]

    dataset_id = infer_dataset_id(dataset_path, dataset.metadata)
    run_id = make_run_id(seed=seed, tag=run_tag)

    run_dir = create_run_dir(
        output_root=output_dir,
        dataset_id=dataset_id,
        model_name=normalized_model_name,
        run_id=run_id,
    )

    X_seq = convert_noise_to_lstm_sequence(X_raw)

    split = split_dataset_indices(
        X_seq.shape[0],
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    X_train = X_seq[split.train]
    y_train = y_log[split.train]
    X_val = X_seq[split.validation]
    y_val = y_log[split.validation]
    X_test = X_seq[split.test]
    y_test = y_log[split.test]

    # Same normalization idea as qel-ml: statistics are fit on training data only.
    # Here every LSTM feature corresponds to one observable.
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
        bidirectional=bool(bidirectional),
        output_size=num_targets,
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
        "dataset_layout": "N,O,T",
        "model_name": normalized_model_name,
        "model_family": "neural_network",
        "architecture": "lstm",
        "seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bool(bidirectional),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "grad_clip": grad_clip,
        "device": str(device),
        "input_shape_raw": list(X_raw.shape),
        "input_shape_sequence": list(X_seq.shape),
        "input_size": int(input_size),
        "target_shape": list(y_log.shape),
        "num_targets": int(num_targets),
        "parameter_names": list(parameter_names),
        "dataset_metadata": dataset.metadata,
    }
    write_json(run_dir / "run_metadata.json", run_metadata)

    print("=" * 88)
    print("Bongo LSTM - qel-ml NoiseDataset format")
    print("=" * 88)
    print(f"Dataset ID:       {dataset_id}")
    print(f"Dataset path:     {dataset_path}")
    print(f"Model:            {normalized_model_name}")
    print(f"Run ID:           {run_id}")
    print(f"Run dir:          {run_dir}")
    print(f"Raw X shape:      {X_raw.shape}  (N,O,T)")
    print(f"LSTM X shape:     {X_seq.shape}  (N,T,O)")
    print(f"Target shape:     {y_log.shape}")
    print(f"Parameters:       {parameter_names}")
    print(f"Input size:       {input_size}")
    print(
        "Train/val/test:   "
        f"{len(split.train)} / {len(split.validation)} / {len(split.test)}"
    )
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
                    "output_size": num_targets,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "bidirectional": bool(bidirectional),
                    "parameter_names": list(parameter_names),
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

        pd.DataFrame(history_rows).to_csv(
            run_dir / "training_history.csv",
            index=False,
        )

        print(
            f"epoch={epoch:04d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_seconds:.2f}s"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch was {best_epoch}."
            )
            break

    training_seconds = time.perf_counter() - training_start

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_train_true, y_train_pred = predict(model, train_loader, device)
    y_val_true, y_val_pred = predict(model, val_loader, device)
    y_test_true, y_test_pred = predict(model, test_loader, device)

    train_metrics = evaluate_predictions(
        y_train_true, y_train_pred, parameter_names
    )
    val_metrics = evaluate_predictions(
        y_val_true, y_val_pred, parameter_names
    )
    test_metrics = evaluate_predictions(
        y_test_true, y_test_pred, parameter_names
    )

    save_predictions_csv(
        run_dir / "train_predictions.csv",
        y_train_true,
        y_train_pred,
        parameter_names,
        indices=split.train,
    )
    save_predictions_csv(
        run_dir / "val_predictions.csv",
        y_val_true,
        y_val_pred,
        parameter_names,
        indices=split.validation,
    )
    save_predictions_csv(
        run_dir / "test_predictions.csv",
        y_test_true,
        y_test_pred,
        parameter_names,
        indices=split.test,
    )

    np.savez(
        run_dir / "split_indices.npz",
        train_idx=split.train,
        val_idx=split.validation,
        test_idx=split.test,
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "metadata": dataset.metadata,
        "model_name": normalized_model_name,
        "model_family": "neural_network",
        "architecture": "lstm",
        "raw_X_shape": list(X_raw.shape),
        "sequence_X_shape": list(X_seq.shape),
        "y_log_shape": list(y_log.shape),
        "parameter_names": list(parameter_names),
        "input_size": int(input_size),
        "seed": seed,
        "timing": {
            "training_seconds": float(training_seconds),
        },
        "requested_params": {
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "batch_size": batch_size,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bool(bidirectional),
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
            "train": int(len(split.train)),
            "val": int(len(split.validation)),
            "test": int(len(split.test)),
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


__all__ = ["convert_noise_to_lstm_sequence", "train_lstm_noise_model"]
