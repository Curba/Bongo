from __future__ import annotations

import csv
import json
import re
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State
except ImportError:
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.simulator import Simulator

from qel_twin.characterization.noise_ml.dataset import (
    NoiseDataset,
    NoiseExperiment,
    generate_noise_dataset,
)
from qel_twin.characterization.noise_ml.preprocessing import split_dataset_indices as split_qel_ml_indices
from qel_twin.characterization.noise_ml.reconstruction import compute_trajectory_metrics, reconstruct_dynamics
from qel_twin.characterization.noise_ml.training import evaluate_noise_model, train_noise_model
from qel_twin.training.classical_ml import AVAILABLE_MODELS
from qel_twin.training.classical_ml_noise import train_classical_noise_model
from qel_twin.training.lstm_noise import train_lstm_noise_model


METHOD_TO_REPRESENTATION = {
    "tjm": "mps",
    "mps": "mps",
    "mcwf": "vector",
    "vector": "vector",
    "lindblad": "density_matrix",
    "density_matrix": "density_matrix",
    "exact": "density_matrix",
}

TORCH_MODELS = {"torch_mlp": "mlp", "cnn2d": "2d_cnn"}
SEQUENCE_MODELS = {"lstm", "bilstm"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return cleaned or "dataset"


def normalize_initial_state(name: str) -> str:
    value = str(name).strip()
    if value.lower() == "neel":
        return "Neel"
    return value or "zeros"


def build_observables(num_sites: int, channels: list[str]) -> list[Observable]:
    return [Observable(channel, sites=site) for channel in channels for site in range(num_sites)]


def build_experiment_from_config(config: dict[str, Any]) -> NoiseExperiment:
    num_sites = int(config["num_sites"])
    channels = [str(v).lower() for v in config["channels"]]
    method = str(config.get("method", "tjm")).lower()
    if method not in METHOD_TO_REPRESENTATION:
        raise ValueError(f"Unsupported YAQS method: {method}")

    representation = METHOD_TO_REPRESENTATION[method]
    state = State(
        num_sites,
        initial=normalize_initial_state(config.get("initial_state", "zeros")),
        representation=representation,
    )

    j_coupling = float(config.get("j_coupling", 1.0))
    transverse_field = float(config.get("transverse_field", 1.0))
    hamiltonian = Hamiltonian.ising(num_sites, j_coupling, transverse_field)
    observables = build_observables(num_sites, channels)

    requested_num_traj = int(config.get("num_traj", 100))
    effective_num_traj = 1 if representation == "density_matrix" else requested_num_traj
    sim_kwargs: dict[str, Any] = {
        "observables": observables,
        "elapsed_time": float(config.get("elapsed_time", 5.0)),
        "dt": float(config.get("dt", 0.1)),
        "num_traj": effective_num_traj,
        "preset": str(config.get("preset", "fast")),
        "order": int(config.get("order", 2)),
        "sample_timesteps": True,
        "random_seed": int(config.get("seed", 1234)),
        "tdvp_sweeps": int(config.get("tdvp_sweeps", 1)),
        "tdvp_mode": str(config.get("tdvp_mode", "2site")),
    }
    if config.get("svd_threshold") not in (None, ""):
        sim_kwargs["svd_threshold"] = float(config["svd_threshold"])
    if config.get("max_bond_dim") not in (None, ""):
        sim_kwargs["max_bond_dim"] = int(config["max_bond_dim"])

    sim_params = AnalogSimParams(**sim_kwargs)
    simulator = Simulator(
        parallel=bool(config.get("parallel", False)),
        max_workers=None if config.get("max_workers") in (None, "") else int(config["max_workers"]),
        show_progress=bool(config.get("show_progress", False)),
    )

    return NoiseExperiment(
        state=state,
        hamiltonian=hamiltonian,
        sim_params=sim_params,
        simulator=simulator,
        hamiltonian_metadata={
            "name": "ising",
            "J": j_coupling,
            "g": transverse_field,
            "channels": channels,
            "method": method,
        },
    )


def build_experiment_from_metadata(metadata: dict[str, Any]) -> NoiseExperiment:
    simulation = dict(metadata.get("simulation", {}))
    simulator_meta = dict(metadata.get("simulator", {}))
    hamiltonian_meta = dict(metadata.get("hamiltonian", {}))
    observable_meta = list(metadata.get("observables", []))
    if not observable_meta:
        raise ValueError("Dataset metadata does not contain observables.")

    observables: list[Observable] = []
    for description in observable_meta:
        gate = str(description["gate"]).lower()
        sites = description["sites"]
        site_value: int | list[int] = int(sites[0]) if len(sites) == 1 else [int(site) for site in sites]
        observables.append(Observable(gate, sites=site_value))

    num_sites = int(metadata["num_sites"])
    representation = str(metadata.get("state_representation", "mps"))
    initial = metadata.get("initial_state", {})
    initial_state = str(initial.get("preset", "zeros")) if isinstance(initial, dict) else str(initial)
    state = State(num_sites, initial=normalize_initial_state(initial_state), representation=representation)

    j_coupling = float(hamiltonian_meta.get("J", 1.0))
    transverse_field = float(hamiltonian_meta.get("g", 1.0))
    hamiltonian = Hamiltonian.ising(num_sites, j_coupling, transverse_field)

    allowed_sim_keys = {
        "elapsed_time",
        "dt",
        "num_traj",
        "preset",
        "max_bond_dim",
        "trunc_mode",
        "svd_threshold",
        "krylov_tol",
        "order",
        "sample_timesteps",
        "random_seed",
        "tdvp_sweeps",
        "tdvp_mode",
    }
    sim_kwargs = {k: v for k, v in simulation.items() if k in allowed_sim_keys and v is not None}
    sim_kwargs["observables"] = observables
    sim_params = AnalogSimParams(**sim_kwargs)
    simulator = Simulator(
        parallel=bool(simulator_meta.get("parallel", False)),
        max_workers=simulator_meta.get("max_workers"),
        show_progress=False,
    )
    return NoiseExperiment(
        state=state,
        hamiltonian=hamiltonian,
        sim_params=sim_params,
        simulator=simulator,
        hamiltonian_metadata=hamiltonian_meta,
    )


def create_dataset_job(*, data_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    channels = [str(v).lower() for v in config.get("channels", [])]
    if not channels:
        raise ValueError("Select at least one observable channel.")
    invalid = set(channels) - {"x", "y", "z"}
    if invalid:
        raise ValueError(f"Unsupported channels: {sorted(invalid)}")
    if int(config["num_samples"]) < 3:
        raise ValueError("Use at least 3 samples so train/validation/test splits are possible.")
    if int(config["num_sites"]) <= 0:
        raise ValueError("num_sites must be positive.")
    if float(config["gamma_min"]) <= 0:
        raise ValueError("gamma_min must be positive.")
    if float(config["gamma_max"]) <= float(config["gamma_min"]):
        raise ValueError("gamma_max must be larger than gamma_min.")

    dataset_name = safe_slug(config.get("dataset_name", "noise_dataset"))
    output_value = str(config.get("output_path", "")).strip()
    output_path = Path(output_value) if output_value else data_root / f"{dataset_name}.npz"
    experiment_config = dict(config)
    experiment_config["channels"] = channels
    experiment = build_experiment_from_config(experiment_config)

    dataset = generate_noise_dataset(
        experiment,
        num_samples=int(config["num_samples"]),
        parameterization=str(config["parameterization"]),
        seed=int(config["seed"]),
        gamma_min=float(config["gamma_min"]),
        gamma_max=float(config["gamma_max"]),
    )
    dataset.metadata["dataset_name"] = dataset_name
    dataset.metadata["ui_created_at"] = datetime.now().isoformat(timespec="seconds")
    dataset.metadata["ui_output_path"] = str(output_path)
    dataset.save(output_path)

    return {
        "dataset_path": str(output_path),
        "dataset_name": dataset_name,
        "shape": list(dataset.expectation_values.shape),
        "target_shape": list(dataset.log10_gamma.shape),
        "parameter_names": list(dataset.parameter_names),
        "parameterization": dataset.metadata["parameterization"],
    }


def available_model_options() -> list[dict[str, str]]:
    options = [{"label": f"Classical · {name}", "value": name} for name in AVAILABLE_MODELS]
    options.extend(
        [
            {"label": "PyTorch · MLP", "value": "torch_mlp"},
            {"label": "PyTorch · 2D CNN", "value": "cnn2d"},
            {"label": "Sequence · LSTM", "value": "lstm"},
            {"label": "Sequence · BiLSTM", "value": "bilstm"},
        ]
    )
    return options


def _make_ui_run_dir(output_root: str | Path, dataset_path: str | Path, model_name: str) -> Path:
    dataset_id = safe_slug(Path(dataset_path).stem)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / dataset_id / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_torch_predictions_csv(
    path: Path,
    *,
    indices: np.ndarray,
    parameter_names: tuple[str, ...],
    true_log: np.ndarray,
    pred_log: np.ndarray,
) -> None:
    fieldnames = ["dataset_index"]
    for name in parameter_names:
        fieldnames.extend(
            [
                f"{name}_true_log10",
                f"{name}_pred_log10",
                f"{name}_true",
                f"{name}_pred",
                f"{name}_factor_error",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, dataset_index in enumerate(indices.tolist()):
            row: dict[str, Any] = {"dataset_index": int(dataset_index)}
            for parameter_index, name in enumerate(parameter_names):
                true_log_value = float(true_log[row_index, parameter_index])
                pred_log_value = float(pred_log[row_index, parameter_index])
                true_gamma = 10.0 ** true_log_value
                pred_gamma = 10.0 ** pred_log_value
                factor = max(
                    pred_gamma / max(true_gamma, 1e-300),
                    true_gamma / max(pred_gamma, 1e-300),
                )
                row[f"{name}_true_log10"] = true_log_value
                row[f"{name}_pred_log10"] = pred_log_value
                row[f"{name}_true"] = true_gamma
                row[f"{name}_pred"] = pred_gamma
                row[f"{name}_factor_error"] = factor
            writer.writerow(row)


def train_torch_noise_model_ui(
    *,
    dataset_path: str | Path,
    output_root: str | Path,
    model_name: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str | None,
) -> dict[str, Any]:
    dataset = NoiseDataset.load(dataset_path)
    parameterization = str(dataset.metadata["parameterization"])
    split = split_qel_ml_indices(
        len(dataset.expectation_values),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    trained = train_noise_model(
        dataset,
        split,
        model_name=TORCH_MODELS[model_name],
        parameterization=parameterization,
        seed=seed,
        max_epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device or None,
    )

    run_dir = _make_ui_run_dir(output_root, dataset_path, model_name)
    trained.save(run_dir / "model.npz")
    test_metrics_raw = evaluate_noise_model(trained, dataset, split.test)
    _, test_log_pred = trained.predict(dataset.expectation_values[split.test])
    _save_torch_predictions_csv(
        run_dir / "test_predictions.csv",
        indices=split.test,
        parameter_names=dataset.parameter_names,
        true_log=dataset.log10_gamma[split.test],
        pred_log=test_log_pred,
    )
    np.savez(
        run_dir / "split_indices.npz",
        train_idx=split.train,
        val_idx=split.validation,
        test_idx=split.test,
    )
    with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "training_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(trained.training_history)

    result = {
        "run_id": run_dir.name,
        "dataset_id": Path(dataset_path).stem,
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "model_name": model_name,
        "model_family": "qel_ml_torch",
        "parameter_names": list(dataset.parameter_names),
        "metadata": dataset.metadata,
        "test_metrics": {
            "mae_log10_mean": float(test_metrics_raw["log10_mae"]),
            "rmse_log10_mean": float(test_metrics_raw["log10_rmse"]),
            "median_factor_error_gamma": float(test_metrics_raw["median_factor_error"]),
        },
        "effective_params": {
            "best_epoch": int(trained.best_epoch),
            "best_val_loss": float(trained.best_validation_loss),
            "device": device or "auto",
        },
        "split_sizes": {
            "train": int(len(split.train)),
            "val": int(len(split.validation)),
            "test": int(len(split.test)),
        },
    }
    write_json(run_dir / "metrics.json", result)
    return result


def _extract_prediction_rows(run_dir: Path, parameter_names: tuple[str, ...]) -> list[tuple[int, np.ndarray]]:
    predictions_path = run_dir / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing test predictions for reconstruction: {predictions_path}")

    rows: list[tuple[int, np.ndarray]] = []
    with predictions_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset_index = int(row["dataset_index"])
            prediction = np.asarray(
                [float(row[f"{name}_pred_log10"]) for name in parameter_names],
                dtype=np.float64,
            )
            rows.append((dataset_index, prediction))
    return rows


def evaluate_run_reconstruction(
    *,
    dataset_path: str | Path,
    run_dir: str | Path,
    reconstruction_samples: int,
) -> dict[str, Any]:
    dataset = NoiseDataset.load(dataset_path)
    run_dir = Path(run_dir)
    experiment = build_experiment_from_metadata(dataset.metadata)
    parameterization = str(dataset.metadata["parameterization"])
    prediction_rows = _extract_prediction_rows(run_dir, dataset.parameter_names)
    if not prediction_rows:
        raise RuntimeError("No test predictions are available for reconstruction.")

    count = min(max(1, int(reconstruction_samples)), len(prediction_rows))
    selected_rows = prediction_rows[:count]
    reconstruction_root = run_dir / "reconstruction"
    reconstruction_root.mkdir(parents=True, exist_ok=True)
    sample_summaries: list[dict[str, Any]] = []

    for dataset_index, predicted_log10_gamma in selected_rows:
        predicted_gamma = np.power(10.0, predicted_log10_gamma)
        reconstructed, times = reconstruct_dynamics(
            experiment,
            predicted_gamma,
            parameterization=parameterization,
        )
        original = np.asarray(dataset.expectation_values[dataset_index], dtype=np.float64)
        metrics = compute_trajectory_metrics(original, reconstructed)
        sample_file = reconstruction_root / f"sample_{dataset_index}.npz"
        np.savez_compressed(
            sample_file,
            dataset_index=np.asarray(dataset_index),
            times=np.asarray(times, dtype=np.float64),
            original=original,
            reconstructed=np.asarray(reconstructed, dtype=np.float64),
            predicted_log10_gamma=predicted_log10_gamma,
            predicted_gamma=predicted_gamma,
            true_log10_gamma=dataset.log10_gamma[dataset_index],
            true_gamma=dataset.gamma[dataset_index],
            parameter_names=np.asarray(dataset.parameter_names),
        )
        sample_summaries.append({"dataset_index": int(dataset_index), "file": str(sample_file), **metrics})

    aggregate = {
        "samples_evaluated": len(sample_summaries),
        "trajectory_mae_mean": float(np.mean([row["trajectory_mae"] for row in sample_summaries])),
        "trajectory_rmse_mean": float(np.mean([row["trajectory_rmse"] for row in sample_summaries])),
        "max_abs_trajectory_error_mean": float(
            np.mean([row["max_abs_trajectory_error"] for row in sample_summaries])
        ),
        "trajectory_rmse_median": float(np.median([row["trajectory_rmse"] for row in sample_summaries])),
        "samples": sample_summaries,
    }
    write_json(reconstruction_root / "summary.json", aggregate)
    metrics_path = run_dir / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics_payload["reconstruction"] = aggregate
    write_json(metrics_path, metrics_payload)
    return aggregate


def train_model_job(*, dataset_path: str | Path, output_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    model_name = str(config["model_name"])
    common = {
        "dataset_path": dataset_path,
        "output_dir": output_root,
        "seed": int(config.get("seed", 1234)),
        "train_fraction": float(config.get("train_fraction", 0.60)),
        "validation_fraction": float(config.get("validation_fraction", 0.20)),
        "run_tag": config.get("run_tag") or None,
    }

    if model_name in SEQUENCE_MODELS:
        result = train_lstm_noise_model(
            **common,
            model_name=model_name,
            batch_size=int(config.get("batch_size", 64)),
            epochs=int(config.get("epochs", 200)),
            hidden_size=int(config.get("hidden_size", 128)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            patience=int(config.get("patience", 30)),
            grad_clip=float(config.get("grad_clip", 1.0)),
            device_name=config.get("device") or None,
        )
    elif model_name in TORCH_MODELS:
        result = train_torch_noise_model_ui(
            dataset_path=dataset_path,
            output_root=output_root,
            model_name=model_name,
            seed=int(config.get("seed", 1234)),
            train_fraction=float(config.get("train_fraction", 0.60)),
            validation_fraction=float(config.get("validation_fraction", 0.20)),
            epochs=int(config.get("epochs", 150)),
            patience=int(config.get("patience", 20)),
            batch_size=int(config.get("batch_size", 32)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            device=config.get("device") or None,
        )
    else:
        result = train_classical_noise_model(
            **common,
            model_name=model_name,
            feature_mode=str(config.get("feature_mode", "flatten")),
            n_neighbors=int(config.get("n_neighbors", 7)),
            pca_components=int(config.get("pca_components", 64)),
            n_estimators=int(config.get("n_estimators", 500)),
        )

    run_dir = Path(result["run_dir"])
    reconstruction = evaluate_run_reconstruction(
        dataset_path=dataset_path,
        run_dir=run_dir,
        reconstruction_samples=max(1, int(config.get("reconstruction_samples", 3))),
    )
    return {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "dataset_path": str(dataset_path),
        "test_metrics": result.get("test_metrics", {}),
        "reconstruction": reconstruction,
    }


@dataclass
class JobState:
    id: str
    kind: str
    status: str
    message: str
    created_at: str
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, function: Callable[..., dict[str, Any]], **kwargs: Any) -> str:
        job_id = uuid.uuid4().hex
        state = JobState(
            id=job_id,
            kind=kind,
            status="queued",
            message="Queued",
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        with self._lock:
            self._jobs[job_id] = state

        def runner() -> None:
            with self._lock:
                self._jobs[job_id].status = "running"
                self._jobs[job_id].message = (
                    "Generating dataset..."
                    if kind == "dataset"
                    else "Training model and reconstructing trajectories..."
                )
            try:
                result = function(**kwargs)
            except Exception:
                error = traceback.format_exc()
                with self._lock:
                    self._jobs[job_id].status = "failed"
                    self._jobs[job_id].message = "Job failed"
                    self._jobs[job_id].error = error
                    self._jobs[job_id].finished_at = datetime.now().isoformat(timespec="seconds")
                return
            with self._lock:
                self._jobs[job_id].status = "complete"
                self._jobs[job_id].message = "Complete"
                self._jobs[job_id].result = result
                self._jobs[job_id].finished_at = datetime.now().isoformat(timespec="seconds")

        self._executor.submit(runner)
        return job_id

    def get(self, job_id: str | None) -> dict[str, Any] | None:
        if not job_id:
            return None
        with self._lock:
            state = self._jobs.get(job_id)
            return None if state is None else _json_safe(state.__dict__)


JOB_MANAGER = JobManager(max_workers=2)


def scan_datasets(data_root: str | Path) -> list[dict[str, Any]]:
    root = Path(data_root)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.npz")):
        try:
            dataset = NoiseDataset.load(path)
        except Exception:
            continue
        rows.append(
            {
                "path": str(path),
                "name": str(dataset.metadata.get("dataset_name", path.stem)),
                "samples": int(dataset.expectation_values.shape[0]),
                "observables": int(dataset.expectation_values.shape[1]),
                "times": int(dataset.expectation_values.shape[2]),
                "sites": int(dataset.metadata.get("num_sites", 0)),
                "parameterization": str(dataset.metadata.get("parameterization", "unknown")),
                "targets": int(dataset.log10_gamma.shape[1]),
            }
        )
    return rows


def dataset_details(path: str | Path) -> dict[str, Any]:
    dataset = NoiseDataset.load(path)
    return {
        "path": str(path),
        "shape": list(dataset.expectation_values.shape),
        "target_shape": list(dataset.log10_gamma.shape),
        "parameter_names": list(dataset.parameter_names),
        "metadata": dataset.metadata,
    }


def load_dataset_preview(path: str | Path, sample_index: int, observable_index: int) -> dict[str, Any]:
    dataset = NoiseDataset.load(path)
    sample_index = max(0, min(int(sample_index), len(dataset.expectation_values) - 1))
    observable_index = max(0, min(int(observable_index), dataset.expectation_values.shape[1] - 1))
    return {
        "times": dataset.times,
        "values": dataset.expectation_values[sample_index, observable_index],
        "sample_index": sample_index,
        "observable_index": observable_index,
    }


def scan_runs(output_root: str | Path) -> list[dict[str, Any]]:
    root = Path(output_root)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        reconstruction = metrics.get("reconstruction", {}) or {}
        test_metrics = metrics.get("test_metrics", {}) or {}
        run_dir = metrics_path.parent
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_id": str(metrics.get("run_id", run_dir.name)),
                "dataset": str(metrics.get("dataset_id", run_dir.parents[1].name)),
                "model": str(metrics.get("model_name", run_dir.parent.name)),
                "log10_mae": test_metrics.get("mae_log10_mean", test_metrics.get("log10_mae")),
                "log10_rmse": test_metrics.get("rmse_log10_mean", test_metrics.get("log10_rmse")),
                "factor_error": test_metrics.get(
                    "median_factor_error_gamma", test_metrics.get("median_factor_error")
                ),
                "reconstruction_mae": reconstruction.get("trajectory_mae_mean"),
                "reconstruction_rmse": reconstruction.get("trajectory_rmse_mean"),
                "reconstruction_max": reconstruction.get("max_abs_trajectory_error_mean"),
                "reconstruction_samples": reconstruction.get("samples_evaluated", 0),
            }
        )
    return rows


def load_run_details(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    history: list[dict[str, Any]] = []
    history_path = run_dir / "training_history.csv"
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))

    reconstruction_summary_path = run_dir / "reconstruction" / "summary.json"
    reconstruction = (
        json.loads(reconstruction_summary_path.read_text(encoding="utf-8"))
        if reconstruction_summary_path.exists()
        else {}
    )

    samples: list[dict[str, Any]] = []
    for sample_file in sorted((run_dir / "reconstruction").glob("sample_*.npz")):
        try:
            with np.load(sample_file, allow_pickle=False) as payload:
                dataset_index = int(np.asarray(payload["dataset_index"]).item())
            samples.append({"label": f"Dataset sample {dataset_index}", "value": str(sample_file)})
        except Exception:
            continue

    return {"metrics": metrics, "history": history, "reconstruction": reconstruction, "samples": samples}


def load_reconstruction_sample(sample_path: str | Path, observable_index: int) -> dict[str, Any]:
    with np.load(sample_path, allow_pickle=False) as payload:
        times = np.asarray(payload["times"], dtype=np.float64)
        original = np.asarray(payload["original"], dtype=np.float64)
        reconstructed = np.asarray(payload["reconstructed"], dtype=np.float64)
        dataset_index = int(np.asarray(payload["dataset_index"]).item())
        parameter_names = [str(v) for v in payload["parameter_names"].tolist()]
        predicted_gamma = np.asarray(payload["predicted_gamma"], dtype=np.float64)
        true_gamma = np.asarray(payload["true_gamma"], dtype=np.float64)
    observable_index = max(0, min(int(observable_index), original.shape[0] - 1))
    return {
        "times": times,
        "original": original[observable_index],
        "reconstructed": reconstructed[observable_index],
        "observable_index": observable_index,
        "dataset_index": dataset_index,
        "parameter_names": parameter_names,
        "predicted_gamma": predicted_gamma,
        "true_gamma": true_gamma,
    }
