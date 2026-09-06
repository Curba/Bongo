from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True, slots=True)
class NoiseDatasetArrays:
    """Arrays loaded from the canonical qel-ml NoiseDataset NPZ format.

    Shapes:
        dynamics: (N, O, T)
        gamma: (N, P)
        log10_gamma: (N, P)
        times: (T,)
    """

    dynamics: np.ndarray
    gamma: np.ndarray
    log10_gamma: np.ndarray
    times: np.ndarray
    parameter_names: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Disjoint sample indices for train/validation/test."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_noise_dataset(path: str | Path) -> NoiseDatasetArrays:
    """Load Wanlin/qel-ml's canonical ``NoiseDataset.save()`` NPZ format.

    Required keys:
        expectation_values
        gamma
        log10_gamma
        times
        parameter_names
        metadata_json
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    with np.load(source, allow_pickle=False) as payload:
        required = {
            "expectation_values",
            "gamma",
            "log10_gamma",
            "times",
            "parameter_names",
            "metadata_json",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(
                "This trainer expects the canonical qel-ml NoiseDataset format. "
                f"Missing fields: {sorted(missing)}. Found: {list(payload.files)}"
            )

        dynamics = np.asarray(payload["expectation_values"], dtype=np.float32)
        gamma = np.asarray(payload["gamma"], dtype=np.float32)
        log10_gamma = np.asarray(payload["log10_gamma"], dtype=np.float32)
        times = np.asarray(payload["times"], dtype=np.float64)
        parameter_names = tuple(
            str(name) for name in payload["parameter_names"].tolist()
        )
        metadata = json.loads(str(payload["metadata_json"].item()))

    if dynamics.ndim != 3:
        raise ValueError(
            f"expectation_values must have shape (N, O, T), got {dynamics.shape}"
        )

    expected_targets = (dynamics.shape[0], len(parameter_names))
    if gamma.shape != expected_targets:
        raise ValueError(
            f"gamma must have shape {expected_targets}, got {gamma.shape}"
        )
    if log10_gamma.shape != expected_targets:
        raise ValueError(
            f"log10_gamma must have shape {expected_targets}, got {log10_gamma.shape}"
        )

    if times.shape != (dynamics.shape[2],):
        raise ValueError(
            f"times must have shape {(dynamics.shape[2],)}, got {times.shape}"
        )

    if not np.all(np.isfinite(dynamics)):
        raise ValueError("expectation_values contains non-finite values.")
    if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0):
        raise ValueError("gamma must contain finite, strictly positive values.")
    if not np.all(np.isfinite(log10_gamma)):
        raise ValueError("log10_gamma contains non-finite values.")
    if not np.allclose(
        np.log10(gamma.astype(np.float64)),
        log10_gamma.astype(np.float64),
        rtol=0.0,
        atol=2e-6,
    ):
        raise ValueError("gamma and log10_gamma are inconsistent.")

    metadata = dict(metadata)
    metadata.setdefault("dataset_id", source.stem)
    metadata["dataset_path"] = str(source)
    metadata["dataset_file"] = source.name
    metadata["dataset_layout"] = "N,O,T"
    metadata["dataset_shape"] = list(dynamics.shape)
    metadata["target_shape"] = list(log10_gamma.shape)
    metadata["parameter_names"] = list(parameter_names)
    metadata["num_samples"] = int(dynamics.shape[0])
    metadata["num_observables"] = int(dynamics.shape[1])
    metadata["num_times"] = int(dynamics.shape[2])

    return NoiseDatasetArrays(
        dynamics=dynamics,
        gamma=gamma,
        log10_gamma=log10_gamma,
        times=times,
        parameter_names=parameter_names,
        metadata=metadata,
    )


def split_dataset_indices(
    num_samples: int,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    seed: int = 1234,
) -> DatasetSplit:
    """Use the same default 60/20/20 sample-level split as qel-ml."""
    if num_samples < 3:
        raise ValueError("At least three samples are required.")

    if (
        train_fraction <= 0
        or validation_fraction <= 0
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError(
            "train_fraction and validation_fraction must be positive "
            "and sum to less than one."
        )

    n_train = int(np.floor(num_samples * train_fraction))
    n_validation = int(np.floor(num_samples * validation_fraction))

    if n_train == 0 or n_validation == 0 or n_train + n_validation >= num_samples:
        raise ValueError(
            "The requested fractions produce an empty train, validation, or test split."
        )

    indices = np.random.default_rng(seed).permutation(num_samples)

    split = DatasetSplit(
        train=indices[:n_train],
        validation=indices[n_train : n_train + n_validation],
        test=indices[n_train + n_validation :],
    )

    # Defensive overlap checks.
    train_set = set(split.train.tolist())
    val_set = set(split.validation.tolist())
    test_set = set(split.test.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Dataset split overlap detected.")

    return split


def ensure_prediction_matrix(values: np.ndarray, num_targets: int) -> np.ndarray:
    """Normalize estimator output to shape ``(N, P)``."""
    array = np.asarray(values, dtype=np.float32)

    if array.ndim == 1:
        array = array[:, np.newaxis]

    if array.ndim != 2 or array.shape[1] != num_targets:
        raise ValueError(
            f"Expected prediction shape (N, {num_targets}), got {array.shape}"
        )

    return array


def evaluate_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    parameter_names: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """Dynamic-P regression metrics in log10 and physical gamma space."""
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)

    if y_true_log.shape != y_pred_log.shape:
        raise ValueError(
            f"Prediction shape mismatch: {y_true_log.shape} vs {y_pred_log.shape}"
        )

    if y_true_log.ndim != 2:
        raise ValueError(f"Expected target shape (N, P), got {y_true_log.shape}")

    names = [str(name) for name in parameter_names]
    if len(names) != y_true_log.shape[1]:
        raise ValueError(
            f"Expected {y_true_log.shape[1]} parameter names, got {len(names)}"
        )

    y_true_gamma = np.power(10.0, y_true_log)
    y_pred_gamma = np.power(10.0, y_pred_log)

    relative_error = np.abs(y_pred_gamma - y_true_gamma) / np.maximum(
        np.abs(y_true_gamma), 1e-12
    )
    factor_error = np.maximum(
        y_pred_gamma / np.maximum(y_true_gamma, 1e-300),
        y_true_gamma / np.maximum(y_pred_gamma, 1e-300),
    )

    metrics: dict[str, float] = {
        "mae_log10_mean": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log10_mean": float(
            np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        ),
        "r2_log10_mean": float(r2_score(y_true_log, y_pred_log)),
        "median_relative_error_gamma": float(np.median(relative_error)),
        "mean_relative_error_gamma": float(np.mean(relative_error)),
        "median_factor_error_gamma": float(np.median(factor_error)),
    }

    for index, name in enumerate(names):
        metrics[f"{name}_mae_log10"] = float(
            mean_absolute_error(y_true_log[:, index], y_pred_log[:, index])
        )
        metrics[f"{name}_rmse_log10"] = float(
            np.sqrt(
                mean_squared_error(
                    y_true_log[:, index],
                    y_pred_log[:, index],
                )
            )
        )
        metrics[f"{name}_r2_log10"] = float(
            r2_score(y_true_log[:, index], y_pred_log[:, index])
        )
        metrics[f"{name}_median_relative_error_gamma"] = float(
            np.median(relative_error[:, index])
        )
        metrics[f"{name}_mean_relative_error_gamma"] = float(
            np.mean(relative_error[:, index])
        )
        metrics[f"{name}_median_factor_error_gamma"] = float(
            np.median(factor_error[:, index])
        )

    return metrics


def save_predictions_csv(
    output_path: str | Path,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    parameter_names: tuple[str, ...] | list[str],
    *,
    indices: np.ndarray | None = None,
) -> None:
    """Save one set of predictions with dynamic parameter columns."""
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)
    names = [str(name) for name in parameter_names]

    rows: dict[str, np.ndarray] = {}

    if indices is not None:
        rows["dataset_index"] = np.asarray(indices, dtype=int)

    for index, name in enumerate(names):
        true_log = y_true_log[:, index]
        pred_log = y_pred_log[:, index]
        true_gamma = np.power(10.0, true_log)
        pred_gamma = np.power(10.0, pred_log)

        rows[f"{name}_true_log10"] = true_log
        rows[f"{name}_pred_log10"] = pred_log
        rows[f"{name}_true"] = true_gamma
        rows[f"{name}_pred"] = pred_gamma
        rows[f"{name}_abs_error_log10"] = np.abs(pred_log - true_log)
        rows[f"{name}_relative_error"] = (
            np.abs(pred_gamma - true_gamma) / np.maximum(true_gamma, 1e-12)
        )
        rows[f"{name}_factor_error"] = np.maximum(
            pred_gamma / np.maximum(true_gamma, 1e-300),
            true_gamma / np.maximum(pred_gamma, 1e-300),
        )

    pd.DataFrame(rows).to_csv(Path(output_path), index=False)
