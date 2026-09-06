"""Verify saved classical estimators and refit explicitly selected configurations."""

from __future__ import annotations

import hashlib
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from qel_twin.training.classical_ml import write_json
from qel_twin.training.noise_dataset import (
    ensure_prediction_matrix,
    evaluate_predictions,
)


def check_metrics(y, prediction, names, expected):
    metrics = evaluate_predictions(y, prediction, names)
    for key, value in expected.items():
        if key not in metrics:
            raise ValueError(f"Unknown recorded metric: {key}")
        if not np.isclose(metrics[key], value, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Metric mismatch: {key}: {metrics[key]} != {value}")
    return metrics


def refit_and_save(
    source,
    destination,
    X_train,
    y_train,
    X_test,
    y_test,
    names,
    expected,
    metadata,
    threads=2,
    blas_threads=None,
):
    """Refit a cloned saved pipeline, reload it, and verify every recorded metric."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    start = time.perf_counter()
    with (
        threadpool_limits(limits=threads),
        threadpool_limits(
            limits=threads if blas_threads is None else blas_threads, user_api="blas"
        ),
    ):
        model = clone(joblib.load(source))
        model.fit(X_train, y_train)
        pred = ensure_prediction_matrix(model.predict(X_test), len(names))
        metrics = check_metrics(y_test, pred, names, expected)
        destination.mkdir(parents=True)
        path = destination / "model.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        reloaded_pred = ensure_prediction_matrix(loaded.predict(X_test), len(names))
        np.testing.assert_array_equal(pred, reloaded_pred)
    record = dict(metadata)
    record.update(
        {
            "model_path": str(path.resolve()),
            "source_model": str(Path(source).resolve()),
            "test_metrics": metrics,
            "recorded_test_metrics": expected,
            "estimator_params": {
                k: v if v is None or isinstance(v, (str, int, float, bool)) else repr(v)
                for k, v in model.get_params(deep=True).items()
            },
            "versions": {
                p: version(p) for p in ["scikit-learn", "numpy", "scipy", "joblib"]
            },
            "verification": {
                "load_predict_passed": True,
                "roundtrip_predictions_exact": True,
                "all_recorded_metrics_match": True,
                "rtol": 1e-6,
                "atol": 1e-7,
            },
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "wall_seconds": time.perf_counter() - start,
        }
    )
    write_json(destination / "metadata.json", record)
    return record


def reconstruct_manifest(
    manifest_path, dataset_path, samples=15, threads=2, blas_threads=1
):
    """Export verified predictions and reconstruct the same sample IDs for every model."""
    import json

    from qel_twin.training.classical_ml import make_features
    from qel_twin.training.noise_dataset import load_noise_dataset, save_predictions_csv
    from qel_twin.visualization.control_center.services import (
        evaluate_run_reconstruction,
    )

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    dataset = load_noise_dataset(dataset_path)
    reference = None
    for name, record in manifest.items():
        root = Path(record["model_path"]).parent
        with np.load(root / "split_indices.npz") as split:
            indices = split["test_idx"]
        if reference is None:
            reference = indices
        np.testing.assert_array_equal(indices, reference)
        with (
            threadpool_limits(limits=threads),
            threadpool_limits(limits=blas_threads, user_api="blas"),
        ):
            model = joblib.load(root / "model.joblib")
            prediction = ensure_prediction_matrix(
                model.predict(make_features(dataset.dynamics[indices], "flatten")),
                len(dataset.parameter_names),
            )
            check_metrics(
                dataset.log10_gamma[indices],
                prediction,
                dataset.parameter_names,
                record["test_metrics"],
            )
            save_predictions_csv(
                root / "test_predictions.csv",
                dataset.log10_gamma[indices],
                prediction,
                dataset.parameter_names,
                indices=indices,
            )
            del model
            metrics_path = root / "metrics.json"
            metrics = (
                json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            )
            metrics.update(
                {
                    "dataset_path": str(Path(dataset_path).resolve()),
                    "model_name": name,
                    "test_metrics": record["test_metrics"],
                }
            )
            write_json(metrics_path, metrics)
            print(f"Reconstructing {name}: {samples} samples", flush=True)
            result = evaluate_run_reconstruction(
                dataset_path=dataset_path, run_dir=root, reconstruction_samples=samples
            )
        record["reconstruction"] = result
        write_json(manifest_path, manifest)
        print(
            f"{name}: reconstruction RMSE={result['trajectory_rmse_mean']:.12f}",
            flush=True,
        )
    return manifest
