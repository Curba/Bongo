from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qel_twin.training.classical_ml import (
    AVAILABLE_MODELS,
    build_model,
    create_run_dir,
    infer_dataset_id,
    make_features,
    make_run_id,
    safe_component_count,
    save_training_history_for_classical_model,
    set_seed,
    write_json,
)
from qel_twin.training.noise_dataset import (
    ensure_prediction_matrix,
    evaluate_predictions,
    load_noise_dataset,
    save_predictions_csv,
    split_dataset_indices,
)


def build_dynamic_model(
    model_name: str,
    *,
    seed: int,
    num_targets: int,
    n_neighbors: int,
    pca_components: int,
    n_estimators: int,
) -> Any:
    """Reuse Bongo's existing model zoo, fixing only dynamic RegressorChain order."""
    normalized = model_name.lower()

    if normalized == "chain_ridge":
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=seed)),
            ]
        )
        return RegressorChain(
            estimator=base,
            order=list(range(num_targets)),
            random_state=seed,
        )

    if normalized == "chain_extra_trees":
        base = ExtraTreesRegressor(
            n_estimators=max(100, n_estimators // 3),
            random_state=seed,
            n_jobs=-1,
        )
        return RegressorChain(
            estimator=base,
            order=list(range(num_targets)),
            random_state=seed,
        )

    return build_model(
        model_name=normalized,
        seed=seed,
        n_neighbors=n_neighbors,
        pca_components=pca_components,
        n_estimators=n_estimators,
    )


def train_classical_noise_model(
    dataset_path: str | Path,
    output_dir: str | Path = "outputs/noise_ml_runs",
    model_name: str = "extra_trees",
    feature_mode: str = "flatten",
    seed: int = 1234,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    n_neighbors: int = 7,
    pca_components: int = 64,
    n_estimators: int = 500,
    run_tag: str | None = None,
) -> dict[str, Any]:
    """Train any existing Bongo classical model directly on qel-ml ``(N,O,T)`` data."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {', '.join(AVAILABLE_MODELS)}"
        )

    set_seed(seed)
    dataset_path = Path(dataset_path)

    dataset = load_noise_dataset(dataset_path)
    X = dataset.dynamics
    y_log = dataset.log10_gamma
    parameter_names = dataset.parameter_names
    num_targets = y_log.shape[1]

    dataset_id = infer_dataset_id(dataset_path, dataset.metadata)
    run_id = make_run_id(seed=seed, tag=run_tag)

    run_dir = create_run_dir(
        output_root=output_dir,
        dataset_id=dataset_id,
        model_name=model_name,
        run_id=run_id,
    )

    # Existing make_features works for (N,O,T): it always preserves N,
    # uses the last axis as time, and flattens the remaining feature axes.
    X_features = make_features(X, feature_mode=feature_mode)

    split = split_dataset_indices(
        X.shape[0],
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    X_train = X_features[split.train]
    y_train = y_log[split.train]
    X_val = X_features[split.validation]
    y_val = y_log[split.validation]
    X_test = X_features[split.test]
    y_test = y_log[split.test]

    safe_pca_components = safe_component_count(
        requested=pca_components,
        n_samples=X_train.shape[0],
        n_features=X_train.shape[1],
    )
    safe_n_neighbors = min(n_neighbors, max(1, X_train.shape[0]))

    model = build_dynamic_model(
        model_name,
        seed=seed,
        num_targets=num_targets,
        n_neighbors=safe_n_neighbors,
        pca_components=safe_pca_components,
        n_estimators=n_estimators,
    )

    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_layout": "N,O,T",
        "model_name": model_name,
        "model_family": "classical_ml",
        "feature_mode": feature_mode,
        "seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "n_neighbors_requested": n_neighbors,
        "n_neighbors_effective": safe_n_neighbors,
        "pca_components_requested": pca_components,
        "pca_components_effective": safe_pca_components,
        "n_estimators": n_estimators,
        "num_targets": num_targets,
        "parameter_names": list(parameter_names),
        "dataset_metadata": dataset.metadata,
    }
    write_json(run_dir / "run_metadata.json", run_metadata)

    print("=" * 88)
    print("Bongo Classical ML - qel-ml NoiseDataset format")
    print("=" * 88)
    print(f"Dataset ID:      {dataset_id}")
    print(f"Dataset path:    {dataset_path}")
    print(f"Model:           {model_name}")
    print(f"Run ID:          {run_id}")
    print(f"Run dir:         {run_dir}")
    print(f"Dynamics shape:  {X.shape}  (N,O,T)")
    print(f"Feature mode:    {feature_mode}")
    print(f"Feature shape:   {X_features.shape}")
    print(f"Target shape:    {y_log.shape}")
    print(f"Parameters:      {parameter_names}")
    print(
        "Train/val/test:  "
        f"{len(split.train)} / {len(split.validation)} / {len(split.test)}"
    )
    print("=" * 88)

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_train_pred = ensure_prediction_matrix(
        model.predict(X_train), num_targets
    )
    y_val_pred = ensure_prediction_matrix(
        model.predict(X_val), num_targets
    )
    y_test_pred = ensure_prediction_matrix(
        model.predict(X_test), num_targets
    )
    predict_seconds = time.perf_counter() - predict_start

    train_metrics = evaluate_predictions(
        y_train, y_train_pred, parameter_names
    )
    val_metrics = evaluate_predictions(
        y_val, y_val_pred, parameter_names
    )
    test_metrics = evaluate_predictions(
        y_test, y_test_pred, parameter_names
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "metadata": dataset.metadata,
        "raw_X_shape": list(X.shape),
        "feature_shape": list(X_features.shape),
        "y_log_shape": list(y_log.shape),
        "parameter_names": list(parameter_names),
        "model_name": model_name,
        "model_family": "classical_ml",
        "feature_mode": feature_mode,
        "seed": seed,
        "timing": {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
        },
        "requested_params": {
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "n_neighbors": n_neighbors,
            "pca_components": pca_components,
            "n_estimators": n_estimators,
        },
        "effective_params": {
            "n_neighbors": safe_n_neighbors,
            "pca_components": safe_pca_components,
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

    joblib.dump(model, run_dir / "model.joblib")

    np.savez(
        run_dir / "split_indices.npz",
        train_idx=split.train,
        val_idx=split.validation,
        test_idx=split.test,
    )

    save_predictions_csv(
        run_dir / "train_predictions.csv",
        y_train,
        y_train_pred,
        parameter_names,
        indices=split.train,
    )
    save_predictions_csv(
        run_dir / "val_predictions.csv",
        y_val,
        y_val_pred,
        parameter_names,
        indices=split.validation,
    )
    save_predictions_csv(
        run_dir / "test_predictions.csv",
        y_test,
        y_test_pred,
        parameter_names,
        indices=split.test,
    )

    save_training_history_for_classical_model(
        run_dir / "training_history.csv",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        fit_seconds=fit_seconds,
    )

    write_json(run_dir / "metrics.json", result)

    print("\nTrain metrics:")
    print(json.dumps(train_metrics, indent=2))
    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nTiming:")
    print(f"Fit seconds:     {fit_seconds:.3f}")
    print(f"Predict seconds: {predict_seconds:.3f}")
    print(f"\nSaved run to: {run_dir}")

    return result


__all__ = ["train_classical_noise_model"]
