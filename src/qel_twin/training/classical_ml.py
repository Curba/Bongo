from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


PARAM_NAMES = ["gamma_x", "gamma_y", "gamma_z"]

AVAILABLE_MODELS = [
    "dummy_mean",
    "linear",
    "ridge",
    "lasso",
    "elastic_net",
    "bayesian_ridge",
    "huber",
    "sgd",
    "poly_ridge",
    "pls",
    "knn",
    "knn_pca",
    "radius_neighbors",
    "svr_rbf",
    "svr_linear",
    "nusvr",
    "kernel_ridge_rbf",
    "kernel_ridge_poly",
    "gaussian_process",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "adaboost",
    "bagging_trees",
    "chain_ridge",
    "chain_extra_trees",
    "mlp",
    "mlp_pca",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_run_id(seed: int, tag: str | None = None) -> str:
    parts = [now_string(), f"seed{seed}"]

    if tag:
        clean_tag = tag.replace(" ", "_").replace("/", "_")
        parts.append(clean_tag)

    return "_".join(parts)


def safe_name(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def infer_dataset_id(dataset_path: str | Path, metadata: dict[str, Any] | None = None) -> str:
    if metadata:
        if metadata.get("dataset_id"):
            return safe_name(str(metadata["dataset_id"]))

        if metadata.get("name"):
            return safe_name(str(metadata["name"]))

    return safe_name(Path(dataset_path).stem)


def create_run_dir(
    output_root: str | Path,
    dataset_id: str,
    model_name: str,
    run_id: str,
) -> Path:
    run_dir = Path(output_root) / dataset_id / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
        raise ValueError(f"Expected target shape (N, 3), got {y_log.shape}")

    if X.shape[0] != y_log.shape[0]:
        raise ValueError(
            f"X and target sample count mismatch: {X.shape[0]} vs {y_log.shape[0]}"
        )

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


def make_features(X: np.ndarray, feature_mode: str = "flatten") -> np.ndarray:
    """
    Convert X from (N, C, L, T) to a 2D matrix for sklearn.

    flatten:
        Uses all trajectory values.
        Example: (N, 3, 5, 101) -> (N, 1515)

    stats:
        Uses compact summary statistics.

    both:
        Concatenates flatten + stats.
    """
    n_samples = X.shape[0]
    flat = X.reshape(n_samples, -1).astype(np.float32)

    if feature_mode == "flatten":
        return flat

    mean = X.mean(axis=-1)
    std = X.std(axis=-1)
    xmin = X.min(axis=-1)
    xmax = X.max(axis=-1)
    first = X[..., 0]
    last = X[..., -1]
    delta = last - first
    abs_mean = np.abs(X).mean(axis=-1)
    abs_max = np.abs(X).max(axis=-1)

    # simple slope estimate between first and last time points
    slope = delta / max(1, X.shape[-1] - 1)

    stats = np.concatenate(
        [
            mean.reshape(n_samples, -1),
            std.reshape(n_samples, -1),
            xmin.reshape(n_samples, -1),
            xmax.reshape(n_samples, -1),
            first.reshape(n_samples, -1),
            last.reshape(n_samples, -1),
            delta.reshape(n_samples, -1),
            slope.reshape(n_samples, -1),
            abs_mean.reshape(n_samples, -1),
            abs_max.reshape(n_samples, -1),
        ],
        axis=1,
    ).astype(np.float32)

    if feature_mode == "stats":
        return stats

    if feature_mode == "both":
        return np.concatenate([flat, stats], axis=1).astype(np.float32)

    raise ValueError("feature_mode must be one of: flatten, stats, both")


def safe_component_count(
    requested: int,
    n_samples: int,
    n_features: int,
    min_value: int = 1,
) -> int:
    return int(max(min_value, min(requested, n_samples - 1, n_features)))


def build_model(
    model_name: str,
    seed: int,
    n_neighbors: int = 7,
    pca_components: int = 64,
    n_estimators: int = 500,
) -> Any:
    model_name = model_name.lower()

    if model_name == "dummy_mean":
        return DummyRegressor(strategy="mean")

    if model_name == "linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    if model_name == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=seed)),
            ]
        )

    if model_name == "lasso":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        Lasso(
                            alpha=1e-4,
                            max_iter=20000,
                            random_state=seed,
                        )
                    ),
                ),
            ]
        )

    if model_name == "elastic_net":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        ElasticNet(
                            alpha=1e-4,
                            l1_ratio=0.3,
                            max_iter=20000,
                            random_state=seed,
                        )
                    ),
                ),
            ]
        )

    if model_name == "bayesian_ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(BayesianRidge())),
            ]
        )

    if model_name == "huber":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(HuberRegressor(max_iter=2000))),
            ]
        )

    if model_name == "sgd":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        SGDRegressor(
                            loss="squared_error",
                            penalty="elasticnet",
                            alpha=1e-4,
                            max_iter=5000,
                            tol=1e-4,
                            random_state=seed,
                        )
                    ),
                ),
            ]
        )

    if model_name == "poly_ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=10.0, random_state=seed)),
            ]
        )

    if model_name == "pls":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", PLSRegression(n_components=max(1, min(16, pca_components)))),
            ]
        )

    if model_name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=n_neighbors,
                        weights="distance",
                    ),
                ),
            ]
        )

    if model_name == "knn_pca":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=n_neighbors,
                        weights="distance",
                    ),
                ),
            ]
        )

    if model_name == "radius_neighbors":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "model",
                    RadiusNeighborsRegressor(
                        radius=5.0,
                        weights="distance",
                    ),
                ),
            ]
        )

    if model_name == "svr_rbf":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "model",
                    MultiOutputRegressor(
                        SVR(
                            kernel="rbf",
                            C=10.0,
                            epsilon=0.03,
                            gamma="scale",
                        )
                    ),
                ),
            ]
        )

    if model_name == "svr_linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        LinearSVR(
                            C=1.0,
                            epsilon=0.03,
                            max_iter=20000,
                            random_state=seed,
                        )
                    ),
                ),
            ]
        )

    if model_name == "nusvr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "model",
                    MultiOutputRegressor(
                        NuSVR(
                            kernel="rbf",
                            C=10.0,
                            nu=0.5,
                            gamma="scale",
                        )
                    ),
                ),
            ]
        )

    if model_name == "kernel_ridge_rbf":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "model",
                    KernelRidge(
                        alpha=1e-2,
                        kernel="rbf",
                        gamma=None,
                    ),
                ),
            ]
        )

    if model_name == "kernel_ridge_poly":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "model",
                    KernelRidge(
                        alpha=1e-2,
                        kernel="polynomial",
                        degree=2,
                    ),
                ),
            ]
        )

    if model_name == "gaussian_process":
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
            noise_level=1e-3
        )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=min(pca_components, 32), random_state=seed)),
                (
                    "model",
                    MultiOutputRegressor(
                        GaussianProcessRegressor(
                            kernel=kernel,
                            alpha=1e-6,
                            normalize_y=True,
                            random_state=seed,
                        )
                    ),
                ),
            ]
        )

    if model_name == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "gradient_boosting":
        base = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            random_state=seed,
        )
        return MultiOutputRegressor(base)

    if model_name == "hist_gradient_boosting":
        base = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=31,
            random_state=seed,
        )
        return MultiOutputRegressor(base)

    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=6,
            tree_method="hist",
            multi_strategy="multi_output_tree",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "adaboost":
        base = AdaBoostRegressor(
            n_estimators=400,
            learning_rate=0.03,
            random_state=seed,
        )
        return MultiOutputRegressor(base)

    if model_name == "bagging_trees":
        base_tree = DecisionTreeRegressor(
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
        )
        return BaggingRegressor(
            estimator=base_tree,
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "chain_ridge":
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=seed)),
            ]
        )
        return RegressorChain(
            estimator=base,
            order=[0, 1, 2],
            random_state=seed,
        )

    if model_name == "chain_extra_trees":
        base = ExtraTreesRegressor(
            n_estimators=max(100, n_estimators // 3),
            random_state=seed,
            n_jobs=-1,
        )
        return RegressorChain(
            estimator=base,
            order=[0, 1, 2],
            random_state=seed,
        )

    if model_name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_name == "mlp_pca":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unknown model '{model_name}'. Available models: {', '.join(AVAILABLE_MODELS)}"
    )


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


def save_training_history_for_classical_model(
    output_path: str | Path,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    fit_seconds: float,
) -> None:
    """
    Classical models do not have epochs, but the dashboard expects an optional
    training_history.csv. We save one row so classical and NN runs share a format.
    """
    row = {
        "epoch": 1,
        "train_loss": train_metrics["rmse_log10_mean"],
        "val_loss": val_metrics["rmse_log10_mean"],
        "test_loss": test_metrics["rmse_log10_mean"],
        "train_mae": train_metrics["mae_log10_mean"],
        "val_mae": val_metrics["mae_log10_mean"],
        "test_mae": test_metrics["mae_log10_mean"],
        "train_r2": train_metrics["r2_log10_mean"],
        "val_r2": val_metrics["r2_log10_mean"],
        "test_r2": test_metrics["r2_log10_mean"],
        "fit_seconds": fit_seconds,
    }

    pd.DataFrame([row]).to_csv(output_path, index=False)


def train_classical_model(
    dataset_path: str | Path,
    output_dir: str | Path = "outputs/ml_runs",
    model_name: str = "hist_gradient_boosting",
    feature_mode: str = "flatten",
    seed: int = 1234,
    test_size: float = 0.15,
    val_size: float = 0.15,
    n_neighbors: int = 7,
    pca_components: int = 64,
    n_estimators: int = 500,
    run_tag: str | None = None,
) -> dict[str, Any]:
    set_seed(seed)

    dataset_path = Path(dataset_path)

    X, y_log, dataset_metadata = load_p3_npz(dataset_path)
    dataset_id = infer_dataset_id(dataset_path, dataset_metadata)
    run_id = make_run_id(seed=seed, tag=run_tag)

    run_dir = create_run_dir(
        output_root=output_dir,
        dataset_id=dataset_id,
        model_name=model_name,
        run_id=run_id,
    )

    X_features = make_features(X, feature_mode=feature_mode)

    indices = np.arange(X_features.shape[0])

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

    X_train = X_features[train_idx]
    y_train = y_log[train_idx]

    X_val = X_features[val_idx]
    y_val = y_log[val_idx]

    X_test = X_features[test_idx]
    y_test = y_log[test_idx]

    safe_pca_components = safe_component_count(
        requested=pca_components,
        n_samples=X_train.shape[0],
        n_features=X_train.shape[1],
    )

    safe_n_neighbors = min(n_neighbors, max(1, X_train.shape[0]))

    model = build_model(
        model_name=model_name,
        seed=seed,
        n_neighbors=safe_n_neighbors,
        pca_components=safe_pca_components,
        n_estimators=n_estimators,
    )

    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "model_name": model_name,
        "model_family": "classical_ml",
        "feature_mode": feature_mode,
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "n_neighbors_requested": n_neighbors,
        "n_neighbors_effective": safe_n_neighbors,
        "pca_components_requested": pca_components,
        "pca_components_effective": safe_pca_components,
        "n_estimators": n_estimators,
        "dataset_metadata": dataset_metadata,
    }

    write_json(run_dir / "run_metadata.json", run_metadata)

    print("=" * 88)
    print("QEL Twin Classical ML Training")
    print("=" * 88)
    print(f"Dataset ID:      {dataset_id}")
    print(f"Dataset path:    {dataset_path}")
    print(f"Model:           {model_name}")
    print(f"Run ID:          {run_id}")
    print(f"Run dir:         {run_dir}")
    print(f"Raw X shape:     {X.shape}")
    print(f"Feature mode:    {feature_mode}")
    print(f"Feature shape:   {X_features.shape}")
    print(f"Target shape:    {y_log.shape}")
    print(f"Train/val/test:  {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"PCA components:  {pca_components} -> {safe_pca_components}")
    print(f"KNN neighbors:   {n_neighbors} -> {safe_n_neighbors}")
    print("=" * 88)

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_train_pred = np.asarray(model.predict(X_train), dtype=np.float32)
    y_val_pred = np.asarray(model.predict(X_val), dtype=np.float32)
    y_test_pred = np.asarray(model.predict(X_test), dtype=np.float32)
    predict_seconds = time.perf_counter() - predict_start

    train_metrics = evaluate_predictions(y_train, y_train_pred)
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    result: dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "metadata": dataset_metadata,
        "raw_X_shape": list(X.shape),
        "feature_shape": list(X_features.shape),
        "y_log_shape": list(y_log.shape),
        "model_name": model_name,
        "model_family": "classical_ml",
        "feature_mode": feature_mode,
        "seed": seed,
        "timing": {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
        },
        "requested_params": {
            "test_size": test_size,
            "val_size": val_size,
            "n_neighbors": n_neighbors,
            "pca_components": pca_components,
            "n_estimators": n_estimators,
        },
        "effective_params": {
            "n_neighbors": safe_n_neighbors,
            "pca_components": safe_pca_components,
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

    joblib.dump(model, run_dir / "model.joblib")

    np.savez(
        run_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    save_predictions_csv(
        run_dir / "train_predictions.csv",
        y_train,
        y_train_pred,
        indices=train_idx,
    )
    save_predictions_csv(
        run_dir / "val_predictions.csv",
        y_val,
        y_val_pred,
        indices=val_idx,
    )
    save_predictions_csv(
        run_dir / "test_predictions.csv",
        y_test,
        y_test_pred,
        indices=test_idx,
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