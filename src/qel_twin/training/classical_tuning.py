"""Training-only CV pilot; held-out predictions occur after all searches finish."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from threadpoolctl import threadpool_limits

from qel_twin.training.classical_ml import build_model, make_features, write_json
from qel_twin.training.noise_dataset import (
    ensure_prediction_matrix,
    evaluate_predictions,
    load_noise_dataset,
    save_predictions_csv,
    split_dataset_indices,
)

PILOT_MODELS = {"knn", "svr_rbf", "hist_gradient_boosting", "xgboost"}


class StopAfterNoImprovement:
    """Stop a minimizing study after patience completed trials without a new best.

    Initialize from the baseline score. Ties count as non-improvements; each
    strictly lower score resets patience. No fold or estimator fit is interrupted.
    """

    def __init__(self, patience: int, best_value: float):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError(
                "early_stopping_patience must be a positive integer or null"
            )
        self.patience = patience
        self.best_value = best_value
        self.stale_trials = 0
        self.stopped = False

    def __call__(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if trial.value < self.best_value:
            self.best_value = trial.value
            self.stale_trials = 0
        else:
            self.stale_trials += 1
        if self.stale_trials >= self.patience:
            self.stopped = True
            study.set_user_attr("stop_reason", "no_improvement")
            study.set_user_attr("early_stopping_patience", self.patience)
            print(
                f"{study.study_name}: stopping after {self.stale_trials} trials "
                f"without improvement; best CV MAE={self.best_value:.9f}",
                flush=True,
            )
            study.stop()


def suggest_params(trial, space):
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif kind in {"int", "float"}:
            method = trial.suggest_int if kind == "int" else trial.suggest_float
            params[name] = method(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        else:
            raise ValueError(f"Unknown distribution: {kind}")
    return params


def cv_mae(model, X_train, y_train, folds):
    """Clone the entire pipeline per fold, including scaling and PCA."""
    return -cross_val_score(
        model,
        X_train,
        y_train,
        cv=folds,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        error_score="raise",
    )


def run_pilot(config):
    start = time.perf_counter()
    if not config["models"] or set(config["models"]) - PILOT_MODELS:
        raise ValueError(f"Supported tuning models: {sorted(PILOT_MODELS)}")
    patience = config.get("early_stopping_patience", 10)
    if patience is not None:
        StopAfterNoImprovement(
            patience, float("inf")
        )  # Validate before creating artifacts.
    for spec in config["models"].values():
        if set(spec.get("fixed_params", {})) & set(spec["space"]):
            raise ValueError("fixed_params and search space must not overlap")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    dataset = load_noise_dataset(config["dataset_path"])
    split = split_dataset_indices(
        len(dataset.dynamics),
        seed=config["seed"],
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
    )
    X_train = make_features(dataset.dynamics[split.train], config["feature_mode"])
    y_train = dataset.log10_gamma[split.train]
    folds = list(
        KFold(
            n_splits=config["folds"], shuffle=True, random_state=config["cv_seed"]
        ).split(X_train)
    )
    np.savez(
        output / "split_indices.npz",
        train_idx=split.train,
        val_idx=split.validation,
        test_idx=split.test,
    )
    write_json(
        output / "fold_indices.json",
        [
            {"fit": split.train[a].tolist(), "score": split.train[b].tolist()}
            for a, b in folds
        ],
    )
    report = {
        "config": config,
        "dataset_sha256": hashlib.sha256(
            Path(config["dataset_path"]).read_bytes()
        ).hexdigest(),
        "versions": {
            p: version(p) for p in ["optuna", "scikit-learn", "numpy", "scipy"]
        },
        "platform": platform.platform(),
        "split_sizes": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "models": {},
    }
    fitted = {}
    if "xgboost" in config["models"]:
        report["versions"]["xgboost"] = version("xgboost")
    with (
        threadpool_limits(limits=config["threads"]),
        threadpool_limits(
            limits=config.get("blas_threads", config["threads"]), user_api="blas"
        ),
    ):
        for name, spec in config["models"].items():
            model_start = time.perf_counter()
            baseline_dir = Path(spec["baseline_run"])
            baseline = json.loads((baseline_dir / "metrics.json").read_text())
            with np.load(baseline_dir / "split_indices.npz") as saved:
                for key, actual in [
                    ("train_idx", split.train),
                    ("val_idx", split.validation),
                    ("test_idx", split.test),
                ]:
                    if not np.array_equal(saved[key], actual):
                        raise ValueError(f"Baseline split mismatch: {name}/{key}")
            if (
                Path(baseline["dataset_path"]).resolve()
                != Path(config["dataset_path"]).resolve()
                or baseline["feature_mode"] != config["feature_mode"]
                or baseline["seed"] != config["seed"]
            ):
                raise ValueError("Baseline dataset/features/seed mismatch")
            params = baseline["effective_params"]
            model = build_model(
                name,
                seed=config["seed"],
                n_neighbors=params["n_neighbors"],
                pca_components=params["pca_components"],
                n_estimators=baseline["requested_params"]["n_estimators"],
            )
            model.set_params(**spec.get("fixed_params", {}))
            if name == "xgboost":
                model.set_params(n_jobs=config["threads"])
            cv_start = time.perf_counter()
            baseline_scores = cv_mae(model, X_train, y_train, folds)
            baseline_seconds = time.perf_counter() - cv_start
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=config["seed"]),
                study_name=name,
                storage=f"sqlite:///{output / 'studies.sqlite3'}",
            )
            # Include the exact untuned model, even where defaults are outside distributions.
            study.add_trial(
                optuna.trial.create_trial(
                    value=float(baseline_scores.mean()),
                    user_attrs={"baseline": True, "fold_mae": baseline_scores.tolist()},
                )
            )

            def objective(trial, model=model, spec=spec):
                candidate = clone(model).set_params(
                    **suggest_params(trial, spec["space"])
                )
                scores = cv_mae(candidate, X_train, y_train, folds)
                trial.set_user_attr("fold_mae", scores.tolist())
                return float(scores.mean())

            search_start = time.perf_counter()
            stopping = (
                StopAfterNoImprovement(patience, study.best_value)
                if patience is not None
                else None
            )
            study.optimize(
                objective,
                n_trials=spec["trials"],
                callbacks=[stopping] if stopping is not None else [],
            )
            search_seconds = time.perf_counter() - search_start
            best = study.best_trial
            chosen = clone(model).set_params(**best.params)
            refit_start = time.perf_counter()
            chosen.fit(X_train, y_train)
            refit_seconds = time.perf_counter() - refit_start
            model_dir = output / name
            model_dir.mkdir()
            joblib.dump(chosen, model_dir / "model.joblib")
            study.trials_dataframe().to_csv(model_dir / "trials.csv", index=False)
            fitted[name] = chosen
            report["models"][name] = {
                "best_params": best.params,
                "selected_params": {
                    key: chosen.get_params()[key] for key in spec["space"]
                },
                "baseline_selected": best.number == 0,
                "search": {
                    "trial_budget": spec["trials"],
                    "completed_search_trials": len(study.trials) - 1,
                    "early_stopping_patience": patience,
                    "stop_reason": (
                        "no_improvement"
                        if stopping is not None and stopping.stopped
                        else "trial_budget_exhausted"
                    ),
                    "fixed_params": spec.get("fixed_params", {}),
                },
                "baseline_cv_mae": float(baseline_scores.mean()),
                "baseline_fold_mae": baseline_scores.tolist(),
                "best_cv_mae": best.value,
                "best_fold_mae": best.user_attrs["fold_mae"],
                "baseline_test_metrics": baseline["test_metrics"],
                "timing": {
                    "baseline_cv_seconds": baseline_seconds,
                    "search_seconds": search_seconds,
                    "refit_seconds": refit_seconds,
                    "model_seconds": time.perf_counter() - model_start,
                },
            }
            write_json(output / "search_results.json", report)
        # All choices are frozen before any test prediction. Validation is never used.
        X_test = make_features(dataset.dynamics[split.test], config["feature_mode"])
        for name, chosen in fitted.items():
            pred = ensure_prediction_matrix(
                chosen.predict(X_test), len(dataset.parameter_names)
            )
            report["models"][name]["tuned_test_metrics"] = evaluate_predictions(
                dataset.log10_gamma[split.test], pred, dataset.parameter_names
            )
            save_predictions_csv(
                output / name / "test_predictions.csv",
                dataset.log10_gamma[split.test],
                pred,
                dataset.parameter_names,
                indices=split.test,
            )
    report["wall_seconds"] = time.perf_counter() - start
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_report(report))
    return report


def render_report(report):
    """Render recorded metrics without loading models or evaluating data again."""
    lines = [
        "# Classical tuning pilot results",
        "",
        (
            f"{report['config']['folds']}-fold training-only CV; "
            f"split sizes: {report['split_sizes']}. "
            "Validation is unused. Baseline test metrics are historical; "
            "chosen models were evaluated once."
        ),
        "",
        "| Model | Baseline CV MAE | Selected CV MAE | Reduction | Wall seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["models"].items():
        baseline, best = result["baseline_cv_mae"], result["best_cv_mae"]
        lines.append(
            f"| {name} | {baseline:.9f} | {best:.9f} | "
            f"{100 * (baseline - best) / baseline:.2f}% | "
            f"{result['timing']['model_seconds']:.3f} |"
        )
    lines += [
        "",
        (
            "| Model | Version | mae_log10_mean | rmse_log10_mean | "
            "r2_log10_mean | median_relative_error_gamma |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    keys = [
        "mae_log10_mean",
        "rmse_log10_mean",
        "r2_log10_mean",
        "median_relative_error_gamma",
    ]
    for name, result in report["models"].items():
        for label, key in [
            ("Baseline", "baseline_test_metrics"),
            ("Selected", "tuned_test_metrics"),
        ]:
            values = " | ".join(f"{result[key][metric]:.9f}" for metric in keys)
            lines.append(f"| {name} | {label} | {values} |")
    lines += ["", "## Selected hyperparameters", ""]
    for name, result in report["models"].items():
        params = result.get("selected_params", result["best_params"])
        lines += [
            f"**{name}**"
            + (" (baseline retained)" if result["baseline_selected"] else ""),
            "",
            "```json",
            json.dumps(params, indent=2),
            "```",
            "",
        ]
    lines += [
        f"Total measured wall time: **{report['wall_seconds']:.3f} seconds**.",
        "",
        "Numerical threads: " + str(report["config"]["threads"]) + ".",
        "",
        "Dataset SHA256: `" + report["dataset_sha256"] + "`.",
        "",
        "Versions: `" + json.dumps(report["versions"], sort_keys=True) + "`.",
        "",
    ]
    return "\n".join(lines)
