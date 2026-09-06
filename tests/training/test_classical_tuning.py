"""Verify fold-local preprocessing and held-out isolation in the tuning pilot."""

import json
from pathlib import Path

import numpy as np
import optuna
import pytest
import yaml
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qel_twin.training import classical_tuning as tuning
from qel_twin.training.noise_dataset import split_dataset_indices


def test_cv_fits_scaler_only_on_fold_training_rows(monkeypatch):
    seen = []
    original = StandardScaler.fit

    def record(self, X, y=None, **kwargs):
        seen.append(np.array(X))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record)
    X = np.arange(24).reshape(12, 2).astype(float)
    folds = [(np.arange(6), np.arange(6, 12)), (np.arange(6, 12), np.arange(6))]
    model = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(2))])
    scores = tuning.cv_mae(model, X, X[:, :1], folds)
    assert scores.shape == (2,)
    assert len(seen) == 2
    for actual, (fit, _) in zip(seen, folds):
        np.testing.assert_array_equal(actual, X[fit])


def test_pilot_refits_train_and_predicts_test_once(tmp_path, monkeypatch):
    n = 30
    X = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
    y = -3 + X.reshape(n, 1) / 100
    data = tmp_path / "data.npz"
    np.savez(
        data,
        expectation_values=X,
        gamma=10**y,
        log10_gamma=y,
        times=[0.0],
        parameter_names=["gamma"],
        metadata_json="{}",
    )
    split = split_dataset_indices(n, seed=1234)
    base = tmp_path / "baseline"
    base.mkdir()
    np.savez(
        base / "split_indices.npz",
        train_idx=split.train,
        val_idx=split.validation,
        test_idx=split.test,
    )
    (base / "metrics.json").write_text(
        json.dumps(
            {
                "dataset_path": str(data),
                "feature_mode": "flatten",
                "seed": 1234,
                "effective_params": {"n_neighbors": 7, "pca_components": 1},
                "requested_params": {"n_estimators": 500},
                "test_metrics": {
                    "mae_log10_mean": 1.0,
                    "rmse_log10_mean": 1.0,
                    "r2_log10_mean": 0.0,
                    "median_relative_error_gamma": 1.0,
                },
            }
        )
    )
    config = {
        "dataset_path": str(data),
        "output_dir": str(tmp_path / "run"),
        "seed": 1234,
        "cv_seed": 2026,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "feature_mode": "flatten",
        "folds": 3,
        "threads": 1,
        "models": {
            "knn": {
                "baseline_run": str(base),
                "trials": 2,
                "space": {"model__n_neighbors": {"type": "int", "low": 1, "high": 3}},
            }
        },
    }
    fits, predictions = [], []
    original_fit, original_predict = Pipeline.fit, Pipeline.predict

    def fit(self, X, y=None, **kwargs):
        fits.append(np.array(X))
        return original_fit(self, X, y, **kwargs)

    def predict(self, X, **kwargs):
        predictions.append(np.array(X))
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(Pipeline, "predict", predict)
    result = tuning.run_pilot(config)
    train_rows = set(split.train.tolist())
    assert all(set(a[:, 0].astype(int)) <= train_rows for a in fits)
    np.testing.assert_array_equal(fits[-1], X[split.train].reshape(-1, 1))
    assert len(predictions) == 10  # 3 baseline + 2*3 search folds + one test
    assert all(set(a[:, 0].astype(int)) <= train_rows for a in predictions[:-1])
    np.testing.assert_array_equal(predictions[-1], X[split.test].reshape(-1, 1))
    assert result["models"]["knn"]["baseline_test_metrics"]["mae_log10_mean"] == 1.0
    assert (tmp_path / "run" / "report.md").exists()
    assert result["models"]["knn"]["search"] == {
        "trial_budget": 2,
        "completed_search_trials": 2,
        "early_stopping_patience": 10,
        "stop_reason": "trial_budget_exhausted",
        "fixed_params": {},
    }
    assert (
        result["models"]["knn"]["best_cv_mae"]
        <= result["models"]["knn"]["baseline_cv_mae"]
    )
    with pytest.raises(FileExistsError):
        tuning.run_pilot(config)


def test_rejects_out_of_scope_model(tmp_path):
    with pytest.raises(ValueError, match="Supported tuning models"):
        tuning.run_pilot({"models": {"ridge": {}}, "output_dir": str(tmp_path / "run")})


@pytest.mark.parametrize(
    ("scores", "expected_trials"),
    [([1.0] * 20, 10), ([1.0] * 9 + [0.5] + [0.5] * 20, 20)],
)
def test_no_improvement_stops_synthetic_search(scores, expected_trials):
    study = optuna.create_study(direction="minimize")
    study.add_trial(optuna.trial.create_trial(value=1.0))
    stopping = tuning.StopAfterNoImprovement(10, study.best_value)
    study.optimize(
        lambda trial: scores[trial.number - 1],
        n_trials=len(scores),
        callbacks=[stopping],
    )
    assert len(study.trials) - 1 == expected_trials
    assert stopping.stopped
    assert study.user_attrs["stop_reason"] == "no_improvement"
    assert study.best_value == min(scores[:expected_trials])


@pytest.mark.parametrize("patience", [0, -1, True, 1.5])
def test_invalid_patience(patience):
    with pytest.raises(ValueError, match="positive integer"):
        tuning.StopAfterNoImprovement(patience, 1.0)


@pytest.mark.parametrize("suffix", ["svr_extended", "xgboost"])
def test_followup_config_estimator_parameters_are_valid(suffix):
    """Check the real YAMLs against estimator APIs without fitting or searching."""
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / f"configs/sweeps/classical_tuning_{suffix}.yaml").read_text()
    )
    name, spec = next(iter(config["models"].items()))
    assert name in tuning.PILOT_MODELS
    model = tuning.build_model(name, seed=config["seed"])
    values = {
        key: setting["choices"][0]
        if setting["type"] == "categorical"
        else setting["low"]
        for key, setting in spec["space"].items()
    }
    sampled = tuning.suggest_params(optuna.trial.FixedTrial(values), spec["space"])
    assert set(sampled) <= set(model.get_params())
    model.set_params(**spec.get("fixed_params", {}), **sampled)
    if name == "xgboost":
        assert model.get_params()["tree_method"] == "hist"
        assert model.get_params()["multi_strategy"] == "multi_output_tree"
