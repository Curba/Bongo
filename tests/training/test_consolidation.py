"""Verify refit/serialization and rejection of mismatching recorded metrics."""

import json

import joblib
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor

from qel_twin.training.consolidation import check_metrics, refit_and_save
from qel_twin.training.noise_dataset import evaluate_predictions


def test_refit_roundtrip_and_metric_guard(tmp_path):
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = (-3 + X / 10).astype(np.float32)
    model = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    source = tmp_path / "source.joblib"
    joblib.dump(model, source)
    expected = evaluate_predictions(y, y, ["gamma"])
    record = refit_and_save(
        source, tmp_path / "final", X, y, X, y, ["gamma"], expected, {}
    )
    assert record["verification"]["roundtrip_predictions_exact"]
    assert (tmp_path / "final" / "metadata.json").exists()
    np.testing.assert_array_equal(joblib.load(record["model_path"]).predict(X), y)
    with pytest.raises(ValueError, match="Metric mismatch"):
        check_metrics(y, y + 0.1, ["gamma"], expected)


def test_reconstruction_manifest_exports_verified_predictions(tmp_path, monkeypatch):
    from qel_twin.training.consolidation import reconstruct_manifest
    from qel_twin.visualization.control_center import services

    X = np.arange(10, dtype=np.float32).reshape(-1, 1)
    y = (-3 + X / 10).astype(np.float32)
    source = tmp_path / "dataset.npz"
    np.savez(
        source,
        expectation_values=X.reshape(10, 1, 1),
        gamma=10**y,
        log10_gamma=y,
        times=[0.0],
        parameter_names=["gamma"],
        metadata_json="{}",
    )
    model_path = tmp_path / "model.joblib"
    joblib.dump(KNeighborsRegressor(n_neighbors=1).fit(X, y), model_path)
    np.savez(tmp_path / "split_indices.npz", test_idx=np.arange(3))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "knn": {
                    "model_path": str(model_path),
                    "test_metrics": evaluate_predictions(y[:3], y[:3], ["gamma"]),
                }
            }
        )
    )

    def fake_reconstruct(*, dataset_path, run_dir, reconstruction_samples):
        assert dataset_path == source
        assert reconstruction_samples == 3
        assert (run_dir / "test_predictions.csv").exists()
        assert json.loads((run_dir / "metrics.json").read_text())["model_name"] == "knn"
        return {"samples_evaluated": 3, "trajectory_rmse_mean": 0.0}

    monkeypatch.setattr(services, "evaluate_run_reconstruction", fake_reconstruct)
    result = reconstruct_manifest(manifest_path, source, samples=3)
    assert result["knn"]["reconstruction"]["samples_evaluated"] == 3
    assert json.loads(manifest_path.read_text()) == result
