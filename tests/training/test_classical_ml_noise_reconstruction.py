# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests that a classical run's test_predictions.csv flows through reconstruction."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from qel_twin.training.classical_ml_noise import train_classical_noise_model
from qel_twin.visualization.control_center.services import evaluate_run_reconstruction

def _test_split_indices(run_dir: Path) -> list[int]:
    with (run_dir / "test_predictions.csv").open("r", newline="", encoding="utf-8") as handle:
        return [int(row["dataset_index"]) for row in csv.DictReader(handle)]


def test_evaluate_run_reconstruction_updates_metrics_and_writes_samples(
    tmp_path: Path,
    noise_dataset_path: Path,
) -> None:
    """Reconstruction populates reconstruction/ and adds exactly one metrics.json key."""
    train_result = train_classical_noise_model(
        dataset_path=noise_dataset_path,
        output_dir=tmp_path,
        model_name="dummy_mean",
        seed=1234,
    )
    run_dir = Path(train_result["run_dir"])
    assert (run_dir / "test_predictions.csv").exists()

    metrics_before = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    aggregate = evaluate_run_reconstruction(
        dataset_path=noise_dataset_path,
        run_dir=run_dir,
        reconstruction_samples=3,
    )

    # --- reconstruction/summary.json exists and is valid JSON ---
    summary_path = run_dir / "reconstruction" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["samples_evaluated"] == aggregate["samples_evaluated"]

    # --- sample_<index>.npz exists per evaluated test-split index, with expected keys/shapes ---
    with np.load(noise_dataset_path, allow_pickle=False) as raw:
        num_observables, num_times = raw["expectation_values"].shape[1:]
        num_targets = raw["parameter_names"].shape[0]

    evaluated_indices = [row["dataset_index"] for row in summary["samples"]]
    test_indices = _test_split_indices(run_dir)
    assert evaluated_indices == test_indices[: len(evaluated_indices)]
    assert evaluated_indices, "Expected at least one test-split sample to be reconstructed."

    expected_vector_keys = (
        "predicted_log10_gamma",
        "predicted_gamma",
        "true_log10_gamma",
        "true_gamma",
        "parameter_names",
    )
    for dataset_index in evaluated_indices:
        sample_path = run_dir / "reconstruction" / f"sample_{dataset_index}.npz"
        assert sample_path.exists()
        with np.load(sample_path) as sample:
            assert sample["dataset_index"].shape == ()
            assert int(sample["dataset_index"]) == dataset_index
            assert sample["times"].shape == (num_times,)
            assert sample["original"].shape == (num_observables, num_times)
            assert sample["reconstructed"].shape == (num_observables, num_times)
            for key in expected_vector_keys:
                assert sample[key].shape == (num_targets,), f"{key} has unexpected shape {sample[key].shape}"

    # --- metrics.json gained exactly one new top-level key, no existing value changed ---
    metrics_after = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics_after) - set(metrics_before) == {"reconstruction"}
    assert set(metrics_before) - set(metrics_after) == set()
    metrics_after_without_reconstruction = {k: v for k, v in metrics_after.items() if k != "reconstruction"}
    assert metrics_before == metrics_after_without_reconstruction
    assert metrics_after["reconstruction"] == aggregate
