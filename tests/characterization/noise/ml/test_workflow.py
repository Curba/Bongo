# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for preprocessing, regressors, persistence, and external inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from mqt.yaqs.characterization.noise.ml import (
    DatasetSplit,
    MLNoiseCharacterizer,
    ModelName,
    NoiseDataset,
    NoiseExperiment,
    TrainedNoiseModel,
    TrajectoryPreprocessor,
    generate_noise_dataset,
    simulate_pauli_xyz,
    split_dataset_indices,
    train_noise_model,
)
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.simulation_parameters import (
    AnalogSimParams,
    Observable,
)
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.simulator import Simulator

if TYPE_CHECKING:
    from pathlib import Path


def _array_dataset(num_samples: int = 8) -> NoiseDataset:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(num_samples, 3, 5))
    log_gamma = rng.uniform(-3, -1, size=(num_samples, 1))
    return NoiseDataset(
        expectation_values=values,
        gamma=10.0**log_gamma,
        log10_gamma=log_gamma,
        times=np.linspace(0, 0.4, 5),
        parameter_names=("gamma",),
        metadata={"num_sites": 2, "gamma_min": 1e-3, "gamma_max": 1e-1},
    )


def test_split_is_disjoint_and_reproducible() -> None:
    """Sample-level split prevents copies of one simulation entering two sets."""
    first = split_dataset_indices(10, seed=9)
    second = split_dataset_indices(10, seed=9)
    assert np.array_equal(first.train, second.train)
    assert not set(first.train) & set(first.validation)
    assert not set(first.train) & set(first.test)
    assert not set(first.validation) & set(first.test)


def test_preprocessor_fits_train_only_and_has_explicit_model_shapes() -> None:
    """External values do not alter saved normalization statistics."""
    training = np.arange(60, dtype=float).reshape(4, 3, 5)
    external = np.full((3, 5), 10000.0)
    mlp = TrajectoryPreprocessor.fit(training, model_name="mlp")
    cnn = TrajectoryPreprocessor.fit(training, model_name="2d_cnn")
    expected_mean = training.mean(axis=(0, 2), keepdims=True)
    assert np.array_equal(mlp.mean, expected_mean)
    assert mlp.transform(external).shape == (1, 3, 5)
    assert cnn.transform(external).shape == (1, 1, 3, 5)
    assert np.array_equal(mlp.mean, expected_mean)


@pytest.mark.parametrize("model_name", ["mlp", "2d_cnn"])
def test_model_training_checkpoint_and_prediction(model_name: str, tmp_path: Path) -> None:
    """Both regressors train, validate, restore, infer, and exponentiate outputs."""
    dataset = _array_dataset()
    split = DatasetSplit(train=np.arange(5), validation=np.array([5]), test=np.array([6, 7]))
    trained = train_noise_model(
        dataset,
        split,
        model_name=cast("ModelName", model_name),
        parameterization="super_global",
        seed=3,
        max_epochs=2,
        patience=1,
        batch_size=2,
        device="cpu",
    )
    gamma, log_gamma = trained.predict(dataset.expectation_values[split.test])
    assert gamma.shape == (2, 1)
    assert log_gamma.shape == (2, 1)
    assert np.allclose(gamma, 10.0**log_gamma)
    checkpoint = tmp_path / f"{model_name}.npz"
    trained.save(checkpoint)
    restored = TrainedNoiseModel.load(checkpoint)
    restored_gamma, restored_log = restored.predict(dataset.expectation_values[split.test])
    assert np.allclose(restored_log, log_gamma)
    assert np.allclose(restored_gamma, gamma)
    assert np.array_equal(restored.preprocessor.mean, trained.preprocessor.mean)


def test_end_to_end_external_trajectory_characterization() -> None:
    """A held-out synthetic trajectory exercises the production external-data API."""
    experiment = NoiseExperiment(
        state=State(2, initial="zeros", representation="density_matrix"),
        hamiltonian=Hamiltonian.ising(2, J=1.0, g=0.5),
        sim_params=AnalogSimParams(
            observables=[Observable("z", sites=0), Observable("x", sites=1)],
            elapsed_time=0.2,
            dt=0.1,
            num_traj=1,
            preset="exact",
        ),
        simulator=Simulator(parallel=False, show_progress=False),
    )
    training_dataset = generate_noise_dataset(
        experiment,
        num_samples=9,
        parameterization="super_global",
        seed=12,
    )
    split = split_dataset_indices(9, seed=18)
    trained = train_noise_model(
        training_dataset,
        split,
        model_name="mlp",
        parameterization="super_global",
        seed=5,
        max_epochs=2,
        patience=1,
        batch_size=3,
        device="cpu",
    )

    # This parameter was not sampled from training_dataset: it stands in for external data.
    original, times = simulate_pauli_xyz(experiment, [0.017], parameterization="super_global")
    result = MLNoiseCharacterizer(experiment, trained).characterize(original, times=times)
    assert result.predicted_gamma.shape == (1,)
    assert result.parameter_names == ("gamma",)
    assert result.original_dynamics.shape == (2, len(times))
    assert result.reconstructed_dynamics.shape == (2, len(times))
    assert np.isfinite(cast("float", result.metrics["trajectory_mae"]))
    assert np.isfinite(cast("float", result.metrics["trajectory_rmse"]))
