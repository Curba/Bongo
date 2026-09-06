# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Pauli-XYZ parameter mapping and dataset generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from qel_twin.characterization.noise_ml import (
    NoiseDataset,
    NoiseExperiment,
    build_pauli_xyz_noise_model,
    generate_noise_dataset,
    parameter_names,
    sample_pauli_xyz_parameters,
)
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.simulator import Simulator

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def _strengths(model: NoiseModel) -> dict[tuple[str, int], float]:
    return {(process["name"], process["sites"][0]): process["strength"] for process in model.processes}


def test_super_global_strengths() -> None:
    """One parameter controls all six L=2 Pauli processes."""
    model = build_pauli_xyz_noise_model([0.01], parameterization="super_global", num_sites=2)
    assert _strengths(model) == {
        (process, site): 0.01 for site in range(2) for process in ("pauli_x", "pauli_y", "pauli_z")
    }


def test_global_strengths() -> None:
    """Three global parameters remain channel-specific at every site."""
    model = build_pauli_xyz_noise_model([0.01, 0.02, 0.03], parameterization="global", num_sites=2)
    assert _strengths(model) == {
        ("pauli_x", 0): 0.01,
        ("pauli_y", 0): 0.02,
        ("pauli_z", 0): 0.03,
        ("pauli_x", 1): 0.01,
        ("pauli_y", 1): 0.02,
        ("pauli_z", 1): 0.03,
    }


@pytest.mark.parametrize("num_sites", [2, 3])
def test_local_strengths_are_site_major_xyz(num_sites: int) -> None:
    """Distinct rates map to their declared site-major XYZ process."""
    values = np.arange(1, 3 * num_sites + 1, dtype=float) / 100
    model = build_pauli_xyz_noise_model(values, parameterization="local", num_sites=num_sites)
    assert parameter_names("local", num_sites) == tuple(
        f"gamma_{channel}_{site}" for site in range(num_sites) for channel in ("x", "y", "z")
    )
    assert [process["strength"] for process in model.processes] == values.tolist()


def test_sampling_matches_explicit_log_uniform_recipe() -> None:
    """Sampling is reproducible uniform-in-log10, not linear uniform."""
    gamma, log_gamma, names = sample_pauli_xyz_parameters(
        5,
        parameterization="global",
        num_sites=4,
        seed=17,
    )
    expected_log = np.random.default_rng(17).uniform(-3.0, -1.0, size=(5, 3))
    assert names == ("gamma_x", "gamma_y", "gamma_z")
    assert np.array_equal(log_gamma, expected_log)
    assert np.allclose(gamma, 10.0**log_gamma)
    assert np.all((gamma >= 1e-3) & (gamma <= 1e-1))
    repeated = sample_pauli_xyz_parameters(5, parameterization="global", num_sites=4, seed=17)
    assert np.array_equal(repeated[0], gamma)


def test_dataset_generation_uses_result_order_and_times(tmp_path: Path) -> None:
    """A real small Lindblad run produces canonical raw data and round-trips safely."""
    experiment = NoiseExperiment(
        state=State(2, initial="zeros", representation="density_matrix"),
        hamiltonian=Hamiltonian.ising(2, J=1.0, g=0.5),
        sim_params=AnalogSimParams(
            observables=[Observable("z", sites=1), Observable("x", sites=0)],
            elapsed_time=0.1,
            dt=0.1,
            num_traj=1,
            preset="exact",
        ),
        simulator=Simulator(parallel=False, show_progress=False),
        hamiltonian_metadata={"type": "Ising", "J": 1.0, "g": 0.5, "boundary": "open"},
    )
    dataset = generate_noise_dataset(experiment, num_samples=2, parameterization="super_global", seed=8)
    assert dataset.expectation_values.shape == (2, 2, 2)
    assert np.array_equal(dataset.times, np.array([0.0, 0.1]))
    assert dataset.metadata["observables"] == [
        {"gate": "z", "sites": [1]},
        {"gate": "x", "sites": [0]},
    ]
    assert dataset.metadata["initial_state"] == {"preset": "zeros"}
    assert dataset.metadata["hamiltonian"] == {
        "type": "Ising",
        "J": 1.0,
        "g": 0.5,
        "boundary": "open",
    }
    simulation_metadata = dataset.metadata["simulation"]
    assert isinstance(simulation_metadata, dict)
    assert simulation_metadata["evolution_mode"] == "tdvp"
    destination = tmp_path / "dataset.npz"
    dataset.save(destination)
    restored = NoiseDataset.load(destination)
    assert np.array_equal(restored.expectation_values, dataset.expectation_values)
    assert restored.metadata == dataset.metadata
