# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pauli-XYZ datasets and forward simulations for ML noise characterization."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.result import Result
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.simulator import Simulator

Parameterization = Literal["super_global", "global", "local"]

_PAULI_CHANNELS = ("x", "y", "z")
_PAULI_PROCESSES = ("pauli_x", "pauli_y", "pauli_z")
DEFAULT_GAMMA_MIN = 1e-3
DEFAULT_GAMMA_MAX = 1e-1
DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class NoiseExperiment:
    """YAQS-native physical configuration shared by generation and reconstruction.

    Attributes:
        state: Initial state, including the selected analog representation/backend.
        hamiltonian: Hamiltonian used for every forward simulation.
        sim_params: Analog settings and observables used for every simulation.
        simulator: Configured YAQS simulator. Dataset samples are still orchestrated serially;
            this object controls only YAQS-native trajectory parallelism.
        hamiltonian_metadata: Optional JSON-safe description of the Hamiltonian configuration.
    """

    state: State
    hamiltonian: Hamiltonian
    sim_params: AnalogSimParams
    simulator: Simulator
    hamiltonian_metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate the immutable experiment boundary.

        Raises:
            ValueError: If state/Hamiltonian lengths differ or observables are empty.
        """
        if self.state.length != self.hamiltonian.length:
            msg = f"State length {self.state.length} does not match Hamiltonian length {self.hamiltonian.length}."
            raise ValueError(msg)
        if not self.sim_params.observables:
            msg = "sim_params.observables must contain at least one observable."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class NoiseDataset:
    """Synthetic Pauli-XYZ forward data with canonical ``(N, O, T)`` dynamics."""

    expectation_values: NDArray[np.float64]
    gamma: NDArray[np.float64]
    log10_gamma: NDArray[np.float64]
    times: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    metadata: dict[str, object]

    def save(self, path: str | Path) -> None:
        """Save arrays and JSON-safe metadata without pickled YAQS objects.

        Args:
            path: Destination ``.npz`` path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(self.metadata, sort_keys=True)
        np.savez_compressed(
            destination,
            expectation_values=self.expectation_values,
            gamma=self.gamma,
            log10_gamma=self.log10_gamma,
            times=self.times,
            parameter_names=np.asarray(self.parameter_names),
            metadata_json=np.asarray(metadata_json),
        )

    @classmethod
    def load(cls, path: str | Path) -> NoiseDataset:
        """Load and validate a dataset saved by :meth:`save`.

        Args:
            path: Source ``.npz`` path.

        Returns:
            Restored dataset.

        Raises:
            ValueError: If required fields or array relationships are invalid.
        """
        with np.load(Path(path), allow_pickle=False) as payload:
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
                msg = f"Dataset is missing fields: {sorted(missing)}."
                raise ValueError(msg)
            dataset = cls(
                expectation_values=np.asarray(payload["expectation_values"], dtype=np.float64),
                gamma=np.asarray(payload["gamma"], dtype=np.float64),
                log10_gamma=np.asarray(payload["log10_gamma"], dtype=np.float64),
                times=np.asarray(payload["times"], dtype=np.float64),
                parameter_names=tuple(str(name) for name in payload["parameter_names"].tolist()),
                metadata=json.loads(str(payload["metadata_json"].item())),
            )
        _validate_dataset(dataset)
        return dataset


def parameter_names(parameterization: Parameterization, num_sites: int) -> tuple[str, ...]:
    """Return the unambiguous parameter-column order.

    Local rates use site-major XYZ ordering: ``gamma_x_0, gamma_y_0, gamma_z_0, ...``.

    Args:
        parameterization: Parameter-sharing mode.
        num_sites: Number of physical sites.

    Returns:
        Ordered parameter names.

    Raises:
        ValueError: If the site count or parameterization is invalid.
    """
    if num_sites <= 0:
        msg = "num_sites must be positive."
        raise ValueError(msg)
    if parameterization == "super_global":
        return ("gamma",)
    if parameterization == "global":
        return ("gamma_x", "gamma_y", "gamma_z")
    if parameterization == "local":
        return tuple(f"gamma_{channel}_{site}" for site in range(num_sites) for channel in _PAULI_CHANNELS)
    msg = f"Unsupported parameterization: {parameterization!r}."
    raise ValueError(msg)


def sample_pauli_xyz_parameters(
    num_samples: int,
    *,
    parameterization: Parameterization,
    num_sites: int,
    seed: int,
    gamma_min: float = DEFAULT_GAMMA_MIN,
    gamma_max: float = DEFAULT_GAMMA_MAX,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    """Sample rates uniformly in ``log10(gamma)``.

    Args:
        num_samples: Number of independent parameter points.
        parameterization: Parameter-sharing mode.
        num_sites: Number of physical sites.
        seed: NumPy random-generator seed.
        gamma_min: Inclusive physical lower range bound.
        gamma_max: Inclusive physical upper range bound.

    Returns:
        Physical rates, log10 rates, and ordered parameter names.

    Raises:
        ValueError: If the sample count, site count, mode, or gamma range is invalid.
    """
    if num_samples <= 0:
        msg = "num_samples must be positive."
        raise ValueError(msg)
    if not np.isfinite(gamma_min) or not np.isfinite(gamma_max) or gamma_min <= 0 or gamma_min >= gamma_max:
        msg = "gamma_min and gamma_max must be finite, positive, and strictly increasing."
        raise ValueError(msg)
    names = parameter_names(parameterization, num_sites)
    generator = np.random.default_rng(seed)
    log10_gamma = generator.uniform(
        np.log10(gamma_min),
        np.log10(gamma_max),
        size=(num_samples, len(names)),
    )
    return np.power(10.0, log10_gamma), log10_gamma, names


def build_pauli_xyz_noise_model(
    parameters: Sequence[float] | NDArray[np.floating],
    *,
    parameterization: Parameterization,
    num_sites: int,
) -> NoiseModel:
    """Map one parameter vector to a standard YAQS single-site Pauli-XYZ model.

    Args:
        parameters: Physical gamma values in :func:`parameter_names` order.
        parameterization: Parameter-sharing mode.
        num_sites: Number of physical sites.

    Returns:
        Concrete YAQS noise model.

    Raises:
        ValueError: If the parameter vector cannot represent the requested model.
    """
    names = parameter_names(parameterization, num_sites)
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (len(names),):
        msg = f"Expected parameter shape {(len(names),)}, received {values.shape}."
        raise ValueError(msg)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        msg = "All physical gamma values must be finite and strictly positive."
        raise ValueError(msg)

    processes: list[dict[str, object]] = []
    for site in range(num_sites):
        for channel_index, process_name in enumerate(_PAULI_PROCESSES):
            if parameterization == "super_global":
                strength = values[0]
            elif parameterization == "global":
                strength = values[channel_index]
            else:
                strength = values[3 * site + channel_index]
            processes.append({"name": process_name, "sites": [site], "strength": float(strength)})
    return NoiseModel(processes=processes)


def simulate_pauli_xyz(
    experiment: NoiseExperiment,
    parameters: Sequence[float] | NDArray[np.floating],
    *,
    parameterization: Parameterization,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one YAQS forward simulation and return ``(O, T)`` dynamics and times.

    Args:
        experiment: Physical configuration shared with training/reconstruction.
        parameters: Physical gamma vector.
        parameterization: Parameter-sharing mode.

    Returns:
        Observable expectations in user order and authoritative ``Result.times``.
    """
    noise_model = build_pauli_xyz_noise_model(
        parameters,
        parameterization=parameterization,
        num_sites=experiment.state.length,
    )
    result = experiment.simulator.run(
        copy.deepcopy(experiment.state),
        experiment.hamiltonian,
        experiment.sim_params,
        noise_model,
    )
    return _extract_result(result, expected_observables=len(experiment.sim_params.observables))


def generate_noise_dataset(
    experiment: NoiseExperiment,
    *,
    num_samples: int,
    parameterization: Parameterization,
    seed: int,
    gamma_min: float = DEFAULT_GAMMA_MIN,
    gamma_max: float = DEFAULT_GAMMA_MAX,
) -> NoiseDataset:
    """Generate serially orchestrated synthetic training data through YAQS.

    Args:
        experiment: Fixed physical experiment configuration.
        num_samples: Number of sampled parameter points.
        parameterization: Parameter-sharing mode.
        seed: Parameter-sampling seed.
        gamma_min: Physical lower range bound.
        gamma_max: Physical upper range bound.

    Returns:
        Synthetic dataset with raw ``(N, O, T)`` trajectories.

    Raises:
        RuntimeError: If YAQS returns incompatible grids across samples.
    """
    gamma, log10_gamma, names = sample_pauli_xyz_parameters(
        num_samples,
        parameterization=parameterization,
        num_sites=experiment.state.length,
        seed=seed,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    samples: list[NDArray[np.float64]] = []
    times: NDArray[np.float64] | None = None
    for sample_index, parameters in enumerate(gamma):
        sample, candidate_times = simulate_pauli_xyz(
            experiment,
            parameters,
            parameterization=parameterization,
        )
        if times is None:
            times = candidate_times.copy()
        elif not np.array_equal(times, candidate_times):
            msg = f"YAQS returned an incompatible time grid at sample {sample_index}."
            raise RuntimeError(msg)
        samples.append(sample)
    if times is None:  # pragma: no cover - num_samples validation makes this defensive
        msg = "No dataset samples were generated."
        raise RuntimeError(msg)

    metadata = _dataset_metadata(
        experiment,
        num_samples=num_samples,
        parameterization=parameterization,
        parameter_names_=names,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        seed=seed,
        times=times,
    )
    dataset = NoiseDataset(
        expectation_values=np.stack(samples),
        gamma=gamma,
        log10_gamma=log10_gamma,
        times=times,
        parameter_names=names,
        metadata=metadata,
    )
    _validate_dataset(dataset)
    return dataset


def _extract_result(result: Result, *, expected_observables: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate one YAQS result without reconstructing its time grid.

    Returns:
        Observable array and authoritative returned time grid.

    Raises:
        RuntimeError: If observables or times are missing or incompatible.
    """
    if len(result.expectation_values) != expected_observables:
        msg = f"Expected {expected_observables} observables, received {len(result.expectation_values)}."
        raise RuntimeError(msg)
    if result.times is None:
        msg = "YAQS Result.times is required for a noise-characterization dataset."
        raise RuntimeError(msg)
    times = np.asarray(result.times, dtype=np.float64)
    rows = [np.asarray(row, dtype=np.float64) for row in result.expectation_values]
    if any(row.ndim != 1 for row in rows):
        msg = "Each YAQS expectation-value trajectory must be one-dimensional."
        raise RuntimeError(msg)
    if any(row.shape != times.shape for row in rows):
        msg = "YAQS expectation-value rows do not match Result.times."
        raise RuntimeError(msg)
    return np.stack(rows), times


def _observable_descriptions(observables: list[Observable]) -> list[dict[str, object]]:
    """Serialize observable identity and user ordering without matrices.

    Returns:
        JSON-safe descriptions in the supplied order.
    """
    descriptions: list[dict[str, object]] = []
    for observable in observables:
        sites = observable.sites if isinstance(observable.sites, list) else [observable.sites]
        descriptions.append({"gate": str(observable.gate.name), "sites": [int(site) for site in sites]})
    return descriptions


def _dataset_metadata(
    experiment: NoiseExperiment,
    *,
    num_samples: int,
    parameterization: Parameterization,
    parameter_names_: tuple[str, ...],
    gamma_min: float,
    gamma_max: float,
    seed: int,
    times: NDArray[np.float64],
) -> dict[str, object]:
    """Build JSON-safe metadata from the settings actually used.

    Returns:
        Dataset metadata.
    """
    params = experiment.sim_params
    simulator = experiment.simulator
    metadata: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "parameterization": parameterization,
        "parameter_names": list(parameter_names_),
        "parameter_order": "site_major_xyz" if parameterization == "local" else parameterization,
        "num_sites": experiment.state.length,
        "noise_type": "single_site_pauli_xyz",
        "gamma_min": float(gamma_min),
        "gamma_max": float(gamma_max),
        "sampling": "log10_gamma_uniform",
        "seed": int(seed),
        "num_samples": int(num_samples),
        "dataset_layout": "N,O,T",
        "dataset_shape": [num_samples, len(params.observables), len(times)],
        "observables": _observable_descriptions(params.observables),
        "initial_state": {"preset": str(experiment.state.initial)},
        "state_representation": str(experiment.state.representation),
        "simulation": {
            "elapsed_time": float(params.elapsed_time),
            "dt": float(params.dt),
            "num_traj": int(params.num_traj),
            "sample_timesteps": bool(params.sample_timesteps),
            "preset": str(params.preset),
            "max_bond_dim": params.max_bond_dim,
            "trunc_mode": str(params.trunc_mode),
            "svd_threshold": float(params.svd_threshold),
            "krylov_tol": float(params.krylov_tol),
            "order": int(params.order),
            "random_seed": params.random_seed,
            "tdvp_sweeps": int(params.tdvp_sweeps),
            "tdvp_mode": str(params.tdvp_mode),
            "evolution_mode": params.evolution_mode.value,
        },
        "simulator": {
            "parallel": simulator.parallel,
            "max_workers": simulator.max_workers,
            "show_progress": simulator.show_progress,
            "mp_context": simulator.mp_context,
            "max_retries": simulator.max_retries,
        },
        "times": times.tolist(),
    }
    if experiment.hamiltonian_metadata is not None:
        metadata["hamiltonian"] = copy.deepcopy(experiment.hamiltonian_metadata)
    return metadata


def _validate_dataset(dataset: NoiseDataset) -> None:
    """Reject ambiguous or inconsistent saved/generated datasets.

    Raises:
        ValueError: If shapes, axes, or target transforms are inconsistent.
    """
    x = dataset.expectation_values
    if x.ndim != 3:
        msg = f"expectation_values must have shape (N, O, T), received {x.shape}."
        raise ValueError(msg)
    expected_targets = (x.shape[0], len(dataset.parameter_names))
    if dataset.gamma.shape != expected_targets or dataset.log10_gamma.shape != expected_targets:
        msg = "Gamma arrays do not align with samples and parameter_names."
        raise ValueError(msg)
    if dataset.times.shape != (x.shape[2],):
        msg = "times does not align with the trajectory axis."
        raise ValueError(msg)
    if not np.allclose(np.log10(dataset.gamma), dataset.log10_gamma, rtol=0.0, atol=1e-12):
        msg = "gamma and log10_gamma are inconsistent."
        raise ValueError(msg)


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_GAMMA_MAX",
    "DEFAULT_GAMMA_MIN",
    "NoiseDataset",
    "NoiseExperiment",
    "Parameterization",
    "build_pauli_xyz_noise_model",
    "generate_noise_dataset",
    "parameter_names",
    "sample_pauli_xyz_parameters",
    "simulate_pauli_xyz",
]
