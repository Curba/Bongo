# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Physics reconstruction and raw trajectory comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qel_twin.characterization.noise_ml.dataset import NoiseExperiment, Parameterization, simulate_pauli_xyz

if TYPE_CHECKING:
    from numpy.typing import NDArray


def reconstruct_dynamics(
    experiment: NoiseExperiment,
    predicted_gamma: NDArray[np.floating],
    *,
    parameterization: Parameterization,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct dynamics through the shared Pauli-XYZ YAQS forward path.

    Args:
        experiment: Original physical experiment configuration.
        predicted_gamma: Physical rate vector in the declared parameter order.
        parameterization: Parameter-sharing mode.

    Returns:
        Reconstructed ``(O, T)`` dynamics and authoritative YAQS time grid.
    """
    return simulate_pauli_xyz(experiment, predicted_gamma, parameterization=parameterization)


def compute_trajectory_metrics(
    original: NDArray[np.floating],
    reconstructed: NDArray[np.floating],
) -> dict[str, object]:
    """Compute raw-space overall and per-observable reconstruction errors.

    Args:
        original: Reference trajectories shaped ``(O, T)``.
        reconstructed: Reconstructed trajectories with the same shape.

    Returns:
        Overall MAE, RMSE, maximum error, and per-observable MAE/RMSE.

    Raises:
        ValueError: If trajectory shapes differ or are not two-dimensional.
    """
    reference = np.asarray(original, dtype=np.float64)
    candidate = np.asarray(reconstructed, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        msg = f"Trajectory arrays must have matching (O, T) shapes, got {reference.shape} and {candidate.shape}."
        raise ValueError(msg)
    difference = candidate - reference
    return {
        "trajectory_mae": float(np.mean(np.abs(difference))),
        "trajectory_rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs_trajectory_error": float(np.max(np.abs(difference))),
        "observable_mae": np.mean(np.abs(difference), axis=1).tolist(),
        "observable_rmse": np.sqrt(np.mean(difference**2, axis=1)).tolist(),
    }


__all__ = ["compute_trajectory_metrics", "reconstruct_dynamics"]
