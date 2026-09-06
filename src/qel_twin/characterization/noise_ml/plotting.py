# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optional plotting helpers for completed ML characterization results."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qel_twin.characterization.noise_ml.results import MLNoiseCharacterizationResult


def plot_trajectory_overlay(
    result: MLNoiseCharacterizationResult,
    path: str | Path,
    *,
    observable_labels: list[str] | None = None,
) -> None:
    """Plot reference and reconstructed trajectories using optional Matplotlib.

    Args:
        result: Completed characterization result.
        path: Destination image path.
        observable_labels: Optional labels in experiment observable order.

    Raises:
        ImportError: If Matplotlib is unavailable.
        ValueError: If the label count does not match the observables.
    """
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as error:
        msg = "Trajectory plotting requires Matplotlib."
        raise ImportError(msg) from error
    num_observables = result.original_dynamics.shape[0]
    labels = observable_labels or [f"observable {index}" for index in range(num_observables)]
    if len(labels) != num_observables:
        msg = f"Expected {num_observables} observable labels, received {len(labels)}."
        raise ValueError(msg)
    figure, axes = plt.subplots(num_observables, 1, figsize=(7, max(3, 2 * num_observables)), squeeze=False)
    for index, label in enumerate(labels):
        axis = axes[index, 0]
        axis.plot(result.times, result.original_dynamics[index], color="black", label="original")
        axis.plot(result.times, result.reconstructed_dynamics[index], "--", label="reconstructed")
        axis.set_ylabel(label)
    axes[-1, 0].set_xlabel("time")
    axes[0, 0].legend()
    figure.suptitle(f"{result.model_name}: trajectory RMSE={result.metrics['trajectory_rmse']:.4g}")
    figure.tight_layout()
    figure.savefig(Path(path))
    plt.close(figure)


__all__ = ["plot_trajectory_overlay"]
