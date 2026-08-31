from __future__ import annotations

from typing import Any

import numpy as np

from qel_twin.characterization.noise_ml.reconstruction import (
    compute_trajectory_metrics,
    reconstruct_dynamics,
)


def to_observable_time(trajectory: np.ndarray) -> np.ndarray:
    """Normalize a single trajectory to qel-ml's canonical ``(O, T)`` form.

    Accepted inputs:
      - (O, T): already canonical
      - (C, L, T): legacy Bongo layout, flattened to O=C*L
    """
    array = np.asarray(trajectory, dtype=np.float64)

    if array.ndim == 2:
        return array

    if array.ndim == 3:
        channels, sites, times = array.shape
        return array.reshape(channels * sites, times)

    raise ValueError(
        "Expected one trajectory with shape (O,T) or (C,L,T), "
        f"got {array.shape}"
    )


def reconstruct_from_log10_prediction(
    *,
    experiment: Any,
    original_trajectory: np.ndarray,
    predicted_log10_gamma: np.ndarray,
    parameterization: str,
) -> dict[str, Any]:
    """Use a model prediction to rebuild the YAQS dynamics and score the twin.

    The ML model predicts log10(gamma).  This adapter converts back to physical
    gamma, calls qel-ml's reconstruction path, and evaluates the reconstructed
    trajectory against the original observed dynamics.

    Reconstruction fidelity is the primary output.  Parameter values are
    returned for diagnostics.
    """
    predicted_log10_gamma = np.asarray(
        predicted_log10_gamma,
        dtype=np.float64,
    ).reshape(-1)

    if predicted_log10_gamma.size == 0:
        raise ValueError("predicted_log10_gamma must contain at least one value.")

    if not np.all(np.isfinite(predicted_log10_gamma)):
        raise ValueError("predicted_log10_gamma contains non-finite values.")

    predicted_gamma = np.power(10.0, predicted_log10_gamma)

    reconstructed, times = reconstruct_dynamics(
        experiment,
        predicted_gamma,
        parameterization=parameterization,
    )

    original = to_observable_time(original_trajectory)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if original.shape != reconstructed.shape:
        raise ValueError(
            "Original and reconstructed trajectory shapes differ: "
            f"{original.shape} vs {reconstructed.shape}"
        )

    metrics = compute_trajectory_metrics(
        original,
        reconstructed,
    )

    return {
        "predicted_log10_gamma": predicted_log10_gamma,
        "predicted_gamma": predicted_gamma,
        "times": np.asarray(times),
        "original_trajectory": original,
        "reconstructed_trajectory": reconstructed,
        "reconstruction_metrics": metrics,
    }


def reconstruction_score(result: dict[str, Any]) -> float:
    """Return trajectory RMSE when qel-ml metrics expose it.

    This is convenient for ranking models by reconstructed dynamics rather than
    by parameter error.
    """
    metrics = result["reconstruction_metrics"]

    for key in ("rmse", "trajectory_rmse", "rmse_trajectory"):
        if key in metrics:
            return float(metrics[key])

    raise KeyError(
        "Could not find a trajectory RMSE key in reconstruction_metrics. "
        f"Available keys: {sorted(metrics)}"
    )


__all__ = [
    "to_observable_time",
    "reconstruct_from_log10_prediction",
    "reconstruction_score",
]
