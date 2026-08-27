# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""External-trajectory orchestration for ML noise characterization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.characterization.noise.ml.dataset import NoiseExperiment, parameter_names
from mqt.yaqs.characterization.noise.ml.reconstruction import compute_trajectory_metrics, reconstruct_dynamics
from mqt.yaqs.characterization.noise.ml.results import MLNoiseCharacterizationResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.characterization.noise.ml.training import TrainedNoiseModel


class MLNoiseCharacterizer:
    """Infer Pauli-XYZ rates from external dynamics and reconstruct them with YAQS."""

    def __init__(self, experiment: NoiseExperiment, trained_model: TrainedNoiseModel) -> None:
        """Bind a fitted model to the same physical experiment used for training.

        Args:
            experiment: Original state, Hamiltonian, observables, simulation parameters, and Simulator.
            trained_model: Fitted regressor and train-only preprocessing.

        Raises:
            ValueError: If parameter ordering or trajectory shape is incompatible.
        """
        expected_names = parameter_names(trained_model.parameterization, experiment.state.length)
        if trained_model.parameter_names != expected_names:
            msg = "The trained model parameter order does not match the experiment site count."
            raise ValueError(msg)
        expected_observables = len(experiment.sim_params.observables)
        if trained_model.preprocessor.input_shape[0] != expected_observables:
            msg = (
                f"Model expects {trained_model.preprocessor.input_shape[0]} observables, but the experiment "
                f"contains {expected_observables}."
            )
            raise ValueError(msg)
        self.experiment = experiment
        self.trained_model = trained_model

    def characterize(
        self,
        original_dynamics: NDArray[np.floating],
        *,
        times: NDArray[np.floating] | None = None,
    ) -> MLNoiseCharacterizationResult:
        """Characterize one external trajectory and run a YAQS reconstruction.

        Args:
            original_dynamics: External raw observable data shaped ``(O, T)`` in the experiment's order.
            times: Optional external time axis. When supplied it must exactly match YAQS reconstruction times.

        Returns:
            Structured prediction, reconstruction, and trajectory metrics.

        Raises:
            ValueError: If external dynamics or its time grid is incompatible.
            RuntimeError: If reconstruction changes the trajectory shape.
        """
        original = np.asarray(original_dynamics, dtype=np.float64)
        if original.shape != self.trained_model.preprocessor.input_shape:
            msg = (
                f"original_dynamics has shape {original.shape}; expected {self.trained_model.preprocessor.input_shape}."
            )
            raise ValueError(msg)
        gamma_batch, log_batch = self.trained_model.predict(original)
        predicted_gamma = gamma_batch[0]
        predicted_log = log_batch[0]
        reconstructed, reconstructed_times = reconstruct_dynamics(
            self.experiment,
            predicted_gamma,
            parameterization=self.trained_model.parameterization,
        )
        if times is not None and not np.array_equal(np.asarray(times, dtype=np.float64), reconstructed_times):
            msg = "External time grid does not match the YAQS reconstruction time grid."
            raise ValueError(msg)
        if reconstructed.shape != original.shape:
            msg = f"Reconstructed shape {reconstructed.shape} does not match original shape {original.shape}."
            raise RuntimeError(msg)
        return MLNoiseCharacterizationResult(
            predicted_gamma=predicted_gamma,
            predicted_log10_gamma=predicted_log,
            parameter_names=self.trained_model.parameter_names,
            original_dynamics=original,
            reconstructed_dynamics=reconstructed,
            times=reconstructed_times,
            metrics=compute_trajectory_metrics(original, reconstructed),
            model_name=self.trained_model.model_name,
            metadata={"parameterization": self.trained_model.parameterization},
        )


__all__ = ["MLNoiseCharacterizer"]
