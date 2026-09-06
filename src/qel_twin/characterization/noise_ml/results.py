# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Structured results for ML noise characterization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from qel_twin.characterization.noise_ml.models import ModelName


@dataclass(frozen=True, slots=True)
class MLNoiseCharacterizationResult:
    """Prediction and physics reconstruction for one external trajectory."""

    predicted_gamma: NDArray[np.float64]
    predicted_log10_gamma: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    original_dynamics: NDArray[np.float64]
    reconstructed_dynamics: NDArray[np.float64]
    times: NDArray[np.float64]
    metrics: dict[str, object]
    model_name: ModelName
    metadata: dict[str, object] = field(default_factory=dict)


__all__ = ["MLNoiseCharacterizationResult"]
