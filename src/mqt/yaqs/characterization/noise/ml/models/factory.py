# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unified selection boundary for noise-regression models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .cnn2d import build_cnn2d
from .mlp import build_mlp

if TYPE_CHECKING:
    from torch import nn

ModelName = Literal["mlp", "2d_cnn"]


def build_regression_model(model_name: ModelName, *, input_shape: tuple[int, int], num_targets: int) -> nn.Module:
    """Construct the selected regression backend through one stable boundary.

    Args:
        model_name: ``"mlp"`` or ``"2d_cnn"``.
        input_shape: Raw trajectory shape ``(O, T)``.
        num_targets: Number of log10-gamma outputs.

    Returns:
        Selected PyTorch module.

    Raises:
        ValueError: If the model name or target count is invalid.
    """
    if num_targets <= 0:
        msg = "num_targets must be positive."
        raise ValueError(msg)
    if model_name == "mlp":
        return build_mlp(input_shape=input_shape, num_targets=num_targets)
    if model_name == "2d_cnn":
        return build_cnn2d(num_targets=num_targets)
    msg = f"Unknown model_name {model_name!r}; expected 'mlp' or '2d_cnn'."
    raise ValueError(msg)


__all__ = ["ModelName", "build_regression_model"]
