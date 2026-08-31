# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Two-dimensional CNN backend for noise regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._torch import require_torch

if TYPE_CHECKING:
    from torch import nn


def build_cnn2d(*, num_targets: int) -> nn.Module:
    """Build the P15-derived small 2D CNN architecture.

    The input is shaped ``(N, 1, O, T)`` by the shared preprocessor.

    Args:
        num_targets: Number of log10-gamma outputs.

    Returns:
        PyTorch 2D CNN module.
    """
    nn = require_torch().nn
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, num_targets),
    )


__all__ = ["build_cnn2d"]
