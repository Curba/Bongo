# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Multilayer-perceptron backend for noise regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._torch import require_torch

if TYPE_CHECKING:
    from torch import nn


def build_mlp(*, input_shape: tuple[int, int], num_targets: int) -> nn.Module:
    """Build the P15-derived MLP architecture.

    Args:
        input_shape: Raw trajectory shape ``(O, T)``.
        num_targets: Number of log10-gamma outputs.

    Returns:
        PyTorch MLP module.
    """
    nn = require_torch().nn
    num_observables, num_times = input_shape
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_observables * num_times, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, num_targets),
    )


__all__ = ["build_mlp"]
