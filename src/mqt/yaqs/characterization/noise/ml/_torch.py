# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lazy access to the optional PyTorch dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types


def require_torch() -> types.ModuleType:
    """Import PyTorch without making it a core YAQS dependency.

    Returns:
        Imported PyTorch module.

    Raises:
        ImportError: If the optional PyTorch dependency is unavailable.
    """
    try:
        import torch  # ruff: ignore[import-outside-top-level]
    except ImportError as error:
        msg = "ML noise characterization requires PyTorch; install YAQS with the 'torch' extra."
        raise ImportError(msg) from error
    return torch


__all__ = ["require_torch"]
