# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Leakage-safe splitting and preprocessing for noise trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from qel_twin.characterization.noise_ml.models import ModelName


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Disjoint sample indices used for training, validation, and final evaluation."""

    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Reject overlaps and duplicate indices.

        Raises:
            ValueError: If an index set is empty, duplicated, or overlapping.
        """
        arrays = [np.asarray(indices, dtype=np.int64) for indices in (self.train, self.validation, self.test)]
        if any(indices.ndim != 1 or len(indices) == 0 for indices in arrays):
            msg = "Every dataset split must be a non-empty one-dimensional index array."
            raise ValueError(msg)
        sets = [set(indices.tolist()) for indices in arrays]
        if any(len(indices) != len(index_set) for indices, index_set in zip(arrays, sets, strict=True)):
            msg = "Dataset splits must not contain duplicate indices."
            raise ValueError(msg)
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            msg = "Training, validation, and test indices must be disjoint."
            raise ValueError(msg)


def split_dataset_indices(
    num_samples: int,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int,
) -> DatasetSplit:
    """Create one deterministic, disjoint sample-level split.

    Args:
        num_samples: Number of independent simulation samples.
        train_fraction: Fraction assigned to training.
        validation_fraction: Fraction assigned to validation.
        seed: Permutation seed.

    Returns:
        Disjoint split indices. Remaining samples form the test split.

    Raises:
        ValueError: If the sample count or fractions cannot form three non-empty splits.
    """
    if num_samples < 3:
        msg = "At least three samples are required for train/validation/test splitting."
        raise ValueError(msg)
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        msg = "train_fraction and validation_fraction must be positive and sum to less than one."
        raise ValueError(msg)
    n_train = int(np.floor(num_samples * train_fraction))
    n_validation = int(np.floor(num_samples * validation_fraction))
    if n_train == 0 or n_validation == 0 or n_train + n_validation >= num_samples:
        msg = "Fractions produce an empty train, validation, or test split."
        raise ValueError(msg)
    indices = np.random.default_rng(seed).permutation(num_samples)
    return DatasetSplit(
        train=indices[:n_train],
        validation=indices[n_train : n_train + n_validation],
        test=indices[n_train + n_validation :],
    )


@dataclass(frozen=True, slots=True)
class TrajectoryPreprocessor:
    """Train-fitted per-observable normalization and explicit model reshape."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    input_shape: tuple[int, int]
    model_name: ModelName

    @classmethod
    def fit(cls, training_dynamics: NDArray[np.floating], *, model_name: ModelName) -> TrajectoryPreprocessor:
        """Fit normalization on training trajectories only.

        Args:
            training_dynamics: Raw training data shaped ``(N, O, T)``.
            model_name: Selected model reshape.

        Returns:
            Fitted immutable preprocessing state.

        Raises:
            ValueError: If the training array or model name is invalid.
        """
        values = np.asarray(training_dynamics, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0:
            msg = f"training_dynamics must have non-empty shape (N, O, T), received {values.shape}."
            raise ValueError(msg)
        if model_name not in {"mlp", "2d_cnn"}:
            msg = f"Unknown model_name {model_name!r}; expected 'mlp' or '2d_cnn'."
            raise ValueError(msg)
        mean = values.mean(axis=(0, 2), keepdims=True)
        std = np.maximum(values.std(axis=(0, 2), keepdims=True), 1e-6)
        return cls(mean=mean, std=std, input_shape=(values.shape[1], values.shape[2]), model_name=model_name)

    def transform(self, dynamics: NDArray[np.floating]) -> NDArray[np.float32]:
        """Normalize trajectories with saved training statistics and reshape explicitly.

        Args:
            dynamics: One ``(O, T)`` trajectory or batch ``(N, O, T)``.

        Returns:
            MLP input ``(N, O, T)`` or 2D-CNN input ``(N, 1, O, T)``.

        Raises:
            ValueError: If the trajectory shape differs from the training shape.
        """
        values = np.asarray(dynamics, dtype=np.float64)
        if values.ndim == 2:
            values = values[np.newaxis, ...]
        if values.ndim != 3 or values.shape[1:] != self.input_shape:
            msg = f"Expected dynamics shape (N, {self.input_shape[0]}, {self.input_shape[1]}), received {values.shape}."
            raise ValueError(msg)
        normalized = ((values - self.mean) / self.std).astype(np.float32)
        if self.model_name == "2d_cnn":
            return normalized[:, np.newaxis, :, :]
        return normalized


__all__ = ["DatasetSplit", "TrajectoryPreprocessor", "split_dataset_indices"]
