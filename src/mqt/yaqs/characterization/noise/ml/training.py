# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Model training, prediction, evaluation, and checkpoint persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mqt.yaqs.characterization.noise.ml.dataset import (
    DEFAULT_GAMMA_MAX,
    DEFAULT_GAMMA_MIN,
    NoiseDataset,
    Parameterization,
    parameter_names,
)
from mqt.yaqs.characterization.noise.ml.models import ModelName, build_regression_model
from mqt.yaqs.characterization.noise.ml.preprocessing import DatasetSplit, TrajectoryPreprocessor

from ._torch import require_torch

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import nn


@dataclass(slots=True)
class TrainedNoiseModel:
    """A fitted regressor together with the preprocessing required for inference."""

    model_name: ModelName
    model: nn.Module
    preprocessor: TrajectoryPreprocessor
    parameterization: Parameterization
    parameter_names: tuple[str, ...]
    gamma_min: float = DEFAULT_GAMMA_MIN
    gamma_max: float = DEFAULT_GAMMA_MAX
    training_history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float = float("inf")
    metadata: dict[str, object] = field(default_factory=dict)

    def predict(self, dynamics: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Infer physical and log10 rates from external dynamics.

        Args:
            dynamics: One ``(O, T)`` trajectory or batch ``(N, O, T)``.

        Returns:
            ``(gamma, log10_gamma)`` with columns in :attr:`parameter_names` order.

        Raises:
            RuntimeError: If the model returns an incompatible output shape.
        """
        torch = require_torch()
        model_input = self.preprocessor.transform(dynamics)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            predicted_log = self.model(torch.from_numpy(model_input).to(device)).cpu().numpy().astype(np.float64)
        expected = (model_input.shape[0], len(self.parameter_names))
        if predicted_log.shape != expected:
            msg = f"Model returned shape {predicted_log.shape}; expected {expected}."
            raise RuntimeError(msg)
        return np.power(10.0, predicted_log), predicted_log

    def save(self, path: str | Path) -> None:
        """Save model tensors and JSON metadata in a non-pickled ``.npz`` checkpoint.

        Args:
            path: Destination checkpoint path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state_arrays = {
            f"state::{name}": tensor.detach().cpu().numpy() for name, tensor in self.model.state_dict().items()
        }
        checkpoint_metadata = {
            "schema_version": 1,
            "model_name": self.model_name,
            "input_shape": list(self.preprocessor.input_shape),
            "parameterization": self.parameterization,
            "parameter_names": list(self.parameter_names),
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "training_history": self.training_history,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            destination,
            normalization_mean=self.preprocessor.mean,
            normalization_std=self.preprocessor.std,
            checkpoint_json=np.asarray(json.dumps(checkpoint_metadata, sort_keys=True)),
            **state_arrays,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> TrainedNoiseModel:
        """Restore a model checkpoint without allowing pickle deserialization.

        Args:
            path: Source checkpoint path.
            device: PyTorch device for the restored model.

        Returns:
            Restored trained model and preprocessing state.

        Raises:
            ValueError: If the checkpoint input shape is invalid.
        """
        torch = require_torch()
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["checkpoint_json"].item()))
            input_dimensions = [int(value) for value in metadata["input_shape"]]
            if len(input_dimensions) != 2:
                msg = f"Checkpoint input_shape must have two dimensions, received {input_dimensions}."
                raise ValueError(msg)
            input_shape = (input_dimensions[0], input_dimensions[1])
            parameter_names_ = tuple(str(value) for value in metadata["parameter_names"])
            model_name = metadata["model_name"]
            model = build_regression_model(
                model_name,
                input_shape=input_shape,
                num_targets=len(parameter_names_),
            )
            state_dict = {
                name.removeprefix("state::"): torch.from_numpy(np.asarray(payload[name]))
                for name in payload.files
                if name.startswith("state::")
            }
            model.load_state_dict(state_dict)
            model.to(torch.device(device))
            preprocessor = TrajectoryPreprocessor(
                mean=np.asarray(payload["normalization_mean"], dtype=np.float64),
                std=np.asarray(payload["normalization_std"], dtype=np.float64),
                input_shape=input_shape,
                model_name=model_name,
            )
        return cls(
            model_name=model_name,
            model=model,
            preprocessor=preprocessor,
            parameterization=metadata["parameterization"],
            parameter_names=parameter_names_,
            gamma_min=float(metadata["gamma_min"]),
            gamma_max=float(metadata["gamma_max"]),
            training_history=metadata["training_history"],
            best_epoch=int(metadata["best_epoch"]),
            best_validation_loss=float(metadata["best_validation_loss"]),
            metadata=metadata["metadata"],
        )


def train_noise_model(
    dataset: NoiseDataset,
    split: DatasetSplit,
    *,
    model_name: ModelName,
    parameterization: Parameterization,
    seed: int,
    max_epochs: int = 150,
    patience: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> TrainedNoiseModel:
    """Train on train indices, select checkpoints on validation indices, and ignore test data.

    Args:
        dataset: Synthetic raw YAQS dataset.
        split: Explicit disjoint sample split. Test indices are recorded but never read here.
        model_name: Regression backend.
        parameterization: Dataset parameter-sharing mode.
        seed: PyTorch and data-loader seed.
        max_epochs: Maximum training epochs.
        patience: Validation-loss early-stopping patience.
        batch_size: Training mini-batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        device: Optional PyTorch device; defaults to CUDA when available, otherwise CPU.

    Returns:
        Best-validation checkpoint with fitted preprocessing.

    Raises:
        ValueError: If training settings, indices, or parameter metadata are invalid.
        RuntimeError: If training produces no valid checkpoint.
    """
    torch = require_torch()
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0:
        msg = "max_epochs, patience, and batch_size must be positive."
        raise ValueError(msg)
    all_indices = np.concatenate((split.train, split.validation, split.test))
    if np.any(all_indices < 0) or np.any(all_indices >= len(dataset.expectation_values)):
        msg = "Dataset split contains an out-of-range sample index."
        raise ValueError(msg)
    expected_names = parameter_names(parameterization, int(cast("int", dataset.metadata["num_sites"])))
    if dataset.parameter_names != expected_names:
        msg = "Dataset parameter names do not match the requested parameterization."
        raise ValueError(msg)

    training_raw = dataset.expectation_values[split.train]
    validation_raw = dataset.expectation_values[split.validation]
    preprocessor = TrajectoryPreprocessor.fit(training_raw, model_name=model_name)
    training_input = preprocessor.transform(training_raw)
    validation_input = preprocessor.transform(validation_raw)
    training_targets = dataset.log10_gamma[split.train].astype(np.float32)
    validation_targets = dataset.log10_gamma[split.validation].astype(np.float32)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_regression_model(
        model_name,
        input_shape=preprocessor.input_shape,
        num_targets=len(dataset.parameter_names),
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    loader_generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(training_input),
            torch.from_numpy(training_targets),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_x = torch.from_numpy(validation_input).to(resolved_device)
    validation_y = torch.from_numpy(validation_targets).to(resolved_device)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(resolved_device)), targets.to(resolved_device))
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y).detach().cpu())
        history.append({
            "epoch": float(epoch),
            "training_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:  # pragma: no cover - finite MSE on a non-empty dataset is expected
        msg = "Training did not produce a valid checkpoint."
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
    model.to(resolved_device)
    return TrainedNoiseModel(
        model_name=model_name,
        model=model,
        preprocessor=preprocessor,
        parameterization=parameterization,
        parameter_names=dataset.parameter_names,
        gamma_min=float(cast("float", dataset.metadata["gamma_min"])),
        gamma_max=float(cast("float", dataset.metadata["gamma_max"])),
        training_history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        metadata={
            "training_indices": split.train.tolist(),
            "validation_indices": split.validation.tolist(),
            "held_out_test_indices": split.test.tolist(),
            "normalization": "per_observable_mean_std_fitted_on_training_only",
            "target": "log10_gamma",
            "dataset": dataset.metadata,
        },
    )


def evaluate_noise_model(
    trained_model: TrainedNoiseModel,
    dataset: NoiseDataset,
    test_indices: NDArray[np.integer],
) -> dict[str, float]:
    """Evaluate a frozen model on a held-out split without refitting anything.

    Args:
        trained_model: Frozen trained model and preprocessing.
        dataset: Source dataset.
        test_indices: Held-out indices.

    Returns:
        Log-target MAE/RMSE and physical median factor error.
    """
    indices = np.asarray(test_indices, dtype=np.int64)
    gamma_pred, log_pred = trained_model.predict(dataset.expectation_values[indices])
    log_true = dataset.log10_gamma[indices]
    gamma_true = dataset.gamma[indices]
    residual = log_pred - log_true
    factor_error = np.maximum(gamma_pred / gamma_true, gamma_true / gamma_pred)
    return {
        "log10_mae": float(np.mean(np.abs(residual))),
        "log10_rmse": float(np.sqrt(np.mean(residual**2))),
        "median_factor_error": float(np.median(factor_error)),
    }


__all__ = ["TrainedNoiseModel", "evaluate_noise_model", "train_noise_model"]
