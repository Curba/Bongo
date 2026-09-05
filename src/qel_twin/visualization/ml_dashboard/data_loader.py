from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PARAMS = ["gamma_x", "gamma_y", "gamma_z"]
SPLITS = ["train", "val", "test"]


@dataclass(frozen=True)
class RunRef:
    dataset: str
    model: str
    run_id: str
    path: Path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parameter_names(run: RunRef) -> list[str]:
    """Load target names recorded by a run, with legacy global fallback."""
    metadata = read_json(run.path / "run_metadata.json")
    metrics = read_json(run.path / "metrics.json")
    names = metadata.get("parameter_names") or metrics.get("parameter_names")
    if isinstance(names, list) and names and all(isinstance(name, str) for name in names):
        return names
    return PARAMS.copy()


def scan_runs(results_root: str | Path) -> list[RunRef]:
    """
    Supports new structure:

        outputs/ml_runs/<dataset>/<model>/<run_id>/metrics.json

    Also supports older structure:

        outputs/classical_ml/<model>/metrics.json
    """
    root = Path(results_root)

    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")

    runs: list[RunRef] = []

    # New structure: dataset/model/run_id
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                if (run_dir / "metrics.json").exists():
                    runs.append(
                        RunRef(
                            dataset=dataset_dir.name,
                            model=model_dir.name,
                            run_id=run_dir.name,
                            path=run_dir,
                        )
                    )

    if runs:
        return sorted(runs, key=lambda r: (r.dataset, r.model, r.run_id))

    # Old structure: model/metrics.json
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue

        if (model_dir / "metrics.json").exists():
            runs.append(
                RunRef(
                    dataset="default_dataset",
                    model=model_dir.name,
                    run_id="default_run",
                    path=model_dir,
                )
            )

    if not runs:
        raise RuntimeError(f"No runs found under {root}")

    return sorted(runs, key=lambda r: (r.dataset, r.model, r.run_id))


def runs_to_frame(runs: list[RunRef]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": r.dataset,
                "model": r.model,
                "run_id": r.run_id,
                "path": str(r.path),
            }
            for r in runs
        ]
    )


def load_metrics_for_run(run: RunRef) -> pd.DataFrame:
    metrics = read_json(run.path / "metrics.json")
    metadata = read_json(run.path / "run_metadata.json")

    rows = []
    parameter_names = load_parameter_names(run)

    for split in SPLITS:
        key = f"{split}_metrics"
        if key not in metrics:
            continue

        m = metrics[key]

        row = {
            "dataset": run.dataset,
            "model": run.model,
            "run_id": run.run_id,
            "split": split,
            "run_path": str(run.path),
            "mae_log10_mean": m.get("mae_log10_mean", np.nan),
            "rmse_log10_mean": m.get("rmse_log10_mean", np.nan),
            "r2_log10_mean": m.get("r2_log10_mean", np.nan),
            "median_relative_error_gamma": m.get("median_relative_error_gamma", np.nan),
            "mean_relative_error_gamma": m.get("mean_relative_error_gamma", np.nan),
            "seed": metadata.get("seed", metrics.get("seed", np.nan)),
            "created_at": metadata.get("created_at", ""),
        }

        for p in parameter_names:
            row[f"{p}_mae_log10"] = m.get(f"{p}_mae_log10", np.nan)
            row[f"{p}_rmse_log10"] = m.get(f"{p}_rmse_log10", np.nan)
            row[f"{p}_r2_log10"] = m.get(f"{p}_r2_log10", np.nan)
            row[f"{p}_median_relative_error_gamma"] = m.get(
                f"{p}_median_relative_error_gamma", np.nan
            )
            row[f"{p}_mean_relative_error_gamma"] = m.get(
                f"{p}_mean_relative_error_gamma", np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def load_all_metrics(runs: list[RunRef]) -> pd.DataFrame:
    frames = [load_metrics_for_run(r) for r in runs]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_predictions(run: RunRef, split: str) -> pd.DataFrame:
    path = run.path / f"{split}_predictions.csv"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["dataset"] = run.dataset
    df["model"] = run.model
    df["run_id"] = run.run_id
    df["split"] = split
    df["sample_id"] = np.arange(len(df))

    return df


def load_training_history(run: RunRef) -> pd.DataFrame:
    """
    Optional file for future neural networks.

    Expected columns can be:
        epoch, train_loss, val_loss
        epoch, train_mae, val_mae
        epoch, learning_rate
    """
    path = run.path / "training_history.csv"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["dataset"] = run.dataset
    df["model"] = run.model
    df["run_id"] = run.run_id

    return df


def get_run(
    runs: list[RunRef],
    dataset: str,
    model: str,
    run_id: str,
) -> RunRef:
    for r in runs:
        if r.dataset == dataset and r.model == model and r.run_id == run_id:
            return r

    raise KeyError(f"Run not found: dataset={dataset}, model={model}, run_id={run_id}")
