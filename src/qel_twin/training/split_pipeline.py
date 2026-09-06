"""Sequential split/baseline/verification/tuning orchestration using existing CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from qel_twin.training.classical_ml import infer_dataset_id, write_json
from qel_twin.training.noise_dataset import load_noise_dataset, split_dataset_indices


def verify_run_splits(run_dirs, expected, model_names):
    """Reject missing models and mutually consistent but incorrect partitions."""
    if set(run_dirs) != set(model_names):
        raise ValueError("Baseline manifest does not contain exactly the requested models")
    for name, run_dir in run_dirs.items():
        with np.load(Path(run_dir) / "split_indices.npz") as saved:
            for key, indices in expected.items():
                if not np.array_equal(saved[key], indices):
                    raise ValueError(f"Expected split mismatch: {name}/{key}")


def make_tuning_config(config, name, spec, baseline_run, root):
    """Give each sequential study its own budget, patience and output directory."""
    study_spec = {k: v for k, v in spec.items() if k not in {
        "threads", "early_stopping_patience"
    }}
    study_spec["baseline_run"] = baseline_run
    return {
        "dataset_path": config["dataset_path"],
        "output_dir": str(root / "tuning" / name),
        "feature_mode": config["feature_mode"],
        "seed": config["seed"],
        "cv_seed": config["cv_seed"],
        "train_fraction": config["train_fraction"],
        "validation_fraction": config["validation_fraction"],
        "folds": config["folds"],
        "threads": spec["threads"],
        "blas_threads": config["blas_threads"],
        "early_stopping_patience": spec["early_stopping_patience"],
        "models": {name: study_spec},
    }


def run_pipeline(config, repo_root):
    """Run only when explicitly invoked; refuse overwrites and fail before tuning on errors."""
    started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    config = dict(config)
    dataset_path = (repo_root / config["dataset_path"]).resolve()
    config["dataset_path"] = str(dataset_path)
    root = (repo_root / config["output_dir"]).resolve()
    if root.exists():
        raise FileExistsError(f"Use a fresh output_dir; automatic resume is unsupported: {root}")
    dataset = load_noise_dataset(dataset_path)
    if len(dataset.dynamics) != config["expected_samples"]:
        raise ValueError("Unexpected dataset sample count")
    split = split_dataset_indices(
        len(dataset.dynamics), seed=config["seed"],
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
    )
    expected = {"train_idx": split.train, "val_idx": split.validation, "test_idx": split.test}
    dataset_id = infer_dataset_id(dataset_path, dataset.metadata)
    root.mkdir(parents=True)
    generated = root / "configs"
    generated.mkdir()
    np.savez(root / "split_indices.npz", **expected)
    write_json(root / "pipeline_config.json", config)
    status = {"state": "running", "stage_seconds": {}, "split_sizes": {
        key: len(value) for key, value in expected.items()
    }}
    write_json(root / "pipeline_status.json", status)

    def stage(label, command):
        status["stage"] = label
        write_json(root / "pipeline_status.json", status)
        print(f"Starting {label}: {' '.join(command)}", flush=True)
        stage_start = time.perf_counter()
        subprocess.run(command, cwd=repo_root, env=os.environ.copy(), check=True)
        status["stage_seconds"][label] = time.perf_counter() - stage_start
        write_json(root / "pipeline_status.json", status)

    try:
        manifest_path = root / "baseline_manifest.json"
        batch = {
            "output_dir": str(root / "baselines"),
            "manifest_path": str(manifest_path),
            "seed": config["seed"], "models": config["baseline_models"],
            "feature_mode": config["feature_mode"],
            "train_fraction": config["train_fraction"],
            "validation_fraction": config["validation_fraction"],
            "threads": config["baseline_threads"], "blas_threads": config["blas_threads"],
        }
        batch_path = generated / "baselines.yaml"
        batch_path.write_text(yaml.safe_dump(batch, sort_keys=False))
        stage("baselines", [sys.executable, str(repo_root / "scripts/run_classical_ml_batch.py"),
                            "--dataset", str(dataset_path), "--config", str(batch_path)])
        manifest = json.loads(manifest_path.read_text())
        if manifest["failures"]:
            raise RuntimeError(f"Baseline failures: {manifest['failures']}")
        run_dirs = manifest["run_dirs"]
        stage("split_consistency", [sys.executable, str(repo_root / "scripts/check_split_consistency.py"),
                                    str(root / "baselines" / dataset_id)])
        verify_run_splits(run_dirs, expected, config["baseline_models"])
        status["expected_split_verified"] = True
        for name, spec in config["tuning"].items():
            study_config = make_tuning_config(config, name, spec, run_dirs[name], root)
            study_path = generated / f"tuning_{name}.yaml"
            study_path.write_text(yaml.safe_dump(study_config, sort_keys=False))
            stage(f"tuning_{name}", [sys.executable, str(repo_root / "scripts/tune_classical_pilot.py"),
                                      "--config", str(study_path)])
        status["state"] = "complete"
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = repr(exc)
        raise
    finally:
        status["wall_seconds"] = time.perf_counter() - started
        write_json(root / "pipeline_status.json", status)
    return status
