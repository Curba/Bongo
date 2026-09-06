"""Prepare a new split, train baselines, verify splits, and tune sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sweeps/classical_80_10_10.yaml")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo_root / args.config).read_text())
    # Set limits before importing NumPy/sklearn/XGBoost, and pass them to children.
    values = [config["baseline_threads"], config["blas_threads"]]
    values.extend(spec["threads"] for spec in config["tuning"].values())
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("Thread limits must be positive integers")
    max_threads = max(values)
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT"):
        os.environ[key] = str(max_threads)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(config["blas_threads"])
    os.environ["PYTHONUNBUFFERED"] = "1"

    from qel_twin.training.split_pipeline import run_pipeline

    print(json.dumps(run_pipeline(config, repo_root), indent=2))


if __name__ == "__main__":
    main()
