from __future__ import annotations

import argparse
import json
import traceback
from contextlib import ExitStack
from pathlib import Path

import yaml
from threadpoolctl import threadpool_limits

from qel_twin.training.classical_ml import write_json
from qel_twin.training.classical_ml_noise import train_classical_noise_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train every classical model listed in a batch config against one "
            "dataset, sharing identical train/val/test split membership via a "
            "fixed seed (see configs/sweeps/classical_ml_batch.yaml)."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/sweeps/classical_ml_batch.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Overrides the config's output_dir if given.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = args.output_dir or config["output_dir"]
    seed = int(config["seed"])
    models = list(config["models"])

    run_dirs: dict[str, str] = {}
    failures: dict[str, str] = {}

    for model_name in models:
        print("=" * 88)
        print(f"Training: {model_name}")
        print("=" * 88)
        try:
            with ExitStack() as limits:
                if "threads" in config:
                    limits.enter_context(threadpool_limits(limits=config["threads"]))
                    limits.enter_context(threadpool_limits(
                        limits=config.get("blas_threads", 1), user_api="blas"
                    ))
                result = train_classical_noise_model(
                    dataset_path=args.dataset,
                    output_dir=output_dir,
                    model_name=model_name,
                    seed=seed,
                    feature_mode=config.get("feature_mode", "flatten"),
                    train_fraction=config.get("train_fraction", 0.60),
                    validation_fraction=config.get("validation_fraction", 0.20),
                    n_neighbors=config.get("n_neighbors", 7),
                    pca_components=config.get("pca_components", 64),
                    n_estimators=config.get("n_estimators", 500),
                    n_jobs=config.get("threads"),
                )
        except Exception:
            print(f"FAILED: {model_name}")
            traceback.print_exc()
            failures[model_name] = traceback.format_exc()
            continue
        run_dirs[model_name] = result["run_dir"]
        if config.get("manifest_path"):
            write_json(config["manifest_path"], {"run_dirs": run_dirs, "failures": failures})

    print("\n" + "=" * 88)
    print("Batch summary")
    print("=" * 88)
    print(json.dumps({"run_dirs": run_dirs, "failures": list(failures)}, indent=2))
    if config.get("manifest_path"):
        write_json(config["manifest_path"], {"run_dirs": run_dirs, "failures": failures})

    if failures:
        raise SystemExit(f"{len(failures)} model(s) failed: {sorted(failures)}")


if __name__ == "__main__":
    main()
