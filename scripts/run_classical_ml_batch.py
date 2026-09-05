from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import yaml

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
            result = train_classical_noise_model(
                dataset_path=args.dataset,
                output_dir=output_dir,
                model_name=model_name,
                seed=seed,
            )
        except Exception:
            print(f"FAILED: {model_name}")
            traceback.print_exc()
            failures[model_name] = traceback.format_exc()
            continue
        run_dirs[model_name] = result["run_dir"]

    print("\n" + "=" * 88)
    print("Batch summary")
    print("=" * 88)
    print(json.dumps({"run_dirs": run_dirs, "failures": list(failures)}, indent=2))

    if failures:
        raise SystemExit(f"{len(failures)} model(s) failed: {sorted(failures)}")


if __name__ == "__main__":
    main()
