from __future__ import annotations

import argparse
import json

from qel_twin.training.classical_ml import AVAILABLE_MODELS
from qel_twin.training.classical_ml_noise import train_classical_noise_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Bongo classical ML models on qel-ml NoiseDataset NPZ files."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/noise_ml_runs")
    parser.add_argument("--model", choices=AVAILABLE_MODELS, default="extra_trees")
    parser.add_argument(
        "--feature-mode",
        choices=["flatten", "stats", "both"],
        default="flatten",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--n-neighbors", type=int, default=7)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--run-tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_classical_noise_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model,
        feature_mode=args.feature_mode,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        n_neighbors=args.n_neighbors,
        pca_components=args.pca_components,
        n_estimators=args.n_estimators,
        run_tag=args.run_tag,
    )
    print(json.dumps({"run_dir": result["run_dir"]}, indent=2))


if __name__ == "__main__":
    main()
