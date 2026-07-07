#!/usr/bin/env python3

from __future__ import annotations

import argparse

from qel_twin.training.classical_ml import AVAILABLE_MODELS, train_classical_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train classical ML baselines on a P3 Lindblad .npz dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset_p3_lindblad.npz",
        help="Path to .npz dataset containing X and y_log.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ml_runs",
        help="Root output directory. Runs are saved as dataset/model/run_id.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="hist_gradient_boosting",
        choices=AVAILABLE_MODELS,
    )

    parser.add_argument(
        "--feature-mode",
        type=str,
        default="flatten",
        choices=["flatten", "stats", "both"],
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    parser.add_argument("--n-neighbors", type=int, default=7)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--n-estimators", type=int, default=500)

    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional label added to run_id, e.g. first_test or larger_dataset.",
    )

    args = parser.parse_args()

    train_classical_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model,
        feature_mode=args.feature_mode,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        n_neighbors=args.n_neighbors,
        pca_components=args.pca_components,
        n_estimators=args.n_estimators,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()