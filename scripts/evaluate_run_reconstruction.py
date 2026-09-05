from __future__ import annotations

import argparse
import json
from pathlib import Path

from qel_twin.visualization.control_center.services import evaluate_run_reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run physics reconstruction for a completed training run's test "
            "predictions, bypassing the Control Center UI/job system. Calls "
            "evaluate_run_reconstruction() directly and updates the run's "
            "metrics.json with the aggregate reconstruction block."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed run directory (must contain test_predictions.csv).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Path to the NoiseDataset NPZ used for training. Defaults to the "
            "'dataset_path' field recorded in the run's metrics.json."
        ),
    )
    parser.add_argument(
        "--reconstruction-samples",
        type=int,
        default=3,
        help="Number of test-set samples to reconstruct (default: 3).",
    )
    return parser.parse_args()


def resolve_dataset_path(run_dir: Path, dataset_arg: str | None) -> Path:
    if dataset_arg:
        return Path(dataset_arg)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"--dataset not given and no metrics.json found at {metrics_path} to infer it from."
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    dataset_path = metrics.get("dataset_path")
    if not dataset_path:
        raise ValueError(f"metrics.json at {metrics_path} has no 'dataset_path' field.")
    return Path(dataset_path)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    dataset_path = resolve_dataset_path(run_dir, args.dataset)

    aggregate = evaluate_run_reconstruction(
        dataset_path=dataset_path,
        run_dir=run_dir,
        reconstruction_samples=args.reconstruction_samples,
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
