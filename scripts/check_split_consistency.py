from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SPLIT_KEYS = ("train_idx", "val_idx", "test_idx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm split_indices.npz (train_idx/val_idx/test_idx) is byte-identical "
            "across every model subdirectory of a batch output directory, e.g. "
            "outputs/ml_runs/<dataset>/."
        )
    )
    parser.add_argument("batch_dir", help="Directory containing one subdirectory per model.")
    return parser.parse_args()


def latest_split_file(model_dir: Path) -> Path:
    """Pick the most recent run for a model (run_id directories sort lexicographically by timestamp)."""
    candidates = sorted(model_dir.glob("*/split_indices.npz"))
    if not candidates:
        raise FileNotFoundError(f"No split_indices.npz found under {model_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        print(f"Not a directory: {batch_dir}", file=sys.stderr)
        raise SystemExit(2)

    model_dirs = sorted(p for p in batch_dir.iterdir() if p.is_dir())
    if not model_dirs:
        print(f"No model subdirectories found under {batch_dir}", file=sys.stderr)
        raise SystemExit(2)

    splits: dict[str, dict[str, np.ndarray]] = {}
    split_files: dict[str, Path] = {}
    for model_dir in model_dirs:
        try:
            split_path = latest_split_file(model_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        with np.load(split_path) as data:
            splits[model_dir.name] = {key: data[key].copy() for key in SPLIT_KEYS}
        split_files[model_dir.name] = split_path

    reference_model = next(iter(splits))
    reference = splits[reference_model]

    print(f"Batch dir:       {batch_dir}")
    print(f"Reference model: {reference_model} ({split_files[reference_model]})")
    for key in SPLIT_KEYS:
        print(f"  {key}: {reference[key].tolist()}")
    print()

    mismatches: list[str] = []
    for model, values in splits.items():
        if model == reference_model:
            continue
        disagreements = [key for key in SPLIT_KEYS if not np.array_equal(values[key], reference[key])]
        status = "OK" if not disagreements else f"MISMATCH on {disagreements}"
        print(f"{model:<28} {split_files[model]}  {status}")
        if disagreements:
            mismatches.append(model)

    print()
    if mismatches:
        print(f"FAIL: {len(mismatches)} model(s) disagree with '{reference_model}': {sorted(mismatches)}")
        raise SystemExit(1)

    print(f"OK: all {len(splits)} models share identical split membership.")


if __name__ == "__main__":
    main()
