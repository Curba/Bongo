from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from qel_twin.characterization.noise_ml import generate_noise_dataset
from qel_twin.visualization.control_center.services import build_experiment_from_config


REQUIRED_KEYS = {
    "dataset_id",
    "output_path",
    "num_samples",
    "num_sites",
    "channels",
    "parameterization",
    "gamma_min",
    "gamma_max",
    "seed",
    "elapsed_time",
    "dt",
    "method",
    "preset",
    "num_traj",
    "initial_state",
    "j_coupling",
    "transverse_field",
    "order",
    "tdvp_sweeps",
    "tdvp_mode",
    "parallel",
    "max_workers",
    "show_progress",
    "svd_threshold",
    "max_bond_dim",
    "yaqs_checkout",
}


def load_config(path: Path) -> dict[str, Any]:
    """Load a fully explicit dataset configuration without implicit experiment defaults."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset config must be a YAML mapping.")
    missing = REQUIRED_KEYS - payload.keys()
    unknown = payload.keys() - REQUIRED_KEYS
    if missing or unknown:
        raise ValueError(f"Config keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return payload


def validate_config(config: dict[str, Any]) -> None:
    """Validate the generator-facing values before constructing YAQS objects."""
    if int(config["num_samples"]) <= 0:
        raise ValueError("num_samples must be positive.")
    if int(config["num_sites"]) <= 0:
        raise ValueError("num_sites must be positive.")
    if config["parameterization"] not in {"super_global", "global", "local"}:
        raise ValueError("parameterization must be super_global, global, or local.")
    channels = [str(value).lower() for value in config["channels"]]
    if not channels or set(channels) - {"x", "y", "z"}:
        raise ValueError("channels must be a non-empty list containing only x, y, and z.")
    gamma_min = float(config["gamma_min"])
    gamma_max = float(config["gamma_max"])
    if gamma_min <= 0 or gamma_max <= gamma_min:
        raise ValueError("gamma_min and gamma_max must be positive and strictly increasing.")
    if float(config["elapsed_time"]) <= 0 or float(config["dt"]) <= 0:
        raise ValueError("elapsed_time and dt must be positive.")


def yaqs_commit(checkout: Path) -> str:
    """Return the exact YAQS revision recorded for a generated dataset."""
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolved_config(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    resolved = dict(config)
    resolved["output_path"] = str((config_path.parent / str(config["output_path"])).resolve())
    resolved["yaqs_checkout"] = str((config_path.parent / str(config["yaqs_checkout"])).resolve())
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a canonical (N,O,T) Pauli-XYZ noise dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run YAQS and save the dataset. Without this flag, only print the resolved config.",
    )
    args = parser.parse_args()

    config = resolved_config(args.config.resolve())
    print(json.dumps(config, indent=2, sort_keys=True))
    if not args.execute:
        print("Dry run only. Pass --execute to generate the dataset.")
        return

    experiment = build_experiment_from_config(config)
    dataset = generate_noise_dataset(
        experiment,
        num_samples=int(config["num_samples"]),
        parameterization=config["parameterization"],
        seed=int(config["seed"]),
        gamma_min=float(config["gamma_min"]),
        gamma_max=float(config["gamma_max"]),
    )
    dataset.metadata["dataset_id"] = str(config["dataset_id"])
    dataset.metadata["yaqs_commit"] = yaqs_commit(Path(config["yaqs_checkout"]))
    dataset.metadata["source_config"] = str(args.config.resolve())
    output_path = Path(config["output_path"])
    dataset.save(output_path)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
