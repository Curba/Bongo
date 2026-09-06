"""Run config-driven classical CV tuning with optional trial-level early stopping."""

import argparse
import json
from pathlib import Path

import yaml

from qel_twin.training.classical_tuning import run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_pilot(yaml.safe_load(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
