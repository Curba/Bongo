"""Reconstruct verified final models listed in a consolidation manifest."""

import argparse
from pathlib import Path

import yaml

from qel_twin.training.consolidation import reconstruct_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    config = yaml.safe_load(parser.parse_args().config.read_text())
    reconstruct_manifest(
        config["manifest_path"],
        config["dataset_path"],
        config["reconstruction_samples"],
        config["threads"],
        config.get("blas_threads", config["threads"]),
    )


if __name__ == "__main__":
    main()
