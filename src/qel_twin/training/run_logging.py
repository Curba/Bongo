from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_id(seed: int | None = None, tag: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parts = [timestamp]

    if seed is not None:
        parts.append(f"seed{seed}")

    if tag:
        parts.append(tag)

    return "_".join(parts)


def create_run_dir(
    output_root: str | Path,
    dataset_id: str,
    model_name: str,
    run_id: str,
) -> Path:
    run_dir = Path(output_root) / dataset_id / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(
    run_dir: str | Path,
    metadata: dict[str, Any],
) -> None:
    run_dir = Path(run_dir)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)