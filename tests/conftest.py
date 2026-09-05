from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--noise-dataset",
        type=Path,
        default=None,
        help="Canonical noise-dataset NPZ used by data-dependent integration tests.",
    )


@pytest.fixture
def noise_dataset_path(request: pytest.FixtureRequest) -> Path:
    path = request.config.getoption("--noise-dataset")
    if path is None:
        pytest.skip("Pass --noise-dataset PATH to run data-dependent integration tests.")
    path = path.resolve()
    if not path.is_file():
        pytest.fail(f"Noise dataset does not exist: {path}")
    return path
