from pathlib import Path

import yaml


def test_numba_cache_is_repository_local_and_gitignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo_root / ".numba_config.yaml").read_text(encoding="utf-8"))
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert config["cache_dir"] == ".cache/numba"
    assert "/.cache/" in gitignore
