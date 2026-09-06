"""Pipeline orchestration checks with synthetic data and mocked child processes."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from qel_twin.training import split_pipeline


@pytest.mark.parametrize("wrong_split", [False, True])
def test_pipeline_order_and_split_guard(tmp_path, monkeypatch, wrong_split):
    repo = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((repo / "configs/sweeps/classical_80_10_10.yaml").read_text())
    config["output_dir"] = str(tmp_path / "pipeline")
    root = Path(config["output_dir"])
    monkeypatch.setattr(
        split_pipeline, "load_noise_dataset",
        lambda _: SimpleNamespace(dynamics=np.zeros((1000, 1, 1)), metadata={"dataset_id": "synthetic"}),
    )
    calls = []
    tuning_configs = []

    def fake_run(command, **kwargs):
        assert kwargs["check"] is True
        script = Path(command[1]).name
        calls.append(script)
        if script == "run_classical_ml_batch.py":
            batch = yaml.safe_load(Path(command[-1]).read_text())
            assert batch["train_fraction"] == 0.8
            assert batch["validation_fraction"] == 0.1
            assert batch["threads"] == 4
            assert batch["blas_threads"] == 1
            with np.load(root / "split_indices.npz") as source:
                splits = {k: source[k] for k in source.files}
            assert [len(splits[k]) for k in ("train_idx", "val_idx", "test_idx")] == [800, 100, 100]
            if wrong_split:
                # All models agree with one another but differ from the intended split.
                splits["train_idx"] = splits["train_idx"][::-1]
            run_dirs = {}
            for name in batch["models"]:
                run = Path(batch["output_dir"]) / "synthetic" / name / "run"
                run.mkdir(parents=True)
                np.savez(run / "split_indices.npz", **splits)
                run_dirs[name] = str(run)
            Path(batch["manifest_path"]).write_text(json.dumps({"run_dirs": run_dirs, "failures": {}}))
        elif script == "tune_classical_pilot.py":
            tuning_configs.append(yaml.safe_load(Path(command[-1]).read_text()))

    monkeypatch.setattr(split_pipeline.subprocess, "run", fake_run)
    if wrong_split:
        with pytest.raises(ValueError, match="Expected split mismatch"):
            split_pipeline.run_pipeline(config, repo)
        assert not tuning_configs
        assert json.loads((root / "pipeline_status.json").read_text())["state"] == "failed"
    else:
        status = split_pipeline.run_pipeline(config, repo)
        assert status["state"] == "complete"
        assert status["expected_split_verified"]
        assert calls == ["run_classical_ml_batch.py", "check_split_consistency.py"] + ["tune_classical_pilot.py"] * 3
        for study, name, budget, patience in zip(
            tuning_configs, ["svr_rbf", "hist_gradient_boosting", "xgboost"],
            [20, 20, 12], [20, 10, 8],
        ):
            assert study["train_fraction"] == 0.8
            assert study["validation_fraction"] == 0.1
            assert study["folds"] == 5
            assert study["early_stopping_patience"] == patience
            assert study["models"][name]["trials"] == budget
        assert tuning_configs[-1]["models"]["xgboost"]["fixed_params"]["n_jobs"] == 4
        assert tuning_configs[-1]["threads"] == 4
        with pytest.raises(FileExistsError):
            split_pipeline.run_pipeline(config, repo)
