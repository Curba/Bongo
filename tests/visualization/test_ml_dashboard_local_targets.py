from __future__ import annotations

import json

import pandas as pd

from qel_twin.visualization.ml_dashboard.data_loader import (
    RunRef,
    load_metrics_for_run,
    load_parameter_names,
)
from qel_twin.visualization.ml_dashboard.figures import (
    per_target_metrics,
    relative_error_box,
)


def test_local_parameter_names_drive_metrics_and_figures(tmp_path) -> None:
    names = ["gamma_x_0", "gamma_y_0", "gamma_z_0", "gamma_x_1", "gamma_y_1", "gamma_z_1"]
    run = RunRef(dataset="local", model="ridge", run_id="run", path=tmp_path)
    (tmp_path / "run_metadata.json").write_text(json.dumps({"parameter_names": names, "seed": 1234}))
    test_metrics = {"mae_log10_mean": 0.1, "rmse_log10_mean": 0.2, "r2_log10_mean": 0.3}
    for index, name in enumerate(names):
        test_metrics[f"{name}_mae_log10"] = index / 100
        test_metrics[f"{name}_r2_log10"] = index / 10
        test_metrics[f"{name}_median_relative_error_gamma"] = index / 20
    (tmp_path / "metrics.json").write_text(json.dumps({"test_metrics": test_metrics}))

    assert load_parameter_names(run) == names
    metrics = load_metrics_for_run(run).iloc[0]
    assert metrics["gamma_z_1_mae_log10"] == 0.05

    predictions = pd.DataFrame({f"{name}_relative_error": [0.1] for name in names})
    box = relative_error_box(predictions, names)
    assert set(box.data[0].x) == set(names)

    bars = per_target_metrics(metrics, names)
    assert list(bars.data[0].x) == names


def test_legacy_run_falls_back_to_three_global_targets(tmp_path) -> None:
    run = RunRef(dataset="global", model="ridge", run_id="run", path=tmp_path)
    assert load_parameter_names(run) == ["gamma_x", "gamma_y", "gamma_z"]
