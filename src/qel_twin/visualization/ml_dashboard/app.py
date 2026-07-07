from __future__ import annotations

import argparse
from pathlib import Path

from dash import Dash

from qel_twin.visualization.ml_dashboard.callbacks import register_callbacks
from qel_twin.visualization.ml_dashboard.data_loader import scan_runs
from qel_twin.visualization.ml_dashboard.layout import create_layout


def create_app(results_root: str | Path) -> Dash:
    runs = scan_runs(results_root)
    datasets = sorted({r.dataset for r in runs})

    app = Dash(__name__)
    app.title = "QEL Twin ML Visualizer"

    app.layout = create_layout(datasets)
    register_callbacks(app, runs)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/ml_runs",
        help="Root folder containing dataset/model/run result folders.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    app = create_app(args.results_dir)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )