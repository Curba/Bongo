from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from .benchmark_services import (
    export_benchmark_excel,
    load_benchmark,
    run_benchmark_job,
    scan_benchmarks,
)
from .services import JOB_MANAGER, available_model_options, scan_datasets

PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#d9e3f0"},
    "margin": {"l": 55, "r": 20, "t": 45, "b": 90},
    "xaxis": {"gridcolor": "#233044", "zerolinecolor": "#233044"},
    "yaxis": {"gridcolor": "#233044", "zerolinecolor": "#233044"},
}

TABLE_CELL = {
    "backgroundColor": "transparent",
    "color": "#d9e3f0",
    "border": "1px solid #233044",
    "fontFamily": "Inter, sans-serif",
    "fontSize": 12,
    "textAlign": "left",
    "padding": "10px",
}
TABLE_HEADER = {
    "backgroundColor": "#111a28",
    "fontWeight": 700,
    "color": "#f6f8fb",
}


def _field(label: str, component: Any, help_text: str | None = None) -> html.Div:
    children: list[Any] = [html.Label(label, className="field-label"), component]
    if help_text:
        children.append(html.Div(help_text, className="field-help"))
    return html.Div(children, className="field")


def _number(
    component_id: str,
    value: Any,
    *,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    step: float | int | str | None = None,
) -> dcc.Input:
    return dcc.Input(
        id=component_id,
        type="number",
        value=value,
        min=min_value,
        max=max_value,
        step=step,
        debounce=True,
        className="control-input",
    )


def _metric(title: str, component_id: str, subtitle: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div("—", id=component_id, className="metric-value"),
            html.Div(subtitle, className="metric-subtitle"),
        ],
        className="metric-card",
    )


def _empty_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title, **PLOT_LAYOUT)
    return figure


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return "—"


def create_benchmark_tab() -> html.Div:
    model_options = available_model_options()
    # Start with all classical models selected. Torch/sequence models remain available
    # but are intentionally opt-in because they can make an accidental benchmark last hours.
    default_models = [
        option["value"]
        for option in model_options
        if str(option["label"]).startswith("Classical ·")
    ]

    return html.Div(
        [
            dcc.Store(id="benchmark-job-store"),
            dcc.Download(id="benchmark-excel-download"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Comparative Benchmark"),
                            html.P(
                                "Train selected Bongo models on the same split, reconstruct held-out trajectories with YAQS, and rank by digital-twin fidelity.",
                                className="section-copy",
                            ),
                            _field(
                                "Dataset",
                                dcc.Dropdown(
                                    id="benchmark-dataset",
                                    options=[],
                                    clearable=False,
                                ),
                            ),
                            _field(
                                "Models",
                                dcc.Checklist(
                                    id="benchmark-models",
                                    options=model_options,
                                    value=default_models,
                                    className="benchmark-model-list",
                                ),
                                "Classical models are selected by default. Use Select all to include Torch, CNN, LSTM and BiLSTM.",
                            ),
                            html.Div(
                                [
                                    _field(
                                        "Seed",
                                        _number("benchmark-seed", 1234, min_value=0, step=1),
                                    ),
                                    _field(
                                        "Train fraction",
                                        _number(
                                            "benchmark-train-fraction",
                                            0.60,
                                            min_value=0.05,
                                            max_value=0.90,
                                            step=0.05,
                                        ),
                                    ),
                                    _field(
                                        "Validation fraction",
                                        _number(
                                            "benchmark-validation-fraction",
                                            0.20,
                                            min_value=0.05,
                                            max_value=0.45,
                                            step=0.05,
                                        ),
                                    ),
                                    _field(
                                        "Reconstruction samples",
                                        _number(
                                            "benchmark-reconstruction-samples",
                                            3,
                                            min_value=1,
                                            step=1,
                                        ),
                                    ),
                                    _field(
                                        "Feature mode",
                                        dcc.Dropdown(
                                            id="benchmark-feature-mode",
                                            options=[
                                                {"label": "Flatten O×T", "value": "flatten"},
                                                {"label": "Statistics", "value": "stats"},
                                                {"label": "Flatten + statistics", "value": "both"},
                                            ],
                                            value="flatten",
                                            clearable=False,
                                        ),
                                    ),
                                    _field(
                                        "Trees / estimators",
                                        _number(
                                            "benchmark-estimators",
                                            500,
                                            min_value=10,
                                            step=10,
                                        ),
                                    ),
                                    _field(
                                        "KNN neighbors",
                                        _number(
                                            "benchmark-neighbors",
                                            7,
                                            min_value=1,
                                            step=1,
                                        ),
                                    ),
                                    _field(
                                        "PCA components",
                                        _number(
                                            "benchmark-pca",
                                            64,
                                            min_value=1,
                                            step=1,
                                        ),
                                    ),
                                ],
                                className="form-grid",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Run comparative test",
                                        id="benchmark-run-btn",
                                        n_clicks=0,
                                        className="primary-button",
                                    ),
                                    html.Button(
                                        "Select all",
                                        id="benchmark-select-all",
                                        n_clicks=0,
                                        className="secondary-button benchmark-action",
                                    ),
                                    html.Button(
                                        "Classical only",
                                        id="benchmark-classical-only",
                                        n_clicks=0,
                                        className="secondary-button benchmark-action",
                                    ),
                                    html.Button(
                                        "Clear",
                                        id="benchmark-clear",
                                        n_clicks=0,
                                        className="secondary-button benchmark-action",
                                    ),
                                ],
                                className="benchmark-actions",
                            ),
                            html.Div(id="benchmark-job-status", className="job-status"),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H2("Benchmark Pipeline"),
                            html.Div(
                                [
                                    html.Div("NoiseDataset", className="flow-node"),
                                    html.Div("→", className="flow-arrow"),
                                    html.Div("Same split", className="flow-node"),
                                    html.Div("→", className="flow-arrow"),
                                    html.Div("Selected models", className="flow-node"),
                                    html.Div("→", className="flow-arrow"),
                                    html.Div("Predicted γ", className="flow-node"),
                                    html.Div("→", className="flow-arrow"),
                                    html.Div(
                                        "YAQS reconstruction",
                                        className="flow-node flow-primary",
                                    ),
                                ],
                                className="flow",
                            ),
                            html.P(
                                "Ranking uses reconstructed-trajectory RMSE first. Parameter-estimation metrics are diagnostics, matching the existing Control Center objective.",
                                className="callout",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Strong("Fair comparison"),
                                            html.P(
                                                "Every model receives the same dataset, seed and split fractions."
                                            ),
                                        ],
                                        className="mini-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Strong("Existing training code"),
                                            html.P(
                                                "Each row is produced by the repository's train_model_job + reconstruction path."
                                            ),
                                        ],
                                        className="mini-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Strong("Excel export"),
                                            html.P(
                                                "Exports leaderboard, config, failures and best-model held-out predictions."
                                            ),
                                        ],
                                        className="mini-card",
                                    ),
                                ],
                                className="mini-grid",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column training-columns",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Benchmark Results"),
                                    html.Div(
                                        [
                                            _field(
                                                "Benchmark",
                                                dcc.Dropdown(
                                                    id="benchmark-result",
                                                    options=[],
                                                    clearable=False,
                                                ),
                                            ),
                                            html.Button(
                                                "Refresh",
                                                id="benchmark-refresh",
                                                n_clicks=0,
                                                className="secondary-button",
                                            ),
                                            html.Button(
                                                "Export Excel",
                                                id="benchmark-export",
                                                n_clicks=0,
                                                className="secondary-button",
                                            ),
                                        ],
                                        className="benchmark-result-selector",
                                    ),
                                ],
                                className="section-header benchmark-section-header",
                            ),
                            html.Div(
                                [
                                    _metric(
                                        "Best model",
                                        "benchmark-best-model",
                                        "Lowest trajectory RMSE",
                                    ),
                                    _metric(
                                        "Trajectory RMSE",
                                        "benchmark-best-rmse",
                                        "Primary score",
                                    ),
                                    _metric(
                                        "log10 γ RMSE",
                                        "benchmark-best-log-rmse",
                                        "Diagnostic",
                                    ),
                                    _metric(
                                        "Models completed",
                                        "benchmark-completed",
                                    ),
                                    _metric("Failures", "benchmark-failures"),
                                ],
                                className="metric-grid",
                            ),
                            dash_table.DataTable(
                                id="benchmark-table",
                                columns=[
                                    {"name": "Model", "id": "model"},
                                    {
                                        "name": "Trajectory RMSE",
                                        "id": "trajectory_rmse",
                                        "type": "numeric",
                                    },
                                    {
                                        "name": "Trajectory MAE",
                                        "id": "trajectory_mae",
                                        "type": "numeric",
                                    },
                                    {
                                        "name": "Max trajectory error",
                                        "id": "trajectory_max_error",
                                        "type": "numeric",
                                    },
                                    {
                                        "name": "log10 γ RMSE",
                                        "id": "log10_gamma_rmse",
                                        "type": "numeric",
                                    },
                                    {
                                        "name": "log10 γ MAE",
                                        "id": "log10_gamma_mae",
                                        "type": "numeric",
                                    },
                                    {"name": "R²", "id": "r2_log10", "type": "numeric"},
                                    {
                                        "name": "Median factor error",
                                        "id": "median_factor_error",
                                        "type": "numeric",
                                    },
                                    {
                                        "name": "Fit s",
                                        "id": "fit_seconds",
                                        "type": "numeric",
                                    },
                                    {
                                        "name": "Predict s",
                                        "id": "predict_seconds",
                                        "type": "numeric",
                                    },
                                ],
                                data=[],
                                sort_action="native",
                                page_size=15,
                                style_table={"overflowX": "auto"},
                                style_cell=TABLE_CELL,
                                style_header=TABLE_HEADER,
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="benchmark-rmse-graph",
                                        config={"displaylogo": False},
                                    ),
                                    dcc.Graph(
                                        id="benchmark-param-graph",
                                        config={"displaylogo": False},
                                    ),
                                ],
                                className="plot-grid",
                            ),
                        ],
                        className="panel",
                    )
                ],
                className="results-stack benchmark-results",
            ),
        ],
        className="tab-content",
    )


def register_benchmark_callbacks(
    app: Dash,
    *,
    data_root: str,
    output_root: str,
) -> None:
    model_options = available_model_options()
    all_models = [option["value"] for option in model_options]
    classical_models = [
        option["value"]
        for option in model_options
        if str(option["label"]).startswith("Classical ·")
    ]

    @app.callback(
        Output("benchmark-models", "value"),
        Input("benchmark-select-all", "n_clicks"),
        Input("benchmark-classical-only", "n_clicks"),
        Input("benchmark-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def select_models(_all_clicks: int, _classical_clicks: int, _clear_clicks: int):
        if ctx.triggered_id == "benchmark-select-all":
            return all_models
        if ctx.triggered_id == "benchmark-classical-only":
            return classical_models
        return []

    @app.callback(
        Output("benchmark-dataset", "options"),
        Output("benchmark-dataset", "value"),
        Input("catalog-poller", "n_intervals"),
        State("benchmark-dataset", "value"),
    )
    def refresh_benchmark_datasets(_tick: int, current: str | None):
        rows = scan_datasets(data_root)
        options = [
            {
                "label": (
                    f"{row['name']} · {row['parameterization']} · "
                    f"{row['samples']} samples"
                ),
                "value": row["path"],
            }
            for row in rows
        ]
        values = {option["value"] for option in options}
        selected = current if current in values else (
            options[0]["value"] if options else None
        )
        return options, selected

    @app.callback(
        Output("benchmark-job-store", "data"),
        Output("benchmark-job-status", "children"),
        Input("benchmark-run-btn", "n_clicks"),
        State("benchmark-dataset", "value"),
        State("benchmark-models", "value"),
        State("benchmark-seed", "value"),
        State("benchmark-train-fraction", "value"),
        State("benchmark-validation-fraction", "value"),
        State("benchmark-reconstruction-samples", "value"),
        State("benchmark-feature-mode", "value"),
        State("benchmark-estimators", "value"),
        State("benchmark-neighbors", "value"),
        State("benchmark-pca", "value"),
        prevent_initial_call=True,
    )
    def start_benchmark(
        n_clicks: int,
        dataset: str | None,
        models: list[str] | None,
        seed: Any,
        train_fraction: Any,
        validation_fraction: Any,
        reconstruction_samples: Any,
        feature_mode: str,
        estimators: Any,
        neighbors: Any,
        pca: Any,
    ):
        if not n_clicks:
            return no_update, no_update
        if not dataset:
            return no_update, "Select a dataset first."
        if not models:
            return no_update, "Select at least one model."
        try:
            train_fraction = float(train_fraction)
            validation_fraction = float(validation_fraction)
            if train_fraction <= 0 or validation_fraction <= 0:
                return no_update, "Train and validation fractions must be positive."
            if train_fraction + validation_fraction >= 1.0:
                return no_update, "Train + validation fractions must sum to less than 1."
            config = {
                "seed": int(seed),
                "train_fraction": train_fraction,
                "validation_fraction": validation_fraction,
                "reconstruction_samples": max(1, int(reconstruction_samples)),
                "feature_mode": str(feature_mode or "flatten"),
                "n_estimators": max(10, int(estimators)),
                "n_neighbors": max(1, int(neighbors)),
                "pca_components": max(1, int(pca)),
            }
        except (TypeError, ValueError) as exc:
            return no_update, f"Invalid benchmark configuration: {exc}"

        job_id = JOB_MANAGER.submit(
            "benchmark",
            run_benchmark_job,
            dataset_path=dataset,
            output_root=output_root,
            models=list(models),
            config=config,
        )
        return (
            {"job_id": job_id},
            f"Benchmark queued · {len(models)} models · same split for every model.",
        )

    @app.callback(
        Output("benchmark-job-status", "children", allow_duplicate=True),
        Input("job-poller", "n_intervals"),
        State("benchmark-job-store", "data"),
        prevent_initial_call=True,
    )
    def poll_benchmark(_tick: int, store: dict[str, Any] | None):
        if not store:
            return no_update
        job = JOB_MANAGER.get(store.get("job_id"))
        if job is None:
            return "Job state unavailable."
        status = job.get("status")
        if status == "failed":
            return f"FAILED\n\n{job.get('error') or 'Unknown error'}"
        if status == "complete":
            result = job.get("result") or {}
            return (
                f"COMPLETE · {result.get('completed', 0)} complete · "
                f"{result.get('failed', 0)} failed · "
                f"best={result.get('best_model') or '—'}"
            )
        return f"{str(status).upper()} · training selected models and reconstructing with YAQS"

    @app.callback(
        Output("benchmark-result", "options"),
        Output("benchmark-result", "value"),
        Input("catalog-poller", "n_intervals"),
        Input("benchmark-refresh", "n_clicks"),
        Input("benchmark-job-store", "data"),
        State("benchmark-dataset", "value"),
        State("benchmark-result", "value"),
    )
    def refresh_benchmarks(
        _tick: int,
        _refresh_clicks: int,
        _job_store: dict[str, Any] | None,
        dataset: str | None,
        current: str | None,
    ):
        rows = scan_benchmarks(output_root, dataset)
        options = [
            {
                "label": (
                    f"{row['benchmark_id']} · {row['completed']}/{row['models']} complete · "
                    f"best {row['best_model'] or '—'}"
                ),
                "value": row["path"],
            }
            for row in rows
        ]
        values = {option["value"] for option in options}
        selected = current if current in values else (
            options[0]["value"] if options else None
        )
        return options, selected

    @app.callback(
        Output("benchmark-table", "data"),
        Output("benchmark-rmse-graph", "figure"),
        Output("benchmark-param-graph", "figure"),
        Output("benchmark-best-model", "children"),
        Output("benchmark-best-rmse", "children"),
        Output("benchmark-best-log-rmse", "children"),
        Output("benchmark-completed", "children"),
        Output("benchmark-failures", "children"),
        Input("benchmark-result", "value"),
    )
    def update_benchmark(path: str | None):
        if not path:
            return (
                [],
                _empty_figure("Trajectory RMSE by model"),
                _empty_figure("Parameter RMSE by model"),
                "—",
                "—",
                "—",
                "0",
                "0",
            )
        try:
            payload = load_benchmark(path)
        except Exception as exc:
            return (
                [],
                _empty_figure(f"Benchmark load error: {exc}"),
                _empty_figure("Parameter RMSE by model"),
                "—",
                "—",
                "—",
                "0",
                "0",
            )

        rows = payload.get("leaderboard") or []
        best = rows[0] if rows else {}

        reconstruction_rows = [
            row for row in rows if row.get("trajectory_rmse") is not None
        ]
        rmse_figure = go.Figure()
        if reconstruction_rows:
            rmse_figure.add_trace(
                go.Bar(
                    x=[row["model"] for row in reconstruction_rows],
                    y=[row["trajectory_rmse"] for row in reconstruction_rows],
                    name="Trajectory RMSE",
                )
            )
        rmse_figure.update_layout(
            title="Digital-twin reconstruction fidelity",
            xaxis_title="Model",
            yaxis_title="Trajectory RMSE",
            **PLOT_LAYOUT,
        )

        parameter_rows = [
            row for row in rows if row.get("log10_gamma_rmse") is not None
        ]
        parameter_figure = go.Figure()
        if parameter_rows:
            parameter_figure.add_trace(
                go.Bar(
                    x=[row["model"] for row in parameter_rows],
                    y=[row["log10_gamma_rmse"] for row in parameter_rows],
                    name="log10 γ RMSE",
                )
            )
        parameter_figure.update_layout(
            title="Noise-parameter estimation",
            xaxis_title="Model",
            yaxis_title="log10 γ RMSE",
            **PLOT_LAYOUT,
        )

        return (
            rows,
            rmse_figure,
            parameter_figure,
            best.get("model", "—"),
            _fmt(best.get("trajectory_rmse")),
            _fmt(best.get("log10_gamma_rmse")),
            str(len(rows)),
            str(len(payload.get("failures") or [])),
        )

    @app.callback(
        Output("benchmark-excel-download", "data"),
        Input("benchmark-export", "n_clicks"),
        State("benchmark-result", "value"),
        prevent_initial_call=True,
    )
    def export_excel(_clicks: int, path: str | None):
        if not path:
            return no_update
        try:
            workbook = export_benchmark_excel(path)
        except Exception as exc:
            # dcc.Download cannot display an error, so keep the current page stable.
            print(f"Benchmark Excel export failed: {exc}")
            return no_update
        return dcc.send_file(str(workbook))


__all__ = ["create_benchmark_tab", "register_benchmark_callbacks"]
