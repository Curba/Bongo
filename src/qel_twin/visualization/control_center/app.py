from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

from .benchmark import create_benchmark_tab, register_benchmark_callbacks
from .services import (
    JOB_MANAGER,
    available_model_options,
    create_dataset_job,
    dataset_details,
    load_dataset_preview,
    load_reconstruction_comparison,
    load_reconstruction_sample,
    load_run_details,
    common_reconstruction_samples,
    scan_datasets,
    scan_runs,
    train_model_job,
)

PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#d9e3f0"},
    "margin": {"l": 55, "r": 20, "t": 45, "b": 50},
    "xaxis": {"gridcolor": "#233044", "zerolinecolor": "#233044"},
    "yaxis": {"gridcolor": "#233044", "zerolinecolor": "#233044"},
    "legend": {"orientation": "h", "y": 1.12, "x": 0},
}



def _empty_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title, **PLOT_LAYOUT)
    return figure


def _format_metric(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    if abs(number) >= 1000 or (0 < abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.5f}"


def _field(label: str, component, help_text: str | None = None):
    children = [html.Label(label, className="field-label"), component]
    if help_text:
        children.append(html.Div(help_text, className="field-help"))
    return html.Div(children, className="field")


def _number(component_id: str, value, *, min_value=None, max_value=None, step=None):
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


def _text(component_id: str, value: str = "", placeholder: str = ""):
    return dcc.Input(
        id=component_id,
        type="text",
        value=value,
        placeholder=placeholder,
        debounce=True,
        className="control-input",
    )


def _metric_card(title: str, component_id: str, subtitle: str = ""):
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div("—", id=component_id, className="metric-value"),
            html.Div(subtitle, className="metric-subtitle"),
        ],
        className="metric-card",
    )


def _dataset_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Create Noise Dataset"),
                            html.P(
                                "Generate canonical qel-ml trajectories (N, O, T) directly through YAQS.",
                                className="section-copy",
                            ),
                            html.Div(
                                [
                                    _field("Dataset name", _text("ds-name", "noise_global_5q")),
                                    _field("Output path (optional)", _text("ds-output", "", "Blank = data root / dataset name")),
                                    _field("Samples", _number("ds-samples", 100, min_value=3, step=1)),
                                    _field("Sites / qubits", _number("ds-sites", 5, min_value=1, step=1)),
                                    _field(
                                        "Observable channels",
                                        dcc.Checklist(
                                            id="ds-channels",
                                            options=[
                                                {"label": "X", "value": "x"},
                                                {"label": "Y", "value": "y"},
                                                {"label": "Z", "value": "z"},
                                            ],
                                            value=["x", "y", "z"],
                                            inline=True,
                                            className="checklist",
                                        ),
                                    ),
                                    _field(
                                        "Noise parameterization",
                                        dcc.Dropdown(
                                            id="ds-parameterization",
                                            options=[
                                                {"label": "Super-global · 1 γ", "value": "super_global"},
                                                {"label": "Global · γx, γy, γz", "value": "global"},
                                                {"label": "Local · 3L parameters", "value": "local"},
                                            ],
                                            value="global",
                                            clearable=False,
                                        ),
                                    ),
                                    _field("γ minimum", _number("ds-gamma-min", 1e-3, min_value=0, step="any")),
                                    _field("γ maximum", _number("ds-gamma-max", 1e-1, min_value=0, step="any")),
                                    _field("Elapsed time", _number("ds-elapsed-time", 5.0, min_value=0, step="any")),
                                    _field("dt", _number("ds-dt", 0.1, min_value=0, step="any")),
                                    _field(
                                        "YAQS method",
                                        dcc.Dropdown(
                                            id="ds-method",
                                            options=[
                                                {"label": "TJM / MPS", "value": "tjm"},
                                                {"label": "MCWF / Vector", "value": "mcwf"},
                                                {"label": "Exact Lindblad / Density matrix", "value": "lindblad"},
                                            ],
                                            value="tjm",
                                            clearable=False,
                                        ),
                                    ),
                                    _field(
                                        "Simulation preset",
                                        dcc.Dropdown(
                                            id="ds-preset",
                                            options=[{"label": v.title(), "value": v} for v in ["fast", "balanced", "accurate", "exact"]],
                                            value="fast",
                                            clearable=False,
                                        ),
                                    ),
                                    _field("Trajectories", _number("ds-num-traj", 100, min_value=1, step=1)),
                                    _field("Initial state", _text("ds-initial-state", "zeros")),
                                    _field("Ising J", _number("ds-j", 1.0, step="any")),
                                    _field("Transverse field g", _number("ds-g", 1.0, step="any")),
                                    _field("Seed", _number("ds-seed", 1234, min_value=0, step=1)),
                                    _field("Trotter order", _number("ds-order", 2, min_value=1, step=1)),
                                    _field("TDVP sweeps", _number("ds-tdvp-sweeps", 1, min_value=1, step=1)),
                                    _field(
                                        "TDVP mode",
                                        dcc.Dropdown(
                                            id="ds-tdvp-mode",
                                            options=[
                                                {"label": "1-site", "value": "1site"},
                                                {"label": "2-site", "value": "2site"},
                                                {"label": "Dynamic", "value": "dynamic"},
                                            ],
                                            value="2site",
                                            clearable=False,
                                        ),
                                    ),
                                    _field(
                                        "Parallel YAQS trajectories",
                                        dcc.Checklist(
                                            id="ds-parallel",
                                            options=[{"label": "Enable", "value": "parallel"}],
                                            value=["parallel"],
                                        ),
                                    ),
                                    _field(
                                        "CPU workers",
                                        _number(
                                            "ds-max-workers",
                                            max(1, (os.cpu_count() or 2) - 1),
                                            min_value=1,
                                            max_value=os.cpu_count() or 1,
                                            step=1,
                                        ),
                                    ),
                                ],
                                className="form-grid",
                            ),
                            html.Button("Generate dataset", id="generate-dataset-btn", n_clicks=0, className="primary-button"),
                            html.Div(id="dataset-job-status", className="job-status"),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [html.H2("Dataset Catalog"), html.Button("Refresh", id="refresh-datasets-btn", className="secondary-button")],
                                className="section-header",
                            ),
                            dash_table.DataTable(
                                id="dataset-table",
                                columns=[
                                    {"name": "Name", "id": "name"},
                                    {"name": "Samples", "id": "samples"},
                                    {"name": "O", "id": "observables"},
                                    {"name": "T", "id": "times"},
                                    {"name": "Sites", "id": "sites"},
                                    {"name": "Parameterization", "id": "parameterization"},
                                    {"name": "Targets", "id": "targets"},
                                    {"name": "Path", "id": "path"},
                                ],
                                data=[],
                                page_size=8,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "backgroundColor": "transparent",
                                    "color": "#d9e3f0",
                                    "border": "1px solid #233044",
                                    "fontFamily": "Inter, sans-serif",
                                    "fontSize": 12,
                                    "textAlign": "left",
                                    "padding": "10px",
                                },
                                style_header={"backgroundColor": "#111a28", "fontWeight": 700, "color": "#f6f8fb"},
                            ),
                            html.Div(
                                [
                                    _field("Preview dataset", dcc.Dropdown(id="dataset-preview-path", options=[], clearable=False)),
                                    _field("Sample index", _number("dataset-preview-sample", 0, min_value=0, step=1)),
                                    _field("Observable index", _number("dataset-preview-observable", 0, min_value=0, step=1)),
                                ],
                                className="preview-controls",
                            ),
                            dcc.Graph(id="dataset-preview-graph", config={"displaylogo": False}),
                            html.Pre(id="dataset-metadata", className="json-box"),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column",
            )
        ],
        className="tab-content",
    )


def _training_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Train Model"),
                            html.P(
                                "Every training job automatically reconstructs held-out trajectories through YAQS.",
                                className="section-copy",
                            ),
                            _field("Dataset", dcc.Dropdown(id="train-dataset", options=[], clearable=False)),
                            _field(
                                "Model",
                                dcc.Dropdown(
                                    id="train-model",
                                    options=available_model_options(),
                                    value="extra_trees",
                                    clearable=False,
                                ),
                            ),
                            html.Div(
                                [
                                    _field("Seed", _number("train-seed", 1234, min_value=0, step=1)),
                                    _field("Train fraction", _number("train-fraction", 0.60, min_value=0.05, max_value=0.90, step=0.05)),
                                    _field("Validation fraction", _number("validation-fraction", 0.20, min_value=0.05, max_value=0.45, step=0.05)),
                                    _field("Reconstruction samples", _number("reconstruction-samples", 3, min_value=1, step=1)),
                                    _field(
                                        "Feature mode",
                                        dcc.Dropdown(
                                            id="feature-mode",
                                            options=[
                                                {"label": "Flatten O×T", "value": "flatten"},
                                                {"label": "Statistics", "value": "stats"},
                                                {"label": "Flatten + statistics", "value": "both"},
                                            ],
                                            value="flatten",
                                            clearable=False,
                                        ),
                                    ),
                                    _field("Trees / estimators", _number("n-estimators", 500, min_value=10, step=10)),
                                    _field("KNN neighbors", _number("n-neighbors", 7, min_value=1, step=1)),
                                    _field("PCA components", _number("pca-components", 64, min_value=1, step=1)),
                                    _field("Epochs", _number("train-epochs", 150, min_value=1, step=10)),
                                    _field("Patience", _number("train-patience", 20, min_value=1, step=1)),
                                    _field("Batch size", _number("train-batch-size", 32, min_value=1, step=1)),
                                    _field("Learning rate", _number("learning-rate", 1e-3, min_value=0, step="any")),
                                    _field("Weight decay", _number("weight-decay", 1e-4, min_value=0, step="any")),
                                    _field("LSTM hidden size", _number("hidden-size", 128, min_value=1, step=16)),
                                    _field("LSTM layers", _number("num-layers", 2, min_value=1, step=1)),
                                    _field("Dropout", _number("dropout", 0.1, min_value=0, max_value=0.9, step="any")),
                                    _field("Device", _text("train-device", "", "Blank = auto, or cpu / cuda")),
                                ],
                                className="form-grid",
                            ),
                            html.Button("Train + reconstruct", id="train-btn", n_clicks=0, className="primary-button"),
                            html.Div(id="training-job-status", className="job-status"),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H2("Pipeline"),
                            html.Div(
                                [
                                    html.Div("NoiseDataset", className="flow-node"),
                                    html.Div("→", className="flow-arrow"),
                                    html.Div("ML model", className="flow-node"),
                                    html.Div("→", className="flow-arrow"),
                                    html.Div("Predicted γ", className="flow-node"),
                                    html.Div("→", className="flow-arrow"),
                                    html.Div("YAQS", className="flow-node"),
                                    html.Div("→", className="flow-arrow"),
                                    html.Div("Reconstructed trajectory", className="flow-node flow-primary"),
                                ],
                                className="flow",
                            ),
                            html.P(
                                "Parameter error is diagnostic. Reconstructed-trajectory fidelity is the primary digital-twin metric.",
                                className="callout",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column training-columns",
            )
        ],
        className="tab-content",
    )


def _results_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Results"),
                            html.Div(
                                [
                                    _field("Run", dcc.Dropdown(id="results-run", options=[], clearable=False)),
                                    html.Button("Refresh", id="refresh-runs-btn", className="secondary-button"),
                                ],
                                className="result-selector",
                            ),
                            html.Div(
                                [
                                    _metric_card("Trajectory RMSE", "metric-recon-rmse", "Primary score · lower is better"),
                                    _metric_card("Trajectory MAE", "metric-recon-mae"),
                                    _metric_card("Max trajectory error", "metric-recon-max"),
                                    _metric_card("log10 γ RMSE", "metric-log-rmse", "Diagnostic"),
                                    _metric_card("Median factor error", "metric-factor-error", "Diagnostic"),
                                ],
                                className="metric-grid",
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="training-history-graph", config={"displaylogo": False}),
                                    dcc.Graph(id="leaderboard-graph", config={"displaylogo": False}),
                                ],
                                className="plot-grid",
                            ),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H2("Trajectory Reconstruction"),
                            html.Div(
                                [
                                    _field("Reconstructed test sample", dcc.Dropdown(id="reconstruction-sample", options=[], clearable=False)),
                                    _field("Observable", _number("reconstruction-observable", 0, min_value=0, step=1)),
                                ],
                                className="preview-controls result-preview-controls",
                            ),
                            dcc.Graph(id="reconstruction-graph", config={"displaylogo": False}),
                            html.H3("Noise parameters"),
                            dash_table.DataTable(
                                id="parameter-table",
                                columns=[
                                    {"name": "Parameter", "id": "parameter"},
                                    {"name": "True γ", "id": "true_gamma"},
                                    {"name": "Predicted γ", "id": "predicted_gamma"},
                                    {"name": "Factor error", "id": "factor_error"},
                                ],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "backgroundColor": "transparent",
                                    "color": "#d9e3f0",
                                    "border": "1px solid #233044",
                                    "fontFamily": "Inter, sans-serif",
                                    "fontSize": 12,
                                    "padding": "10px",
                                },
                                style_header={"backgroundColor": "#111a28", "fontWeight": 700, "color": "#f6f8fb"},
                            ),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H2("Cross-model Trajectory Comparison"),
                            html.P(
                                "Overlay the real held-out trajectory with reconstructions from multiple trained models "
                                "for the same dataset sample. The sample selector only shows samples reconstructed by all selected runs.",
                                className="section-copy",
                            ),
                            html.Div(
                                [
                                    _field(
                                        "Models / runs",
                                        dcc.Dropdown(
                                            id="comparison-runs",
                                            options=[],
                                            value=[],
                                            multi=True,
                                            placeholder="Select LSTM, CNN, classical models, ...",
                                        ),
                                    ),
                                    _field(
                                        "Common reconstructed sample",
                                        dcc.Dropdown(
                                            id="comparison-sample",
                                            options=[],
                                            clearable=False,
                                            placeholder="Select a common sample",
                                        ),
                                    ),
                                    _field(
                                        "Site / qubit",
                                        _number("comparison-site", 0, min_value=0, step=1),
                                        "Plots X, Y and Z for this site when those observables are available.",
                                    ),
                                ],
                                className="preview-controls result-preview-controls",
                            ),
                            html.Div(id="comparison-status", className="field-help"),
                            dcc.Graph(
                                id="comparison-trajectory-graph",
                                figure=_empty_figure("Cross-model trajectory comparison"),
                                config={
                                    "displaylogo": False,
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "trajectory_model_comparison",
                                        "scale": 3,
                                    },
                                },
                            ),
                            html.H3("Selected-sample reconstruction error"),
                            dash_table.DataTable(
                                id="comparison-metrics-table",
                                columns=[
                                    {"name": "Model", "id": "model"},
                                    {"name": "Trajectory RMSE", "id": "trajectory_rmse"},
                                    {"name": "Trajectory MAE", "id": "trajectory_mae"},
                                    {"name": "Run", "id": "run"},
                                ],
                                data=[],
                                sort_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "backgroundColor": "transparent",
                                    "color": "#d9e3f0",
                                    "border": "1px solid #233044",
                                    "fontFamily": "Inter, sans-serif",
                                    "fontSize": 12,
                                    "padding": "10px",
                                },
                                style_header={"backgroundColor": "#111a28", "fontWeight": 700, "color": "#f6f8fb"},
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="results-stack",
            )
        ],
        className="tab-content",
    )

def create_layout(data_root: str, output_root: str):
    return html.Div(
        [
            dcc.Store(id="dataset-job-store"),
            dcc.Store(id="training-job-store"),
            dcc.Interval(id="job-poller", interval=1200, n_intervals=0),
            dcc.Interval(id="catalog-poller", interval=5000, n_intervals=0),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("QEL", className="brand-mark"),
                            html.Div(
                                [
                                    html.H1("Twin Control Center"),
                                    html.P("Dataset → training → YAQS reconstruction → benchmark", className="brand-subtitle"),
                                ]
                            ),
                        ],
                        className="brand",
                    ),
                    html.Div(
                        [html.Span("Data root"), html.Code(data_root), html.Span("Results root"), html.Code(output_root)],
                        className="path-strip",
                    ),
                ],
                className="topbar",
            ),
            dcc.Tabs(
                id="main-tabs",
                value="dataset",
                children=[
                    dcc.Tab(label="1 · Dataset", value="dataset", children=_dataset_tab()),
                    dcc.Tab(label="2 · Train", value="train", children=_training_tab()),
                    dcc.Tab(label="3 · Results", value="results", children=_results_tab()),
                    # CHANGED: fourth tab uses the new benchmark module.
                    dcc.Tab(label="4 · Benchmark", value="benchmark", children=create_benchmark_tab()),
                ],
                className="tabs",
            ),
        ],
        className="app-shell",
    )


def register_callbacks(app: Dash, *, data_root: str, output_root: str) -> None:
    @app.callback(
        Output("dataset-job-store", "data"),
        Output("dataset-job-status", "children"),
        Input("generate-dataset-btn", "n_clicks"),
        State("ds-name", "value"), State("ds-output", "value"), State("ds-samples", "value"), State("ds-sites", "value"),
        State("ds-channels", "value"), State("ds-parameterization", "value"), State("ds-gamma-min", "value"), State("ds-gamma-max", "value"),
        State("ds-elapsed-time", "value"), State("ds-dt", "value"), State("ds-method", "value"), State("ds-preset", "value"),
        State("ds-num-traj", "value"), State("ds-initial-state", "value"), State("ds-j", "value"), State("ds-g", "value"),
        State("ds-seed", "value"), State("ds-order", "value"), State("ds-tdvp-sweeps", "value"), State("ds-tdvp-mode", "value"),
        State("ds-parallel", "value"), State("ds-max-workers", "value"),
        prevent_initial_call=True,
    )
    def start_dataset_job(n_clicks, name, output, samples, sites, channels, parameterization, gamma_min, gamma_max, elapsed_time, dt, method, preset, num_traj, initial_state, j_coupling, transverse_field, seed, order, tdvp_sweeps, tdvp_mode, parallel_values, max_workers):
        if not n_clicks:
            return no_update, no_update
        try:
            gamma_min, gamma_max = float(gamma_min), float(gamma_max)
            elapsed_time, dt = float(elapsed_time), float(dt)
            if gamma_min <= 0:
                return no_update, "γ minimum must be greater than zero."
            if gamma_max <= gamma_min:
                return no_update, "γ maximum must be greater than γ minimum."
            if elapsed_time <= 0 or dt <= 0 or dt > elapsed_time:
                return no_update, "Check elapsed time and dt."
            config = {
                "dataset_name": name,
                "output_path": output,
                "num_samples": int(samples),
                "num_sites": int(sites),
                "channels": channels or [],
                "parameterization": parameterization,
                "gamma_min": gamma_min,
                "gamma_max": gamma_max,
                "elapsed_time": elapsed_time,
                "dt": dt,
                "method": method,
                "preset": preset,
                "num_traj": int(num_traj),
                "initial_state": initial_state,
                "j_coupling": float(j_coupling),
                "transverse_field": float(transverse_field),
                "seed": int(seed),
                "order": int(order),
                "tdvp_sweeps": int(tdvp_sweeps),
                "tdvp_mode": tdvp_mode,
                "parallel": "parallel" in (parallel_values or []),
                "max_workers": int(max_workers),
                "show_progress": False,
            }
        except (TypeError, ValueError) as exc:
            return no_update, f"Invalid dataset configuration: {exc}"
        job_id = JOB_MANAGER.submit("dataset", create_dataset_job, data_root=data_root, config=config)
        return {"job_id": job_id}, f"Dataset job queued · parallel={config['parallel']} · workers={config['max_workers']}"

    @app.callback(
        Output("training-job-store", "data"),
        Output("training-job-status", "children"),
        Input("train-btn", "n_clicks"),
        State("train-dataset", "value"), State("train-model", "value"), State("train-seed", "value"), State("train-fraction", "value"),
        State("validation-fraction", "value"), State("reconstruction-samples", "value"), State("feature-mode", "value"), State("n-estimators", "value"),
        State("n-neighbors", "value"), State("pca-components", "value"), State("train-epochs", "value"), State("train-patience", "value"),
        State("train-batch-size", "value"), State("learning-rate", "value"), State("weight-decay", "value"), State("hidden-size", "value"),
        State("num-layers", "value"), State("dropout", "value"), State("train-device", "value"),
        prevent_initial_call=True,
    )
    def start_training_job(n_clicks, dataset_path, model_name, seed, train_fraction, validation_fraction, reconstruction_samples, feature_mode, n_estimators, n_neighbors, pca_components, epochs, patience, batch_size, learning_rate, weight_decay, hidden_size, num_layers, dropout, device):
        if not n_clicks:
            return no_update, no_update
        if not dataset_path:
            return no_update, "Select a dataset first."
        if float(train_fraction) + float(validation_fraction) >= 1.0:
            return no_update, "Train + validation fractions must sum to less than 1."
        config = {
            "model_name": model_name,
            "seed": int(seed),
            "train_fraction": float(train_fraction),
            "validation_fraction": float(validation_fraction),
            "reconstruction_samples": max(1, int(reconstruction_samples)),
            "feature_mode": feature_mode,
            "n_estimators": int(n_estimators),
            "n_neighbors": int(n_neighbors),
            "pca_components": int(pca_components),
            "epochs": int(epochs),
            "patience": int(patience),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "device": (device or "").strip() or None,
        }
        job_id = JOB_MANAGER.submit("training", train_model_job, dataset_path=dataset_path, output_root=output_root, config=config)
        return {"job_id": job_id}, "Training + reconstruction job queued."

    @app.callback(
        Output("dataset-job-status", "children", allow_duplicate=True),
        Output("training-job-status", "children", allow_duplicate=True),
        Input("job-poller", "n_intervals"),
        State("dataset-job-store", "data"),
        State("training-job-store", "data"),
        prevent_initial_call=True,
    )
    def poll_jobs(_tick, dataset_store, training_store):
        def status_text(store):
            if not store:
                return no_update
            job = JOB_MANAGER.get(store.get("job_id"))
            if job is None:
                return "Job state unavailable."
            if job["status"] == "failed":
                return f"FAILED\n\n{job['error']}"
            if job["status"] == "complete":
                return "COMPLETE\n\n" + json.dumps(job["result"], indent=2)
            return f"{job['status'].upper()} · {job['message']}"
        return status_text(dataset_store), status_text(training_store)

    @app.callback(
        Output("dataset-table", "data"),
        Output("dataset-preview-path", "options"), Output("dataset-preview-path", "value"),
        Output("train-dataset", "options"), Output("train-dataset", "value"),
        Input("catalog-poller", "n_intervals"), Input("refresh-datasets-btn", "n_clicks"),
        State("dataset-preview-path", "value"), State("train-dataset", "value"),
    )
    def refresh_datasets(_tick, _clicks, current_preview, current_train):
        rows = scan_datasets(data_root)
        options = [{"label": f"{row['name']} · {row['parameterization']} · {row['samples']} samples", "value": row["path"]} for row in rows]
        values = {option["value"] for option in options}
        preview = current_preview if current_preview in values else (options[0]["value"] if options else None)
        train = current_train if current_train in values else (options[0]["value"] if options else None)
        return rows, options, preview, options, train

    @app.callback(
        Output("dataset-preview-graph", "figure"), Output("dataset-metadata", "children"),
        Input("dataset-preview-path", "value"), Input("dataset-preview-sample", "value"), Input("dataset-preview-observable", "value"),
    )
    def update_dataset_preview(path, sample_index, observable_index):
        if not path:
            return _empty_figure("Dataset preview"), "No dataset selected."
        try:
            preview = load_dataset_preview(path, int(sample_index or 0), int(observable_index or 0))
            details = dataset_details(path)
        except Exception as exc:
            return _empty_figure("Dataset preview"), f"ERROR: {exc}"
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=preview["times"], y=preview["values"], mode="lines", name="Expectation value"))
        figure.update_layout(title=f"Sample {preview['sample_index']} · Observable {preview['observable_index']}", xaxis_title="Time", yaxis_title="Expectation value", **PLOT_LAYOUT)
        return figure, json.dumps(details, indent=2)

    @app.callback(
        Output("results-run", "options"), Output("results-run", "value"), Output("leaderboard-graph", "figure"),
        Input("catalog-poller", "n_intervals"), Input("refresh-runs-btn", "n_clicks"), State("results-run", "value"),
    )
    def refresh_runs(_tick, _clicks, current_run):
        runs = scan_runs(output_root)
        options = [
            {"label": f"{row['model']} · {row['dataset']} · RMSE={_format_metric(row['reconstruction_rmse'])}", "value": row["run_dir"]}
            for row in runs
        ]
        values = {option["value"] for option in options}
        selected = current_run if current_run in values else (options[0]["value"] if options else None)
        leaderboard_rows = [row for row in runs if row["reconstruction_rmse"] is not None]
        leaderboard_rows.sort(key=lambda row: float(row["reconstruction_rmse"]))
        figure = go.Figure()
        if leaderboard_rows:
            figure.add_trace(go.Bar(x=[f"{row['model']} · {row['dataset']}" for row in leaderboard_rows], y=[float(row["reconstruction_rmse"]) for row in leaderboard_rows], name="Trajectory RMSE"))
        figure.update_layout(title="Reconstruction leaderboard", xaxis_title="Run", yaxis_title="Mean trajectory RMSE", **PLOT_LAYOUT)
        return options, selected, figure

    @app.callback(
        Output("metric-recon-rmse", "children"), Output("metric-recon-mae", "children"), Output("metric-recon-max", "children"),
        Output("metric-log-rmse", "children"), Output("metric-factor-error", "children"), Output("training-history-graph", "figure"),
        Output("reconstruction-sample", "options"), Output("reconstruction-sample", "value"),
        Input("results-run", "value"),
    )
    def update_run_summary(run_dir):
        if not run_dir:
            return "—", "—", "—", "—", "—", _empty_figure("Training history"), [], None
        try:
            details = load_run_details(run_dir)
        except Exception:
            return "—", "—", "—", "—", "—", _empty_figure("Training history"), [], None
        metrics = details["metrics"]
        test_metrics = metrics.get("test_metrics", {}) or {}
        reconstruction = details["reconstruction"] or metrics.get("reconstruction", {})
        figure = go.Figure()
        history = details["history"]
        if history:
            keys = set(history[0])
            epoch_key = "epoch" if "epoch" in keys else None
            train_key = "training_loss" if "training_loss" in keys else ("train_loss" if "train_loss" in keys else None)
            val_key = "validation_loss" if "validation_loss" in keys else ("val_loss" if "val_loss" in keys else None)
            if epoch_key and train_key:
                figure.add_trace(go.Scatter(x=[float(row[epoch_key]) for row in history], y=[float(row[train_key]) for row in history], mode="lines", name="Train"))
            if epoch_key and val_key:
                figure.add_trace(go.Scatter(x=[float(row[epoch_key]) for row in history], y=[float(row[val_key]) for row in history], mode="lines", name="Validation"))
        figure.update_layout(title="Training history", xaxis_title="Epoch", yaxis_title="Loss", **PLOT_LAYOUT)
        sample_options = [{"label": row["label"], "value": row["value"]} for row in details["samples"]]
        sample_value = sample_options[0]["value"] if sample_options else None
        return (
            _format_metric(reconstruction.get("trajectory_rmse_mean")),
            _format_metric(reconstruction.get("trajectory_mae_mean")),
            _format_metric(reconstruction.get("max_abs_trajectory_error_mean")),
            _format_metric(test_metrics.get("rmse_log10_mean", test_metrics.get("log10_rmse"))),
            _format_metric(test_metrics.get("median_factor_error_gamma", test_metrics.get("median_factor_error"))),
            figure,
            sample_options,
            sample_value,
        )

    @app.callback(
        Output("reconstruction-graph", "figure"), Output("parameter-table", "data"),
        Input("reconstruction-sample", "value"), Input("reconstruction-observable", "value"),
    )
    def update_reconstruction(sample_path, observable_index):
        if not sample_path:
            return _empty_figure("Reconstructed trajectory"), []
        try:
            sample = load_reconstruction_sample(sample_path, int(observable_index or 0))
        except Exception as exc:
            return _empty_figure(f"Reconstruction error: {exc}"), []
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=sample["times"], y=sample["original"], mode="lines", name="Original"))
        figure.add_trace(go.Scatter(x=sample["times"], y=sample["reconstructed"], mode="lines", name="Reconstructed"))
        figure.update_layout(title=f"Dataset sample {sample['dataset_index']} · Observable {sample['observable_index']}", xaxis_title="Time", yaxis_title="Expectation value", **PLOT_LAYOUT)
        table = []
        for name, true_gamma, predicted_gamma in zip(sample["parameter_names"], sample["true_gamma"], sample["predicted_gamma"], strict=True):
            factor = max(float(predicted_gamma) / max(float(true_gamma), 1e-300), float(true_gamma) / max(float(predicted_gamma), 1e-300))
            table.append({
                "parameter": name,
                "true_gamma": f"{float(true_gamma):.6e}",
                "predicted_gamma": f"{float(predicted_gamma):.6e}",
                "factor_error": f"{factor:.4f}",
            })
        return figure, table


    @app.callback(
        Output("comparison-runs", "options"),
        Output("comparison-runs", "value"),
        Input("catalog-poller", "n_intervals"),
        Input("refresh-runs-btn", "n_clicks"),
        State("comparison-runs", "value"),
    )
    def refresh_comparison_runs(_tick, _clicks, current_runs):
        runs = [row for row in scan_runs(output_root) if int(row.get("reconstruction_samples") or 0) > 0]
        options = [
            {
                "label": f"{row['model']} · {row['dataset']} · RMSE={_format_metric(row['reconstruction_rmse'])}",
                "value": row["run_dir"],
            }
            for row in runs
        ]
        valid = {option["value"] for option in options}
        selected = [value for value in (current_runs or []) if value in valid]

        if not selected:
            preferred_names = ("lstm", "bilstm", "gradient_boosting", "gaussian_process", "cnn2d")
            by_name = {}
            for row in runs:
                by_name.setdefault(str(row["model"]).lower(), row["run_dir"])
            selected = [by_name[name] for name in preferred_names if name in by_name][:5]
            if not selected:
                selected = [option["value"] for option in options[:4]]

        return options, selected

    @app.callback(
        Output("comparison-sample", "options"),
        Output("comparison-sample", "value"),
        Output("comparison-status", "children"),
        Input("comparison-runs", "value"),
        State("comparison-sample", "value"),
    )
    def refresh_comparison_samples(run_dirs, current_sample):
        run_dirs = list(run_dirs or [])
        if not run_dirs:
            return [], None, "Select at least one model run."

        try:
            samples = common_reconstruction_samples(run_dirs)
        except Exception as exc:
            return [], None, f"Could not inspect reconstruction samples: {exc}"

        options = [{"label": f"Dataset sample {index}", "value": int(index)} for index in samples]
        values = {option["value"] for option in options}
        selected = int(current_sample) if current_sample in values else (options[0]["value"] if options else None)

        if not options:
            return [], None, (
                "No common reconstructed sample exists across these runs. "
                "Re-run reconstruction for the same held-out samples or increase the reconstruction sample count."
            )
        return options, selected, f"{len(options)} common reconstructed sample(s)."

    @app.callback(
        Output("comparison-trajectory-graph", "figure"),
        Output("comparison-metrics-table", "data"),
        Output("comparison-status", "children", allow_duplicate=True),
        Input("comparison-runs", "value"),
        Input("comparison-sample", "value"),
        Input("comparison-site", "value"),
        prevent_initial_call=True,
    )
    def update_model_comparison(run_dirs, dataset_index, site):
        run_dirs = list(run_dirs or [])
        if not run_dirs or dataset_index is None:
            return _empty_figure("Cross-model trajectory comparison"), [], "Select runs and a common sample."

        try:
            comparison = load_reconstruction_comparison(
                run_dirs=run_dirs,
                dataset_index=int(dataset_index),
            )
        except Exception as exc:
            return _empty_figure(f"Comparison error: {exc}"), [], f"Comparison failed: {exc}"

        time = np.asarray(comparison["times"], dtype=np.float64)
        original = np.asarray(comparison["original"], dtype=np.float64)
        site = max(0, int(site or 0))

        observable_map = comparison.get("observable_map", {}) or {}
        panel_specs = []
        for channel in ("x", "y", "z"):
            channel_map = observable_map.get(channel, {}) or {}
            if site in channel_map:
                panel_specs.append((channel.upper(), int(channel_map[site])))

        if not panel_specs:
            # Safe fallback for legacy reconstruction files without dataset metadata.
            if original.shape[0] % 3 == 0:
                num_sites = original.shape[0] // 3
                site = min(site, num_sites - 1)
                panel_specs = [
                    ("X", site),
                    ("Y", num_sites + site),
                    ("Z", 2 * num_sites + site),
                ]
            else:
                panel_specs = [("Observable 0", 0)]

        figure = make_subplots(
            rows=len(panel_specs),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[f"⟨{name}_{site + 1}(t)⟩" if len(name) == 1 else name for name, _ in panel_specs],
        )

        for row_index, (_name, observable_index) in enumerate(panel_specs, start=1):
            figure.add_trace(
                go.Scatter(
                    x=time,
                    y=original[observable_index],
                    mode="lines",
                    name="Real / reference",
                    legendgroup="reference",
                    showlegend=row_index == 1,
                    line={"width": 4},
                ),
                row=row_index,
                col=1,
            )

            for model in comparison["models"]:
                reconstructed = np.asarray(model["reconstructed"], dtype=np.float64)
                figure.add_trace(
                    go.Scatter(
                        x=time,
                        y=reconstructed[observable_index],
                        mode="lines",
                        name=model["label"],
                        legendgroup=model["run_dir"],
                        showlegend=row_index == 1,
                    ),
                    row=row_index,
                    col=1,
                )

        figure.update_xaxes(title_text="Time", row=len(panel_specs), col=1)
        figure.update_yaxes(title_text="Expectation value")
        figure.update_layout(
            title=f"Dataset sample {int(dataset_index)} · site {site + 1}",
            height=max(420, 240 * len(panel_specs)),
            **PLOT_LAYOUT,
        )

        table = [
            {
                "model": model["label"],
                "trajectory_rmse": _format_metric(model["trajectory_rmse"]),
                "trajectory_mae": _format_metric(model["trajectory_mae"]),
                "run": Path(model["run_dir"]).name,
            }
            for model in sorted(comparison["models"], key=lambda row: row["trajectory_rmse"])
        ]

        return figure, table, f"Comparing {len(comparison['models'])} model(s) on dataset sample {int(dataset_index)}."


def create_app(*, data_root: str | Path = "data/noise_datasets", output_root: str | Path = "outputs/noise_ml_runs") -> Dash:
    package_dir = Path(__file__).resolve().parent
    app = Dash(__name__, assets_folder=str(package_dir / "assets"), suppress_callback_exceptions=True)
    app.title = "QEL Twin Control Center"
    app.layout = create_layout(str(data_root), str(output_root))
    register_callbacks(app, data_root=str(data_root), output_root=str(output_root))
    register_benchmark_callbacks(app, data_root=str(data_root), output_root=str(output_root))
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="QEL Twin end-to-end dataset, training, and reconstruction UI.")
    parser.add_argument("--data-dir", default="data/noise_datasets")
    parser.add_argument("--results-dir", default="outputs/noise_ml_runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app = create_app(data_root=args.data_dir, output_root=args.results_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
