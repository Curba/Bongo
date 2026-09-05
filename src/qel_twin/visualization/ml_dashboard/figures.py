from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from qel_twin.visualization.ml_dashboard.data_loader import PARAMS


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 18},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=420)
    return fig


def model_metric_bar(
    metrics_df: pd.DataFrame,
    metric: str,
    title: str,
    ascending: bool,
) -> go.Figure:
    if metrics_df.empty or metric not in metrics_df.columns:
        return empty_figure(f"No metric found: {metric}")

    df = metrics_df.sort_values(metric, ascending=ascending)

    fig = px.bar(
        df,
        x="model",
        y=metric,
        hover_data=["dataset", "run_id", "split"],
        title=title,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=430,
        margin=dict(l=30, r=30, t=70, b=120),
    )
    return fig


def true_vs_pred_log(pred_df: pd.DataFrame, param: str) -> go.Figure:
    true_col = f"{param}_true_log10"
    pred_col = f"{param}_pred_log10"

    if pred_df.empty or true_col not in pred_df.columns or pred_col not in pred_df.columns:
        return empty_figure("No prediction data found for this split/run.")

    mn = min(pred_df[true_col].min(), pred_df[pred_col].min())
    mx = max(pred_df[true_col].max(), pred_df[pred_col].max())

    fig = px.scatter(
        pred_df,
        x=true_col,
        y=pred_col,
        hover_data=["sample_id"],
        title=f"{param}: true vs predicted log10(gamma)",
        labels={
            true_col: "True log10(gamma)",
            pred_col: "Predicted log10(gamma)",
        },
    )
    fig.add_trace(
        go.Scatter(
            x=[mn, mx],
            y=[mn, mx],
            mode="lines",
            name="perfect prediction",
        )
    )
    fig.update_layout(height=500)
    return fig


def true_vs_pred_gamma(pred_df: pd.DataFrame, param: str) -> go.Figure:
    true_col = f"{param}_true"
    pred_col = f"{param}_pred"

    if pred_df.empty or true_col not in pred_df.columns or pred_col not in pred_df.columns:
        return empty_figure("No gamma-scale prediction data found.")

    mn = min(pred_df[true_col].min(), pred_df[pred_col].min())
    mx = max(pred_df[true_col].max(), pred_df[pred_col].max())

    fig = px.scatter(
        pred_df,
        x=true_col,
        y=pred_col,
        hover_data=["sample_id"],
        title=f"{param}: true vs predicted gamma",
        labels={
            true_col: "True gamma",
            pred_col: "Predicted gamma",
        },
    )
    fig.add_trace(
        go.Scatter(
            x=[mn, mx],
            y=[mn, mx],
            mode="lines",
            name="perfect prediction",
        )
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.update_layout(height=500)
    return fig


def abs_error_histogram(pred_df: pd.DataFrame, param: str) -> go.Figure:
    col = f"{param}_abs_error_log10"

    if pred_df.empty or col not in pred_df.columns:
        return empty_figure("No absolute error column found.")

    fig = px.histogram(
        pred_df,
        x=col,
        nbins=40,
        title=f"{param}: absolute log10 error distribution",
        labels={col: "|prediction - truth| in log10 scale"},
    )
    fig.update_layout(height=430)
    return fig


def relative_error_box(pred_df: pd.DataFrame, parameter_names: list[str] | None = None) -> go.Figure:
    if pred_df.empty:
        return empty_figure("No prediction data found.")

    rows = []

    for p in parameter_names or PARAMS:
        col = f"{p}_relative_error"
        if col not in pred_df.columns:
            continue

        rows.append(
            pd.DataFrame(
                {
                    "parameter": p,
                    "relative_error_percent": 100.0 * pred_df[col],
                }
            )
        )

    if not rows:
        return empty_figure("No relative error columns found.")

    long_df = pd.concat(rows, ignore_index=True)

    fig = px.box(
        long_df,
        x="parameter",
        y="relative_error_percent",
        points="outliers",
        title="Relative gamma error by parameter",
        labels={
            "relative_error_percent": "Relative error (%)",
            "parameter": "Parameter",
        },
    )
    fig.update_layout(height=430)
    return fig


def per_target_metrics(metrics_row: pd.Series, parameter_names: list[str] | None = None) -> go.Figure:
    rows = []

    for p in parameter_names or PARAMS:
        rows.append(
            {
                "parameter": p,
                "mae_log10": metrics_row.get(f"{p}_mae_log10", np.nan),
                "r2_log10": metrics_row.get(f"{p}_r2_log10", np.nan),
                "median_relative_error_percent": 100.0
                * metrics_row.get(f"{p}_median_relative_error_gamma", np.nan),
            }
        )

    df = pd.DataFrame(rows)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["parameter"],
            y=df["mae_log10"],
            name="MAE log10",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["parameter"],
            y=df["median_relative_error_percent"],
            name="Median relative error (%)",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Per-target metrics",
        xaxis_title="Parameter",
        yaxis=dict(title="MAE log10"),
        yaxis2=dict(
            title="Median relative error (%)",
            overlaying="y",
            side="right",
        ),
        barmode="group",
        height=460,
    )

    return fig


def training_history_figure(history_df: pd.DataFrame) -> go.Figure:
    if history_df.empty:
        return empty_figure(
            "No training_history.csv found for this run. "
            "Classical models may not have epoch-wise training logs."
        )

    if "epoch" not in history_df.columns:
        return empty_figure("training_history.csv exists but has no epoch column.")

    fig = go.Figure()

    possible_cols = [
        "train_loss",
        "val_loss",
        "test_loss",
        "train_mae",
        "val_mae",
        "train_rmse",
        "val_rmse",
        "learning_rate",
    ]

    plotted = False

    for col in possible_cols:
        if col in history_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=history_df["epoch"],
                    y=history_df[col],
                    mode="lines+markers",
                    name=col,
                )
            )
            plotted = True

    if not plotted:
        return empty_figure(
            "training_history.csv found, but no supported metric columns were found."
        )

    fig.update_layout(
        title="Training history",
        xaxis_title="Epoch",
        yaxis_title="Metric value",
        height=480,
    )

    return fig
