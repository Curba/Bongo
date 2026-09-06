from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output

from qel_twin.visualization.ml_dashboard.data_loader import (
    get_run,
    load_all_metrics,
    load_parameter_names,
    load_predictions,
    load_training_history,
)
from qel_twin.visualization.ml_dashboard.figures import (
    abs_error_histogram,
    empty_figure,
    model_metric_bar,
    per_target_metrics,
    relative_error_box,
    training_history_figure,
    true_vs_pred_gamma,
    true_vs_pred_log,
)
from qel_twin.visualization.ml_dashboard.layout import metric_card


def register_callbacks(app, runs):
    runs_df = pd.DataFrame(
        [
            {
                "dataset": r.dataset,
                "model": r.model,
                "run_id": r.run_id,
                "path": str(r.path),
            }
            for r in runs
        ]
    )

    metrics_df = load_all_metrics(runs)

    @app.callback(
        Output("model-dropdown", "options"),
        Output("model-dropdown", "value"),
        Input("dataset-dropdown", "value"),
    )
    def update_models(dataset):
        filtered = runs_df[runs_df["dataset"] == dataset]
        models = sorted(filtered["model"].unique())

        options = [{"label": m, "value": m} for m in models]
        value = "hist_gradient_boosting" if "hist_gradient_boosting" in models else models[0]

        return options, value

    @app.callback(
        Output("run-dropdown", "options"),
        Output("run-dropdown", "value"),
        Input("dataset-dropdown", "value"),
        Input("model-dropdown", "value"),
    )
    def update_runs(dataset, model):
        filtered = runs_df[
            (runs_df["dataset"] == dataset)
            & (runs_df["model"] == model)
        ]

        run_ids = sorted(filtered["run_id"].unique(), reverse=True)

        options = [{"label": r, "value": r} for r in run_ids]
        value = run_ids[0] if run_ids else None

        return options, value

    @app.callback(
        Output("param-dropdown", "options"),
        Output("param-dropdown", "value"),
        Input("dataset-dropdown", "value"),
        Input("model-dropdown", "value"),
        Input("run-dropdown", "value"),
    )
    def update_parameters(dataset, model, run_id):
        if not dataset or not model or not run_id:
            return [], None
        run = get_run(runs, dataset=dataset, model=model, run_id=run_id)
        parameter_names = load_parameter_names(run)
        return [{"label": name, "value": name} for name in parameter_names], parameter_names[0]

    @app.callback(
        Output("kpi-cards", "children"),
        Output("mae-ranking", "figure"),
        Output("r2-ranking", "figure"),
        Output("relative-error-ranking", "figure"),
        Output("scatter-log", "figure"),
        Output("scatter-gamma", "figure"),
        Output("abs-error-hist", "figure"),
        Output("relative-error-box", "figure"),
        Output("per-target-metrics", "figure"),
        Output("training-history", "figure"),
        Output("metrics-table", "data"),
        Output("metrics-table", "columns"),
        Input("dataset-dropdown", "value"),
        Input("model-dropdown", "value"),
        Input("run-dropdown", "value"),
        Input("split-dropdown", "value"),
        Input("param-dropdown", "value"),
    )
    def update_dashboard(dataset, model, run_id, split, param):
        if not dataset or not model or not run_id:
            empty = empty_figure("No run selected.")
            return [], empty, empty, empty, empty, empty, empty, empty, empty, empty, [], []

        dataset_split_metrics = metrics_df[
            (metrics_df["dataset"] == dataset)
            & (metrics_df["split"] == split)
        ].copy()

        if dataset_split_metrics.empty:
            empty = empty_figure(f"No metrics found for dataset={dataset}, split={split}")
            return [], empty, empty, empty, empty, empty, empty, empty, empty, empty, [], []

        run_metrics = dataset_split_metrics[
            (dataset_split_metrics["model"] == model)
            & (dataset_split_metrics["run_id"] == run_id)
        ]

        if run_metrics.empty:
            selected_metrics = dataset_split_metrics.iloc[0]
        else:
            selected_metrics = run_metrics.iloc[0]

        cards = [
            metric_card("Dataset", str(dataset)),
            metric_card("Model", str(model)),
            metric_card("MAE log10", f"{selected_metrics['mae_log10_mean']:.4f}"),
            metric_card("R² log10", f"{selected_metrics['r2_log10_mean']:.4f}"),
            metric_card(
                "Median gamma error",
                f"{100.0 * selected_metrics['median_relative_error_gamma']:.2f}%",
            ),
        ]

        fig_mae = model_metric_bar(
            dataset_split_metrics,
            metric="mae_log10_mean",
            title=f"{dataset}: MAE log10 comparison on {split}",
            ascending=True,
        )

        fig_r2 = model_metric_bar(
            dataset_split_metrics,
            metric="r2_log10_mean",
            title=f"{dataset}: R² comparison on {split}",
            ascending=False,
        )

        fig_rel = model_metric_bar(
            dataset_split_metrics,
            metric="median_relative_error_gamma",
            title=f"{dataset}: median relative gamma error on {split}",
            ascending=True,
        )

        run = get_run(runs, dataset=dataset, model=model, run_id=run_id)
        parameter_names = load_parameter_names(run)

        pred_df = load_predictions(run, split=split)
        history_df = load_training_history(run)

        fig_scatter_log = true_vs_pred_log(pred_df, param)
        fig_scatter_gamma = true_vs_pred_gamma(pred_df, param)
        fig_abs_error = abs_error_histogram(pred_df, param)
        fig_rel_box = relative_error_box(pred_df, parameter_names)
        fig_per_target = per_target_metrics(selected_metrics, parameter_names)
        fig_history = training_history_figure(history_df)

        table_df = dataset_split_metrics.copy()
        numeric_cols = table_df.select_dtypes(include=[np.number]).columns
        table_df[numeric_cols] = table_df[numeric_cols].round(5)

        columns = [{"name": c, "id": c} for c in table_df.columns]

        return (
            cards,
            fig_mae,
            fig_r2,
            fig_rel,
            fig_scatter_log,
            fig_scatter_gamma,
            fig_abs_error,
            fig_rel_box,
            fig_per_target,
            fig_history,
            table_df.to_dict("records"),
            columns,
        )
