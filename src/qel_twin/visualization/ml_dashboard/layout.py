from __future__ import annotations

from dash import dash_table, dcc, html


def metric_card(title: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "13px", "color": "#666"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "700"}),
        ],
        style={
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "padding": "14px",
            "backgroundColor": "white",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
        },
    )


def create_layout(datasets: list[str]) -> html.Div:
    default_dataset = datasets[0] if datasets else None

    return html.Div(
        [
            html.H1("QEL Twin ML Visualizer"),
            html.Div(
                "Compare classical ML and future neural-network models across datasets, runs, and splits.",
                style={"color": "#666", "marginBottom": "20px"},
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Dataset"),
                            dcc.Dropdown(
                                id="dataset-dropdown",
                                options=[{"label": d, "value": d} for d in datasets],
                                value=default_dataset,
                                clearable=False,
                            ),
                        ],
                        style={"width": "22%"},
                    ),
                    html.Div(
                        [
                            html.Label("Model"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                clearable=False,
                            ),
                        ],
                        style={"width": "24%"},
                    ),
                    html.Div(
                        [
                            html.Label("Run"),
                            dcc.Dropdown(
                                id="run-dropdown",
                                clearable=False,
                            ),
                        ],
                        style={"width": "28%"},
                    ),
                    html.Div(
                        [
                            html.Label("Split"),
                            dcc.Dropdown(
                                id="split-dropdown",
                                options=[
                                    {"label": "train", "value": "train"},
                                    {"label": "val", "value": "val"},
                                    {"label": "test", "value": "test"},
                                ],
                                value="test",
                                clearable=False,
                            ),
                        ],
                        style={"width": "12%"},
                    ),
                    html.Div(
                        [
                            html.Label("Target"),
                            dcc.Dropdown(
                                id="param-dropdown",
                                options=[],
                                value=None,
                                clearable=False,
                            ),
                        ],
                        style={"width": "12%"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "alignItems": "center",
                    "marginBottom": "22px",
                },
            ),

            html.Div(
                id="kpi-cards",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(5, 1fr)",
                    "gap": "14px",
                    "marginBottom": "24px",
                },
            ),

            html.H2("Dataset-level model comparison"),
            html.Div(
                [
                    dcc.Graph(id="mae-ranking"),
                    dcc.Graph(id="r2-ranking"),
                    dcc.Graph(id="relative-error-ranking"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, 1fr)",
                    "gap": "14px",
                },
            ),

            html.H2("Run-level prediction quality"),
            html.Div(
                [
                    dcc.Graph(id="scatter-log"),
                    dcc.Graph(id="scatter-gamma"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, 1fr)",
                    "gap": "14px",
                },
            ),

            html.H2("Run-level error analysis"),
            html.Div(
                [
                    dcc.Graph(id="abs-error-hist"),
                    dcc.Graph(id="relative-error-box"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, 1fr)",
                    "gap": "14px",
                },
            ),

            html.H2("Per-target metrics"),
            dcc.Graph(id="per-target-metrics"),

            html.H2("Training graph"),
            dcc.Graph(id="training-history"),

            html.H2("Metrics table"),
            dash_table.DataTable(
                id="metrics-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "Arial",
                    "fontSize": "13px",
                    "padding": "6px",
                    "textAlign": "left",
                },
                style_header={
                    "fontWeight": "bold",
                    "backgroundColor": "#f2f2f2",
                },
            ),
        ],
        style={
            "fontFamily": "Arial, sans-serif",
            "backgroundColor": "#fafafa",
            "padding": "28px",
        },
    )
