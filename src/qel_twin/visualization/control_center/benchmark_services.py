from __future__ import annotations

import csv
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape

from .services import train_model_job


def _pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_metrics(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "metrics.json"
    return _load_json(path) if path.exists() else {}


def _per_target_rows(model_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    suffixes = (
        "_mae_log10",
        "_rmse_log10",
        "_r2_log10",
        "_median_relative_error_gamma",
        "_mean_relative_error_gamma",
        "_median_factor_error_gamma",
    )
    parameter_names: set[str] = set()
    for key in metrics:
        for suffix in suffixes:
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                if prefix and prefix not in {"mean", "median"}:
                    parameter_names.add(prefix)

    rows: list[dict[str, Any]] = []
    for name in sorted(parameter_names):
        rows.append(
            {
                "model": model_name,
                "parameter": name,
                "mae_log10": _finite_number(metrics.get(f"{name}_mae_log10")),
                "rmse_log10": _finite_number(metrics.get(f"{name}_rmse_log10")),
                "r2_log10": _finite_number(metrics.get(f"{name}_r2_log10")),
                "median_relative_error_gamma": _finite_number(
                    metrics.get(f"{name}_median_relative_error_gamma")
                ),
                "mean_relative_error_gamma": _finite_number(
                    metrics.get(f"{name}_mean_relative_error_gamma")
                ),
                "median_factor_error_gamma": _finite_number(
                    metrics.get(f"{name}_median_factor_error_gamma")
                ),
            }
        )
    return rows


def run_benchmark_job(
    *,
    dataset_path: str | Path,
    output_root: str | Path,
    models: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train and reconstruct selected models using one shared deterministic split."""
    if not models:
        raise ValueError("Select at least one model.")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_fraction = float(config.get("train_fraction", 0.60))
    validation_fraction = float(config.get("validation_fraction", 0.20))
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train + validation fractions must sum to less than 1.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    benchmark_dir = output_root / "_benchmarks" / dataset_path.stem / timestamp
    benchmark_dir.mkdir(parents=True, exist_ok=False)

    shared_config = dict(config)
    shared_config["run_tag"] = shared_config.get("run_tag") or f"benchmark_{timestamp}"

    leaderboard: list[dict[str, Any]] = []
    per_target_metrics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for model_name in models:
        model_name = str(model_name)
        model_config = dict(shared_config)
        model_config["model_name"] = model_name

        try:
            result = train_model_job(
                dataset_path=dataset_path,
                output_root=output_root,
                config=model_config,
            )
            run_dir = Path(result["run_dir"])
            persisted = _load_run_metrics(run_dir)
            test_metrics = dict(
                result.get("test_metrics") or persisted.get("test_metrics") or {}
            )
            reconstruction = dict(
                result.get("reconstruction") or persisted.get("reconstruction") or {}
            )
            timing = dict(persisted.get("timing") or {})

            leaderboard.append(
                {
                    "model": model_name,
                    "trajectory_rmse": _finite_number(
                        _pick(reconstruction, "trajectory_rmse_mean", "trajectory_rmse")
                    ),
                    "trajectory_mae": _finite_number(
                        _pick(reconstruction, "trajectory_mae_mean", "trajectory_mae")
                    ),
                    "trajectory_max_error": _finite_number(
                        _pick(
                            reconstruction,
                            "max_abs_trajectory_error_mean",
                            "max_abs_trajectory_error",
                        )
                    ),
                    "log10_gamma_rmse": _finite_number(
                        _pick(test_metrics, "rmse_log10_mean", "log10_rmse")
                    ),
                    "log10_gamma_mae": _finite_number(
                        _pick(test_metrics, "mae_log10_mean", "log10_mae")
                    ),
                    "r2_log10": _finite_number(
                        _pick(test_metrics, "r2_log10_mean", "r2_log10", "log10_r2")
                    ),
                    "median_factor_error": _finite_number(
                        _pick(
                            test_metrics,
                            "median_factor_error_gamma",
                            "median_factor_error",
                        )
                    ),
                    "median_relative_error": _finite_number(
                        _pick(
                            test_metrics,
                            "median_relative_error_gamma",
                            "median_relative_error",
                        )
                    ),
                    "mean_relative_error": _finite_number(
                        _pick(
                            test_metrics,
                            "mean_relative_error_gamma",
                            "mean_relative_error",
                        )
                    ),
                    "fit_seconds": _finite_number(
                        _pick(timing, "fit_seconds", "training_seconds")
                    ),
                    "predict_seconds": _finite_number(timing.get("predict_seconds")),
                    "run_dir": str(run_dir),
                }
            )
            per_target_metrics.extend(_per_target_rows(model_name, test_metrics))
        except Exception as exc:
            failures.append(
                {
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    leaderboard.sort(
        key=lambda row: (
            row["trajectory_rmse"] is None,
            row["trajectory_rmse"]
            if row["trajectory_rmse"] is not None
            else float("inf"),
            row["log10_gamma_rmse"] is None,
            row["log10_gamma_rmse"]
            if row["log10_gamma_rmse"] is not None
            else float("inf"),
        )
    )

    payload = {
        "benchmark_id": timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "dataset_id": dataset_path.stem,
        "models_requested": [str(model) for model in models],
        "config": shared_config,
        "leaderboard": leaderboard,
        "per_target_metrics": per_target_metrics,
        "failures": failures,
    }
    benchmark_json = benchmark_dir / "benchmark.json"
    benchmark_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "benchmark_dir": str(benchmark_dir),
        "benchmark_json": str(benchmark_json),
        "completed": len(leaderboard),
        "failed": len(failures),
        "best_model": leaderboard[0]["model"] if leaderboard else None,
    }


def scan_benchmarks(
    output_root: str | Path,
    dataset_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(output_root) / "_benchmarks"
    if not root.exists():
        return []

    dataset_id = Path(dataset_path).stem if dataset_path else None
    rows: list[dict[str, Any]] = []
    for path in root.rglob("benchmark.json"):
        try:
            payload = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if dataset_id and payload.get("dataset_id") != dataset_id:
            continue
        leaderboard = payload.get("leaderboard") or []
        best = leaderboard[0] if leaderboard else {}
        rows.append(
            {
                "benchmark_id": str(payload.get("benchmark_id", path.parent.name)),
                "dataset_id": str(payload.get("dataset_id", "unknown")),
                "models": len(payload.get("models_requested") or []),
                "completed": len(leaderboard),
                "failed": len(payload.get("failures") or []),
                "best_model": best.get("model"),
                "best_trajectory_rmse": best.get("trajectory_rmse"),
                "path": str(path),
            }
        )
    rows.sort(key=lambda row: row["benchmark_id"], reverse=True)
    return rows


def load_benchmark(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    return _load_json(source)


# Minimal dependency-free XLSX writer. This keeps the benchmark patch entirely
# inside src/ and does not require adding openpyxl to the project dependencies.


def _excel_col(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_cell(reference: str, value: Any) -> str:
    if value is None:
        return f'<c r="{reference}"/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return f'<c r="{reference}"><v>{number:.17g}</v></c>'
    text = escape(str(value))
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(rows: Iterable[Iterable[Any]]) -> str:
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = [
            _xml_cell(f"{_excel_col(column_index)}{row_index}", value)
            for column_index, value in enumerate(row)
        ]
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(row_xml)
        + "</sheetData></worksheet>"
    )


def _dict_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    if not records:
        return [["No rows"]]
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    return [columns] + [
        [record.get(column) for column in columns] for record in records
    ]


def _csv_rows(path: Path) -> list[list[Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [list(row) for row in csv.reader(handle)]


def _sanitize_sheet_name(name: str) -> str:
    invalid = set('[]:*?/\\')
    cleaned = "".join("_" if char in invalid else char for char in name).strip()
    return (cleaned or "Sheet")[:31]


def _write_xlsx(path: Path, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_sheets = [(_sanitize_sheet_name(name), rows) for name, rows in sheets]

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for index in range(1, len(safe_sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _) in enumerate(safe_sheets, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets></workbook>"
    )

    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index in range(1, len(safe_sheets) + 1):
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    workbook_rels.append("</Relationships>")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for index, (_, rows) in enumerate(safe_sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _sheet_xml(rows),
            )


def export_benchmark_excel(benchmark_json: str | Path) -> Path:
    benchmark_json = Path(benchmark_json)
    payload = load_benchmark(benchmark_json)
    output = benchmark_json.parent / (
        f"{payload['dataset_id']}_{payload['benchmark_id']}_comparison.xlsx"
    )

    config_rows = [["key", "value"]] + [
        [key, json.dumps(value) if isinstance(value, (dict, list)) else value]
        for key, value in (payload.get("config") or {}).items()
    ]
    sheets: list[tuple[str, list[list[Any]]]] = [
        ("leaderboard", _dict_rows(payload.get("leaderboard") or [])),
        (
            "per_target_metrics",
            _dict_rows(payload.get("per_target_metrics") or []),
        ),
        ("benchmark_config", config_rows),
        ("failures", _dict_rows(payload.get("failures") or [])),
    ]

    leaderboard = payload.get("leaderboard") or []
    if leaderboard:
        run_dir = leaderboard[0].get("run_dir")
        if run_dir:
            predictions = Path(str(run_dir)) / "test_predictions.csv"
            if predictions.exists():
                sheets.append(("best_predictions", _csv_rows(predictions)))

    _write_xlsx(output, sheets)
    return output


__all__ = [
    "export_benchmark_excel",
    "load_benchmark",
    "run_benchmark_job",
    "scan_benchmarks",
]
