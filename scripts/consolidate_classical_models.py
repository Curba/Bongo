"""Consolidate explicitly selected models; never perform hyperparameter searches."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from qel_twin.training.classical_ml import make_features, write_json
from qel_twin.training.consolidation import check_metrics, refit_and_save
from qel_twin.training.noise_dataset import ensure_prediction_matrix, load_noise_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    config = yaml.safe_load(parser.parse_args().config.read_text())
    pilot = json.loads(Path(config["pilot_report"]).read_text())
    dataset_path = Path(config["dataset_path"])
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if digest != pilot["dataset_sha256"]:
        raise ValueError("Dataset differs from pilot")
    dataset = load_noise_dataset(dataset_path)
    X = make_features(dataset.dynamics, pilot["config"]["feature_mode"])
    with np.load(Path(config["pilot_report"]).parent / "split_indices.npz") as source:
        splits = {k: source[k] for k in source.files}
    train, test = splits["train_idx"], splits["test_idx"]
    manifest = {}
    for name, selection in config["models"].items():
        runs = list((Path(config["baseline_root"]) / name).glob("*/metrics.json"))
        if len(runs) != 1:
            raise ValueError(
                f"Expected exactly one baseline for {name}, found {len(runs)}"
            )
        baseline = json.loads(runs[0].read_text())
        with np.load(runs[0].parent / "split_indices.npz") as source:
            for key, value in splits.items():
                np.testing.assert_array_equal(source[key], value)
        if selection == "pilot":
            report_path = Path(
                config.get("pilot_reports", {}).get(name, config["pilot_report"])
            )
            model_report = json.loads(report_path.read_text())
            if model_report["dataset_sha256"] != digest:
                raise ValueError(f"Dataset differs from tuning report for {name}")
            if (
                model_report["config"]["feature_mode"]
                != pilot["config"]["feature_mode"]
            ):
                raise ValueError(f"Feature mode differs for {name}")
            with np.load(report_path.parent / "split_indices.npz") as source:
                for key, value in splits.items():
                    np.testing.assert_array_equal(source[key], value)
            chosen = model_report["models"][name]
            record = refit_and_save(
                report_path.parent / name / "model.joblib",
                Path(config["tuned_output"]) / name,
                X[train],
                dataset.log10_gamma[train],
                X[test],
                dataset.log10_gamma[test],
                dataset.parameter_names,
                chosen["tuned_test_metrics"],
                {
                    "model_name": name,
                    "hyperparameters": chosen["selected_params"],
                    "dataset_sha256": digest,
                    "dataset_path": str(dataset_path.resolve()),
                    "feature_mode": pilot["config"]["feature_mode"],
                    "parameter_names": list(dataset.parameter_names),
                    "seed": pilot["config"]["seed"],
                    "split_sizes": pilot["split_sizes"],
                    "baseline_cv_mae": chosen["baseline_cv_mae"],
                    "best_cv_mae": chosen["best_cv_mae"],
                    "source_report": str(report_path),
                    "evaluation_purpose": "Reproduction check only; no model selection using test data",
                },
                config["threads"],
                config.get("blas_threads"),
            )
            np.savez(
                Path(config["tuned_output"]) / name / "split_indices.npz", **splits
            )
        elif selection == "baseline":
            path = runs[0].parent / "model.joblib"
            with (
                threadpool_limits(limits=config["threads"]),
                threadpool_limits(
                    limits=config.get("blas_threads", config["threads"]),
                    user_api="blas",
                ),
            ):
                model = joblib.load(path)
                pred = ensure_prediction_matrix(
                    model.predict(X[test]), len(dataset.parameter_names)
                )
            metrics = check_metrics(
                dataset.log10_gamma[test],
                pred,
                dataset.parameter_names,
                baseline["test_metrics"],
            )
            record = {
                "model_path": str(path.resolve()),
                "test_metrics": metrics,
                "load_predict_passed": True,
                "all_recorded_metrics_match": True,
                "baseline_metrics_path": str(runs[0].resolve()),
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "estimator_params": {
                    k: v
                    if v is None or isinstance(v, (str, int, float, bool))
                    else repr(v)
                    for k, v in model.get_params(deep=True).items()
                },
            }
            if hasattr(model, "named_steps") and "pca" in model.named_steps:
                record["fitted_pca_components"] = model.named_steps["pca"].n_components_
        else:
            raise ValueError(f"Unknown selection: {selection}")
        record["selection"] = selection
        manifest[name] = record
        write_json(config["manifest_path"], manifest)
        print(f"{name}: verified {record['model_path']}", flush=True)


if __name__ == "__main__":
    main()
