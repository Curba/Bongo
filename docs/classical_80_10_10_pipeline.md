# 80/10/10 classical pipeline

Entry point: `scripts/run_80_10_10_pipeline.py`.
Configuration: `configs/sweeps/classical_80_10_10.yaml`.

The entry point sets native-library thread limits before importing numerical
libraries, then calls the library orchestrator in `training/split_pipeline.py`.
It creates an 800/100/100 split with the existing default_rng seed convention
(1234), runs the existing batch CLI for all ten models, runs
`scripts/check_split_consistency.py`, and additionally compares every saved
partition against the intended split. A failed baseline or split check prevents
tuning. The three tuning CLIs run sequentially, using their generated YAMLs and
the exact baseline paths returned by the batch manifest.

Five-fold tuning only sees the 800 training samples: 640 fit / 160 score per
fold. Each selected model is refitted on all 800 training samples and evaluated
on the 100 test samples. The baseline trainer also reports validation/test
metrics as before; tuning does not use either held-out split. The new test set
is a subset of the previously examined test set, not a fresh blind evaluation.

All outputs are below `outputs/ml_runs/local_5q_1000_80_10_10/`:

- `split_indices.npz`: intended split.
- `baselines/local_5q_1000/<model>/<run_id>/`: ten baseline models and metrics.
- `baseline_manifest.json`: exact baseline run paths and failures.
- `configs/`: generated batch and per-model tuning configs.
- `tuning/<model>/<model>/model.joblib`: selected refitted model.
- `tuning/<model>/report.json`: CV, test metrics, timing and stopping reason.
- `pipeline_status.json`: current stage, completed-stage times and final status.

Existing output roots are rejected. This is not an automatic-resume workflow.
Child processes inherit stdout/stderr, so one detached log captures all stages.

Concurrency: models, CV folds and Optuna trials run sequentially. Baseline
top-level forest/XGBoost estimators use four workers/threads; multi-output
wrappers and nested estimator job settings use one. Tuning uses two numerical
threads for SVR/HGB, four for XGBoost (`n_jobs=4`), and one BLAS thread. The
OpenMP thread limit is four with one active nesting level. Limits reduce
oversubscription; they do not impose a memory budget on deep XGBoost trees.
The existing Gaussian-process PCA cap of 32 remains unchanged.

Budgets: SVR 20 candidates/patience 20, HGB 20/patience 10, XGBoost 12/patience 8.
Each study also evaluates an untuned baseline in CV. The original winning tuned
parameters are not inserted as candidates; the ranges are narrowed as requested.

Runtime planning estimate, not a new benchmark: baselines 5–20 minutes, SVR
5–20 minutes, HGB 1.5–3 hours, XGBoost 2–8 hours. Reserve roughly 4–12 hours
overall; XGBoost can exceed this if deep configurations still cause memory
pressure. Previous HGB search took 2.88 hours for 40 candidates on 600 training
samples; this run halves the candidate count but uses 800 training samples and
600–1000 boosting iterations. Previous XGBoost candidate durations were heavily
affected by oversubscription/swap and do not support a precise speedup estimate.

Implementation and mocked tests were prepared but not executed, per request.
