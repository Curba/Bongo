# Classical tuning pilot

Run with `.venv/bin/python scripts/tune_classical_pilot.py --config configs/sweeps/classical_tuning_pilot.yaml`.
The output directory must not already exist. Use a new output path for a separately authorized experiment; the runner refuses to overwrite a completed or partial run. All generated studies, models, fold indices, predictions and metrics live under the gitignored `results/` directory.

The checked-in pilot config requires the existing `local_5q_1000.npz` dataset and the three historical baseline runs referenced in the YAML. Generate those artifacts first on a fresh checkout. This is an explicit comparison against those baselines, not automatic selection of whichever run happens to be newest.

## Dependency check

Neither Optuna nor scikit-optimize was declared or installed. `uv pip check` initially passed for 87 packages. A dry run with all installed versions constrained resolved Optuna 4.9.0 plus alembic 1.19.2, colorlog 6.12.0, greenlet 3.5.5, mako 1.4.1 and SQLAlchemy 2.0.52 without replacing any existing packages. After installation, all 93 packages passed `uv pip check`. The project dependency is `optuna>=4,<5`; universal `uv lock` resolution also succeeded. Bootstrap now installs project dependencies instead of skipping them.

## Protocol fixed before test evaluation

- Dataset: 1,000 samples, 15 observables × 51 times = 765 features, 15 log10 noise targets. Features remain flattened as in the baselines.
- Original split, seed 1234: 600 training, 200 validation, 200 test. Saved baseline indices must match exactly; dataset path, feature mode and seed must also match.
- Five shuffled KFold partitions, CV seed 2026: each fit uses 480 training samples, each objective score uses 120 other training samples. All models and candidates share the same folds. This is sample-level CV: each complete trajectory stays together.
- Objective: average fold MAE over all 15 log10 targets, lower is better. Scaling and PCA are cloned and fitted inside each fold. Five folds provide a reasonable balance at N=600; ten folds would approximately double the fitting cost.
- Baseline: recompute its CV scores using the same folds and include it as a candidate. Reuse its stored test metrics.
- Optuna uses sequential, seeded TPE with its default 10 startup trials, no pruning, and fixed trial counts. The baseline candidate is an additional completed trial. See [Optuna's TPE documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).
- Refit the chosen model on all 600 training samples. Do not add validation data.
- Freeze all three searches before predicting on the 200 test samples once per chosen model. Validation is never used. No tuning decisions follow test evaluation.
- Numerical libraries use two threads; folds, trials and target wrappers run sequentially. Timing includes baseline CV, search, refits, serialization and final evaluation, excluding dependency installation and development.

The existing test set has already been evaluated historically; “untouched” here means excluded from this tuning process, not a newly collected blind test set. Best-search CV scores are selection scores and can be optimistic after trying many candidates. Test performance is the separate evaluation. Fold differences are descriptive, not independent samples for a significance claim.

## Search spaces

| Model | Additional trials | Space and rationale |
|---|---:|---|
| KNN | 40 | Neighbors 1–40, uniform/distance weights, Minkowski p 1/2. Varies smoothing and distance geometry. |
| RBF SVR | 60 | C 0.1–1,000, epsilon 0.001–0.3, gamma 0.00001–0.1, all log-uniform; PCA 16/32/64/128. Varies regularization, error tolerance, kernel scale and retained trajectory detail. Exact baseline gamma=`scale` is included separately. |
| HGB | 40 | Learning rate 0.01–0.2 (log), iterations 200–1,000, leaves 7/15/31/63, minimum leaf samples 5–60, L2 0.000001–10 (log). Varies capacity and regularization; exact baseline L2=0 is included separately. |

Unsearched parameters retain existing baseline defaults. In particular HGB's `early_stopping='auto'` does not activate for these small training sets; no external validation data enters fitting.

## Validation

The test suite includes explicit spies verifying fold-local scaling, training-only fits, a full-training final refit, exactly one final test prediction, reuse of historical baseline metrics, refusal to overwrite results, and rejection of out-of-scope models. Repository tests: 17 passed, 1 skipped. Changed Python files pass Ruff. Existing YAQS smoke test passes (`<Z_0>(t_final)=0.980146`); bootstrap shell syntax passes. A fresh-clone bootstrap was not executed, because it installs into a sibling environment; dependency resolution and the current environment smoke test were checked instead.

## Follow-up searches and trial-level early stopping

The runner now also accepts XGBoost. The prepared follow-up configs are
`configs/sweeps/classical_tuning_svr_extended.yaml` (C up to 10,000; at most 60
candidates) and `configs/sweeps/classical_tuning_xgboost.yaml` (at most 15 candidates).
Both use the same five training-only folds and `early_stopping_patience: 10`.
XGBoost fixes `tree_method: hist` and `multi_strategy: multi_output_tree` and uses
the configured numerical thread count for `n_jobs`.

`early_stopping_patience` defaults to 10 when omitted; set it to YAML `null` to
restore the historical fixed-budget behavior described above. A callback calls
Optuna's `study.stop()` after N completed candidates fail to strictly improve the
best training CV MAE, initialized from the baseline. Ties count toward patience;
a strict improvement resets the counter. The baseline is additional to the trial
budget and does not itself consume patience. The selected model is still refitted
on the full training split and evaluated once on the test split after stopping.
Reports record the budget, actual completed candidate count and stop reason.

No MedianPruner is used: `cross_val_score` returns after all folds finish. Adding
fold-level pruning would require a manual fold loop and decisions based on partial
fold averages. Every evaluated candidate instead receives the full five-fold score.

Patience is a compute/accuracy tradeoff: on the original SVR trial sequence, N=10
would have stopped at trial 14, before the winner at trial 32. Even a 15-trial
XGBoost search can stop after ten candidates if none beats the baseline. The new
SVR search retains the historical untuned baseline as its fallback; it does not
seed the prior tuned winner, so a better result than the prior pilot is not guaranteed.

Follow-up output directories must be new; automatic resume is still unsupported.
The follow-up configs have been prepared and checked without launching either search.
