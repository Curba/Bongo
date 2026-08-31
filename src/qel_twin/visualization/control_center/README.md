# QEL Twin Control Center

QEL Twin Control Center is the end-to-end user interface for the QEL/Bongo noise digital-twin workflow.

It provides one place to:

1. create quantum-noise datasets with YAQS,
2. train machine-learning models,
3. infer Lindblad noise parameters,
4. reconstruct the quantum dynamics with YAQS,
5. compare the reconstructed trajectory with the original trajectory,
6. inspect and compare model results.

The main evaluation target is **trajectory reconstruction fidelity**. Parameter-estimation error is still reported, but it is treated as a diagnostic rather than the final success criterion.

---

## 1. Core idea

The program implements the following loop:

```text
Physical / simulated system
        |
        v
      YAQS
        |
        v
Observable trajectories
   NoiseDataset (N, O, T)
        |
        v
Machine-learning model
        |
        v
Predicted log10(gamma)
        |
        v
Predicted Lindblad gamma
        |
        v
      YAQS
        |
        v
Reconstructed trajectory
        |
        v
Compare with original trajectory
        |
        v
Digital-twin reconstruction score
```

The goal is not only to estimate the Lindblad parameters correctly.

The more important question is:

> Do the inferred noise parameters reproduce the observed dynamics?

That is why the results page puts trajectory RMSE, trajectory MAE, and maximum trajectory error before parameter-estimation metrics.

---

## 2. Canonical dataset format

The Control Center keeps the qel-ml `NoiseDataset` representation as the common format.

The trajectory array has shape:

```text
(N, O, T)
```

where:

- `N` = number of samples / simulated experiments,
- `O` = number of observables,
- `T` = number of time points.

For example, if we measure X, Y, and Z on five sites:

```text
3 observable channels x 5 sites = 15 observables
```

then the dataset may look like:

```text
expectation_values.shape = (1000, 15, 101)
```

The dataset stores:

```text
expectation_values : (N, O, T)
gamma              : (N, P)
log10_gamma        : (N, P)
times              : (T,)
parameter_names    : P names
metadata           : simulation and dataset configuration
```

No conversion back to the old `(N, C, L, T)` representation is required.

---

## 3. Noise parameterizations

Three Lindblad parameterizations are supported.

### Super-global

One value controls all Pauli noise channels and all sites.

```text
P = 1
```

### Global

One shared value is learned for each Pauli noise channel.

```text
gamma_x
gamma_y
gamma_z

P = 3
```

### Local

Every site gets its own X, Y, and Z noise parameters.

For `L` sites:

```text
P = 3L
```

For five sites:

```text
gamma_x_0
gamma_y_0
gamma_z_0
gamma_x_1
gamma_y_1
gamma_z_1
...
gamma_x_4
gamma_y_4
gamma_z_4
```

The training integration therefore uses a dynamic target dimension `P` instead of assuming that every model predicts exactly three values.

---

# 4. Program structure

The Control Center UI lives under:

```text
src/qel_twin/visualization/control_center/
```

Typical structure:

```text
src/qel_twin/
|
|-- characterization/
|   |-- noise_ml/
|   |   |-- dataset.py
|   |   |-- preprocessing.py
|   |   |-- training.py
|   |   |-- reconstruction.py
|   |   |-- results.py
|   |   |-- plotting.py
|   |   |-- run.py
|   |   `-- models/
|   |       |-- mlp.py
|   |       `-- cnn2d.py
|   |
|   `-- reconstruction_adapter.py
|
|-- training/
|   |-- classical_ml.py
|   |-- lstm.py
|   |-- noise_dataset.py
|   |-- classical_ml_noise.py
|   `-- lstm_noise.py
|
`-- visualization/
    `-- control_center/
        |-- __init__.py
        |-- app.py
        |-- layout.py
        |-- callbacks.py
        |-- services.py
        `-- assets/
            `-- control_center.css

scripts/
|-- run_control_center.py
|-- train_classical_ml_noise.py
`-- train_lstm_noise.py
```

---

# 5. What each UI file does

## `app.py`

Creates and starts the Dash application.

It connects:

```text
layout.py
callbacks.py
services.py
```

and defines the dataset and results directories.

---

## `layout.py`

Defines the visible UI.

The interface contains three main tabs:

```text
1. Dataset
2. Train
3. Results
```

It contains forms, tables, buttons, model selectors, metric cards, and plots.

---

## `callbacks.py`

Contains the interaction logic for the UI.

Examples:

- clicking **Generate dataset** starts a dataset-generation job,
- selecting a dataset updates the trajectory preview,
- clicking **Train + reconstruct** starts training,
- finished runs are automatically added to the results list,
- selecting a run loads reconstruction metrics and plots,
- selecting a reconstructed sample loads the original and reconstructed trajectories.

---

## `services.py`

Contains the backend logic used by the UI.

It handles:

- building YAQS experiments,
- creating datasets,
- training models,
- starting reconstruction,
- calculating reconstruction metrics,
- scanning datasets,
- scanning completed runs,
- loading plots and run details,
- running long jobs without freezing the Dash interface.

This is the main bridge between the UI and the QEL Twin physics/ML code.

---

## `assets/control_center.css`

Contains the UI styling.

It defines the dark dashboard layout, panels, metric cards, forms, responsive behavior, and visual hierarchy.

---

# 6. Dataset tab

The **Dataset** page creates a new `NoiseDataset`.

The user configures:

- dataset name,
- number of samples,
- number of sites / qubits,
- observable channels,
- Lindblad parameterization,
- gamma range,
- elapsed simulation time,
- time step `dt`,
- YAQS simulation method,
- simulation preset,
- number of trajectories,
- initial state,
- Ising coupling `J`,
- transverse field `g`,
- random seed,
- Trotter order,
- TDVP settings,
- YAQS parallel execution.

The flow is:

```text
UI configuration
      |
      v
NoiseExperiment
      |
      v
YAQS simulation
      |
      v
generate_noise_dataset(...)
      |
      v
NoiseDataset
      |
      v
.npz file
```

The created dataset is saved in the configured dataset directory.

Default:

```text
data/noise_datasets/
```

---

# 7. Dataset preview

Existing canonical qel-ml `.npz` datasets are scanned automatically.

The dataset page shows:

- dataset name,
- sample count,
- observable count,
- time-point count,
- site count,
- parameterization,
- target count,
- path.

A selected trajectory can be previewed by choosing:

```text
sample index
observable index
```

The UI plots:

```text
expectation value vs time
```

and displays dataset metadata.

---

# 8. Training tab

The **Train** page lets the user choose:

```text
Dataset
+
Model
+
Training settings
```

The model families are combined into one interface.

---

## Classical ML

The UI reuses the existing Bongo classical model registry.

Depending on the current repository version this may include models such as:

```text
dummy_mean
linear
ridge
elastic_net
bayesian_ridge
pls
knn
knn_pca
svr_rbf
kernel_ridge_rbf
random_forest
extra_trees
gradient_boosting
hist_gradient_boosting
chain_ridge
mlp_pca
...
```

Classical models use the qel-ml trajectory directly.

Input:

```text
(N, O, T)
```

For flat-feature models:

```text
(N, O, T)
      |
      v
(N, O*T)
```

---

## qel-ml PyTorch models

The qel-ml models remain available:

```text
Torch MLP
2D CNN
```

The CNN naturally treats each sample as an:

```text
observable x time
```

signal.

---

## Sequence models

The integration also exposes:

```text
LSTM
BiLSTM
```

For recurrent models:

```text
(N, O, T)
      |
transpose
      |
      v
(N, T, O)
```

Therefore each time step contains all observable values.

The recurrent output dimension is dynamic:

```text
output_size = P
```

rather than hard-coded to three parameters.

---

# 9. Train / validation / test split

The default split is:

```text
60% train
20% validation
20% test
```

The split happens at the sample level.

The sets are disjoint.

For models using normalization, normalization statistics are fit on training data only.

Conceptually:

```text
                dataset
                   |
          -------------------
          |        |        |
        train     val      test
          |
     fit preprocessing
          |
          +--------+--------+
                   |
             transform val/test
```

This prevents validation/test leakage.

---

# 10. What the model predicts

The target is:

```text
log10(gamma)
```

rather than gamma directly.

The ML model therefore learns:

```text
trajectory
    |
    v
predicted log10(gamma)
```

and the physical Lindblad parameters are recovered using:

```text
gamma = 10 ** predicted_log10_gamma
```

The log representation is useful because noise rates can span several orders of magnitude.

---

# 11. Automatic trajectory reconstruction

This is the most important stage.

A training job does not stop after parameter prediction.

The Control Center automatically continues into:

```text
predicted gamma
      |
      v
reconstruct_dynamics(...)
      |
      v
YAQS
      |
      v
reconstructed trajectory
```

For each selected held-out test sample:

```text
original trajectory
        vs
reconstructed trajectory
```

is evaluated.

At least one held-out trajectory should always be reconstructed.

The user can increase the number of reconstruction samples in the training page.

---

# 12. Reconstruction metrics

The primary metrics are trajectory-based.

Typical reconstruction metrics include:

```text
trajectory MAE
trajectory RMSE
maximum absolute trajectory error
per-observable trajectory error
```

These answer the main digital-twin question:

> How accurately does the inferred noise model reproduce the dynamics?

---

# 13. Parameter metrics

Parameter metrics are still calculated.

Typical examples:

```text
log10 gamma MAE
log10 gamma RMSE
R²
relative gamma error
median factor error
```

These are useful for understanding why a reconstruction succeeds or fails.

However, model ranking should prioritize reconstruction metrics.

A model can have a parameter error but still reproduce the observed dynamics well.

Conversely, numerically close parameters do not automatically guarantee a high-fidelity reconstructed trajectory.

---

# 14. Results tab

The **Results** page is reconstruction-first.

The most important cards are:

```text
Trajectory RMSE
Trajectory MAE
Maximum trajectory error
```

Diagnostic cards include:

```text
log10 gamma RMSE
median gamma factor error
```

The page also includes:

- training-history plot,
- reconstruction leaderboard,
- original-vs-reconstructed trajectory plot,
- reconstructed test-sample selector,
- observable selector,
- table of true and predicted Lindblad parameters.

---

# 15. Reconstruction leaderboard

Completed model runs are scanned from the results directory.

Default:

```text
outputs/noise_ml_runs/
```

Runs with reconstruction results can be compared using:

```text
mean trajectory RMSE
```

Lower is better.

This gives one common score for comparing models from different families:

```text
Ridge
KNN
SVR
Extra Trees
Random Forest
Torch MLP
CNN
LSTM
BiLSTM
...
```

---

# 16. Output files

A typical model run directory contains artifacts such as:

```text
model.joblib
or
model.pt / model.npz

metrics.json
training_history.csv
test_predictions.csv
split_indices.npz

reconstruction/
    summary.json
    sample_123.npz
    sample_456.npz
    ...
```

The exact model file depends on the model family.

---

## `metrics.json`

Contains run-level metrics and metadata.

After reconstruction it also contains reconstruction metrics.

---

## `test_predictions.csv`

Contains:

```text
dataset index
true log10 gamma
predicted log10 gamma
true gamma
predicted gamma
factor error
```

for the held-out test set.

---

## `reconstruction/summary.json`

Contains aggregate reconstruction results such as:

```text
samples evaluated
mean trajectory MAE
mean trajectory RMSE
median trajectory RMSE
mean maximum absolute trajectory error
```

---

## `reconstruction/sample_*.npz`

Stores one reconstructed held-out sample.

Typical content:

```text
dataset_index
times
original
reconstructed
true_gamma
predicted_gamma
true_log10_gamma
predicted_log10_gamma
parameter_names
```

These files are used by the Results UI to plot the original and reconstructed dynamics.

---

# 17. Background jobs

Dataset generation and model training may take a long time.

The UI therefore starts these operations as background jobs instead of blocking the Dash request.

Conceptually:

```text
Dash UI
   |
submit job
   |
   v
JobManager
   |
background worker
   |
   +-- dataset generation
   |
   `-- training + reconstruction
```

The browser periodically polls the job state and displays:

```text
QUEUED
RUNNING
COMPLETE
FAILED
```

If a job fails, the traceback is shown in the UI so the problem can be diagnosed.

---

# 18. Running the program

Activate the environment:

```bash
source /home/han/QEL_ws/qel_env/bin/activate
```

Go to the repository:

```bash
cd ~/QEL_ws/qel-digital-twin
```

Start the Control Center:

```bash
python scripts/run_control_center.py
```

Then open:

```text
http://127.0.0.1:8050
```

---

# 19. Custom dataset and results directories

You can specify different folders:

```bash
python scripts/run_control_center.py \
  --data-dir data/noise_datasets \
  --results-dir outputs/noise_ml_runs \
  --port 8050
```

You can also expose the UI on another interface:

```bash
python scripts/run_control_center.py \
  --host 0.0.0.0 \
  --port 8050
```

Only expose the application on a network when that is appropriate for the environment.

---

# 20. Recommended first experiment

For the first end-to-end test, use a small global-noise dataset.

Example configuration:

```text
Samples:              20
Sites:                3
Observables:          X, Y, Z
Parameterization:     global
gamma_min:            1e-3
gamma_max:            1e-1
Elapsed time:         1.0
dt:                   0.1
Method:               TJM
Trajectories:         10
Preset:               fast
```

Then train:

```text
Extra Trees
```

with:

```text
Reconstruction samples: 2
```

The first validation goal is:

```text
dataset generation
      |
      v
training
      |
      v
prediction
      |
      v
YAQS reconstruction
      |
      v
original/reconstructed plot
```

Once this works reliably, increase:

```text
number of samples
number of sites
simulation time
trajectory count
```

and compare additional models.

---

# 21. Recommended experimental workflow

A useful benchmarking sequence is:

### Stage 1 — Global noise

```text
P = 3
```

Start here because it is the easiest problem to debug.

Compare:

```text
Ridge
KNN
Extra Trees
Torch MLP
CNN
LSTM
```

using reconstruction RMSE.

### Stage 2 — More data / longer dynamics

Increase dataset size and simulation duration.

### Stage 3 — Local noise

Switch to:

```text
P = 3L
```

This tests whether the observable dynamics contain enough information to infer spatially varying noise.

### Stage 4 — Robust model comparison

Compare all models using identical dataset splits.

Primary ranking:

```text
reconstruction RMSE
```

Secondary diagnostics:

```text
reconstruction MAE
parameter log10 RMSE
factor error
runtime
```

---

# 22. Why reconstruction is the final metric

Noise characterization is an inverse problem:

```text
observed dynamics
      |
      v
infer hidden noise parameters
```

There may be multiple parameter configurations that produce similar trajectories.

Therefore the final digital-twin criterion should not be:

```text
"Did the model exactly recover every parameter?"
```

It should be:

```text
"Does the inferred model reproduce the system's observable dynamics?"
```

That is why this program is built around the loop:

```text
observed trajectory
        |
        v
machine learning
        |
        v
inferred noise
        |
        v
YAQS
        |
        v
reconstructed trajectory
```

---

# 23. Troubleshooting

## UI cannot import `qel_twin`

Run the UI from the repository root inside the correct environment:

```bash
source /home/han/QEL_ws/qel_env/bin/activate
cd ~/QEL_ws/qel-digital-twin
python scripts/run_control_center.py
```

---

## Dataset does not appear in the catalog

The catalog only accepts canonical qel-ml `NoiseDataset` `.npz` files.

Check that the dataset contains:

```text
expectation_values
gamma
log10_gamma
times
parameter_names
metadata_json
```

---

## Training works but reconstruction fails

Reconstruction requires enough metadata to rebuild the original YAQS experiment.

Check the dataset metadata for:

```text
num_sites
observables
initial_state
state_representation
simulation
simulator
hamiltonian
parameterization
```

---

## Training is slow

Possible causes:

- large sample count,
- long time grid,
- many trajectories,
- many sites,
- local-noise parameterization,
- large tree ensembles,
- long neural-network training,
- many reconstructed test samples.

For debugging, start with a very small dataset and one or two reconstruction samples.

---

## UI shows a failed background job

The job-status box shows the Python traceback.

Use that traceback to determine whether the failure occurred during:

```text
dataset generation
model training
prediction
or
YAQS reconstruction
```

---

# 24. Tests

The qel-ml characterization tests should remain green after the package integration.

Run:

```bash
python -m pytest tests/characterization/noise/ml -q
```

Then run the full project test suite:

```bash
python -m pytest -q
```

Before major changes to the UI/backend, these tests should be used as a regression check.

---

# 25. Summary

QEL Twin Control Center combines:

```text
YAQS physics simulation
+
qel-ml NoiseDataset
+
Bongo classical ML
+
Torch MLP / CNN
+
LSTM / BiLSTM
+
YAQS reconstruction
```

into one end-to-end workflow.

The common pipeline is:

```text
Create dataset
      |
      v
Train model
      |
      v
Predict Lindblad parameters
      |
      v
Reconstruct with YAQS
      |
      v
Compare trajectories
      |
      v
Rank digital-twin quality
```

The central design principle is:

> **Reconstructed-trajectory fidelity is the primary measure of whether the inferred noise model is a useful digital twin.**
