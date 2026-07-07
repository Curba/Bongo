#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-data/processed/dataset_p3_lindblad.npz}"
OUTPUT_DIR="${2:-outputs/ml_runs}"
SEED="${3:-1234}"

FEATURE_MODE="${FEATURE_MODE:-flatten}"
PCA_COMPONENTS="${PCA_COMPONENTS:-64}"
N_NEIGHBORS="${N_NEIGHBORS:-7}"
N_ESTIMATORS="${N_ESTIMATORS:-500}"

RUN_LSTM="${RUN_LSTM:-1}"

CLASSICAL_MODELS=(
  dummy_mean
  linear
  ridge
  lasso
  elastic_net
  bayesian_ridge
  huber
  sgd
  poly_ridge
  pls
  knn
  knn_pca
  svr_rbf
  svr_linear
  nusvr
  kernel_ridge_rbf
  kernel_ridge_poly
  decision_tree
  random_forest
  extra_trees
  gradient_boosting
  hist_gradient_boosting
  adaboost
  bagging_trees
  chain_ridge
  chain_extra_trees
  mlp
  mlp_pca
)

echo "============================================================"
echo "QEL Twin — Run all ML baselines"
echo "============================================================"
echo "Dataset:        ${DATASET}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Seed:           ${SEED}"
echo "Feature mode:   ${FEATURE_MODE}"
echo "PCA components: ${PCA_COMPONENTS}"
echo "KNN neighbors:  ${N_NEIGHBORS}"
echo "N estimators:   ${N_ESTIMATORS}"
echo "Run LSTM:       ${RUN_LSTM}"
echo "============================================================"

if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: Dataset not found: ${DATASET}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p outputs/logs

FAILED_MODELS=()

for MODEL in "${CLASSICAL_MODELS[@]}"; do
  LOG_FILE="outputs/logs/${MODEL}_${SEED}.log"

  echo ""
  echo "============================================================"
  echo "Running classical model: ${MODEL}"
  echo "Log: ${LOG_FILE}"
  echo "============================================================"

  if python scripts/train_classical_ml.py \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --feature-mode "${FEATURE_MODE}" \
    --pca-components "${PCA_COMPONENTS}" \
    --n-neighbors "${N_NEIGHBORS}" \
    --n-estimators "${N_ESTIMATORS}" \
    --seed "${SEED}" \
    --run-tag "all_models" 2>&1 | tee "${LOG_FILE}"; then

    echo "Finished: ${MODEL}"

  else
    echo "FAILED: ${MODEL}"
    FAILED_MODELS+=("${MODEL}")
  fi
done

if [[ "${RUN_LSTM}" == "1" ]]; then
  echo ""
  echo "============================================================"
  echo "Running LSTM baseline"
  echo "============================================================"

  if [[ -f "scripts/train_lstm.py" ]]; then
    LOG_FILE="outputs/logs/lstm_${SEED}.log"

    if python scripts/train_lstm.py \
      --dataset "${DATASET}" \
      --output-dir "${OUTPUT_DIR}" \
      --model-name lstm \
      --epochs 200 \
      --hidden-size 128 \
      --num-layers 2 \
      --dropout 0.1 \
      --batch-size 64 \
      --learning-rate 1e-3 \
      --weight-decay 1e-4 \
      --patience 30 \
      --seed "${SEED}" \
      --run-tag "all_models" 2>&1 | tee "${LOG_FILE}"; then

      echo "Finished: lstm"

    else
      echo "FAILED: lstm"
      FAILED_MODELS+=("lstm")
    fi
  else
    echo "Skipping LSTM: scripts/train_lstm.py not found."
  fi

  echo ""
  echo "============================================================"
  echo "Running BiLSTM baseline"
  echo "============================================================"

  if [[ -f "scripts/train_lstm.py" ]]; then
    LOG_FILE="outputs/logs/bilstm_${SEED}.log"

    if python scripts/train_lstm.py \
      --dataset "${DATASET}" \
      --output-dir "${OUTPUT_DIR}" \
      --model-name bilstm \
      --epochs 200 \
      --hidden-size 128 \
      --num-layers 2 \
      --dropout 0.1 \
      --bidirectional \
      --batch-size 64 \
      --learning-rate 1e-3 \
      --weight-decay 1e-4 \
      --patience 30 \
      --seed "${SEED}" \
      --run-tag "all_models" 2>&1 | tee "${LOG_FILE}"; then

      echo "Finished: bilstm"

    else
      echo "FAILED: bilstm"
      FAILED_MODELS+=("bilstm")
    fi
  fi
fi

echo ""
echo "============================================================"
echo "All requested runs finished."
echo "Results saved under: ${OUTPUT_DIR}"
echo "Logs saved under: outputs/logs"
echo "============================================================"

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo ""
  echo "Some models failed:"
  for MODEL in "${FAILED_MODELS[@]}"; do
    echo "  - ${MODEL}"
  done
  echo ""
  echo "The script continued running other models. Check outputs/logs/*.log."
  exit 1
fi

echo "No failures."