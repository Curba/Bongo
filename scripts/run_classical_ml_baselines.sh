#!/usr/bin/env bash
set -e

DATASET="${1:-data/dataset_p3_lindblad.npz}"
OUTPUT_DIR="${2:-outputs/classical_ml}"

MODELS=(
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
)

echo "Dataset: ${DATASET}"
echo "Output dir: ${OUTPUT_DIR}"

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "============================================================"
  echo "Running model: ${MODEL}"
  echo "============================================================"

  python scripts/train_classical_ml.py \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --feature-mode flatten \
    --pca-components 64 \
    --n-neighbors 7 \
    --n-estimators 500
done

echo ""
echo "All classical ML baselines finished."
echo "Results saved under: ${OUTPUT_DIR}"