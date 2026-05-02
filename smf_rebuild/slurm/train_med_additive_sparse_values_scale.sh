#!/bin/bash

#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=train_med_add_vs
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --output=smf_rebuild/train/%x.log

set -euo pipefail

: "${SMF_MED_RUN_DIR:?Set SMF_MED_RUN_DIR before submitting this job.}"
mkdir -p "$SMF_MED_RUN_DIR" smf_rebuild/train

python -m smf_rebuild.train_sparse \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --init-checkpoint "$SMF_MED_RUN_DIR/qwen_additive_memory_retrofit" \
  --output-dir "$SMF_MED_RUN_DIR/qwen_additive_memory_sparse_kl_values_scale" \
  --dataset-preset medmcqa \
  --train-split train \
  --eval-split validation \
  --sample-size 60000 \
  --eval-sample-size 1000 \
  --background-preset oasst1 \
  --background-split train \
  --background-sample-size 20000 \
  --background-max-batches 2000 \
  --sparse-scoring kl \
  --sparse-top-t 512 \
  --trainable-scope values_scale \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 3 \
  --learning-rate 5e-4 \
  --eval-steps 500 \
  --save-steps 500 \
  --seed "${SMF_SEED:-42}"
