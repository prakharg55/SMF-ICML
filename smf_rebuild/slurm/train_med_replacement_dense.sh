#!/bin/bash

#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=train_med_repl_dense
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

python -m smf_rebuild.train_retrofit \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir "$SMF_MED_RUN_DIR/qwen_memory_retrofit" \
  --memory-mode replacement \
  --layers 6,12,18 \
  --mem-n-keys 128 \
  --mem-heads 4 \
  --mem-knn 16 \
  --mem-k-dim 256 \
  --dataset-preset oasst1 \
  --train-split train \
  --eval-split validation \
  --sample-size 50000 \
  --eval-sample-size 2000 \
  --max-length 1024 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2 \
  --learning-rate 5e-4 \
  --eval-steps 500 \
  --save-steps 500 \
  --seed "${SMF_SEED:-42}"
