#!/bin/bash

#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=train_med_full
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

python -m smf_rebuild.train_full_finetune \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir "$SMF_MED_RUN_DIR/qwen_full_finetune_medmcqa" \
  --dataset-preset medmcqa \
  --train-split train \
  --eval-split validation \
  --sample-size 60000 \
  --eval-sample-size 1000 \
  --max-length 1024 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 3 \
  --learning-rate 5e-5 \
  --eval-steps 500 \
  --save-steps 500 \
  --seed "${SMF_SEED:-42}"
