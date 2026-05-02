#!/bin/bash

#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=train_med_lora
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

python -m smf_rebuild.train_lora \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir "$SMF_MED_RUN_DIR/qwen_lora_medmcqa" \
  --dataset-preset medmcqa \
  --train-split train \
  --eval-split validation \
  --sample-size 60000 \
  --eval-sample-size 1000 \
  --max-length 1024 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --eval-steps 500 \
  --save-steps 500 \
  --seed "${SMF_SEED:-42}"
