#!/bin/bash

#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=eval_med_repl_sparse
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --output=smf_rebuild/eval/%x.log

set -euo pipefail

: "${SMF_MED_RUN_DIR:?Set SMF_MED_RUN_DIR before submitting this job.}"
mkdir -p "$SMF_MED_RUN_DIR" smf_rebuild/eval

python -m smf_rebuild.eval_tasks \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --memory-checkpoint "$SMF_MED_RUN_DIR/qwen_memory_sparse_kl" \
  --tasks medmcqa,wikitext,triviaqa \
  --limit 1000 \
  --output "$SMF_MED_RUN_DIR/eval_replacement_sparse_kl.json"
