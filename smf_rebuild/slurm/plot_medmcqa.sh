#!/bin/bash

set -euo pipefail

: "${SMF_MED_RUN_DIR:?Set SMF_MED_RUN_DIR before running this script.}"
mkdir -p "$SMF_MED_RUN_DIR/figures_values_scale"

python -m smf_rebuild.plot_results \
  --result "Base Qwen=$SMF_MED_RUN_DIR/eval_base_qwen.json" \
  --result "Replacement sparse KL=$SMF_MED_RUN_DIR/eval_replacement_sparse_kl.json" \
  --result "Additive sparse KL=$SMF_MED_RUN_DIR/eval_additive_sparse_kl.json" \
  --result "Additive sparse KL values+scale=$SMF_MED_RUN_DIR/eval_additive_sparse_kl_values_scale.json" \
  --result "LoRA=$SMF_MED_RUN_DIR/eval_lora.json" \
  --result "Full finetune=$SMF_MED_RUN_DIR/eval_full_finetune.json" \
  --loss-log "Replacement sparse KL=$SMF_MED_RUN_DIR/qwen_memory_sparse_kl/trainer_log_history.json" \
  --loss-log "Additive sparse KL=$SMF_MED_RUN_DIR/qwen_additive_memory_sparse_kl/trainer_log_history.json" \
  --loss-log "Additive sparse KL values+scale=$SMF_MED_RUN_DIR/qwen_additive_memory_sparse_kl_values_scale/trainer_log_history.json" \
  --loss-log "LoRA=$SMF_MED_RUN_DIR/qwen_lora_medmcqa/trainer_log_history.json" \
  --loss-log "Full finetune=$SMF_MED_RUN_DIR/qwen_full_finetune_medmcqa/trainer_log_history.json" \
  --output-dir "$SMF_MED_RUN_DIR/figures"
