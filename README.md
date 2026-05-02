# Code release

This repository contains the code for the paper *"Comparing Sparse Memory
Finetuning with LoRA and Full Finetuning"* (anonymous submission).

It implements:
- A retrofit of `Qwen-2.5-0.5B-Instruct` with Product Key Memory (PKM)
  layers at selected transformer layers, in two integration modes
  (replacement and additive).
- A two-stage training pipeline: a dense retrofit phase on
  general-purpose text, followed by a sparse phase that updates only a
  small subset of memory value rows per batch.
- Two slot-selection rules for the sparse phase: a TF-IDF rule and a
  KL-divergence rule.
- Two non-sparse baselines: LoRA and full finetuning.
- An evaluator for MedMCQA accuracy, WikiText sliding-window perplexity,
  and TriviaQA alias-substring accuracy.

## Repository layout

```
.
├── README.md                         # this file
├── requirements.txt
├── memory_layers/                    # Product Key Memory module — see attribution below
│   ├── __init__.py
│   ├── memory.py                     # HashingMemory (PKM with SiLU gating)
│   ├── xformer_embeddingbag.py       # weighted embedding-bag lookup
│   ├── callbacks.py                  # TrainerCallbacks (not used by paper pipeline)
│   ├── data.py                       # data loading helpers (not used by paper pipeline)
│   ├── evaluation.py                 # ModelEvaluator (not used by paper pipeline)
│   ├── ft_callbacks.py               # TrainerCallback variant (not used by paper pipeline)
│   └── sft_callbacks.py              # TrainerCallback variant (not used by paper pipeline)
└── smf_rebuild/
    ├── __init__.py
    ├── datasets.py                   # MedMCQA / OASST1 / WikiText / TriviaQA loaders
    ├── memory_model.py               # patch Qwen MLPs with memory layers
    ├── sparse_memory.py              # per-batch slot scoring + gradient masking
    ├── trainer_utils.py              # SparseMemoryTrainer (HF Trainer subclass)
    ├── train_retrofit.py             # stage 1: dense retrofit on OASST1
    ├── train_sparse.py               # stage 2: sparse value-row training on MedMCQA
    ├── train_lora.py                 # baseline: LoRA on MedMCQA
    ├── train_full_finetune.py        # baseline: full finetune on MedMCQA
    ├── eval_tasks.py                 # MedMCQA + WikiText + TriviaQA evaluation
    ├── plot_results.py               # per-seed bar / loss-curve plots
    ├── pareto_plot.py                 # cross-seed Pareto figure used in the paper
    └── slurm/                        # example slurm wrappers (anonymized)
```

### Attribution for `memory_layers/`

The `memory_layers/` directory contains the Product Key Memory implementation
adapted from the Meta XLM repository
(<https://github.com/facebookresearch/XLM/blob/main/xlm/model/memory>),
released under the original Meta copyright. We include it unmodified for
reproducibility. Of the files in this directory, only `memory.py` and
`xformer_embeddingbag.py` are exercised by the paper pipeline (they
implement the `HashingMemory` module that `smf_rebuild/memory_model.py`
inserts into Qwen's transformer blocks). The remaining files
(`callbacks.py`, `data.py`, `evaluation.py`, `ft_callbacks.py`,
`sft_callbacks.py`) are part of the original module but are not invoked
by any of the training or evaluation commands documented below.

## Setup

Python 3.12 with the dependencies listed in `requirements.txt`:

```bash
python -m venv ~/smf_env
source ~/smf_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Hugging Face caching:

```bash
export HF_HOME=<path/to/cache>
```

## Reproducing the paper

The paper uses three random seeds. For each seed `S`, set:

```bash
export SMF_SEED=S
export SMF_MED_RUN_DIR=<scratch>/medmcqa_seed${SMF_SEED}
mkdir -p "$SMF_MED_RUN_DIR"
```

Then run the conditions below in order. Stage 1 produces the retrofit
checkpoints required by Stage 2.

### Stage 0 — base evaluation

```bash
python -m smf_rebuild.eval_tasks \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks medmcqa,wikitext,triviaqa --limit 1000 \
  --medmcqa-score-mode answer_norm \
  --output "$SMF_MED_RUN_DIR/eval_base_qwen.json"
```

### Stage 1 — dense retrofit (replacement and additive)

Replacement retrofit (used as init for replacement-sparse runs):

```bash
python -m smf_rebuild.train_retrofit \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir "$SMF_MED_RUN_DIR/qwen_memory_retrofit" \
  --memory-mode replacement \
  --layers 6,12,18 --mem-n-keys 128 --mem-heads 4 --mem-knn 16 --mem-k-dim 256 \
  --dataset-preset oasst1 --train-split train --eval-split validation \
  --sample-size 50000 --eval-sample-size 2000 --max-length 1024 \
  --per-device-train-batch-size 2 --gradient-accumulation-steps 8 \
  --num-train-epochs 2 --learning-rate 5e-4 \
  --eval-steps 500 --save-steps 500 --seed "$SMF_SEED"
```

Additive retrofit (used as init for additive-sparse and additive-+S runs):

```bash
python -m smf_rebuild.train_retrofit \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir "$SMF_MED_RUN_DIR/qwen_additive_memory_retrofit" \
  --memory-mode additive --memory-scale-init 0.01 \
  --layers 6,12,18 --mem-n-keys 128 --mem-heads 4 --mem-knn 16 --mem-k-dim 256 \
  --dataset-preset oasst1 --train-split train --eval-split validation \
  --sample-size 50000 --eval-sample-size 2000 --max-length 1024 \
  --per-device-train-batch-size 2 --gradient-accumulation-steps 8 \
  --num-train-epochs 2 --learning-rate 5e-4 \
  --eval-steps 500 --save-steps 500 --seed "$SMF_SEED"
```

### Stage 2 — sparse training

For each of the three sparse architectures and each of the two scoring
rules (six runs total per seed). Replace `<SCORING>` with `kl` or
`tfidf` and `<TAG>` with the matching `kl` or `tfidf` for output paths.

Replacement sparse:

```bash
python -m smf_rebuild.train_sparse \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --init-checkpoint "$SMF_MED_RUN_DIR/qwen_memory_retrofit" \
  --output-dir "$SMF_MED_RUN_DIR/qwen_memory_sparse_<TAG>" \
  --dataset-preset medmcqa \
  --sample-size 60000 --eval-sample-size 1000 --max-length 1024 \
  --background-preset oasst1 --background-sample-size 20000 --background-max-batches 2000 \
  --sparse-scoring <SCORING> --sparse-top-t 512 --trainable-scope values \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 4 \
  --num-train-epochs 3 --learning-rate 5e-4 \
  --eval-steps 500 --save-steps 500 --seed "$SMF_SEED"
```

Additive sparse (same args, but use the additive retrofit and a different
output dir):

```bash
python -m smf_rebuild.train_sparse \
  --init-checkpoint "$SMF_MED_RUN_DIR/qwen_additive_memory_retrofit" \
  --output-dir "$SMF_MED_RUN_DIR/qwen_additive_memory_sparse_<TAG>" \
  ... # same other flags as above with --trainable-scope values
```

Additive sparse +S (also trains the per-layer additive scale `α`):

```bash
python -m smf_rebuild.train_sparse \
  --init-checkpoint "$SMF_MED_RUN_DIR/qwen_additive_memory_retrofit" \
  --output-dir "$SMF_MED_RUN_DIR/qwen_additive_memory_sparse_<TAG>_values_scale" \
  ... # same other flags as above but --trainable-scope values_scale
```

### Stage 2 — non-sparse baselines

LoRA:

```bash
python -m smf_rebuild.train_lora \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir "$SMF_MED_RUN_DIR/qwen_lora_medmcqa" \
  --dataset-preset medmcqa \
  --sample-size 60000 --eval-sample-size 1000 --max-length 1024 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 4 \
  --num-train-epochs 3 --learning-rate 2e-4 \
  --eval-steps 500 --save-steps 500 --seed "$SMF_SEED"
```

Full finetune:

```bash
python -m smf_rebuild.train_full_finetune \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir "$SMF_MED_RUN_DIR/qwen_full_finetune_medmcqa" \
  --dataset-preset medmcqa \
  --sample-size 60000 --eval-sample-size 1000 --max-length 1024 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --num-train-epochs 3 --learning-rate 5e-5 \
  --eval-steps 500 --save-steps 500 --seed "$SMF_SEED"
```

### Stage 3 — evaluation

For sparse / retrofit checkpoints, point `--memory-checkpoint` at the
saved directory. For LoRA, point `--base-model` at the merged checkpoint
that `train_lora.py` writes under `<output_dir>/merged`. For full
finetune, point `--base-model` at the trained directory directly. All
evals share the same flags otherwise:

```bash
python -m smf_rebuild.eval_tasks \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --memory-checkpoint "$SMF_MED_RUN_DIR/qwen_<dir>" \
  --tasks medmcqa,wikitext,triviaqa --limit 1000 \
  --medmcqa-score-mode answer_norm \
  --output "$SMF_MED_RUN_DIR/eval_<name>.json"
```

### Stage 4 — Pareto figure

After all conditions for all seeds have been evaluated, produce the
paper's main figure and results table:

```bash
python -m smf_rebuild.pareto_plot \
  --runs <scratch>/medmcqa_seed1,<scratch>/medmcqa_seed2,<scratch>/medmcqa_seed3 \
  --output-dir <scratch>/figures
```

This writes `pareto.png`, `results_table.csv`, and `results_table.tex`
to `<output-dir>`.

## Slurm wrappers

The `slurm/` directory contains one example slurm script per
training/eval condition, plus `plot_medmcqa.sh` for the per-seed
plotting step. Before submitting, edit the `<YOUR_ACCOUNT>`,
`<YOUR_EMAIL>`, and `<YOUR_PARTITION>` placeholders in the SBATCH
headers, and add or replace any cluster-specific module loads or GPU
flags as appropriate for your environment.

## Hyperparameters at a glance

| Method | Trained params | LR | Epochs | Batch (eff.) |
|---|---|---|---|---|
| Replacement sparse / Additive sparse / Additive sparse +S | top-T=512 value rows per layer per batch | 5e-4 | 3 | 16 |
| LoRA (rank 16, α=32) | ~9M | 2e-4 | 3 | 16 |
| Full finetune | ~494M | 5e-5 | 3 | 16 |
| Dense retrofit (replacement / additive) | ~52M | 5e-4 | 2 | 16 |

Memory layer config: `mem_n_keys=128` (so 16,384 slots per layer),
`mem_heads=4`, `mem_knn=16`, `mem_k_dim=256`, layers `{6, 12, 18}`.
Background statistics for sparse slot selection: 2,000 single-example
batches of OASST1.

Optimizer: AdamW with cosine learning-rate schedule, 100 warmup steps,
max sequence length 1024, gradient clipping at `||g||=1.0`.

## Datasets

All four datasets used in the paper are loaded from Hugging Face Hub
and are public:

- `Qwen/Qwen2.5-0.5B-Instruct` (model)
- `openlifescienceai/medmcqa` (target task)
- `OpenAssistant/oasst1` (retrofit + sparse-selection background)
- `wikitext` (`wikitext-103-v1`, perplexity probe)
- `trivia_qa` (`rc.nocontext`, knowledge-retention probe)

## Notes on the framework

`eval_tasks.py` and `datasets.py` are written as multi-task frameworks
and accept additional task / dataset names beyond the four used in the
paper (e.g.\ HellaSwag, GSM8K). These code paths are not exercised by
the commands above and do not affect the reproduction of any reported
result. The paper's results are obtained exclusively from the
combination `--tasks medmcqa,wikitext,triviaqa` and
`--dataset-preset medmcqa` (task) / `--background-preset oasst1`
(sparse-selection background) / `--dataset-preset oasst1` (retrofit).
