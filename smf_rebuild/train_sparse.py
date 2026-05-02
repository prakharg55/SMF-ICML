from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import DataCollatorForLanguageModeling, TrainingArguments, set_seed

from smf_rebuild.datasets import DatasetSpec, load_lm_dataset
from smf_rebuild.memory_model import (
    MemoryConfig,
    cast_trainable_params,
    count_trainable_params,
    dtype_from_name,
    freeze_for_memory_training,
    load_compatible_state_dict,
    load_qwen_with_memory,
    load_tokenizer,
    parse_layers,
    save_experiment_metadata,
)
from smf_rebuild.sparse_memory import SparseMemoryController, SparseSelectionConfig
from smf_rebuild.trainer_utils import SparseMemoryTrainer, save_log_history


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse finetune only selected memory value rows.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--init-checkpoint", required=True, help="Dense retrofit checkpoint to start from.")
    parser.add_argument("--memory-config", default=None, help="Defaults to <init-checkpoint>/memory_config.json.")
    parser.add_argument("--output-dir", default="outputs/qwen_memory_sparse")
    parser.add_argument("--layers", default=None, help="Override memory-config layers, e.g. 6,12,18.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trainable-scope", default="values", choices=["values", "values_scale", "memory"])
    parser.add_argument("--trainable-param-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])

    parser.add_argument("--dataset-preset", default="hellaswag", choices=["oasst1", "hellaswag", "medmcqa", "wikitext", "hf"])
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--eval-sample-size", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=1024)

    parser.add_argument("--background-preset", default="oasst1", choices=["oasst1", "hellaswag", "wikitext", "hf"])
    parser.add_argument("--background-dataset-name", default=None)
    parser.add_argument("--background-dataset-config", default=None)
    parser.add_argument("--background-text-column", default="text")
    parser.add_argument("--background-split", default="train")
    parser.add_argument("--background-sample-size", type=int, default=10000)
    parser.add_argument("--background-max-batches", type=int, default=1000)
    parser.add_argument("--background-batch-size", type=int, default=1)
    parser.add_argument("--sparse-top-t", type=int, default=256)
    parser.add_argument("--sparse-scoring", default="kl", choices=["kl", "tfidf"])
    parser.add_argument("--num-proc", type=int, default=4)

    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="tensorboard")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = dtype_from_name(args.dtype)

    config_path = Path(args.memory_config) if args.memory_config else Path(args.init_checkpoint) / "memory_config.json"
    memory_config = MemoryConfig.from_json(config_path)
    if args.layers:
        memory_config.layers = parse_layers(args.layers)

    tokenizer = load_tokenizer(args.base_model)
    model = load_qwen_with_memory(args.base_model, memory_config, dtype=model_dtype, device=device)
    missing, skipped = load_compatible_state_dict(model, args.init_checkpoint)
    if skipped:
        print(f"Skipped {len(skipped)} incompatible checkpoint tensors. Check memory_config if this is unexpected.")
    if missing:
        print(f"Missing {len(missing)} tensors after load. This is normal for optimizer-free checkpoints only if listed tensors are frozen.")

    freeze_for_memory_training(model, memory_config.layers, scope=args.trainable_scope)
    cast_trainable_params(model, dtype_from_name(args.trainable_param_dtype))
    trainable, total = count_trainable_params(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    print(f"Sparse selector: {args.sparse_scoring}, top_t={args.sparse_top_t}")

    train_dataset = load_lm_dataset(
        tokenizer,
        DatasetSpec(
            preset=args.dataset_preset,
            split=args.train_split,
            sample_size=args.sample_size,
            max_length=args.max_length,
            seed=args.seed,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            text_column=args.text_column,
            num_proc=args.num_proc,
        ),
    )
    eval_dataset = load_lm_dataset(
        tokenizer,
        DatasetSpec(
            preset=args.dataset_preset,
            split=args.eval_split,
            sample_size=args.eval_sample_size,
            max_length=args.max_length,
            seed=args.seed,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            text_column=args.text_column,
            num_proc=args.num_proc,
        ),
    )
    background_dataset = load_lm_dataset(
        tokenizer,
        DatasetSpec(
            preset=args.background_preset,
            split=args.background_split,
            sample_size=args.background_sample_size,
            max_length=args.max_length,
            seed=args.seed,
            dataset_name=args.background_dataset_name,
            dataset_config=args.background_dataset_config,
            text_column=args.background_text_column,
            num_proc=args.num_proc,
        ),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    sparse_controller = SparseMemoryController(
        model,
        memory_config.layers,
        SparseSelectionConfig(
            top_t=args.sparse_top_t,
            scoring=args.sparse_scoring,
            background_max_batches=args.background_max_batches,
        ),
        device=device,
    )
    print("Computing background memory-slot document frequencies...")
    sparse_controller.compute_background_df(
        background_dataset,
        collator,
        batch_size=args.background_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    sparse_controller.attach_gradient_hooks()

    use_bf16 = model_dtype == torch.bfloat16
    use_fp16 = model_dtype == torch.float16
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        max_grad_norm=1.0,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to.split(",") if args.report_to else [],
        remove_unused_columns=False,
    )
    trainer = SparseMemoryTrainer(
        sparse_controller=sparse_controller,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    try:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        save_experiment_metadata(args.output_dir, memory_config, args.base_model)
        save_log_history(trainer, args.output_dir)
        print(f"Saved sparse-finetuned model to {Path(args.output_dir).resolve()}")
    finally:
        sparse_controller.close()


if __name__ == "__main__":
    main()
