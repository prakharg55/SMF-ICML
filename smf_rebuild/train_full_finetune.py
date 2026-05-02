from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed

from smf_rebuild.datasets import DatasetSpec, load_lm_dataset
from smf_rebuild.memory_model import dtype_from_name, load_tokenizer
from smf_rebuild.trainer_utils import save_log_history


def parse_args():
    parser = argparse.ArgumentParser(description="Full finetune the base model on a task dataset.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/qwen_full_finetune_hellaswag")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    parser.add_argument("--dataset-preset", default="hellaswag", choices=["oasst1", "hellaswag", "medmcqa", "wikitext", "hf"])
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--eval-sample-size", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-proc", type=int, default=4)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
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
    dtype = dtype_from_name(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

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
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        max_grad_norm=1.0,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to.split(",") if args.report_to else [],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_log_history(trainer, args.output_dir)
    (Path(args.output_dir) / "base_model.txt").write_text(args.base_model + "\n")
    print(f"Saved full-finetuned model to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
