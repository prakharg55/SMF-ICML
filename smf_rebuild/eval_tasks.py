from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from smf_rebuild.memory_model import (
    MemoryConfig,
    dtype_from_name,
    load_compatible_state_dict,
    load_qwen_with_memory,
    load_tokenizer,
)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 .-]", " ", text)
    return " ".join(text.split())


def load_eval_model(args, device):
    tokenizer = load_tokenizer(args.base_model)
    if args.memory_checkpoint:
        config_path = Path(args.memory_config) if args.memory_config else Path(args.memory_checkpoint) / "memory_config.json"
        memory_config = MemoryConfig.from_json(config_path)
        model = load_qwen_with_memory(args.base_model, memory_config, dtype_from_name(args.dtype), device=device)
        missing, skipped = load_compatible_state_dict(model, args.memory_checkpoint)
        if skipped:
            print(f"Skipped {len(skipped)} incompatible tensors while loading memory checkpoint.")
        if missing:
            print(f"Missing {len(missing)} tensors while loading memory checkpoint.")
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype_from_name(args.dtype),
            trust_remote_code=True,
        ).to(device)
    model.eval()
    return model, tokenizer


def continuation_score(model, tokenizer, context: str, continuation: str, device: str, normalize: bool = False) -> float:
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    full = tokenizer(context + continuation, return_tensors="pt", add_special_tokens=False)
    input_ids = full.input_ids.to(device)
    labels = input_ids.clone()
    labels[:, : len(ctx_ids)] = -100
    target_tokens = int((labels != -100).sum().item())
    if target_tokens == 0:
        return float("-inf")
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss
    if normalize:
        return float((-loss).item())
    return float((-loss * target_tokens).item())


def eval_hellaswag(model, tokenizer, device: str, limit: int | None, split: str = "validation"):
    dataset = load_dataset("hellaswag", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    correct = 0
    for row in tqdm(dataset, desc="HellaSwag"):
        ctx = row["ctx"]
        scores = [continuation_score(model, tokenizer, ctx, " " + ending, device) for ending in row["endings"]]
        pred = int(np.argmax(scores))
        correct += pred == int(row["label"])
    return {"acc": correct / len(dataset), "n": len(dataset)}


def _medmcqa_answer_index(row) -> int:
    cop = row["cop"]
    if isinstance(cop, str) and cop.strip().lower() in {"a", "b", "c", "d"}:
        return "abcd".index(cop.strip().lower())
    cop = int(cop)
    return cop if 0 <= cop <= 3 else cop - 1


def eval_medmcqa(model, tokenizer, device: str, limit: int | None, split: str = "validation", score_mode: str = "answer_norm"):
    dataset = load_dataset("openlifescienceai/medmcqa", split=split)
    dataset = dataset.filter(lambda x: x.get("choice_type") == "single")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    correct = 0
    for row in tqdm(dataset, desc="MedMCQA"):
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        prompt = (
            "Medical exam question:\n"
            f"Question: {row['question']}\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n"
            "Answer:"
        )
        if score_mode == "letter":
            continuations = [f" {label}" for label in "ABCD"]
        elif score_mode in {"answer", "answer_norm"}:
            continuations = [f" {label}. {option}" for label, option in zip("ABCD", options)]
        else:
            raise ValueError("--medmcqa-score-mode must be 'answer_norm', 'answer', or 'letter'")
        normalize = score_mode == "answer_norm"
        scores = [continuation_score(model, tokenizer, prompt, continuation, device, normalize=normalize) for continuation in continuations]
        pred = int(np.argmax(scores))
        correct += pred == _medmcqa_answer_index(row)
    return {"acc": correct / len(dataset), "n": len(dataset), "score_mode": score_mode}


def eval_wikitext_ppl(model, tokenizer, device: str, limit: int | None, max_length: int = 1024, stride: int = 512):
    if stride > max_length:
        raise ValueError("--wikitext-stride must be less than or equal to --wikitext-max-length")
    raw = load_dataset("wikitext", "wikitext-103-v1", split="test")
    texts = [row["text"].strip() for row in raw if row.get("text") and row["text"].strip()]
    if limit:
        texts = texts[:limit]
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)
    seq_len = encodings.input_ids.size(1)
    if seq_len <= 1:
        return {"ppl": float("inf"), "nll": float("inf"), "tokens": 0, "n": len(texts), "stride": stride}

    nll_sum = 0.0
    token_count = 0
    prev_end = 0
    for begin in tqdm(range(0, seq_len, stride), desc="WikiText perplexity"):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end].to(device)
        labels = input_ids.clone()
        labels[:, :-target_len] = -100
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss
        nll_sum += float(loss.item()) * target_len
        token_count += target_len
        prev_end = end
        if end == seq_len:
            break
    nll = nll_sum / max(token_count, 1)
    return {"ppl": math.exp(nll), "nll": nll, "tokens": int(token_count), "n": len(texts), "stride": stride}


def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def eval_triviaqa(model, tokenizer, device: str, limit: int | None):
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    correct = 0
    for row in tqdm(dataset, desc="TriviaQA"):
        pred = normalize_answer(generate_answer(model, tokenizer, f"Question: {row['question']}\nAnswer:", device, 32))
        aliases = [normalize_answer(x) for x in row["answer"]["aliases"]]
        correct += any(alias and alias in pred for alias in aliases)
    return {"alias_contains_acc": correct / len(dataset), "n": len(dataset)}


def eval_gsm8k(model, tokenizer, device: str, limit: int | None):
    dataset = load_dataset("gsm8k", "main", split="test")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    correct = 0
    for row in tqdm(dataset, desc="GSM8K"):
        response = generate_answer(
            model,
            tokenizer,
            f"Question: {row['question']}\nLet's think step by step.\nAnswer:",
            device,
            256,
        )
        pred_numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", response.replace(",", ""))
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        correct += bool(pred_numbers) and pred_numbers[-1] == gold
    return {"exact_number_acc": correct / len(dataset), "n": len(dataset)}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base or patched-memory Qwen models.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--memory-checkpoint", default=None)
    parser.add_argument("--memory-config", default=None)
    parser.add_argument("--tasks", default="hellaswag,wikitext", help="Comma-separated: medmcqa,hellaswag,wikitext,triviaqa,gsm8k")
    parser.add_argument("--medmcqa-score-mode", default="answer_norm", choices=["answer_norm", "answer", "letter"])
    parser.add_argument("--wikitext-max-length", type=int, default=1024)
    parser.add_argument("--wikitext-stride", type=int, default=512)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--output", default="outputs/eval_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_eval_model(args, device)
    results = {}
    for task in [x.strip().lower() for x in args.tasks.split(",") if x.strip()]:
        if task == "hellaswag":
            results[task] = eval_hellaswag(model, tokenizer, device, args.limit)
        elif task == "medmcqa":
            results[task] = eval_medmcqa(model, tokenizer, device, args.limit, score_mode=args.medmcqa_score_mode)
        elif task == "wikitext":
            results[task] = eval_wikitext_ppl(
                model,
                tokenizer,
                device,
                args.limit,
                max_length=args.wikitext_max_length,
                stride=args.wikitext_stride,
            )
        elif task == "triviaqa":
            results[task] = eval_triviaqa(model, tokenizer, device, args.limit)
        elif task == "gsm8k":
            results[task] = eval_gsm8k(model, tokenizer, device, args.limit)
        else:
            raise ValueError(f"Unknown task {task!r}")
        print(task, results[task])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
