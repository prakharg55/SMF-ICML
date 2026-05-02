from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, load_dataset

@dataclass
class DatasetSpec:
    preset: str
    split: str
    sample_size: Optional[int] = None
    max_length: int = 1024
    seed: int = 42
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    text_column: str = "text"
    min_text_chars: int = 50
    num_proc: int = 4


def _limit(dataset: Dataset, sample_size: Optional[int], seed: int) -> Dataset:
    if sample_size is None or sample_size <= 0 or sample_size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_size))


def _load_texts(spec: DatasetSpec) -> Dataset:
    preset = spec.preset.lower()
    if preset == "oasst1":
        dataset = load_dataset("OpenAssistant/oasst1", split=spec.split)

        def keep(example):
            rank = example.get("rank")
            return (
                example.get("lang") == "en"
                and example.get("role") == "assistant"
                and rank in (0.0, 1.0)
                and len(example.get("text") or "") >= spec.min_text_chars
            )

        return dataset.filter(keep, num_proc=spec.num_proc)

    if preset == "hellaswag":
        dataset = load_dataset("hellaswag", split=spec.split)

        def to_text(example):
            label = int(example["label"])
            ending = example["endings"][label]
            ctx = example.get("ctx") or f"{example.get('ctx_a', '')} {example.get('ctx_b', '')}".strip()
            return {"text": f"{ctx} {ending}".strip()}

        return dataset.map(to_text, remove_columns=dataset.column_names, num_proc=spec.num_proc)

    if preset == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa", split=spec.split)
        dataset = dataset.filter(lambda x: x.get("choice_type") == "single", num_proc=spec.num_proc)

        def answer_index(example):
            cop = example["cop"]
            if isinstance(cop, str) and cop.strip().lower() in {"a", "b", "c", "d"}:
                return "abcd".index(cop.strip().lower())
            cop = int(cop)
            return cop if 0 <= cop <= 3 else cop - 1

        def to_text(example):
            options = [example["opa"], example["opb"], example["opc"], example["opd"]]
            idx = answer_index(example)
            label = "ABCD"[idx]
            answer = options[idx]
            explanation = (example.get("exp") or "").strip()
            text = (
                "Medical exam question:\n"
                f"Question: {example['question']}\n"
                f"A. {options[0]}\n"
                f"B. {options[1]}\n"
                f"C. {options[2]}\n"
                f"D. {options[3]}\n"
                f"Answer: {label}. {answer}"
            )
            if explanation:
                text += f"\nExplanation: {explanation}"
            return {"text": text}

        return dataset.map(to_text, remove_columns=dataset.column_names, num_proc=spec.num_proc)

    if preset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=spec.split)
        return dataset.filter(lambda x: len((x.get("text") or "").strip()) >= spec.min_text_chars, num_proc=spec.num_proc)

    if preset == "hf":
        if not spec.dataset_name:
            raise ValueError("--dataset-name is required when --dataset-preset hf")
        dataset = load_dataset(spec.dataset_name, spec.dataset_config, split=spec.split)
        if spec.text_column != "text":
            dataset = dataset.rename_column(spec.text_column, "text")
        return dataset

    raise ValueError(f"Unknown dataset preset {spec.preset!r}")


def load_lm_dataset(tokenizer, spec: DatasetSpec) -> Dataset:
    dataset = _limit(_load_texts(spec), spec.sample_size, spec.seed)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=spec.max_length,
            padding=False,
        )

    columns = dataset.column_names
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=columns,
        num_proc=spec.num_proc,
        desc=f"Tokenizing {spec.preset}:{spec.split}",
    )
    return tokenized.filter(lambda x: len(x["input_ids"]) > 1, num_proc=spec.num_proc)
