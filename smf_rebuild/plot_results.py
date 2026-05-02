from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("medmcqa", "acc", "MedMCQA accuracy", "higher"),
    ("hellaswag", "acc", "HellaSwag accuracy", "higher"),
    ("wikitext", "ppl", "WikiText perplexity", "lower"),
    ("triviaqa", "alias_contains_acc", "TriviaQA alias accuracy", "higher"),
    ("gsm8k", "exact_number_acc", "GSM8K exact-number accuracy", "higher"),
]

COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#ff9da6", "#9d755d"]


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. If the path looks like '/eval_*.json', "
            "one of the shell variables in your command, such as $SMF_RUN_DIR "
            "or $SMF_ADD_RUN_DIR, was probably not set in this shell."
        )
    return json.loads(path.read_text())


def extract_metrics(label: str, path: str | Path) -> dict:
    payload = load_json(path)
    row = {"model": label, "source": str(path)}
    for task, key, _, _ in METRICS:
        value = payload.get(task, {}).get(key)
        if value is not None:
            row[f"{task}.{key}"] = value
    return row


def write_csv(rows: list[dict], output: Path) -> None:
    keys = ["model", "source"]
    for task, key, _, _ in METRICS:
        metric = f"{task}.{key}"
        if any(metric in row for row in rows):
            keys.append(metric)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_eval_bars(rows: list[dict], output: Path) -> None:
    available = []
    for task, key, title, direction in METRICS:
        metric = f"{task}.{key}"
        if any(metric in row for row in rows):
            available.append((metric, title, direction))

    if not available:
        print("No supported eval metrics found; skipping bar plot.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), squeeze=False)
    names = [row["model"] for row in rows]
    for ax, (metric, title, direction) in zip(axes[0], available):
        values = [row.get(metric) for row in rows]
        x = range(len(names))
        ax.bar(x, [v if v is not None else 0 for v in values], color=COLORS[: len(names)])
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("lower is better" if direction == "lower" else "higher is better")
        for idx, value in enumerate(values):
            if value is not None:
                ax.text(idx, value, f"{value:.4g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved eval comparison plot to {output}")


def load_losses(path: str | Path) -> tuple[list[int], list[float], list[int], list[float]]:
    path = Path(path)
    if not path.exists():
        return [], [], [], []
    history = load_json(path)
    train_steps = [int(row["step"]) for row in history if "loss" in row and "step" in row]
    train_losses = [float(row["loss"]) for row in history if "loss" in row and "step" in row]
    eval_steps = [int(row["step"]) for row in history if "eval_loss" in row and "step" in row]
    eval_losses = [float(row["eval_loss"]) for row in history if "eval_loss" in row and "step" in row]
    return train_steps, train_losses, eval_steps, eval_losses


def plot_loss_curves(series: list[tuple[str, str]], output: Path) -> None:
    has_any = False
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, path) in enumerate(series):
        train_steps, train_losses, eval_steps, eval_losses = load_losses(path)
        color = COLORS[idx % len(COLORS)]
        if train_steps:
            has_any = True
            ax.plot(train_steps, train_losses, color=color, alpha=0.35, label=f"{label} train")
        if eval_steps:
            has_any = True
            ax.plot(eval_steps, eval_losses, color=color, marker="o", linewidth=2, label=f"{label} eval")
    if not has_any:
        plt.close(fig)
        print("No trainer log histories found; skipping loss plot.")
        return
    ax.set_title("Training loss curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved loss plot to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot model evaluation results.")
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Add a labeled eval JSON. Can be repeated.",
    )
    parser.add_argument(
        "--loss-log",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Add a labeled trainer_log_history.json. Can be repeated.",
    )
    parser.add_argument("--base", default="outputs/eval_base_qwen.json")
    parser.add_argument("--retrofit", default="outputs/eval_dense_retrofit.json")
    parser.add_argument("--sparse-kl", default="outputs/eval_sparse_kl.json")
    parser.add_argument("--retrofit-log", default="outputs/qwen_memory_retrofit/trainer_log_history.json")
    parser.add_argument("--sparse-log", default="outputs/qwen_memory_sparse_hellaswag_kl/trainer_log_history.json")
    parser.add_argument("--output-dir", default="outputs/figures")
    return parser.parse_args()


def parse_labeled(items: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        parsed.append((label, path))
    return parsed


def main():
    args = parse_args()
    results = parse_labeled(args.result) or [
        ("Base Qwen", args.base),
        ("Dense retrofit", args.retrofit),
        ("Sparse KL", args.sparse_kl),
    ]
    rows = [extract_metrics(label, path) for label, path in results]
    loss_logs = parse_labeled(args.loss_log) or [
        ("Dense retrofit", args.retrofit_log),
        ("Sparse KL", args.sparse_log),
    ]
    output_dir = Path(args.output_dir)
    write_csv(rows, output_dir / "model_comparison.csv")
    plot_eval_bars(rows, output_dir / "model_comparison.png")
    plot_loss_curves(loss_logs, output_dir / "loss_curves.png")
    print(f"Saved comparison CSV to {output_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
