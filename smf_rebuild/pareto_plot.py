"""Aggregate eval JSONs across seeds and produce a paper-ready Pareto figure.

Usage:
    python3 -m smf_rebuild.pareto_plot \
        --runs /scratch/$USER/smf/medmcqa_seed55,/scratch/$USER/smf/medmcqa_seed5 \
        --output-dir /scratch/$USER/smf/figures_paper

Emits:
    pareto.png        (300 dpi)
    results_table.csv (mean ± std/range per (method, metric))
    results_table.tex (LaTeX-ready booktabs version of the same)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- methods in plot order, with display label, short-label-for-annotation, color, marker
# Convention: same color per architecture, marker shape distinguishes scoring rule.
#   circle (o)   = KL scoring     (or non-sparse baseline)
#   triangle (^) = TF-IDF scoring
#   square (s)   = non-sparse baseline that is its own thing (Base, Full FT)
METHODS = [
    ("eval_base_qwen.json",                          "Base Qwen",                       "Base",      "#7f7f7f", "s"),
    ("eval_replacement_sparse_kl.json",              "Replacement sparse (KL)",         "Repl-KL",   "#d62728", "o"),
    ("eval_replacement_sparse_tfidf.json",           "Replacement sparse (TF-IDF)",     "Repl-TF",   "#d62728", "^"),
    ("eval_additive_sparse_kl.json",                 "Additive sparse (KL)",            "Add-KL",    "#2ca02c", "o"),
    ("eval_additive_sparse_tfidf.json",              "Additive sparse (TF-IDF)",        "Add-TF",    "#2ca02c", "^"),
    ("eval_additive_sparse_kl_values_scale.json",    "Additive sparse +S (KL)",         "Add+S-KL",  "#17becf", "o"),
    ("eval_additive_sparse_tfidf_values_scale.json", "Additive sparse +S (TF-IDF)",     "Add+S-TF",  "#17becf", "^"),
    ("eval_lora.json",                               "LoRA",                            "LoRA",      "#1f77b4", "o"),
    ("eval_full_finetune.json",                      "Full finetune",                   "Full-FT",   "#9467bd", "s"),
]

# Per-(panel, method) label placements: (dx, dy, ha, va).
# Default puts the label above-right of the marker. For close-in pairs we put
# one label on the LEFT of its marker (right-aligned) so the two labels end up
# on opposite sides of their respective markers — no overlap, no arrows needed.
DEFAULT_PLACEMENT = (8, 6, "left", "baseline")
LABEL_PLACEMENTS = {
    # WikiText panel: Add-TF and Base are at almost identical y; Add-KL and
    # Add+S-TF share y; Repl-KL and Repl-TF share y.
    ("wikitext.ppl", "Add-TF"):   (-8, 0, "right", "center"),
    ("wikitext.ppl", "Add-KL"):   (-8, 0, "right", "center"),
    ("wikitext.ppl", "Repl-TF"):  (-8, 0, "right", "center"),
    # TriviaQA panel: same Add-TF/Base, same Add-KL/Add+S-TF (Add+S-TF is on
    # the LEFT here, so flip which one moves left), Repl-KL/Repl-TF spread is
    # large enough that default placement is fine.
    ("triviaqa.alias_contains_acc", "Add-TF"):    (-8, 0, "right", "center"),
    ("triviaqa.alias_contains_acc", "Add+S-TF"):  (-8, 0, "right", "center"),
}

# (json_field_path, label, direction)
FORGETTING = [
    (("wikitext", "ppl"),                  "WikiText perplexity",        "lower"),
    (("triviaqa", "alias_contains_acc"),   "TriviaQA accuracy",          "higher"),
]
TASK = ("medmcqa", "acc", "MedMCQA accuracy")


def collect(run_dirs):
    """Returns {label: {metric_key: [values across seeds]}}, metric_key='task.subkey'."""
    data = defaultdict(lambda: defaultdict(list))
    for d in run_dirs:
        d = Path(d)
        for fname, label, _, _, _ in METHODS:
            p = d / fname
            if not p.exists():
                continue
            payload = json.loads(p.read_text())
            for task, key in [(TASK[0], TASK[1])] + [(t, k) for (t, k), _, _ in FORGETTING]:
                v = payload.get(task, {}).get(key)
                if v is not None:
                    data[label][f"{task}.{key}"].append(float(v))
    return data


def stats(values):
    """Returns (mean, low_err, high_err, n).
    For n=2 returns range bars; for n>=3 returns std."""
    n = len(values)
    if n == 0: return None, 0.0, 0.0, 0
    m = float(np.mean(values))
    if n == 1:
        return m, 0.0, 0.0, 1
    if n == 2:
        return m, m - min(values), max(values) - m, 2
    s = float(np.std(values, ddof=1))
    return m, s, s, n


def is_pareto(points, x_dir, y_dir):
    """Given list of (x, y, label), return labels on the Pareto frontier.
    x_dir / y_dir: 'lower' or 'higher' = better."""
    keep = []
    for i, (xi, yi, li) in enumerate(points):
        dominated = False
        for j, (xj, yj, lj) in enumerate(points):
            if i == j: continue
            x_better = (xj <= xi) if x_dir == "lower" else (xj >= xi)
            y_better = (yj <= yi) if y_dir == "lower" else (yj >= yi)
            x_strict = (xj <  xi) if x_dir == "lower" else (xj >  xi)
            y_strict = (yj <  yi) if y_dir == "lower" else (yj >  yi)
            if x_better and y_better and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            keep.append(li)
    return set(keep)


def plot_pareto(data, output_png, n_seeds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plt.rcParams.update({"font.size": 11})

    for ax, ((xtask, xkey), xlabel, xdir) in zip(axes, FORGETTING):
        # collect points
        rows = []
        for fname, label, short, color, marker in METHODS:
            if label not in data: continue
            xm, xlo, xhi, n = stats(data[label].get(f"{xtask}.{xkey}", []))
            ym, ylo, yhi, _ = stats(data[label].get(f"{TASK[0]}.{TASK[1]}", []))
            if xm is None or ym is None: continue
            rows.append((xm, ym, xlo, xhi, ylo, yhi, label, short, color, marker))

        # Pareto frontier (use task acc on y as 'higher better')
        pareto = is_pareto([(x, y, lbl) for x, y, *_, lbl, _, _, _ in rows], xdir, "higher")
        # connect frontier with a faint dashed line
        front = sorted([(x, y, lbl) for x, y, *_, lbl, _, _, _ in rows if lbl in pareto],
                       key=lambda r: (r[0] if xdir == "lower" else -r[0]))
        if len(front) >= 2:
            ax.plot([r[0] for r in front], [r[1] for r in front],
                    linestyle="--", color="#aaaaaa", alpha=0.7, linewidth=1.2, zorder=1,
                    label=None)

        # plot each method
        for xm, ym, xlo, xhi, ylo, yhi, label, short, color, marker in rows:
            ax.errorbar(xm, ym, xerr=[[xlo],[xhi]], yerr=[[ylo],[yhi]],
                        fmt=marker, markersize=10, color=color, ecolor=color,
                        capsize=3, elinewidth=1.0,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=3)
            # annotate, with offset chosen per-point to avoid overlap
            key = (f"{xtask}.{xkey}", short)
            dx, dy, ha, va = LABEL_PLACEMENTS.get(key, DEFAULT_PLACEMENT)
            ax.annotate(short, (xm, ym), xytext=(dx, dy),
                        textcoords="offset points",
                        ha=ha, va=va,
                        fontsize=10, color="black", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        ax.set_xlabel(f"{xlabel}  ({'← lower = less forgetting' if xdir=='lower' else 'higher = retained →'})")
        ax.set_ylabel(f"{TASK[2]}  (higher = more learning →)")
        ax.grid(True, alpha=0.25, zorder=0)

        # subtle "best corner" hint
        if xdir == "lower":
            ax.text(0.02, 0.98, "best", transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, alpha=0.5, style="italic")
        else:
            ax.text(0.98, 0.98, "best", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, alpha=0.5, style="italic")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_png}")


def write_csv_and_tex(data, output_csv, output_tex, n_seeds):
    err_label = "range" if n_seeds == 2 else "std"
    metric_keys = [(TASK[0], TASK[1], TASK[2], "higher")] + \
                  [(t, k, lbl, dirn) for (t, k), lbl, dirn in FORGETTING]

    # CSV
    cols = ["method", "n_seeds"]
    for _, _, mlbl, _ in metric_keys:
        cols += [f"{mlbl} mean", f"{mlbl} {err_label}"]
    rows = []
    for _, label, _, _, _ in METHODS:
        if label not in data: continue
        n = max((len(v) for v in data[label].values()), default=0)
        row = [label, n]
        for t, k, _, _ in metric_keys:
            m, lo, hi, _ = stats(data[label].get(f"{t}.{k}", []))
            err = max(lo, hi)
            row += [f"{m:.4f}" if m is not None else "",
                    f"{err:.4f}" if m is not None else ""]
        rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {output_csv}")

    # LaTeX (booktabs)
    with output_tex.open("w") as f:
        f.write("% requires \\usepackage{booktabs}\n")
        f.write("\\begin{tabular}{l" + "c" * len(metric_keys) + "}\n")
        f.write("\\toprule\n")
        header = ["Method"] + [f"{lbl} ({'$\\downarrow$' if d=='lower' else '$\\uparrow$'})" for _, _, lbl, d in metric_keys]
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")
        for label, _, *_ in [(l,) + tuple(rest) for fname, l, *rest in METHODS]:
            if label not in data: continue
            row = [label]
            for t, k, _, _ in metric_keys:
                m, lo, hi, n = stats(data[label].get(f"{t}.{k}", []))
                err = max(lo, hi)
                if m is None:
                    row.append("---")
                elif n == 1 or err == 0:
                    row.append(f"{m:.3f}")
                else:
                    row.append(f"{m:.3f} $\\pm$ {err:.3f}")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"wrote {output_tex}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True,
                   help="Comma-separated run dirs, one per seed.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    data = collect(runs)
    n_seeds = max((max((len(v) for v in d.values()), default=0) for d in data.values()), default=0)
    print(f"Aggregated {len(data)} methods across {n_seeds} seeds.")
    for _, label, *_ in METHODS:
        if label not in data: continue
        bits = []
        for t, k in [(TASK[0], TASK[1])] + [(t,k) for (t,k),_,_ in FORGETTING]:
            m, lo, hi, _ = stats(data[label].get(f"{t}.{k}", []))
            err = max(lo, hi)
            bits.append(f"{t}={m:.3f}±{err:.3f}" if m is not None else f"{t}=—")
        print(f"  {label:38s}  " + "  ".join(bits))
    print()

    plot_pareto(data, out / "pareto.png", n_seeds)
    write_csv_and_tex(data, out / "results_table.csv", out / "results_table.tex", n_seeds)


if __name__ == "__main__":
    main()
