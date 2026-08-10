"""Turn run artefacts into the figures the writeup embeds.

Colour and form follow one rule set throughout: three categorical series maximum
(KDA, Gated DeltaNet, Mamba2 -- the same three the paper's Figure 4 plots), softmax
attention as a neutral dashed reference rather than a fourth colour, direct labels on
every line so identity never depends on colour alone, and a CSV beside every figure so
the numbers are readable without the picture.

Usage::

    python -m scripts.make_figures --runs /mnt/data/kda/runs --out figures/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots 1-3 (all-pairs: worst CVD dE 9.2, normal-vision dE 24.0).
SERIES = {
    "kda": ("KDA", "#2a78d6"),
    "gated_deltanet": ("Gated DeltaNet", "#eb6834"),
    "mamba2": ("Mamba2", "#1baf7a"),
}
REFERENCE = {"softmax": ("Softmax attention", "#6b7280")}
INK, INK2, GRID = "#111827", "#4b5563", "#e5e7eb"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def load_synthetic(runs: Path):
    """-> {task: {variant: {seq_len: [best_acc per seed]}}}"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted((runs / "synthetic").glob("*/*.json")):
        if f.name == "best_lr.json":
            continue
        d = json.loads(f.read_text())
        c = d["config"]
        out[c["task"]][c["variant"]][c["seq_len"]].append(d["best_acc"])
    return out


def place_labels(ax, entries, min_gap=0.075):
    """Direct-label line ends, pushed apart so overlapping series stay readable.

    These tasks saturate -- several models sit at exactly 1.0 -- so without this the
    labels stack into an unreadable smear precisely where the interesting comparison
    is. Entries are ``(x, y, text, colour, bold)``; y is in axis data units.
    """
    entries = sorted(entries, key=lambda e: -e[1])
    placed = []
    for x, y, text, color, bold in entries:
        ty = y
        for py in placed:
            if abs(ty - py) < min_gap:
                ty = py - min_gap
        placed.append(ty)
        # a leader line when the label had to move off its point
        if abs(ty - y) > 1e-6:
            ax.annotate("", xy=(x, y), xytext=(x, ty), textcoords="data",
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.6))
        ax.annotate(text, (x, ty), textcoords="offset points", xytext=(6, 0),
                    fontsize=7.5, color=color, va="center",
                    fontweight="bold" if bold else "normal")


def style_axes(ax, xlabel, ylabel, title):
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)


def fig_accuracy_vs_length(data, out: Path):
    """Figure 4 top row: peak accuracy as the sequence gets longer."""
    tasks = [t for t in ("mqar", "palindrome", "stack") if t in data]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.5 * len(tasks), 3.0), squeeze=False)
    rows = []

    for ax, task in zip(axes[0], tasks):
        series = {**SERIES, **REFERENCE}
        labels = []
        for variant, (label, color) in series.items():
            if variant not in data[task]:
                continue
            lens = sorted(data[task][variant])
            if not lens:
                continue
            means = [sum(data[task][variant][L]) / len(data[task][variant][L]) for L in lens]
            lo = [min(data[task][variant][L]) for L in lens]
            hi = [max(data[task][variant][L]) for L in lens]
            is_ref = variant in REFERENCE
            ax.plot(lens, means, color=color, lw=2,
                    ls="--" if is_ref else "-",
                    marker="o" if not is_ref else None, ms=4.5,
                    zorder=3 if not is_ref else 2)
            # Individual seeds as dots, not a min/max band. On MQAR the outcome is
            # bimodal -- a seed either solves the task or sits at chance -- so a band
            # spans the whole axis and a mean lands in the empty middle where no run
            # ever was. Showing the raw points makes that visible instead of hiding it.
            if not is_ref:
                for L in lens:
                    pts = data[task][variant][L]
                    ax.scatter([L] * len(pts), pts, color=color, s=11, alpha=0.55,
                               linewidths=0, zorder=4)
            labels.append((lens[-1], means[-1], label, color, not is_ref))
            for L, m, l, h in zip(lens, means, lo, hi):
                rows.append([task, label, L, f"{m:.4f}", f"{l:.4f}", f"{h:.4f}"])

        place_labels(ax, labels)
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.03, 1.10)
        style_axes(ax, "sequence length", "best accuracy" if task == tasks[0] else "", task)
        ax.margins(x=0.34)

    fig.suptitle("Peak accuracy vs sequence length — 2 layers, 2 heads, 3 seeds (band = min/max)",
                 fontsize=9, color=INK2, y=1.02, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_length.png", bbox_inches="tight")
    plt.close(fig)

    with open(out / "accuracy_vs_length.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "model", "seq_len", "mean_acc", "min_acc", "max_acc"])
        w.writerows(rows)


def fig_convergence(data_dir: Path, out: Path, task="mqar", seq_len=1024):
    """Figure 4 bottom row: how fast each model gets there."""
    files = sorted((data_dir / "synthetic" / task).glob(f"*_T{seq_len}_*.json"))
    if not files:
        return
    curves = defaultdict(list)
    for f in files:
        d = json.loads(f.read_text())
        curves[d["config"]["variant"]].append(d["history"])

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    labels = []
    for variant, (label, color) in {**SERIES, **REFERENCE}.items():
        if variant not in curves:
            continue
        h = max(curves[variant], key=lambda x: max(p["acc"] for p in x))  # best seed
        steps = [p["step"] for p in h]
        accs = [p["acc"] for p in h]
        is_ref = variant in REFERENCE
        ax.plot(steps, accs, color=color, lw=2, ls="--" if is_ref else "-")
        labels.append((steps[-1], accs[-1], label, color, not is_ref))
    place_labels(ax, labels)
    ax.set_ylim(-0.03, 1.10)
    style_axes(ax, "training step", "accuracy",
               f"Convergence on {task} at T={seq_len} (best of 3 seeds)")
    ax.margins(x=0.22)
    fig.tight_layout()
    fig.savefig(out / f"convergence_{task}_{seq_len}.png", bbox_inches="tight")
    plt.close(fig)


def fig_kernel_bench(runs: Path, out: Path):
    """Figure 2: our chunk kernel against the general DPLR form and flash attention."""
    path = runs / "bench" / "kernels.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    rows = d["kernels"]
    lens = [r["seq_len"] for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for key, label, color in [("kda", "KDA (chunk)", "#2a78d6"),
                              ("dplr", "general DPLR", "#eb6834"),
                              ("flash_attn", "flash attention", "#6b7280")]:
        xs = [r["seq_len"] for r in rows if r.get(key)]
        ys = [r[key] for r in rows if r.get(key)]
        if not xs:
            continue
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4,
                ls="--" if key == "flash_attn" else "-")
        ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(4, 0), fontsize=7.5, color=color, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style_axes(ax, "sequence length", "milliseconds", "Kernel time (lower is better)")
    ax.margins(x=0.3)

    speedup = [(r["seq_len"], r["dplr"] / r["kda"])
               for r in rows if r.get("dplr") and r.get("kda")]
    if speedup:
        ax2.plot([s[0] for s in speedup], [s[1] for s in speedup],
                 color="#2a78d6", lw=2, marker="o", ms=4)
        ax2.axhline(1.0, color=GRID, lw=1.5, ls="--")
        ax2.axhline(2.0, color="#9ca3af", lw=1, ls=":")
        ax2.annotate("the paper's ~2×", (speedup[0][0], 2.0), textcoords="offset points",
                     xytext=(2, 4), fontsize=7.5, color=INK2)
        ax2.set_xscale("log", base=2)
        style_axes(ax2, "sequence length", "× faster than general DPLR",
                   f"Speedup from binding a = b = k ({d.get('device','GPU')})")
    fig.tight_layout()
    fig.savefig(out / "kernel_bench.png", bbox_inches="tight")
    plt.close(fig)


def fig_hybrid_ratio(runs: Path, out: Path):
    """Table 1's shape: validation perplexity against the KDA-to-attention ratio."""
    files = sorted((runs / "pretrain").glob("*.json"))
    if not files:
        return
    pts = []
    for f in files:
        d = json.loads(f.read_text())
        pts.append((d["config"]["ratio_name"], d["hybrid_ratio"], d["final_val_ppl"]))
    order = {"attn_only": 0, "1to1": 1, "3to1": 2, "7to1": 3, "kda_only": 4}
    pts.sort(key=lambda p: order.get(p[0], 99))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    labels = [p[0].replace("_", " ") for p in pts]
    vals = [p[2] for p in pts]
    best = min(range(len(vals)), key=lambda i: vals[i])
    colors = ["#2a78d6" if i == best else "#c7d7ee" for i in range(len(vals))]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=8, color=INK)
    style_axes(ax, "KDA layers per attention layer", "validation perplexity",
               "Hybrid ratio (lower is better)")
    ax.set_ylim(min(vals) * 0.97, max(vals) * 1.03)
    fig.tight_layout()
    fig.savefig(out / "hybrid_ratio.png", bbox_inches="tight")
    plt.close(fig)

    with open(out / "hybrid_ratio.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "kda_per_attention", "val_ppl"])
        w.writerows([[a, b, f"{c:.4f}"] for a, b, c in pts])


def fig_solve_rate(runs: Path, out: Path, seq_len=1024):
    """The headline synthetic result: how often each model solves MQAR at all.

    On this task accuracy is bimodal -- a run either reaches ~1.0 or stays at chance --
    so "mean accuracy over 3 seeds" mostly reports how many coin flips came up heads.
    Eight seeds and a solve count is the statistic the outcome actually supports.
    """
    files = sorted((runs / "solverate" / "mqar").glob(f"*_T{seq_len}_*.json"))
    if not files:
        return
    accs = defaultdict(list)
    for f in files:
        d = json.loads(f.read_text())
        accs[d["config"]["variant"]].append(d["best_acc"])

    order = ["deltanet", "kda", "gated_deltanet", "mamba2", "softmax"]
    names = {"deltanet": "DeltaNet\n(no gate)", "kda": "KDA\n(channel gate)",
             "gated_deltanet": "Gated DeltaNet\n(scalar gate)",
             "mamba2": "Mamba2\n(no delta rule)", "softmax": "Softmax\nattention"}
    rows = [(v, accs[v]) for v in order if v in accs]
    if not rows:
        return

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    # left: solve count
    labels = [names[v] for v, _ in rows]
    solved = [sum(a >= 0.9 for a in vals) for _, vals in rows]
    n = len(rows[0][1])
    colors = ["#6b7280" if v == "softmax" else
              ("#2a78d6" if v == "kda" else "#9db8dd") for v, _ in rows]
    bars = ax.bar(labels, solved, color=colors, width=0.62)
    for b, s in zip(bars, solved):
        ax.annotate(f"{s}/{n}", (b.get_x() + b.get_width() / 2, s), ha="center",
                    va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.set_ylim(0, n * 1.18)
    style_axes(ax, "", f"seeds that solved (of {n})",
               f"How often MQAR is solved at all (T={seq_len})")
    ax.tick_params(axis="x", labelsize=7.5)

    # right: every seed's accuracy, showing the bimodality directly
    for i, (v, vals) in enumerate(rows):
        col = "#6b7280" if v == "softmax" else ("#2a78d6" if v == "kda" else "#9db8dd")
        jitter = [i + (j - (len(vals) - 1) / 2) * 0.055 for j in range(len(vals))]
        ax2.scatter(jitter, vals, color=col, s=26, alpha=0.85, linewidths=0)
    ax2.axhline(0.9, color=GRID, lw=1.2, ls="--")
    ax2.annotate("solved", (len(rows) - 0.45, 0.91), fontsize=7.5, color=INK2, va="bottom")
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels([names[v].replace("\n", " ") for v, _ in rows],
                        fontsize=7, rotation=18, ha="right")
    ax2.set_ylim(-0.05, 1.08)
    style_axes(ax2, "", "best accuracy", "Every seed, plotted")

    fig.tight_layout()
    fig.savefig(out / "solve_rate_mqar.png", bbox_inches="tight")
    plt.close(fig)

    with open(out / "solve_rate_mqar.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seeds", "solved", "mean_acc", "accuracies"])
        for v, vals in rows:
            w.writerow([v, len(vals), sum(a >= 0.9 for a in vals),
                        f"{sum(vals)/len(vals):.4f}",
                        " ".join(f"{a:.2f}" for a in sorted(vals, reverse=True))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="/mnt/data/kda/runs")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_synthetic(runs)
    if data:
        fig_accuracy_vs_length(data, out)
        for task in data:
            fig_convergence(runs, out, task=task)
    fig_solve_rate(runs, out)
    fig_kernel_bench(runs, out)
    fig_hybrid_ratio(runs, out)
    made = sorted(f.name for f in out.iterdir())
    print(f"wrote {len(made)} files to {out}: {', '.join(made)}")


if __name__ == "__main__":
    main()


