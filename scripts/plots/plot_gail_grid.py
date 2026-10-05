#!/usr/bin/env python3
"""Plot the GAIL count + noise grid (post-hoc and online noise).

Reads gail_grid_<date>.csv produced by scrape_gail_grid_results.py and
generates four plots (× 2 metrics = 8 total):
  - Count axis: human time vs reward (mean±std over seeds), noise=0%
  - Post-hoc noise axis: noise % vs reward (mean±std), N=100
  - Online noise axis: noise % vs reward (mean±std), N=100
  - Combined noise axis (post-hoc vs online overlaid)

Y-axis can be eval_reward_mean (post-training rollout) or last10_tb_reward
(training-time avg).
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEC_PER_DEMO = 43.2  # human time/demo, from sess2 measurement


def human_time_min(n_demos):
    return n_demos * SEC_PER_DEMO / 60.0


def load_grid(csv_path):
    rows = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows[r["condition_id"]].append(r)
    return rows


def aggregate(rows, y_key):
    vals = []
    for r in rows:
        v = r.get(y_key, "")
        if v != "":
            try:
                vals.append(float(v))
            except ValueError:
                pass
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def plot_count_axis(grid, y_key, ylabel, out_path):
    Ns = [1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150]
    data = []
    for N in Ns:
        cond = f"count_N{N}"
        rows = grid.get(cond, [])
        m, s, n = aggregate(rows, y_key)
        if m is None:
            continue
        data.append({"N": N, "mean": m, "std": s, "n": n, "x": human_time_min(N)})
        print(f"  {cond}: N={N}, n_seeds={n}, mean={m:.2f}±{s:.2f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = [d["x"] for d in data]
    ys = [d["mean"] for d in data]
    es = [d["std"] for d in data]
    ax.errorbar(xs, ys, yerr=es, fmt="-s", color="#1f77b4", markersize=8, capsize=4,
                linewidth=1.6, label="GAIL count axis (noise=0%)")
    for d in data:
        ax.annotate(f"N={d['N']}", (d["x"], d["mean"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel(f"Human Time (min) — assuming {SEC_PER_DEMO} sec/demo from sess2")
    ax.set_ylabel(ylabel)
    ax.set_title(f"GAIL count axis (noise=0%) — 10 seeds per point, mean ± std")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_noise_axis(grid, y_key, ylabel, condition_prefix, out_path, label_suffix, color):
    """condition_prefix like 'noise_p' or 'noise_online_p'."""
    pcts = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    data = []
    for pct in pcts:
        if pct == 0:
            cond = "count_N100"  # 0% comes from the no-noise N=100 run
        else:
            cond = f"{condition_prefix}{pct}"
        rows = grid.get(cond, [])
        m, s, n = aggregate(rows, y_key)
        if m is None:
            continue
        data.append({"pct": pct, "mean": m, "std": s, "n": n})
        print(f"  {cond}: noise={pct}%, n_seeds={n}, mean={m:.2f}±{s:.2f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = [d["pct"] for d in data]
    ys = [d["mean"] for d in data]
    es = [d["std"] for d in data]
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color=color, markersize=8, capsize=4,
                linewidth=1.6, label=f"GAIL noise axis ({label_suffix}, N=100)")
    for d in data:
        ax.annotate(f"{d['mean']:.0f}", (d["pct"], d["mean"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Action-noise (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"GAIL noise axis ({label_suffix}) — N=100, 10 seeds per point, mean ± std")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_noise_combined(grid, y_key, ylabel, out_path):
    """Both post-hoc and online noise on the same axes."""
    pcts = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    series = [
        ("Post-hoc (action-flip on saved demos)", "noise_p", "#d62728", "-o"),
        ("Online (noise during rollout)", "noise_online_p", "purple", "-^"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, prefix, color, fmt in series:
        data = []
        for pct in pcts:
            cond = "count_N100" if pct == 0 else f"{prefix}{pct}"
            rows = grid.get(cond, [])
            m, s, n = aggregate(rows, y_key)
            if m is None:
                continue
            data.append({"pct": pct, "mean": m, "std": s})
        xs = [d["pct"] for d in data]
        ys = [d["mean"] for d in data]
        es = [d["std"] for d in data]
        ax.errorbar(xs, ys, yerr=es, fmt=fmt, color=color, markersize=7, capsize=3,
                    linewidth=1.5, label=label, alpha=0.85)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Action-noise (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"GAIL noise axis — post-hoc vs online (N=100, 10 seeds, mean ± std)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--out-prefix", default="/home/marzii/IRL3/figures/gail_grid")
    args = parser.parse_args()
    grid = load_grid(args.csv)

    for y_key, suffix, ylabel in [
        ("eval_reward_mean", "eval", "Post-training Eval Reward (50 ep, mean over seeds)"),
        ("last10_tb_reward", "tb",   "TB last-10% Reward (mean over seeds)"),
    ]:
        print(f"\n========== {suffix.upper()} ==========")
        print("\n[count axis]")
        plot_count_axis(grid, y_key, ylabel, f"{args.out_prefix}_{suffix}_count_axis.png")
        print("\n[post-hoc noise]")
        plot_noise_axis(grid, y_key, ylabel, "noise_p",
                        f"{args.out_prefix}_{suffix}_noise_posthoc.png", "post-hoc", "#d62728")
        print("\n[online noise]")
        plot_noise_axis(grid, y_key, ylabel, "noise_online_p",
                        f"{args.out_prefix}_{suffix}_noise_online.png", "online", "purple")
        print("\n[combined noise]")
        plot_noise_combined(grid, y_key, ylabel, f"{args.out_prefix}_{suffix}_noise_compare.png")


if __name__ == "__main__":
    main()
