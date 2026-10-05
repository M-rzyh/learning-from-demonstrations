#!/usr/bin/env python3
"""Compute action change rate for human vs expert demonstrations.

Action change rate = (# of steps where a[t] != a[t-1]) / (episode_length - 1).
One number per episode; we aggregate across episodes per group.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk


GROUPS = {
    "Human session 1": ("/scratch/marzii/imitation_runs/human_demos/lunarlander/session_1", None),
    "Human session 2": ("/scratch/marzii/imitation_runs/human_demos/lunarlander/session_2", None),
    "Expert 4615187 (first 50)": ("/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/final.npz", 50),
}


def episode_change_rate(acts):
    acts = np.asarray(acts)
    if len(acts) < 2:
        return np.nan
    return float(np.mean(acts[1:] != acts[:-1]))


def load_rates(path, n_demos=None):
    ds = load_from_disk(path)
    if n_demos is not None:
        ds = ds.select(range(min(n_demos, len(ds))))
    return np.array([episode_change_rate(ep["acts"]) for ep in ds])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/home/marzii/IRL3/figures/action_change_rate.png")
    args = parser.parse_args()

    data = {}
    print(f"{'Group':<22} {'n_eps':>6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for name, (path, n_demos) in GROUPS.items():
        rates = load_rates(path, n_demos)
        data[name] = rates
        print(f"{name:<22} {len(rates):>6d} {rates.mean():>8.4f} {rates.std():>8.4f} "
              f"{rates.min():>8.4f} {rates.max():>8.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(1, len(data) + 1)
    bp = ax.boxplot(list(data.values()), tick_labels=list(data.keys()),
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='green', markersize=6),
                    whis=[0, 100], showfliers=False)
    for i, (name, vals) in enumerate(data.items(), start=1):
        ax.scatter(np.full_like(vals, i) + np.random.uniform(-0.08, 0.08, len(vals)),
                   vals, alpha=0.5, s=18, zorder=3)
        # Add annotation with stats to the right of each box
        med = np.median(vals)
        mean = np.mean(vals)
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        mn = np.min(vals)
        mx = np.max(vals)
        stats_text = (f"mean={mean:.4f}\n"
                      f"median={med:.4f}\n"
                      f"Q1={q1:.4f}\n"
                      f"Q3={q3:.4f}\n"
                      f"min={mn:.4f}\n"
                      f"max={mx:.4f}")
        ax.annotate(stats_text, xy=(i + 0.35, med), fontsize=7, fontfamily='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_ylabel("Action change rate (fraction of steps with a[t] ≠ a[t-1])")
    ax.set_title("Action change rate — human vs expert demonstrations")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
