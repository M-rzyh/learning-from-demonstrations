#!/usr/bin/env python3
"""Plot mean±std PEBBLE learning curves across multiple seeds.

Reads train.csv from each run and averages true_episode_reward over a common
step grid via interpolation (handles seeds with different episode counts).

Usage:
    python plot_pebble_multi_seed.py \\
        --group "label:JOBID,JOBID,..." \\
        [--group ...] --output FILE
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_dir import fig_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PEBBLE_ROOTS = [
    "/scratch/marzii/compare_runs/pebble/lunarlander",
    "/scratch/marzii/compare_runs/pebble/lunarlander_human",
    "/scratch/marzii/compare_runs/pebble/lunarlander_web",
]


def find_train_csv(job_id):
    for root in PEBBLE_ROOTS:
        for seedish in ("", "/seed_12345"):
            p = f"{root}/{job_id}{seedish}/pebble/train.csv"
            if os.path.exists(p):
                return p
    return None


def load_seed(job_id):
    p = find_train_csv(job_id)
    if not p:
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    steps = []
    rewards = []
    for r in rows:
        s = r.get("step")
        v = r.get("true_episode_reward")
        if s in (None, "", "nan") or v in (None, "", "nan"):
            continue
        try:
            steps.append(float(s))
            rewards.append(float(v))
        except ValueError:
            continue
    if not steps:
        return None
    return np.array(steps), np.array(rewards)


def smooth(values, window=51):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='same')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="PEBBLE learning curves")
    p.add_argument("--ylim", default=None)
    p.add_argument("--colors", default=None)
    p.add_argument("--max-step", type=float, default=1_000_000)
    p.add_argument("--n-grid", type=int, default=200, help="Common-step grid points")
    p.add_argument("--smooth", type=int, default=51, help="Moving avg window for raw curves")
    args = p.parse_args()

    grid = np.linspace(0, args.max_step, args.n_grid)
    groups = []
    for g in args.group:
        label, jobs_str = g.split(":", 1)
        jobs = [j.strip() for j in jobs_str.split(",")]
        seed_curves = []
        for j in jobs:
            data = load_seed(j)
            if data is None:
                print(f"  [skip] {j}: no train.csv")
                continue
            steps, rew = data
            # Smooth the noisy per-episode rewards
            rew_s = smooth(rew, window=args.smooth)
            # Interpolate to common step grid; mask outside this seed's range with NaN
            mask = grid <= steps.max()
            interp = np.full(grid.shape, np.nan)
            interp[mask] = np.interp(grid[mask], steps, rew_s)
            seed_curves.append(interp)
            last10_idx = int(len(steps) * 0.9)
            last10 = rew[last10_idx:].mean() if len(rew) > last10_idx else float('nan')
            print(f"  {j}: {len(steps)} eps, max step {steps.max():.0f}, last10% raw mean = {last10:.2f}")
        if not seed_curves:
            continue
        seed_curves = np.array(seed_curves)
        groups.append((label, jobs, seed_curves))

    fig, ax = plt.subplots(figsize=(11, 6))
    if args.colors:
        colors = [c.strip() for c in args.colors.split(",")]
    else:
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for i, (label, jobs, curves) in enumerate(groups):
        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0)
        c = colors[i % len(colors)]
        # Last-10% reward across the grid (final 10% of x range)
        last_idx = int(len(grid) * 0.9)
        last_means = np.nanmean(curves[:, last_idx:], axis=1)  # per-seed
        valid = last_means[~np.isnan(last_means)]
        last_label = (f"last10%={valid.mean():.1f}±{valid.std():.1f}"
                      if len(valid) > 0 else "last10%=N/A")
        ax.plot(grid, mean, color=c, linewidth=1.8,
                label=f"{label} (n={len(curves)} seeds, {last_label})")
        ax.fill_between(grid, mean - std, mean + std, color=c, alpha=0.18)

    ax.set_xlabel("Steps")
    ax.set_ylabel("True Episode Reward")
    ax.set_title(args.title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    if args.ylim:
        ymin, ymax = (float(v) for v in args.ylim.split(","))
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    out = fig_path(args.output)
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
