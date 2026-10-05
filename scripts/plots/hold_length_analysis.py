#!/usr/bin/env python3
"""Compute how long humans (vs expert) hold each action.

For every demo, scan for runs of consecutive identical actions, record run lengths.
Outputs per-group + per-action stats and a histogram plot.

Env was recorded at 30 fps (human_demo.py --fps 30), so hold_length / 30 = seconds.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk


ACTION_NAMES = {0: "noop", 1: "left", 2: "main", 3: "right"}
FPS = 30

GROUPS = {
    "Human session 1": ("/scratch/marzii/imitation_runs/human_demos/lunarlander/session_1", None),
    "Human session 2": ("/scratch/marzii/imitation_runs/human_demos/lunarlander/session_2", None),
    "Expert 4615187 (first 50)": ("/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/final.npz", 50),
}


def run_lengths(acts):
    """Return list of (action, run_length) for consecutive-same-action runs."""
    acts = np.asarray(acts)
    if len(acts) == 0:
        return []
    # find positions where action changes
    change_idx = np.flatnonzero(np.diff(acts) != 0) + 1
    boundaries = np.concatenate([[0], change_idx, [len(acts)]])
    runs = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        runs.append((int(acts[start]), end - start))
    return runs


def analyze(path, n_demos=None):
    ds = load_from_disk(path)
    if n_demos is not None:
        ds = ds.select(range(min(n_demos, len(ds))))
    per_action = defaultdict(list)
    all_runs = []
    for ep in ds:
        for act, length in run_lengths(ep["acts"]):
            per_action[act].append(length)
            all_runs.append(length)
    return all_runs, per_action


def main():
    summaries = {}
    for name, (path, n) in GROUPS.items():
        all_runs, per_action = analyze(path, n)
        summaries[name] = (all_runs, per_action)

    # Print table
    print(f"\n{'Group':<30} {'overall mean':>14} {'median':>8} {'max':>6}")
    for name, (all_runs, _) in summaries.items():
        arr = np.array(all_runs)
        print(f"{name:<30} {arr.mean():>10.2f} frames ({arr.mean()/FPS:>5.2f}s) "
              f"{int(np.median(arr)):>6d}f   {arr.max():>4d}f")

    # Per-action breakdown
    print(f"\n{'Group':<30} {'action':<8} {'mean (frames)':>14} {'mean (sec)':>10} {'count':>7}")
    for name, (_, per_action) in summaries.items():
        for act in sorted(per_action.keys()):
            lens = np.array(per_action[act])
            print(f"{name:<30} {ACTION_NAMES[act]:<8} "
                  f"{lens.mean():>10.2f}   {lens.mean()/FPS:>8.3f}s  {len(lens):>6d}")
        print()

    # Plot: histogram of hold lengths per group (log y)
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(0, 120, 2)
    for name, (all_runs, _) in summaries.items():
        ax.hist(all_runs, bins=bins, alpha=0.5, label=f"{name} (mean={np.mean(all_runs):.1f}f)",
                density=True)
    ax.set_xlabel("Hold length (env steps; 30 fps → divide by 30 for seconds)")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("How long are actions held? (run-length distribution)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "/home/marzii/IRL3/figures/hold_length_distribution.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
