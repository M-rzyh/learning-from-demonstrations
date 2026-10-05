#!/usr/bin/env python3
"""Plot mean±std GAIL learning curves across multiple seeds, one band per demo source.

Usage:
    python plot_gail_multi_seed.py \\
        --group "Humans sess2:4826549,4826550,4826551,4826552,4826553" \\
        --group "FS-expert perframe:4826554,4826555,4826556,4826557,4826558" \\
        --output curves.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_dir import fig_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


GAIL_ROOT = "/scratch/marzii/imitation_runs/gail/lunarlander"


def load_curve(job_id):
    ea = EventAccumulator(f"{GAIL_ROOT}/{job_id}/log/raw/gen")
    ea.Reload()
    events = ea.Scalars("raw/gen/rollout/ep_rew_mean")
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", action="append", required=True,
                   help="Group spec 'Label:JOBID,JOBID,...' (repeatable)")
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="GAIL learning curves — mean ± std across seeds")
    p.add_argument("--ylim", default=None, help="'min,max'")
    p.add_argument("--colors", default=None,
                   help="Comma-separated matplotlib colors, one per --group (e.g. 'blue,green')")
    args = p.parse_args()

    groups = []
    for g in args.group:
        label, jobs_str = g.split(":", 1)
        jobs = jobs_str.split(",")
        curves = []
        ref_steps = None
        for j in jobs:
            try:
                steps, values = load_curve(j)
            except Exception as e:
                print(f"  {j}: skip ({e})")
                continue
            if ref_steps is None:
                ref_steps = steps
            elif len(steps) != len(ref_steps) or steps[0] != ref_steps[0]:
                # Truncate to shorter length
                m = min(len(steps), len(ref_steps))
                ref_steps = steps[:m]
                values = values[:m]
            curves.append(values)
        if not curves:
            print(f"Group '{label}': no data")
            continue
        # Stack: align lengths
        m = min(len(c) for c in curves)
        curves = np.stack([c[:m] for c in curves])
        ref_steps = ref_steps[:m]
        groups.append((label, ref_steps, curves, jobs[:len(curves)]))
        last10 = curves[:, int(m * 0.9):].mean(axis=1)
        print(f"  {label}: {len(curves)} seeds, last-10% per seed = {last10.round(1).tolist()}, "
              f"mean across seeds = {last10.mean():.1f} ± {last10.std():.1f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    if args.colors:
        colors = [c.strip() for c in args.colors.split(",")]
    else:
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, (label, steps, curves, jobs) in enumerate(groups):
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        c = colors[i % len(colors)]
        last10_m = curves[:, int(len(steps) * 0.9):].mean()
        last10_s = curves[:, int(len(steps) * 0.9):].mean(axis=1).std()
        ax.plot(steps, mean, color=c, linewidth=1.8,
                label=f"{label} (n={len(curves)} seeds, last10%={last10_m:.1f}±{last10_s:.1f})")
        ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.18)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Episode Reward (ep_rew_mean)")
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
