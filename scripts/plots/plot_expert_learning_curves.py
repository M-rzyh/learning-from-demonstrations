#!/usr/bin/env python3
"""Plot PPO expert training curves (steps vs ep_rew_mean) for selected expert jobs.

Usage:
    python plot_expert_learning_curves.py --job-ids 4615153,4615170,4615187,4720242 [--output PATH]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EXPERT_ROOT = "/scratch/marzii/imitation_runs/expert/lunarlander"

# Notes to annotate on specific jobs
JOB_NOTES = {
    "4615187": "(used as demo source for all non-FS experiments — highest non-FS performer)",
    "4720242": "(FRAME_SKIP=1, K=10, LunarLander-v2-FS10)",
}


def load_curve(tb_dir):
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    tag = "rollout/ep_rew_mean"
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-ids", required=True,
                        help="Comma-separated expert slurm job IDs")
    parser.add_argument("--output", default="/home/marzii/IRL3/figures/expert_curves.png")
    parser.add_argument("--title", default="Expert PPO learning curves — LunarLander")
    args = parser.parse_args()

    job_ids = args.job_ids.split(",")
    fig, ax = plt.subplots(figsize=(12, 6.5))

    print(f"Loading {len(job_ids)} expert runs:")
    for job in job_ids:
        tb_dir = os.path.join(EXPERT_ROOT, job, "log")
        steps, values = load_curve(tb_dir)
        if not steps:
            print(f"  {job}: NO TB DATA — skipping")
            continue
        last10 = float(np.mean(values[int(len(values) * 0.9):]))
        note = JOB_NOTES.get(job, "")
        fs_tag = "FS10" if job == "4720242" else "noFS"
        label = f"expert {job} [{fs_tag}] — last10%={last10:.1f}"
        if note:
            label += f"  {note}"
        print(f"  {label}: {len(steps)} points")
        ax.plot(steps, values, label=label, alpha=0.85, linewidth=1.4)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Episode Reward (ep_rew_mean)")
    ax.set_title(args.title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
