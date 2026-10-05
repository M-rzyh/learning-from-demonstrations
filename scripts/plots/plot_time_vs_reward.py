#!/usr/bin/env python3
"""Scatter plot: time investment vs. final policy reward for PEBBLE & GAIL runs.

Usage:
    python plot_time_vs_reward.py <config.json> [--output-dir DIR]

Config JSON format — see lunarlander_runs.json for a working example.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def totalcpu_minutes(slurm_job_id):
    """Query sacct for TotalCPU of a job, return minutes. Raises if sacct unavailable."""
    out = subprocess.check_output(
        ["sacct", "-j", str(slurm_job_id), "-P", "--format=JobID,TotalCPU", "--noheader"],
        text=True,
    )
    # Take the main job line (first line; subsequent are .batch/.extern subjobs)
    for line in out.strip().splitlines():
        jid, total_cpu = line.split("|")
        if jid == str(slurm_job_id):
            return _parse_slurm_duration(total_cpu)
    raise RuntimeError(f"sacct returned no main-job line for {slurm_job_id}")


def _parse_slurm_duration(s):
    """Parse sacct TotalCPU (e.g. '26:40.696', '04:07:39', '1-06:07:36') into minutes."""
    s = s.strip()
    days = 0
    if "-" in s:
        days_str, s = s.split("-", 1)
        days = int(days_str)
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h = 0
        m, sec = parts
    else:
        h = 0
        m = 0
        sec = parts[0]
    total_sec = days * 86400 + int(h) * 3600 + int(m) * 60 + float(sec)
    return total_sec / 60.0


# ── extractors ────────────────────────────────────────────────────────────────

def _last_pct_avg(values, pct=0.10):
    n = len(values)
    tail = values[int(n * (1.0 - pct)):]
    return float(np.mean(tail))


def extract_pebble(run_cfg, timing_mode="totalcpu"):
    """Return (human_time_min, total_time_min, last10_avg_reward)."""
    # Y-axis: last 10% avg true_episode_reward from train.csv
    with open(run_cfg["train_csv"]) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rewards = [float(r["true_episode_reward"]) for r in rows]
    y = _last_pct_avg(rewards)

    # Human time: from TB scalar (stored in ms) or from human_labels_csv
    if "human_labels_csv" in run_cfg:
        with open(run_cfg["human_labels_csv"]) as f:
            label_rows = list(csv.DictReader(f))
        human_time_sec = sum(
            float(r["time_sec"]) for r in label_rows
            if r.get("label") not in (None, "None", "")
        )
        human_time_min = human_time_sec / 60.0
    else:
        ea = EventAccumulator(run_cfg["tb_human_time_dir"])
        ea.Reload()
        events = ea.Scalars("pebble/true_reward_vs_human_time")
        human_time_min = max(e.step for e in events) / 1000.0 / 60.0

    # Total time: depends on timing_mode
    # PEBBLE labeling happens during training, so total = training time
    # (don't add human_time on top — that would double count)
    if timing_mode == "totalcpu":
        total_time_min = totalcpu_minutes(run_cfg["slurm_job_id"])
    else:  # wallclock
        total_time_min = sum(float(r["duration"]) for r in rows) / 60.0

    return human_time_min, total_time_min, y


def extract_gail(run_cfg, timing_mode="totalcpu"):
    """Return (human_time_min, total_time_min, last10_avg_reward)."""
    # Y-axis: last 10% avg ep_rew_mean from TensorBoard
    ea = EventAccumulator(run_cfg["tb_dir"])
    ea.Reload()
    events = ea.Scalars("raw/gen/rollout/ep_rew_mean")
    values = [e.value for e in events]
    y = _last_pct_avg(values)

    # Human time: wall-clock time to collect first n_demos saved episodes
    # Uses all_episodes_log.csv which tracks every attempt (saved + discarded)
    with open(run_cfg["demo_log_csv"]) as f:
        reader = csv.DictReader(f)
        log_rows = list(reader)
    n_demos = run_cfg["n_demos"]
    saved_count = 0
    human_time_sec = 0.0
    for r in log_rows:
        if r["status"] == "saved":
            saved_count += 1
            if saved_count == n_demos:
                human_time_sec = float(r["cumulative_sec"])
                break
    human_time_min = human_time_sec / 60.0

    # Training time: depends on timing_mode
    if timing_mode == "totalcpu":
        train_min = totalcpu_minutes(run_cfg["slurm_job_id"])
    else:  # wallclock — stop_time - start_time from sacred run.json
        with open(run_cfg["run_json"]) as f:
            rj = json.load(f)
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        start = datetime.strptime(rj["start_time"], fmt)
        stop = datetime.strptime(rj["stop_time"], fmt)
        train_min = (stop - start).total_seconds() / 60.0

    total_time_min = human_time_min + train_min

    return human_time_min, total_time_min, y


EXTRACTORS = {"pebble": extract_pebble, "gail": extract_gail}


# ── plotting ──────────────────────────────────────────────────────────────────

def make_scatter(data, x_key, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in data:
        ax.scatter(
            d[x_key], d["reward"],
            label=d["label"],
            color=d.get("color", None),
            marker=d.get("marker", "o"),
            s=100, zorder=3,
        )
        ax.annotate(
            d["label"], (d[x_key], d["reward"]),
            textcoords="offset points", xytext=(8, 6), fontsize=8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Last-10% Avg Episode Reward")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON config file listing runs")
    parser.add_argument("--output-dir", default="/home/marzii/IRL3/figures",
                        help="Directory for output PNGs")
    parser.add_argument("--timing-mode", choices=["wallclock", "totalcpu"],
                        default="totalcpu",
                        help="wallclock: Elapsed wall-clock from run.json / train.csv duration. "
                             "totalcpu: sacct TotalCPU (actual CPU seconds).")
    parser.add_argument("--suffix", default="",
                        help="Suffix to append to output filenames (e.g. '_v2')")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.get("output_prefix", "comparison")

    data = []
    for run in cfg["runs"]:
        rtype = run["type"]
        extractor = EXTRACTORS[rtype]
        human_min, total_min, reward = extractor(run, timing_mode=args.timing_mode)
        data.append({
            "label": run["label"],
            "human_time": human_min,
            "total_time": total_min,
            "reward": reward,
            "color": run.get("color"),
            "marker": run.get("marker", "o"),
        })
        print(f"  {run['label']:30s}  human={human_min:7.2f} min  total={total_min:7.2f} min  reward={reward:8.2f}")

    timing_label = "Wall-clock" if args.timing_mode == "wallclock" else "Total CPU"
    make_scatter(data, "human_time", "Human Time (min)",
                 "Human Time vs. Final Reward",
                 out_dir / f"{prefix}_human_time{args.suffix}.png")
    make_scatter(data, "total_time", f"Total Time (min, {timing_label})",
                 f"Total Time vs. Final Reward ({timing_label})",
                 out_dir / f"{prefix}_total_time{args.suffix}.png")


if __name__ == "__main__":
    main()
