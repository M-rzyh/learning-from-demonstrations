#!/usr/bin/env python3
"""Populate the GAIL grid index CSV with results scraped from each run dir.

For each row in the index CSV, look up the run's eval_data/agent_alignment.json
and noisy-demo dir's demo_alignment.json, and write the metrics back.

Usage:
    python scrape_gail_grid_results.py /path/to/gail_grid_<date>.csv
"""

import argparse
import csv
import json
from pathlib import Path

GAIL_ROOT = Path("/scratch/marzii/imitation_runs/gail/lunarlander")


def load_json(p):
    if Path(p).exists():
        with open(p) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_csv")
    args = parser.parse_args()

    with open(args.index_csv) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    fieldnames = (list(rows[0].keys()) +
                  ["demo_alignment", "agent_alignment", "eval_reward_mean", "eval_reward_std", "last10_tb_reward"])
    # Deduplicate column order
    seen = set()
    cols = []
    for k in fieldnames:
        if k not in seen:
            cols.append(k)
            seen.add(k)

    n_done = 0
    n_partial = 0
    for r in rows:
        out = dict(r)
        job_dir = GAIL_ROOT / r["slurm_job_id"]
        eval_dir = job_dir / "eval_data"

        # demo alignment (from the noisy-demo dataset)
        demo_align = load_json(Path(r["demo_path"]) / "demo_alignment.json")
        out["demo_alignment"] = f"{demo_align['alignment_fraction']:.4f}" if demo_align else ""

        # agent alignment + eval rewards (from this run's eval_data)
        agent_align = load_json(eval_dir / "agent_alignment.json")
        if agent_align:
            out["agent_alignment"] = f"{agent_align['alignment_fraction']:.4f}"
            out["eval_reward_mean"] = f"{agent_align['ep_reward_mean']:.2f}"
            out["eval_reward_std"] = f"{agent_align['ep_reward_std']:.2f}"
            n_done += 1
        else:
            out["agent_alignment"] = ""
            out["eval_reward_mean"] = ""
            out["eval_reward_std"] = ""

        # last 10% TB reward
        tb_dir = job_dir / "log" / "raw" / "gen"
        last10 = ""
        if tb_dir.is_dir():
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                ea = EventAccumulator(str(tb_dir))
                ea.Reload()
                tag = "raw/gen/rollout/ep_rew_mean"
                if tag in ea.Tags()["scalars"]:
                    vals = [e.value for e in ea.Scalars(tag)]
                    n = len(vals)
                    if n:
                        import statistics
                        last10 = f"{statistics.mean(vals[int(n*0.9):]):.2f}"
                        if not agent_align:
                            n_partial += 1
            except Exception:
                pass
        out["last10_tb_reward"] = last10

        # Status
        if agent_align:
            out["status"] = "done"
        elif tb_dir.is_dir() and any(tb_dir.iterdir()):
            out["status"] = "training_complete_no_eval"
        else:
            out["status"] = "pending"

        out_rows.append(out)

    with open(args.index_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Updated {args.index_csv}")
    print(f"  Total: {len(rows)}, Done: {n_done}, Training-only (no eval): {n_partial}, Pending: {len(rows)-n_done-n_partial}")


if __name__ == "__main__":
    main()
