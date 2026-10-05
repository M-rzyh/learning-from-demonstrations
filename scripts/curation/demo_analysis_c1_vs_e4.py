"""Two per-demonstration metrics for the C1 (human) vs E4 (expert) demo subsets.

Metrics, computed once per demonstration:

1. action_change_frequency
       changes / (T - 1),  T = number of actions in that demo,
       changes = #{t : a[t] != a[t-1]}
   LunarLander-v2 is Discrete(4); T is the demo's own length, not the action count.

2. unique_discretized_state_ratio
       |{discretised states}| / T,  T = number of observations in that demo.
   Only the first 6 observation dims are used: [x, y, vx, vy, angle, angular_vel].
   Each dim is clipped to the LunarLander-v2 bounds and split into 10 equal-width
   bins:  low  = [-1.5, -1.5, -5, -5, -pi, -5]
          high = [ 1.5,  1.5,  5,  5,  pi,  5]
   The same discretisation is applied to human and expert demos.

Subsets analysed (the exact directories the GAIL/BC runs trained on):
  C1 human   session_3_ext700_ret_only_top{10,50,100,250,400} and the full
             session_3_ext700 (N=699) -- return-ranked, nested, seed-independent
  E4 expert  the first N demos of perframe_demos_500_holdK1 for
             N = 1,10,50,100,250,400 -- SHUFFLE=0, seed-independent

Because both arms' subsets are nested and seed-independent, one pass per (source,
N) covers every seed of the corresponding GAIL/BC runs.

Outputs (per-demo rows + a summary with mean/sd per source and N):
  <out_dir>/demo_analysis_action_change_frequency_C1_vs_E4.csv
  <out_dir>/demo_analysis_unique_discretized_state_ratio_C1_vs_E4.csv
  <out_dir>/demo_analysis_summary_C1_vs_E4.csv

Nothing in the demonstration directories is modified; they are only read.

Usage (compute node, imitation-gail env):
    python scripts/curation/demo_analysis_c1_vs_e4.py
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from datasets import load_from_disk
import datasets

datasets.disable_progress_bar()

DEMOS = Path("/scratch/marzii/imitation_runs/demos/human_baseline/"
             "record_human_demos/lunarlander")
EXPERT = Path("/scratch/marzii/imitation_runs/expert/lunarlander/4615187/"
              "rollouts/perframe_demos_500_holdK1")
OUT_DIR = Path("/scratch/marzii/imitation_runs/_analysis")

LOW = np.array([-1.5, -1.5, -5.0, -5.0, -math.pi, -5.0], dtype=np.float64)
HIGH = np.array([1.5, 1.5, 5.0, 5.0, math.pi, 5.0], dtype=np.float64)
N_BINS = 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--c1_budgets", type=int, nargs="+",
                   default=[10, 50, 100, 250, 400, 699])
    p.add_argument("--e4_budgets", type=int, nargs="+",
                   default=[1, 10, 50, 100, 250, 400])
    p.add_argument("--demos_root", type=Path, default=DEMOS)
    p.add_argument("--expert_pool", type=Path, default=EXPERT)
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    p.add_argument("--bins", type=int, default=N_BINS)
    return p.parse_args()


def action_change_frequency(acts) -> float:
    """changes / (T-1); NaN for T < 2 (no transition exists)."""
    a = np.asarray(acts).reshape(len(acts), -1)
    T = len(a)
    if T < 2:
        return float("nan")
    changes = int(np.sum(np.any(a[1:] != a[:-1], axis=1)))
    return changes / (T - 1)


def unique_discretized_state_ratio(obs, n_bins: int) -> tuple:
    """(|unique discretised states| / T, T, n_unique) over the first 6 dims."""
    o = np.asarray(obs, dtype=np.float64)[:, :6]
    T = len(o)
    if T == 0:
        return float("nan"), 0, 0
    clipped = np.clip(o, LOW, HIGH)
    # equal-width bins over [low, high]; the top edge falls in the last bin
    idx = ((clipped - LOW) / (HIGH - LOW) * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    uniq = len({tuple(row) for row in idx})
    return uniq / T, T, uniq


def subset_dir(demos_root: Path, n: int) -> Path:
    if n == 699:
        return demos_root / "session_3_ext700"
    return demos_root / f"session_3_ext700_ret_only_top{n}"


def rows_for(ds, source: str, n: int, bins: int):
    out = []
    for i, t in enumerate(ds):
        acf = action_change_frequency(t["acts"])
        ratio, T_obs, n_uniq = unique_discretized_state_ratio(t["obs"], bins)
        out.append(dict(source=source, N=n, demo_index=i,
                        n_actions=len(t["acts"]), n_obs=T_obs,
                        episodic_return=round(float(np.sum(t["rews"])), 4),
                        action_change_frequency=acf,
                        n_unique_discrete_states=n_uniq,
                        unique_discretized_state_ratio=ratio))
    return out


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print("C1 (human, return-ranked ext700):")
    for n in a.c1_budgets:
        d = subset_dir(a.demos_root, n)
        if not d.exists():
            print(f"  (missing) {d}")
            continue
        ds = load_from_disk(str(d))
        r = rows_for(ds, "C1_human", n, a.bins)
        all_rows += r
        print(f"  N={n:<5} demos={len(r)}  {d.name}")

    print("E4 (expert, first N of the perframe pool):")
    if a.expert_pool.exists():
        pool = load_from_disk(str(a.expert_pool))
        for n in a.e4_budgets:
            if n > len(pool):
                print(f"  (pool has only {len(pool)}) N={n}")
                continue
            r = rows_for(pool.select(range(n)), "E4_expert", n, a.bins)
            all_rows += r
            print(f"  N={n:<5} demos={len(r)}")
    else:
        print(f"  (missing) {a.expert_pool}")

    # ---- per-metric per-demo CSVs ----
    metrics = {
        "action_change_frequency": dict(
            path=a.out_dir / "demo_analysis_action_change_frequency_C1_vs_E4.csv",
            cols=["source", "N", "demo_index", "n_actions", "episodic_return",
                  "action_change_frequency"]),
        "unique_discretized_state_ratio": dict(
            path=a.out_dir / "demo_analysis_unique_discretized_state_ratio_C1_vs_E4.csv",
            cols=["source", "N", "demo_index", "n_obs", "n_unique_discrete_states",
                  "episodic_return", "unique_discretized_state_ratio"]),
    }
    for m, spec in metrics.items():
        with open(spec["path"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=spec["cols"])
            w.writeheader()
            for row in all_rows:
                w.writerow({c: row[c] for c in spec["cols"]})
        print(f"wrote {spec['path']}")

    # ---- summary: mean +/- sd per (source, N) ----
    summary_path = a.out_dir / "demo_analysis_summary_C1_vs_E4.csv"
    keys = sorted({(r["source"], r["N"]) for r in all_rows},
                  key=lambda k: (k[0], k[1]))
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "N", "n_demos",
                    "action_change_freq_mean", "action_change_freq_sd",
                    "unique_state_ratio_mean", "unique_state_ratio_sd",
                    "mean_episodic_return"])
        print("\n%-10s %5s %6s   %-21s %-21s" %
              ("source", "N", "demos", "action_change_freq", "unique_state_ratio"))
        for src, n in keys:
            sel = [r for r in all_rows if r["source"] == src and r["N"] == n]
            acf = np.array([r["action_change_frequency"] for r in sel], dtype=float)
            usr = np.array([r["unique_discretized_state_ratio"] for r in sel], dtype=float)
            ret = np.array([r["episodic_return"] for r in sel], dtype=float)
            acf_m, acf_s = np.nanmean(acf), (np.nanstd(acf, ddof=1) if len(acf) > 1 else 0.0)
            usr_m, usr_s = np.nanmean(usr), (np.nanstd(usr, ddof=1) if len(usr) > 1 else 0.0)
            w.writerow([src, n, len(sel), round(float(acf_m), 6), round(float(acf_s), 6),
                        round(float(usr_m), 6), round(float(usr_s), 6),
                        round(float(ret.mean()), 3)])
            print("%-10s %5d %6d   %8.4f +/- %-8.4f %8.4f +/- %-8.4f" %
                  (src, n, len(sel), acf_m, acf_s, usr_m, usr_s))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
