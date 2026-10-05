"""ACF and USR for the version-2 demo arms: R2 (random draw from the human
return>200 pool) vs E4 (expert), using the SAME metric definitions as
demo_analysis_c1_vs_e4.py (imported from it):

  action_change_frequency (ACF)        changes / (T - 1)
  unique_discretized_state_ratio (USR) |{discretised states}| / T, first 6 obs
                                       dims, 10 equal-width bins per dim

Difference from the C1 analysis: the R2 subsets are NOT seed-independent. Each
seed trains on ds.shuffle(seed=SEED) of session_3_ext700_ret_gt200 truncated to
N -- exactly what run_gail_lunarlander.sh / bc/run_bc_lunarlander.sh do with
SHUFFLE=1, and identical between GAIL and BC. So every (N, seed) draw is
measured separately, and the summary reports the mean over seeds of the
per-draw mean, with the spread across seeds.

E4 is seed-independent (first N of the expert pool, SHUFFLE=0), so it is
measured once per N, as before.

Outputs:
  <out_dir>/demo_analysis_per_draw_R2_vs_E4.csv     one row per (source, N, seed)
  <out_dir>/demo_analysis_summary_R2_vs_E4.csv      mean +- sd across seeds

Usage (compute node, imitation-gail env):
    python scripts/curation/demo_analysis_r2_vs_e4.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_analysis_c1_vs_e4 import (              # noqa: E402  same metric code
    action_change_frequency, unique_discretized_state_ratio, DEMOS, EXPERT, OUT_DIR, N_BINS)

datasets.disable_progress_bar()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 10, 50, 100, 250, 400])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--pool", type=Path, default=DEMOS / "session_3_ext700_ret_gt200")
    p.add_argument("--expert_pool", type=Path, default=EXPERT)
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    p.add_argument("--bins", type=int, default=N_BINS)
    return p.parse_args()


def draw_stats(ds, bins):
    """Per-draw means of ACF, USR and episodic return."""
    acf, usr, ret = [], [], []
    for t in ds:
        acf.append(action_change_frequency(t["acts"]))
        usr.append(unique_discretized_state_ratio(t["obs"], bins)[0])
        ret.append(float(np.sum(t["rews"])))
    return (float(np.nanmean(acf)), float(np.nanmean(usr)), float(np.mean(ret)))


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    pool = load_from_disk(str(a.pool))
    print(f"R2 human pool: {a.pool.name}, {len(pool)} demos")
    for n in a.budgets:
        for seed in a.seeds:
            sub = pool.shuffle(seed=seed).select(range(min(n, len(pool))))
            acf, usr, ret = draw_stats(sub, a.bins)
            rows.append(dict(source="R2_human_rand_gt200", N=n, seed=seed, n_demos=len(sub),
                             acf=acf, usr=usr, mean_return=ret))
        print(f"  N={n:<5} {len(a.seeds)} seeds done")

    expert = load_from_disk(str(a.expert_pool))
    print(f"E4 expert pool: {len(expert)} demos (first N, seed-independent)")
    for n in a.budgets:
        acf, usr, ret = draw_stats(expert.select(range(min(n, len(expert)))), a.bins)
        rows.append(dict(source="E4_expert", N=n, seed=-1, n_demos=min(n, len(expert)),
                         acf=acf, usr=usr, mean_return=ret))

    per_draw = a.out_dir / "demo_analysis_per_draw_R2_vs_E4.csv"
    with open(per_draw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "N", "seed", "n_demos",
                                          "acf", "usr", "mean_return"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {per_draw}")

    summary = a.out_dir / "demo_analysis_summary_R2_vs_E4.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "acf_human_mean", "acf_human_sd", "acf_expert",
                    "acf_ratio_expert_over_human", "usr_human_mean", "usr_human_sd",
                    "usr_expert", "usr_ratio_human_over_expert",
                    "return_human_mean", "return_expert"])
        print(f"\n{'N':>5} {'ACF human':>16} {'ACF exp':>8} {'exp/hum':>8}   "
              f"{'USR human':>16} {'USR exp':>8} {'hum/exp':>8}")
        for n in a.budgets:
            h = [r for r in rows if r["source"].startswith("R2") and r["N"] == n]
            e = next(r for r in rows if r["source"] == "E4_expert" and r["N"] == n)
            ha = np.array([r["acf"] for r in h])
            hu = np.array([r["usr"] for r in h])
            sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0
            w.writerow([n, round(ha.mean(), 6), round(sd(ha), 6), round(e["acf"], 6),
                        round(e["acf"] / ha.mean(), 4), round(hu.mean(), 6), round(sd(hu), 6),
                        round(e["usr"], 6), round(hu.mean() / e["usr"], 4),
                        round(float(np.mean([r["mean_return"] for r in h])), 2),
                        round(e["mean_return"], 2)])
            print(f"{n:>5} {ha.mean():>8.4f} +-{sd(ha):<6.4f} {e['acf']:>8.4f} "
                  f"{e['acf'] / ha.mean():>8.2f}   {hu.mean():>8.4f} +-{sd(hu):<6.4f} "
                  f"{e['usr']:>8.4f} {hu.mean() / e['usr']:>8.2f}")
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
