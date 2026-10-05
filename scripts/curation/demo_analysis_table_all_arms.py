"""ACF and USR for all three demonstration arms at the count-axis budgets.

Metric definitions are imported from demo_analysis_c1_vs_e4.py (unchanged):
  ACF  action_change_frequency        = changes / (T - 1)
  USR  unique_discretized_state_ratio = |{discretised states}| / T
                                        (first 6 obs dims, 10 bins per dim)

Arms:
  synthetic  first N of the expert pool                  (seed-independent)
  ranked     session_3_ext700_ret_only_top{N}, first N   (seed-independent)
  random     N drawn per seed from the 403 human demos with return > 200,
             i.e. ds.shuffle(seed).select(range(N)) -- the draw the runs use,
             so this arm is reported as mean +- SD over the 10 seeds

Writes one CSV; nothing else is touched.

Usage (compute node, imitation-gail env):
    python scripts/curation/demo_analysis_table_all_arms.py --out <path.csv>
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
    action_change_frequency, unique_discretized_state_ratio, DEMOS, EXPERT, N_BINS)

datasets.disable_progress_bar()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 10, 50, 100, 250, 400])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--bins", type=int, default=N_BINS)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def stats(ds, bins):
    acf = [action_change_frequency(t["acts"]) for t in ds]
    usr = [unique_discretized_state_ratio(t["obs"], bins)[0] for t in ds]
    ret = [float(np.sum(t["rews"])) for t in ds]
    return float(np.nanmean(acf)), float(np.nanmean(usr)), float(np.mean(ret))


def ranked_subset(n):
    """The demos the ranked arm trains on: top-N by return, nested."""
    if n <= 10:
        return load_from_disk(str(DEMOS / "session_3_ext700_ret_only_top10")).select(range(n))
    return load_from_disk(str(DEMOS / f"session_3_ext700_ret_only_top{n}"))


def main():
    a = parse_args()
    expert = load_from_disk(str(EXPERT))
    pool = load_from_disk(str(DEMOS / "session_3_ext700_ret_gt200"))
    print(f"expert pool {len(expert)} demos; human return>200 pool {len(pool)} demos")

    rows = []
    for n in a.budgets:
        e_acf, e_usr, e_ret = stats(expert.select(range(n)), a.bins)
        r_acf, r_usr, r_ret = stats(ranked_subset(n), a.bins)
        draws = [stats(pool.shuffle(seed=s).select(range(min(n, len(pool)))), a.bins)
                 for s in a.seeds]
        d = np.array(draws)                       # [seeds, (acf, usr, ret)]
        sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0
        rows.append(dict(
            N=n,
            synthetic_acf=e_acf, synthetic_usr=e_usr, synthetic_return=e_ret,
            ranked_acf=r_acf, ranked_usr=r_usr, ranked_return=r_ret,
            random_acf_mean=d[:, 0].mean(), random_acf_sd=sd(d[:, 0]),
            random_usr_mean=d[:, 1].mean(), random_usr_sd=sd(d[:, 1]),
            random_return_mean=d[:, 2].mean(), random_return_sd=sd(d[:, 2]),
            acf_ratio_synth_over_ranked=e_acf / r_acf,
            acf_ratio_synth_over_random=e_acf / d[:, 0].mean(),
            usr_ratio_ranked_over_synth=r_usr / e_usr,
            usr_ratio_random_over_synth=d[:, 1].mean() / e_usr))
        print(f"  N={n:<4} ACF synth {e_acf:.4f} | ranked {r_acf:.4f} | "
              f"random {d[:, 0].mean():.4f}+-{sd(d[:, 0]):.4f}   "
              f"USR synth {e_usr:.4f} | ranked {r_usr:.4f} | "
              f"random {d[:, 1].mean():.4f}+-{sd(d[:, 1]):.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
