"""Dump a GAIL training-progress curve at ONE demonstration budget as JSON.

New script; plot_gail_learning_curves.py is imported, not modified.

x = environment steps, y = true environment return read from each run's
monitor/mon00*.monitor.csv (the env's own reward, not the discriminator's).
Monitor rows are per episode and asynchronous across the 8 envs, so they are
aggregated into fixed env-step bins (--bin). That binning is the only
aggregation; --smooth adds a light centred rolling mean per seed on top.

Per bin we report the mean over seeds and the half-width of a 95% CI
(t(0.975, n-1) * SD/sqrt(n)), matching the count-axis figures.

Arms:
  E4  expert demos, first N            (synthetic)
  C1  human demos, return-ranked top-N (ranked)
  R2  human demos, random N per seed from the 403 with return > 200

Usage (repo root, imitation-gail env):
    python scripts/plots/dump_gail_progress_curve.py --human_arm R2 --N 50 \
        --out /path/gail_r2_n50.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_gail_learning_curves import (          # noqa: E402  shared loaders
    RUNS, IDX, build_arms, rows, run_curve)

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 19: 2.093, 29: 2.045}


def roll(v, k):
    """Centred rolling mean over k points, truncating at the edges (no padding).

    Applied per seed BEFORE averaging, so the band reflects smoothed seeds.
    """
    if k <= 1:
        return v
    h, n = k // 2, v.shape[-1]
    out = np.empty_like(v, dtype=float)
    for i in range(n):
        out[..., i] = np.nanmean(v[..., max(0, i - h):min(n, i + h + 1)], axis=-1)
    return out


def ci95(v):
    n = v.shape[0]
    sd = np.nanstd(v, axis=0, ddof=1) if n > 1 else np.zeros(v.shape[1])
    return np.nanmean(v, axis=0), sd / np.sqrt(n) * T95.get(n - 1, 1.96)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human_arm", choices=["C1", "R2"], default="R2")
    p.add_argument("--synth_arm", choices=["E4", "R3"], default="E4",
                   help="E4: fixed first-N expert demos; R3: N redrawn per seed")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--bin", type=int, default=10000, help="env-step bin width")
    p.add_argument("--max_steps", type=int, default=1000000)
    p.add_argument("--runs_root", type=Path, default=RUNS)
    p.add_argument("--index_dir", type=Path, default=IDX)
    p.add_argument("--ppo_baseline_root", type=Path,
                   default=Path("/scratch/marzii/imitation_runs/ppo_truereward_matched/lunarlander"))
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--smooth", type=int, default=5,
                   help="centred rolling mean over this many points, per seed "
                        "(1 = none)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def stack(jobs, root, bin_w, max_steps):
    xs, curves = None, []
    for job in jobs:
        c = run_curve(root / job, bin_w, max_steps)
        if c is not None and not np.all(np.isnan(c[1])):
            xs = c[0]
            curves.append(c[1])
    return (xs, np.vstack(curves)) if curves else (None, None)


def main():
    a = parse_args()
    arms = build_arms(a.index_dir)
    # the random-pool arm postdates plot_gail_learning_curves.py, so read its index here
    for pattern, arm_name, key in (("gail_rand_gt200_*.csv", "rand_gt200", "R2"),
                                   ("gail_rand_expert_*.csv", "rand_expert", "R3")):
        arms[key] = {}
        for f in sorted(a.index_dir.glob(pattern)):
            for r in rows(f):
                job = r.get("slurm_job_id", "").strip()
                if r.get("arm") == arm_name and job:
                    arms[key].setdefault(int(r["N"]), []).append(job)

    label = {"C1": "human (ranked)", "R2": "human"}[a.human_arm]
    title = {"C1": "GAIL (LfD), ranked, N = %d", "R2": "GAIL (LfD), N = %d"}[a.human_arm] % a.N
    out = {"panel": title, "xlabel": "environment steps", "ylabel": "return",
           "bin": a.bin, "series": [], "baseline": None}

    for role, arm in (("synthetic", a.synth_arm), (label, a.human_arm)):
        jobs = arms.get(arm, {}).get(a.N, [])
        xs, v = stack(jobs, a.runs_root, a.bin, a.max_steps)
        if xs is None:
            print(f"  (no runs) {arm} N={a.N}")
            continue
        m, e = ci95(roll(v, a.smooth))
        out["series"].append(dict(label=role, x=xs.tolist(), mean=m.tolist(),
                                  err=e.tolist(), n_seeds=int(v.shape[0])))
        print(f"  {arm} {role:<16} seeds={v.shape[0]} final={m[-1]:.1f}")

    if not a.no_baseline and a.ppo_baseline_root.exists():
        xs, v = stack([p.name for p in sorted(a.ppo_baseline_root.glob("*"))],
                      a.ppo_baseline_root, a.bin, a.max_steps)
        if xs is not None:
            m, e = ci95(roll(v, a.smooth))
            out["baseline"] = dict(label="PPO, true reward", x=xs.tolist(),
                                   mean=m.tolist(), err=e.tolist(),
                                   n_seeds=int(v.shape[0]))
            print(f"  baseline seeds={v.shape[0]} final={m[-1]:.1f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
