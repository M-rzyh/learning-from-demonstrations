"""GAIL learning curves: true-environment return vs ENVIRONMENT steps.

Three panels: E4 (expert demos) | C1 (human, return-ranked top-K) | H6 (human,
random K). One curve per demo count, mean over the seeds that exist, with a band.

WHAT THE AXES ARE (verified in code and on disk, not assumed):
  y  TRUE environment return. Each run writes monitor/mon00{0..7}.monitor.csv
     with columns r,l,t (episode return, length, wall time). Monitor is applied
     to each individual env inside util.make_vec_env (util.py:150), i.e. BELOW
     BufferingWrapper and RewardVecEnvWrapper, which wrap the VecEnv afterwards
     (algorithms/adversarial/common.py:229,236). So `r` is the environment's own
     reward, NOT the discriminator reward the generator optimizes.
  x  ENVIRONMENT steps. GAIL is online: 8 envs step in lockstep, so global env
     steps at an episode boundary = n_envs * (that env's cumulative steps).

Arms are resolved from the experiment index CSVs so the grouping is explicit:
  E4  gail_countaxis_kappa1000_2026-09-07.csv   arm=expert     (10 seeds)
  C1  ret_only top-K: same CSV (arm=human, K=400)
      + gail_ret_only_countaxis_2026-09-03.csv (K=50,100,250,699)
      + gail_2feat_rankings_top10_2026-08-28.csv (ranking=ret_only, K=10)
  H6  gail_session3_curation_2026-08-05.csv     rand{5,10,50,100,200} (5 seeds)

NOTE ON ERROR BARS: E4 and C1 use SHUFFLE=0, so their seeds vary training only
(fixed demo set). H6 uses SHUFFLE=1, so its seeds vary the demo subset AND
training. The H6 band is therefore wider for a different reason -- do not read
the three bands as measuring the same thing.

Usage:
    python scripts/plots/plot_gail_learning_curves.py
    python scripts/plots/plot_gail_learning_curves.py --bin 25000 --err sd
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("/scratch/marzii/imitation_runs/gail/lunarlander")
IDX = Path("/home/marzii/IRL3/experiments/GAIL")
OUT_DEFAULT = Path("/home/marzii/IRL3/figures/gail_learning_curves_E4_C1_H6.png")
N_ENVS = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", type=Path, default=RUNS)
    p.add_argument("--index_dir", type=Path, default=IDX)
    p.add_argument("--bin", type=int, default=8000,
                   help="env-step bin width; smaller = less smoothing")
    p.add_argument("--max_steps", type=int, default=1000000)
    p.add_argument("--err", choices=["se", "sd"], default="se")
    p.add_argument("--ppo_baseline_root", type=Path,
                   default=Path("/scratch/marzii/imitation_runs/ppo_truereward_matched/lunarlander"),
                   help="matched true-reward PPO runs; drawn as a reference in every panel")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--arms", nargs="+", default=["E4", "C1"], choices=["E4", "C1", "H6"],
                   help="one panel per arm (H6 excluded by default)")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    return p.parse_args()


def rows(path: Path):
    if not path.exists():
        print(f"  (missing index) {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def build_arms(index_dir: Path):
    """{arm: {N: [job_ids]}} from the experiment index CSVs."""
    arms = {"E4": defaultdict(list), "C1": defaultdict(list), "H6": defaultdict(list)}

    for r in rows(index_dir / "gail_countaxis_kappa1000_2026-09-07.csv"):
        job = r.get("slurm_job_id", "").strip()
        if not job:
            continue
        if r["arm"] == "expert":
            arms["E4"][int(r["N"])].append(job)
        elif r["arm"] == "human":
            arms["C1"][int(r["N"])].append(job)

    for r in rows(index_dir / "gail_ret_only_countaxis_2026-09-03.csv"):
        if r.get("condition") == "ret_only" and r.get("slurm_job_id", "").strip():
            arms["C1"][int(r["K"])].append(r["slurm_job_id"].strip())

    for r in rows(index_dir / "gail_2feat_rankings_top10_2026-08-28.csv"):
        if r.get("ranking") == "ret_only" and r.get("slurm_job_id", "").strip():
            arms["C1"][int(r["K"])].append(r["slurm_job_id"].strip())

    for r in rows(index_dir / "gail_session3_curation_2026-08-05.csv"):
        cond = r.get("condition", "")
        if cond.startswith("rand") and r.get("slurm_job_id", "").strip():
            arms["H6"][int(cond[4:])].append(r["slurm_job_id"].strip())

    return arms


def run_curve(run_dir: Path, bin_w: int, max_steps: int):
    """Binned (steps, mean true return) for one run, from its monitor files."""
    mons = sorted((run_dir / "monitor").glob("mon*.monitor.csv"))
    if not mons:
        return None
    n_bins = max_steps // bin_w
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for m in mons:
        try:
            data = np.genfromtxt(m, delimiter=",", skip_header=2, usecols=(0, 1))
        except (ValueError, OSError):
            continue
        if data.size == 0:
            continue
        data = np.atleast_2d(data)
        r, l = data[:, 0], data[:, 1]
        # global env steps at each episode end: n_envs * this env's cumulative steps
        gsteps = N_ENVS * np.cumsum(l)
        idx = np.clip((gsteps // bin_w).astype(int), 0, n_bins - 1)
        np.add.at(sums, idx, r)
        np.add.at(counts, idx, 1)
    with np.errstate(invalid="ignore"):
        vals = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    # forward-fill empty bins so the curve stays continuous
    for i in range(1, n_bins):
        if np.isnan(vals[i]):
            vals[i] = vals[i - 1]
    x = (np.arange(n_bins) + 1) * bin_w
    return x, vals


def main():
    a = parse_args()
    arms = build_arms(a.index_dir)

    titles = {"E4": "E4: expert demos",
              "C1": "C1: human demos, return-ranked top-K",
              "H6": "H6: human demos, random K"}
    # standard categorical palette (matplotlib tab10)
    palette = plt.get_cmap("tab10").colors

    # matched true-reward PPO baseline: same PPO config and env-step budget as the
    # GAIL generator, but optimizing the TRUE reward instead of the discriminator.
    base = None
    if not a.no_baseline and a.ppo_baseline_root.exists():
        curves = []
        for run in sorted(a.ppo_baseline_root.glob("*")):
            c = run_curve(run, a.bin, a.max_steps)
            if c is not None and not np.all(np.isnan(c[1])):
                curves.append(c[1])
        if curves:
            base = np.vstack(curves)
            print("PPO true-reward baseline: %d runs, final %.1f"
                  % (base.shape[0], np.nanmean(base, axis=0)[-1]))
    fig, axes = plt.subplots(1, len(a.arms), figsize=(5.2 * len(a.arms), 4.4),
                             sharey=True, sharex=True, squeeze=False)
    axes = axes[0]

    for ax, arm in zip(axes, a.arms):
        budgets = sorted(arms[arm])
        print(f"--- {arm}")
        for i, n in enumerate(budgets):
            curves = []
            for job in arms[arm][n]:
                c = run_curve(a.runs_root / job, a.bin, a.max_steps)
                if c is not None and not np.all(np.isnan(c[1])):
                    curves.append(c[1])
            if not curves:
                print(f"  (no monitor data) N={n}")
                continue
            x = (np.arange(a.max_steps // a.bin) + 1) * a.bin
            v = np.vstack(curves)
            m = np.nanmean(v, axis=0)
            sd = np.nanstd(v, axis=0, ddof=1) if v.shape[0] > 1 else np.zeros_like(m)
            e = sd / np.sqrt(v.shape[0]) if a.err == "se" else sd
            col = palette[i % len(palette)]
            ax.plot(x, m, color=col, lw=1.7, label=f"N={n} (n={v.shape[0]})")
            ax.fill_between(x, m - e, m + e, color=col, alpha=0.18, lw=0)
            print(f"  N={n:<5} seeds={v.shape[0]:<3} final={m[-1]:7.1f}")

        if base is not None:
            x = (np.arange(a.max_steps // a.bin) + 1) * a.bin
            bm = np.nanmean(base, axis=0)
            ax.plot(x, bm, color="0.35", ls="--", lw=1.6,
                    label=f"true-reward PPO (n={base.shape[0]})")

        ax.set_title(titles[arm], fontsize=10.5)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.legend(fontsize=7.5, loc="lower right", ncol=2)

    axes[0].set_ylabel("true environment return")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
