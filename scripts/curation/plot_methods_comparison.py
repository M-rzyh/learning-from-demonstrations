"""Method comparison on LunarLander, one learning-curve panel:

  PPO on TRUE reward, no demos      (the expert's own training run 4615187, 1 seed)
  BC warm-start -> PPO true reward  (9-feature top-10 human demos, 5 seeds; curve parsed
                                     from the PPO verbose tables in the slurm logs)
  GAIL, 15 EXPERT demos (cap1000)   (10 seeds -- no N=10 expert run exists at these settings)
  GAIL, 10 human demos, return-only top-10   (10 seeds)
  GAIL, 10 human demos, satisficing+ad(1/8) top-10   (10 seeds)

All x-axes are environment steps 0..1M. GAIL TB logs index by round -> scaled by 1e6/488.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_methods_comparison.py
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_2feat_learning_curves import discover, curve, smooth, GA, TOTAL_TIMESTEPS, ROUNDS_FULL

BC_IDX = "/home/marzii/IRL3/experiments/GAIL/../BC/bc_warmstart_ext700_2026-08-13.csv"
BC_LOGS = "/scratch/marzii/imitation_runs/_slurm_logs/bc_warmstart/lunarlander"
BW = "/scratch/marzii/imitation_runs/bc_warmstart/lunarlander"
EXP15_IDX = "/home/marzii/IRL3/experiments/GAIL/gail_expert15_cap1000_2026-08-03.csv"
PPO_DIR = "/scratch/marzii/imitation_runs/ppo_baseline/lunarlander"   # 10-seed PPO-from-scratch batch (2026-09-02)


def bc_curve(job):
    """(steps, ep_rew_mean) parsed from the SB3 verbose table in the slurm log."""
    f = glob.glob(f"{BC_LOGS}/*_{job}.out")
    if not f:
        return None
    xs, ys, last_rew = [], [], None
    for line in open(f[0], errors="ignore"):
        m = re.search(r"ep_rew_mean\s*\|\s*(-?[\d.e+]+)", line)
        if m: last_rew = float(m.group(1)); continue
        m = re.search(r"total_timesteps\s*\|\s*(\d+)", line)
        if m and last_rew is not None:
            xs.append(int(m.group(1))); ys.append(last_rew); last_rew = None
    return (np.array(xs), np.array(ys)) if xs else None


def gail_series(jobs, window):
    ys = [y for y in (curve(j) for j in jobs) if y is not None]
    n = min(len(y) for y in ys)
    Y = np.vstack([smooth(y[:n], window) for y in ys])
    x = np.arange(Y.shape[1]) * (TOTAL_TIMESTEPS / ROUNDS_FULL)
    return x, Y.mean(0), Y.std(0), len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=3, help="rolling window (low smoothing)")
    ap.add_argument("--out", default="/home/marzii/IRL3/figures/ext700/methods_comparison_curves.png")
    a = ap.parse_args()
    runs = discover(topk=10)
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # --- PPO true reward, no demos (10-seed baseline batch) ---
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ppo = []
    for d in sorted(glob.glob(f"{PPO_DIR}/*/log/events*")):
        ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
        if "rollout/ep_rew_mean" not in ea.Tags()["scalars"]:
            continue
        pts = [(q.step, q.value) for q in ea.Scalars("rollout/ep_rew_mean") if q.step <= TOTAL_TIMESTEPS]
        ppo.append(pts)
    n = min(len(p_) for p_ in ppo)
    X = np.array([q[0] for q in ppo[0][:n]], float)
    Y = np.vstack([smooth(np.array([q[1] for q in p_[:n]]), a.window) for p_ in ppo])
    Xs = X[:Y.shape[1]]
    ax.plot(Xs, Y.mean(0), lw=2.2, color="black", ls="--",
            label=f"PPO ({len(ppo)} seeds)", zorder=5)
    ax.fill_between(Xs, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color="black", alpha=0.10, lw=0)

    # --- BC warm-start (5 seeds, 9-feature top-10) ---
    jobs = [r["jobid"].strip() for r in csv.DictReader(open(BC_IDX)) if r["K"] == "10"]
    series = [c for j in sorted(set(jobs)) if (c := bc_curve(j)) is not None]
    n = min(len(y) for _, y in series)
    X = series[0][0][:n]
    Y = np.vstack([smooth(y[:n], a.window) for _, y in series])
    Xs = X[:Y.shape[1]]
    line, = ax.plot(Xs, Y.mean(0), lw=2.0, label=f"BC warmstart ({len(series)} seeds)", zorder=4)
    ax.fill_between(Xs, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0),
                    color=line.get_color(), alpha=0.13, lw=0)

    # --- the three GAIL series ---
    exp_jobs = [r["slurm_job_id"].strip() for r in csv.DictReader(open(EXP15_IDX))]
    for jobs, lbl in [
        (exp_jobs, "GAIL expert demos"),
        (sorted(runs["only"].values()), "GAIL human demos (return top10)"),
        (sorted(runs["satad8"].values()), "GAIL human (satisficing+AD top10)"),
    ]:
        x, m_, sd, ns = gail_series(jobs, a.window)
        line, = ax.plot(x, m_, lw=2.0, label=f"{lbl} ({ns} seeds)", zorder=3)
        ax.fill_between(x, m_ - sd, m_ + sd, color=line.get_color(), alpha=0.13, lw=0)

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("Environment steps"); ax.set_xlim(0, TOTAL_TIMESTEPS)
    ax.set_ylabel("Mean episode return")
    ax.grid(True, alpha=0.25); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("BC, PPO, GAIL learning curves", fontsize=12)
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150); print("saved", a.out)


if __name__ == "__main__":
    main()
