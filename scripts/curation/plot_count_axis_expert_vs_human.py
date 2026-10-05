"""ONE count-axis figure combining the expert/oracle and human supervision results:

  GAIL expert demos      (30 seeds)   x = N * (expert mean ep-length * measured human s/step)
                                       = N * 266.3 steps * 0.05008 s/step = N * 13.34 s/demo,
                                       i.e. the time a HUMAN would need to fly the expert's
                                       trajectories at the measured 20 steps/s play rate
  PT   scripted labels   (30 seeds)   x = N *  9.074 s/pair   (pool-mean measured rate)
  GAIL human demos       (10 seeds)   x = measured duration sum of the selected top-K
  PT   human preferences (30 seeds)   x = measured per-pair time sum of the shown pairs

Same runs as figures/grid_count_pt_vs_gail.png (expert half) and
figures/count_axis_human_pt_vs_gail_retonly.png (human half); everything priced in the
measured human-time currency so the four curves share one x-axis.
Solid = human supervision, dashed = expert/oracle. Green = GAIL, blue = PT.

    python scripts/curation/plot_count_axis_expert_vs_human.py
"""
import csv, glob, json, os, pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_count_axis_human_pt_vs_gail_retonly as H

GAIL_CSV = "/home/marzii/IRL3/experiments/GAIL/gail_grid_count_2026-06-18.csv"
MS = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
PAIR_SEC = 9.074
SEC_PER_STEP = 0.05008          # 133.2 min of human play / 159,546 env steps (20 steps/s)
EXPERT_POOL = "/scratch/marzii/imitation_runs/demos/noisy_demos/lunarlander/expert_4615187/n100_p0_clean"
GAIL_C, PT_C = "#2ca02c", "#1f77b4"


def series(ax, pts, fmt, color, ls, label, lw=1.8):
    print(f"{label}:")
    for x, y, e, n in pts:
        print(f"   N={n:>4}  {y:+7.1f} ±{e:5.1f}   {x:7.2f} min")
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; es = [p[2] for p in pts]
    ax.errorbar(xs, ys, yerr=es, fmt=fmt, color=color, ls=ls, ms=7, capsize=4,
                lw=lw, label=label)
    for x, y, _, n in pts:
        ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(4, 6),
                    fontsize=7, color=color)


def main():
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # ---- expert half (same runs as grid_count_pt_vs_gail.png) ----
    per = {}
    for r in csv.DictReader(open(GAIL_CSV)):
        f = f"{H.GA}/{r['slurm_job_id']}/eval_data/agent_rollouts.npz"
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            per.setdefault(int(r["N"]), []).append(np.mean([float(e.sum()) for e in d["rews"]]))
    from datasets import load_from_disk, disable_progress_bar; disable_progress_bar()
    ex = load_from_disk(EXPERT_POOL)
    exp_sec = np.mean([len(ex[i]["acts"]) for i in range(len(ex))]) * SEC_PER_STEP
    print(f"expert demo priced at {exp_sec:.2f} s (mean length x human 20 steps/s rate)")
    ge = [(N * exp_sec / 60, np.mean(v), np.std(v), N)
          for N, v in sorted(per.items()) if N != 5]
    series(ax, ge, "s", GAIL_C, "--", "GAIL, expert demos (30 seeds)")

    pe = []
    for N in [50, 100, 250, 500, 750, 1000]:
        fs = glob.glob(f"{MS}/lunarlander-grid-ms-count-N{N}/seed_*/eval_summary.json")
        v = np.array([json.load(open(f))["last10_eval_reward"] for f in fs])
        pe.append((N * PAIR_SEC / 60, v.mean(), v.std(), N))
    series(ax, pe, "o", PT_C, "--", "PT, scripted labels (30 seeds)")

    # ---- human half (same runs/times as count_axis_human_pt_vs_gail_retonly.png) ----
    dur = H.demo_durations()
    rank = sorted(csv.DictReader(open(H.RANK)), key=lambda r: int(r["rank"]))
    order = [int(r["demo_id"]) for r in rank]
    runs = H.gail_runs_by_tag()
    gh = []
    for K in [10, 50, 100, 250, 699]:
        tag = "session_3_ext700" if K == 699 else f"session_3_ext700_ret_only_top{K}"
        v = H.evals(runs.get((tag, K), {}).values())
        if len(v):
            gh.append((dur[order[:K]].sum() / 60, v.mean(), v.std(), K))
    series(ax, gh, "s", GAIL_C, "-", "GAIL, human demos, return top-K (10 seeds)", lw=2.2)

    w = pickle.load(open(H.POOL, "rb"))
    ptime = {(int(a), int(b)): float(t) for a, b, t in
             zip(w["pair_starts_a"], w["pair_starts_b"], w["time_sec"])}
    ph = []
    for N in [10, 50, 100, 750, 1000]:
        vals, secs = [], []
        for s in range(30):
            p = f"{H.PT_RUNS}/lunarlander-human-count-v2-N{N}/seed_{s}/eval_summary.json"
            if not os.path.exists(p): continue
            vals.append(float(json.load(open(p))["last10_eval_reward"]))
            d = f"{H.LBL}/lunarlander-human-count-v2-N{N}-s{s}"
            A = pickle.load(open(f"{d}/indices_num{N}_q100", "rb"))
            B = pickle.load(open(f"{d}/indices_2_num{N}_q100", "rb"))
            secs.append(sum(ptime[(int(x), int(y))] for x, y in zip(A, B)))
        if vals:
            v = np.array(vals)
            ph.append((np.mean(secs) / 60, v.mean(), v.std(), N))
    series(ax, ph, "o", PT_C, "-", "PT, human preferences (30 seeds)", lw=2.2)

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("Human time (min)  --  measured rates/durations; point labels = N demos or pairs")
    ax.set_ylabel("Mean eval reward (mean over seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Count axis, one currency -- expert/oracle (dashed) vs human (solid) supervision",
                 fontsize=12)
    fig.tight_layout()
    out = "/home/marzii/IRL3/figures/count_axis_expert_vs_human.png"
    fig.savefig(out, dpi=150); print("saved", out)


if __name__ == "__main__":
    main()
