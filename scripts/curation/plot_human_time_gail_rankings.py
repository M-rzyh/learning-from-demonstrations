"""Human supervision on LunarLander -- POST-TRAINING EVAL reward vs human time, for the GAIL
ranking families on the 699-demo pool.

x = human time to record the K demos used (10.8 s/demo, from 300 demos in 54 min).
y = mean of each run's 50-episode deterministic post-training eval; band = +/- 1 std over seeds.
Point labels = K. Runs are auto-discovered from Slurm logs (same machinery as the curve plots).

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_human_time_gail_rankings.py
"""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_2feat_learning_curves import discover, LABEL, GA

DEMO_SEC = 54 * 60 / 300                 # 10.8 s per demo
SERIES = ["only", "ad", "adxy", "satpad", "satad", "satad8"]


def eval_stats(jobs):
    v = [json.load(open(f))["ep_reward_mean"]
         for j in jobs
         if os.path.exists(f := f"{GA}/{j}/eval_data/agent_alignment.json")]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/marzii/IRL3/figures/ext700/human_time_gail_rankings.png")
    a = ap.parse_args()

    runs = discover()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    print(f"{'series':>8} | K -> eval (n seeds)")
    for tag, marker in zip(SERIES, "os^Dv*"):
        cells = {K: st for (t, K), jb in runs.items() if t == tag
                 and (st := eval_stats(jb.values())) is not None}
        if not cells:
            continue
        Ks = sorted(cells)
        x = np.array([K * DEMO_SEC / 60 for K in Ks])
        y = np.array([cells[K][0] for K in Ks])
        e = np.array([cells[K][1] for K in Ks])
        line, = ax.plot(x, y, marker + "-", lw=2.0, ms=7, label=LABEL.get(tag, tag), zorder=4)
        ax.fill_between(x, y - e, y + e, color=line.get_color(), alpha=0.12, lw=0, zorder=1)
        for K, xi, yi in zip(Ks, x, y):
            ax.annotate(f"{K}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=line.get_color())
        print(f"{tag:>8} | " + "  ".join(f"{K}:{cells[K][0]:+.0f}({cells[K][2]})" for K in Ks))

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.axhline(200, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel(f"Human time (min)  --  {DEMO_SEC:.1f}s per demo; point labels = K demos used")
    ax.set_ylabel("Mean eval reward (true env, 50-episode post-training eval)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Human supervision on LunarLander (699-demo pool) -- eval reward vs human time\n"
                 "GAIL on top-K of each ranking; band = +/-1 std over seeds "
                 "(10 seeds; ad K in {5,30,200,300,500}: 5 seeds)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150); print("\nsaved", a.out)


if __name__ == "__main__":
    main()
