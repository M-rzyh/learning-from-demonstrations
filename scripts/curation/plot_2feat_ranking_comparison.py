"""GAIL on the top-10 demos of seven 2-feature rankings (return + one consistency metric),
10 seeds each. Shows per-seed spread, not just the mean.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_2feat_ranking_comparison.py
"""
import argparse, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GA = "/scratch/marzii/imitation_runs/gail/lunarlander"
ORDER = ["ret_only", "ret_sad", "ret_sae", "ret_ae", "ret_cdad", "ret_cdae", "ret_ad"]
LABEL = {"ret_only": "return only\n(baseline)", "ret_ad": "+ action\ndivergence",
         "ret_ae": "+ action\nentropy", "ret_sad": "+ self action\ndivergence",
         "ret_sae": "+ self action\nentropy", "ret_cdad": "+ cross-demo\ndivergence",
         "ret_cdae": "+ cross-demo\nentropy"}
# grey = baseline, orange = self (own demo), green = cross-demo (other demos)
COLOR = {"ret_only": "#6e6e6e", "ret_sad": "#e6820e", "ret_sae": "#e6820e",
         "ret_ae": "#2ca02c", "ret_cdad": "#2ca02c", "ret_cdae": "#2ca02c", "ret_ad": "#2ca02c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/home/marzii/IRL3/experiments/GAIL/"
                                       "gail_2feat_rankings_top10_2026-08-28.csv")
    ap.add_argument("--out", default="/home/marzii/IRL3/figures/ext700/gail_2feat_ranking_top10.png")
    a = ap.parse_args()

    per = {}
    for r in csv.DictReader(open(a.index)):
        f = f"{GA}/{r['slurm_job_id'].strip()}/eval_data/agent_rollouts.npz"
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            per.setdefault(r["ranking"], []).append(
                float(np.mean([float(e.sum()) for e in d["rews"]])))

    fig, ax = plt.subplots(figsize=(11, 6))
    base = np.mean(per["ret_only"])
    for i, k in enumerate(ORDER):
        v = np.array(per[k]); c = COLOR[k]
        ax.bar(i, v.mean(), 0.62, color=c, alpha=0.30, zorder=2)
        ax.errorbar(i, v.mean(), yerr=v.std(), color=c, lw=2.2, capsize=6, zorder=4)
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.13, .13, len(v)),
                   v, color=c, s=26, alpha=0.8, zorder=5, edgecolor="white", linewidth=0.5)
        ax.annotate(f"{v.mean():+.0f}", (i, v.mean()), xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=c, alpha=0.92), zorder=6)
        if k != "ret_only":
            ax.annotate(f"{v.mean()-base:+.0f}", (i, ax.get_ylim()[0]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8.5, color=c)

    ax.axhline(base, color="#6e6e6e", ls="--", lw=1.4, zorder=1)
    ax.annotate("baseline", (len(ORDER) - 0.4, base), xytext=(4, 4), textcoords="offset points",
                fontsize=8.5, color="#6e6e6e")
    ax.axhline(0, color="gray", ls=":", alpha=0.4, zorder=1)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=9)
    ax.set_ylabel("Mean eval reward (true env), 50 episodes")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("GAIL on the top-10 demos of each 2-feature ranking (10 seeds, dots = individual seeds)\n"
                 "orange = within-person consistency   ·   green = across-person consistency",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150); print("saved", a.out)


if __name__ == "__main__":
    main()
