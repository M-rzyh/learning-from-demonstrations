"""Why curation shows an inverted U: the realizability ceiling vs the actual GAIL return.

Left  : irreducible BC error of the top-K subset (fixed 30 demos sampled from each, so only
        WHICH demos changes) -- falls monotonically as K shrinks, but never reaches the expert.
Right : the GAIL result on the same subsets, which peaks and then falls.

    python scripts/curation/plot_curation_ceiling.py
"""
import csv, json, os
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GA = "/scratch/marzii/imitation_runs/gail/lunarlander"
EXP = "/home/marzii/IRL3/experiments"
JS = "/home/marzii/IRL3/figures/ext700/curation_bc_ceiling.json"
OUT = "/home/marzii/IRL3/figures/ext700/curation_realizability_ceiling.png"
CH, CE, CG = "#c2410c", "#1d4ed8", "#2ca02c"


def gail_curve(idx):
    per = defaultdict(list)
    for r in csv.DictReader(open(idx)):
        f = f"{GA}/{r['slurm_job_id'].strip()}/eval_data/agent_rollouts.npz"
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            per[int(r["K"])].append(float(np.mean([float(e.sum()) for e in d["rews"]])))
    return {K: (float(np.mean(v)), float(np.std(v)), len(v)) for K, v in sorted(per.items())}


def main():
    res = json.load(open(JS))
    Ks = sorted(int(k) for k in res["by_K"])
    err = np.array([res["by_K"][str(K)]["err"]["mean"] for K in Ks])
    sd = np.array([res["by_K"][str(K)]["err"]["std"] for K in Ks])
    ret = np.array([res["by_K"][str(K)]["demo_return"] for K in Ks])
    exp = res["expert_reference"]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax[0].plot(Ks, err, "o-", color=CH, lw=2.2, ms=7,
               label=f"human top-K ({res['n_take']} demos sampled from each)")
    ax[0].fill_between(Ks, err - sd, err + sd, color=CH, alpha=.18, lw=0)
    ax[0].axhline(exp["mean"], color=CE, ls="--", lw=1.8,
                  label=f"expert, same {res['n_take']} demos ({exp['mean']:.3f})")
    for K, y, r in zip(Ks, err, ret):
        ax[0].annotate(f"{y:.3f}", (K, y), textcoords="offset points", xytext=(0, 9),
                       ha="center", fontsize=7.5, color=CH, fontweight="bold")
    closed = (err[-1] - err[0]) / (err[-1] - exp["mean"])
    ax[0].annotate("", xy=(Ks[0], err[0]), xytext=(Ks[0], err[-1]),
                   arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.2))
    ax[0].text(Ks[0] * 1.15, (err[0] + err[-1]) / 2,
               f"curation closes only\n{100*closed:.0f}% of the gap", fontsize=8.5, color="0.3")
    ax[0].set_xscale("log"); ax[0].set_xticks(Ks); ax[0].set_xticklabels(map(str, Ks))
    ax[0].set_xlabel("K = size of the top-K curated pool (log)")
    ax[0].set_ylabel("irreducible BC error (k=15)")
    ax[0].set_title("Curation lowers the realizability ceiling…\n…but never to the expert's level")
    ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=.3)

    g = gail_curve(f"{EXP}/gail_session3_ext700_ksweep_2026-08-13.csv")
    gK = sorted(g); gy = np.array([g[K][0] for K in gK]); ge = np.array([g[K][1] for K in gK])
    ax[1].plot(gK, gy, "o-", color=CG, lw=2.2, ms=7, label="GAIL, 9-feature ranking")
    ax[1].fill_between(gK, gy - ge, gy + ge, color=CG, alpha=.15, lw=0)
    for K, y in zip(gK, gy):
        ax[1].annotate(f"{y:+.0f}", (K, y), textcoords="offset points", xytext=(0, 9),
                       ha="center", fontsize=7.5, color=CG, fontweight="bold")
    ax[1].plot(Ks, ret, "s--", color=CH, lw=1.5, ms=5, alpha=.75, label="mean demo return")
    ax[1].axhline(0, color="gray", ls=":", alpha=.5)
    ax[1].set_xscale("log"); ax[1].set_xticks(gK); ax[1].set_xticklabels(map(str, gK))
    ax[1].set_xlabel("K (log)"); ax[1].set_ylabel("eval return")
    ax[1].set_title("The GAIL inverted-U: shrinking K buys realizability\nbut costs data")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    fig.suptitle("Why more human demos hurt: the realizability ceiling rises monotonically with K, "
                 "so curation trades it against data volume", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT, dpi=150); plt.close(fig)
    print("wrote", OUT)
    print("GAIL curve:", {K: round(g[K][0], 1) for K in gK})


if __name__ == "__main__":
    main()
