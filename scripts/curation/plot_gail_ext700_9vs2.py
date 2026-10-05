"""GAIL on the 699 human-demo pool: 9-feature ranking vs 2-feature (return + action_divergence)
ranking. Same K axis, 5 seeds each. Shaded +/-1 std bands.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_gail_ext700_9vs2.py
"""
import csv, os
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GA = "/scratch/marzii/imitation_runs/gail/lunarlander"
EXP = "/home/marzii/IRL3/experiments"
OUT = "/home/marzii/IRL3/figures/gail_ext700_9feat_vs_2feat.png"
C9, C2 = "#2ca02c", "#d62728"


def epret(j):
    f = f"{GA}/{j}/eval_data/agent_rollouts.npz"
    if not os.path.exists(f): return None
    d = np.load(f, allow_pickle=True)
    return float(np.mean([float(e.sum()) for e in d["rews"]]))


def curve(idx):
    per = defaultdict(list)
    for r in csv.DictReader(open(idx)):
        v = epret(r["slurm_job_id"].strip())
        if v is not None: per[int(r["K"])].append(v)
    return {K: (np.mean(v), np.std(v), len(v)) for K, v in sorted(per.items())}


def main():
    g9 = curve(f"{EXP}/gail_session3_ext700_ksweep_2026-08-13.csv")
    g2 = curve(f"{EXP}/gail_session3_ext700_RA_ksweep_2026-08-13.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for d, c, lbl in [(g9, C9, "GAIL — 9-feature ranking"),
                      (g2, C2, "GAIL — 2-feature ranking (return + action_divergence)")]:
        Ks = np.array(sorted(d)); y = np.array([d[K][0] for K in Ks]); e = np.array([d[K][1] for K in Ks])
        ax.fill_between(Ks, y - e, y + e, color=c, alpha=0.13, lw=0, zorder=1)
        ax.plot(Ks, y, "o-", color=c, lw=2.3, ms=7, label=lbl, zorder=4)
        for K, yi in zip(Ks, y):
            ax.annotate(f"{yi:+.0f}", (K, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.5, color=c, fontweight="bold")

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xticks(sorted(g9)); ax.set_xticklabels([str(k) for k in sorted(g9)])
    ax.set_xlabel("K = number of top-ranked human demos (log scale)")
    ax.set_ylabel("Mean eval reward (true env, 5 seeds)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("GAIL on 699 human demos — ranking ablation: 9 features vs 2 (return + consistency)\n"
                 "same top-K subsets, 5 seeds each; shaded = ±1 std", fontsize=10.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150); print("saved", OUT)
    print("9-feat:", {K: round(v[0]) for K, v in g9.items()})
    print("2-feat:", {K: round(v[0]) for K, v in g2.items()})


if __name__ == "__main__":
    main()
