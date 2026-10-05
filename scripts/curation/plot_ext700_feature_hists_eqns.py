"""Histograms of the 9 RANKING features on the 699-demo pool (session_3_ext700), each subplot
labelled with the equation used to compute it and the MEAN marked (threshold reference).

Reads the precomputed features CSV (no recompute).
    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_ext700_feature_hists_eqns.py
"""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CSV = "/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander/session_3_ext700_features.csv"
OUT = "/home/marzii/IRL3/figures/ext700/ext700_9feature_hists_eqns.png"

# (feature, "higher"/"lower" better, equation as mathtext)  — obs = [x, y, v_x, v_y, θ, ω, leg_L, leg_R]
FEATS = [
    ("return",            "higher", r"$R=\sum_t r_t$"),
    ("descent_rate",      "lower",  r"$\langle\max(-v_y,\,0)\rangle_t$"),
    ("land_x_offset",     "lower",  r"$|x_T|$"),
    ("touchdown_vy",      "lower",  r"$|v_{y,T}|$"),
    ("touchdown_vx",      "lower",  r"$|v_{x,T}|$"),
    ("angvel_mean_abs",   "lower",  r"$\langle|\omega|\rangle_t$"),
    ("vx_mean_abs",       "lower",  r"$\langle|v_x|\rangle_t$"),
    ("angle_mean_abs",    "lower",  r"$\langle|\theta|\rangle_t$"),
    ("action_divergence", "lower",  r"$\langle\frac{1}{k}\sum_{j\in kNN'}\mathbf{1}[a_j\neq a_i]\rangle_t$"),
]


def main():
    rows = list(csv.DictReader(open(CSV)))
    outc = np.array([r["outcome"] for r in rows])
    n, nl = len(rows), int((outc == "landed").sum())

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.ravel()
    for ax, (f, better, eqn) in zip(axes, FEATS):
        v = np.array([float(r[f]) for r in rows], float)
        bins = np.histogram_bin_edges(v, bins=30)
        ax.hist(v[outc == "landed"], bins=bins, color="#2ca02c", alpha=.65, label="landed")
        ax.hist(v[outc != "landed"], bins=bins, color="#d62728", alpha=.55, label="crashed")
        m = float(v.mean())
        ax.axvline(m, color="k", ls="-", lw=1.8)
        ax.annotate(f"mean={m:.3f}", (m, ax.get_ylim()[1]), xytext=(4, -4),
                    textcoords="offset points", fontsize=9, fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_title(f"{f}   ({better} better)", fontsize=11, fontweight="bold")
        # equation, top-left inside the axes
        ax.text(0.03, 0.94, eqn, transform=ax.transAxes, fontsize=13, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff7e6", ec="#e6820e", alpha=0.9))
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("feature value", fontsize=8); ax.set_ylabel("count (# demos)", fontsize=8)
    axes[0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.80))
    fig.suptitle(f"Human demos — 9 ranking features on the 699 pool (session_3_ext700): "
                 f"{n} demos, {nl} landed / {n-nl} crashed\n"
                 r"obs $=[x,\,y,\,v_x,\,v_y,\,\theta,\,\omega,\,\mathrm{leg}_L,\,\mathrm{leg}_R]$;"
                 r"  $\langle\cdot\rangle_t$ = mean over the trajectory's timesteps;"
                 r"  $T$ = final step;  solid black = feature MEAN",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140); print("saved", OUT)


if __name__ == "__main__":
    main()
