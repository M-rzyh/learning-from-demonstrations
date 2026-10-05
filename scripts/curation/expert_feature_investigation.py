"""Feature investigation of the EXPERT (agent) demos used in the grid-count PT-vs-GAIL
experiment: demos/noisy_demos/lunarlander/expert_4615187/n100_p0_clean (nonFS, 30-seed count axis).

Reuses the SAME extractors as demo_features.py (so thresholds mean the same thing as for the
human demos), computes every feature, marks the MEAN (not median) on each histogram, and prints
a mean table to use as thresholds.

    export PYTHONPATH=/home/marzii/IRL3/imitation/src
    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/expert_feature_investigation.py
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets; datasets.disable_progress_bar()
from demo_features import traj_features, add_action_divergence, add_ranking, GOOD_DIR, FEATURES

DP = "/scratch/marzii/imitation_runs/demos/noisy_demos/lunarlander/expert_4615187/n100_p0_clean"
OUT_DIR = "/home/marzii/IRL3/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    ds = datasets.load_from_disk(DP)
    rows = []
    for i in range(len(ds)):
        f = traj_features(ds[i]); f["demo_id"] = i
        f["return"] = f.pop("return_")
        rows.append(f)
    add_action_divergence(rows, ds)      # cross-demo feature (needs whole set)
    add_ranking(rows)                    # composite rank (uses ACTIVE_FEATS)

    outc = np.array([r["outcome"] for r in rows])
    print(f"\nEXPERT n100_p0_clean pool: {len(rows)} demos")
    print("outcomes:", dict(Counter(outc)))

    feats = list(FEATURES)               # all 16 computed features
    means = {f: float(np.mean([r[f] for r in rows])) for f in feats}
    stds  = {f: float(np.std([r[f] for r in rows]))  for f in feats}

    # ---- mean table (thresholds) ----
    print(f"\n{'feature':>18} | {'dir':>6} | {'MEAN':>10} | {'std':>9} | landed-mean / crashed-mean")
    print("-" * 78)
    for f in feats:
        d = "higher" if GOOD_DIR[f] > 0 else "lower"
        v = np.array([r[f] for r in rows], float)
        lm = v[outc == "landed"].mean() if (outc == "landed").any() else float("nan")
        cm = v[outc != "landed"].mean() if (outc != "landed").any() else float("nan")
        print(f"{f:>18} | {d:>6} | {means[f]:>10.3f} | {stds[f]:>9.3f} | {lm:>8.2f} / {cm:>8.2f}")

    # ---- histograms with MEAN marked ----
    plot_feats = [f for f in feats if f != "legs_down"]   # legs_down is binary
    ncol = 4; nrow = int(np.ceil(len(plot_feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, plot_feats):
        v = np.array([r[f] for r in rows], float)
        bins = np.histogram_bin_edges(v, bins=25)
        ax.hist(v[outc == "landed"], bins=bins, color="#2ca02c", alpha=.65, label="landed")
        ax.hist(v[outc != "landed"], bins=bins, color="#d62728", alpha=.55, label="crashed/timeout")
        m = means[f]
        ax.axvline(m, color="k", ls="-", lw=1.8)
        ax.annotate(f"mean={m:.2f}", (m, ax.get_ylim()[1]), xytext=(3, -3),
                    textcoords="offset points", fontsize=7.5, fontweight="bold",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_title(f"{f}  ({'higher' if GOOD_DIR[f] > 0 else 'lower'} better)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("feature value", fontsize=7); ax.set_ylabel("count (# demos)", fontsize=7)
    for ax in axes[len(plot_feats):]: ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"EXPERT demos (agent 4615187, n100_p0_clean, {len(rows)} demos) — per-trajectory features\n"
                 f"landed=green vs crashed/timeout=red; solid black = MEAN (threshold reference)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "expert_n100_feature_hists.png")
    fig.savefig(out, dpi=130); print("\nsaved", out)


if __name__ == "__main__":
    main()
