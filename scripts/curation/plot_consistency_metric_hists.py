"""Histograms over the demo pool for the 6 action-consistency metrics + return.

Landed (green) vs crashed (red) overlay, solid black = this pool's mean, dashed grey = the
expert pool's mean for the same metric (the single-deterministic-policy floor).

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_consistency_metric_hists.py \
        --csv <human_features.csv> --expert_csv <expert_features.csv> --out <png>
"""
import argparse, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PANELS = [
    ("return",                       "total episode reward"),
    ("action_divergence",            "cross-demo, multi-vote"),
    ("action_entropy",               "cross-demo, multi-vote"),
    ("self_action_divergence",       "own demo only"),
    ("self_action_entropy",          "own demo only"),
    ("cross_demo_action_divergence", "distinct demos, 1 vote each"),
    ("cross_demo_action_entropy",    "distinct demos, 1 vote each"),
]


def load(path):
    rows = list(csv.DictReader(open(path)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--expert_csv", default="")
    ap.add_argument("--out", default="/home/marzii/IRL3/figures/ext700/consistency_metric_hists.png")
    a = ap.parse_args()

    rows = load(a.csv)
    outc = np.array([r["outcome"] for r in rows])
    exp = load(a.expert_csv) if a.expert_csv else None

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for ax, (f, sub) in zip(axes, PANELS):
        v = np.array([float(r[f]) for r in rows], float)
        bins = np.histogram_bin_edges(v, bins=30)
        ax.hist(v[outc == "landed"], bins=bins, color="#2ca02c", alpha=.65, label="landed")
        ax.hist(v[outc != "landed"], bins=bins, color="#d62728", alpha=.55, label="crashed")
        m = v.mean()
        ax.axvline(m, color="k", lw=1.8)
        txt = f"mean {m:.3f}" if f != "return" else f"mean {m:+.0f}"
        if exp is not None:
            e = np.mean([float(r[f]) for r in exp])
            ax.axvline(e, color="#555", ls="--", lw=1.6)
            txt += f"\nexpert {e:.3f}" if f != "return" else f"\nexpert {e:+.0f}"
        ax.annotate(txt, (0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                    fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_title(f"{f}\n({sub})", fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("value", fontsize=8); ax.set_ylabel("count (# demos)", fontsize=8)
    axes[0].legend(fontsize=8)
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.75,
                  "solid black = pool mean\n"
                  "dashed grey = expert pool mean\n"
                  "  (one deterministic policy =\n"
                  "   the kNN smoothing floor)\n\n"
                  "divergence: was THIS action unusual?\n"
                  "entropy:    is this STATE ambiguous?\n"
                  "            (ignores the taken action)",
                  fontsize=9, va="top", family="monospace")
    fig.suptitle(f"Action-consistency metrics over {len(rows)} human demos "
                 f"({int((outc=='landed').sum())} landed / {int((outc!='landed').sum())} crashed)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140); print("saved", a.out)

    print(f"\n{'metric':>30} | {'mean':>8} | {'std':>7} | {'landed':>8} | {'crashed':>8} | corr(return)")
    print("-" * 92)
    ret = np.array([float(r["return"]) for r in rows])
    for f, _ in PANELS:
        v = np.array([float(r[f]) for r in rows], float)
        print(f"{f:>30} | {v.mean():>8.3f} | {v.std():>7.3f} | "
              f"{v[outc=='landed'].mean():>8.3f} | {v[outc!='landed'].mean():>8.3f} | "
              f"{np.corrcoef(v, ret)[0,1]:>+.3f}")


if __name__ == "__main__":
    main()
