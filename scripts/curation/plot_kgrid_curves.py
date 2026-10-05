"""GAIL learning curves over the (kNN neighbours) x (demo count) grid for one 2-feature ranking.

One panel per demo count K; within a panel, one line per kNN neighbour count k, so the
comparison of interest -- does the neighbourhood size matter at this K? -- is within-panel.
Line = mean over seeds, band = +/- 1 std. The number in each legend entry is the
post-training 50-episode eval, which is the real performance measure; the curves are
on-policy training rollouts and sit lower.

    python scripts/curation/plot_kgrid_curves.py --stem ad
    python scripts/curation/plot_kgrid_curves.py --stem adxy --Ks 10,50,100
"""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_2feat_learning_curves import discover, curve, smooth, GA, TOTAL_TIMESTEPS, ROUNDS_FULL

NAME = {"ad": "return + action_divergence (6-D state)",
        "adxy": "return + action_divergence_xy (x,y only)"}


def tag_for(stem, k):
    return stem if k == 15 else f"{stem}k{k}"


def eval_mean(jobs):
    v = [json.load(open(f"{GA}/{j}/eval_data/agent_alignment.json"))["ep_reward_mean"]
         for j in jobs if os.path.exists(f"{GA}/{j}/eval_data/agent_alignment.json")]
    return (np.mean(v), np.std(v), len(v)) if v else (None, None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="ad", choices=("ad", "adxy"))
    ap.add_argument("--ks", default="15,50,100", help="kNN neighbour counts (rows of the grid)")
    ap.add_argument("--Ks", default="10,50,100", help="demo counts (one panel each)")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ks = [int(x) for x in a.ks.split(",")]
    Ks = [int(x) for x in a.Ks.split(",")]
    runs = discover()
    out = a.out or f"/home/marzii/IRL3/figures/ext700/gail_kgrid_{a.stem}.png"

    fig, axes = plt.subplots(1, len(Ks), figsize=(5.0 * len(Ks), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    print(f"{'cell':>18} | seeds | curve final | eval (50 ep)")
    print("-" * 62)
    for ax, K in zip(axes, Ks):
        for k in ks:
            jobs = runs.get((tag_for(a.stem, k), K), {})
            ys = [y for y in (curve(j) for _, j in sorted(jobs.items())) if y is not None]
            if not ys:
                print(f"{'k=%d K=%d' % (k, K):>18} | no runs"); continue
            n = min(len(y) for y in ys)
            Y = np.vstack([smooth(y[:n], a.window) for y in ys])
            x = np.arange(Y.shape[1]) * (TOTAL_TIMESTEPS / ROUNDS_FULL)
            m, sd = Y.mean(0), Y.std(0)
            em, es, ne = eval_mean(jobs.values())
            lbl = f"k={k}" + (f"   eval {em:+.0f}±{es:.0f}" if em is not None else "")
            line, = ax.plot(x, m, lw=2.0, label=lbl)
            ax.fill_between(x, m - sd, m + sd, color=line.get_color(), alpha=0.15, lw=0)
            print(f"{'k=%d K=%d' % (k, K):>18} | {len(ys):5d} | {m[-1]:+11.1f} | "
                  f"{em:+.1f} +/- {es:.1f} (n={ne})" if em is not None else "")
        ax.axhline(0, color="gray", ls=":", alpha=0.5)
        ax.set_title(f"{K} demos", fontsize=11)
        ax.set_xlabel("steps")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    axes[0].set_ylabel(f"Mean episode return, true env  (rolling {a.window} rounds)")
    fig.suptitle(f"GAIL — {NAME[a.stem]}\n"
                 "columns = number of demos, lines = kNN neighbours used by the metric; "
                 "band = ±1 std over 10 seeds", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
