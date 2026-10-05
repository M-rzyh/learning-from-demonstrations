"""Figures for the human multi-modality / satisficing characterization.

Reads the JSON + NPZ written by `multimodality_analysis.py`.

  fig 1  multimodality_bc_ceiling.png   irreducible BC error vs k, human vs expert (demo-matched)
  fig 2  multimodality_state_cells.png  p(a|s) in shared state cells: human spreads, expert commits
  fig 3  multimodality_error_map.png    where in (x, y) the non-realizability lives
  fig 4  satisficing_landing_sites.png  humans stop "good enough", anywhere; expert kisses the pad

    python scripts/curation/plot_multimodality.py
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots"))
from figures_dir import FIGURES              # noqa: E402


def out_path(subdir, name):
    """figures/<subdir>/<name> — the IRL3 convention (scripts/figures_dir.py)."""
    d = FIGURES / subdir if subdir else FIGURES
    d.mkdir(parents=True, exist_ok=True)
    return d / name

ACT_NAMES = ["noop", "left", "main", "right"]
CH_COL, CE_COL = "#c2410c", "#1d4ed8"       # human = orange, expert = blue
NA = 4


def fig_bc_ceiling(res, out):
    by_k = res["matched"]["by_k"]
    ks = sorted(int(k) for k in by_k)
    hm = np.array([by_k[str(k)]["human"]["err"]["mean"] for k in ks])
    hs = np.array([by_k[str(k)]["human"]["err"]["std"] for k in ks])
    em = np.array([by_k[str(k)]["expert"]["err"]["mean"] for k in ks])
    hd = np.array([by_k[str(k)]["human"]["div"]["mean"] for k in ks])
    ed = np.array([by_k[str(k)]["expert"]["div"]["mean"] for k in ks])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ks, hm, "o-", color=CH_COL, label=f"human ({res['matched']['n_take']} demos)")
    ax[0].fill_between(ks, hm - hs, hm + hs, color=CH_COL, alpha=.2)
    ax[0].plot(ks, em, "s-", color=CE_COL, label=f"expert ({res['matched']['n_take']} demos)")
    for k, a, b in zip(ks, hm, em):
        ax[0].annotate(f"{a/b:.2f}x", (k, (a + b) / 2), fontsize=8, ha="center",
                       color="0.35")
    ax[0].set_xlabel("k (neighbours used to estimate $\\pi(a|s)$)")
    ax[0].set_ylabel("irreducible BC error  (0-1 action loss)")
    ax[0].set_title("The best Markov policy still gets\n1 in 3 human actions wrong")
    ax[0].set_ylim(0, .45); ax[0].legend(); ax[0].grid(alpha=.3)

    w = .35; xs = np.arange(len(ks))
    ax[1].bar(xs - w/2, hd, w, color=CH_COL, alpha=.85, label="human")
    ax[1].bar(xs + w/2, ed, w, color=CE_COL, alpha=.85, label="expert")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"k={k}" for k in ks])
    ax[1].set_ylabel("action_divergence")
    ax[1].set_title("The existing summary statistic\nseparates the two sets far less")
    ax[1].set_ylim(0, .45); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
    for i, (a, b) in enumerate(zip(hd, ed)):
        ax[1].annotate(f"{a/b:.2f}x", (i, max(a, b) + .015), fontsize=8, ha="center", color="0.35")

    fig.suptitle("Human demos are not realizable by a single $\\pi(a|s)$ — demo-count matched, "
                 "shared state metric", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def fig_state_cells(z, res, out, n_show=6):
    CH, CE = z["CH"], z["CE"]
    nH, nE = CH.sum(1), CE.sum(1)
    mc = res["cells"]["min_cell"]
    shared = np.flatnonzero((nH >= mc) & (nE >= mc))
    # rank shared cells by how much MORE spread the human is than the expert there
    def norm_ent(C):
        p = C / np.maximum(C.sum(1, keepdims=True), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(p > 0, np.log(p), 0.0)
        return -(p * lg).sum(1) / np.log(NA)
    gap = norm_ent(CH)[shared] - norm_ent(CE)[shared]
    pick = shared[np.argsort(-gap)[:n_show]]

    mean, std = z["scaler_mean"], z["scaler_std"]
    cen = z["cell_centers"] * std + mean
    fig, axes = plt.subplots(2, n_show // 2, figsize=(3.0 * (n_show // 2), 6.0))
    for ax, c in zip(axes.ravel(), pick):
        ph = CH[c] / CH[c].sum(); pe = CE[c] / CE[c].sum()
        xs = np.arange(NA); w = .38
        ax.bar(xs - w/2, ph, w, color=CH_COL, label="human")
        ax.bar(xs + w/2, pe, w, color=CE_COL, label="expert")
        ax.set_xticks(xs); ax.set_xticklabels(ACT_NAMES, fontsize=8, rotation=20)
        ax.set_ylim(0, 1.18); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
        ax.grid(alpha=.3, axis="y")
        s = cen[c]
        ax.set_title(f"x={s[0]:+.2f} y={s[1]:.2f}\n"
                     f"$v_x$={s[2]:+.2f} $v_y$={s[3]:+.2f} $\\theta$={s[4]:+.2f}",
                     fontsize=8)
        ax.text(.02, .98, f"n={int(CH[c].sum())} / {int(CE[c].sum())}",
                transform=ax.transAxes, fontsize=7, va="top", ha="left", color="0.4")
    axes.ravel()[0].legend(fontsize=8, loc="upper right", framealpha=.9)
    for ax in axes[:, 0]:
        ax.set_ylabel("$P(a \\mid \\mathrm{cell})$")
    fig.suptitle("Same state, different action: $P(a|s)$ in state cells both sets visit\n"
                 "(the 6 shared cells where the human is most spread; expert commits to one action)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def fig_error_map(z, res, out):
    """Per-state irreducible BC error (the discriminative statistic) + state coverage.

    NOTE the honest reading: mean H(a|s) is nearly identical for the two sets (kNN smoothing
    blurs decision boundaries in BOTH), so entropy is NOT what separates them. What separates
    them is (a) how often the demonstrator's own action differs from the local plurality, and
    (b) how much of the state space the human visits at all.
    """
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    mH = z["h_used"] > 0; mE = z["e_used"] > 0
    kw = dict(gridsize=45, extent=(-1.0, 1.0, -0.15, 1.5), mincnt=25, cmap="magma",
              vmin=0, vmax=.55, reduce_C_function=np.mean)
    h0 = ax[0].hexbin(z["h_xy"][mH, 0], z["h_xy"][mH, 1], C=z["h_err"][mH], **kw)
    ax[0].set_title("human — local BC error rate\n(mean %.2f)"
                    % res["full_pool"]["human"]["irreducible_bc_err"])
    ax[1].hexbin(z["e_xy"][mE, 0], z["e_xy"][mE, 1], C=z["e_err"][mE], **kw)
    ax[1].set_title("expert (single policy) — mean %.2f"
                    % res["full_pool"]["expert"]["irreducible_bc_err"])
    fig.colorbar(h0, ax=ax[:2].tolist(), label="P(own action $\\neq$ local plurality)",
                 fraction=.03)

    h2 = ax[2].hexbin(z["h_xy"][:, 0], z["h_xy"][:, 1], gridsize=45,
                      extent=(-1.0, 1.0, -0.15, 1.5), cmap="viridis", bins="log")
    ax[2].hexbin(z["e_xy"][:, 0], z["e_xy"][:, 1], gridsize=45,
                 extent=(-1.0, 1.0, -0.15, 1.5), cmap="Greys", bins="log", alpha=.0)
    xe, ye = z["e_xy"][:, 0], z["e_xy"][:, 1]
    ax[2].plot(np.percentile(xe, [1, 99])[[0, 0, 1, 1, 0]],
               [ye.min(), np.percentile(ye, 99), np.percentile(ye, 99), ye.min(), ye.min()],
               color="r", lw=1.2, ls="--", label="expert 1-99% x-range")
    ax[2].legend(fontsize=8, loc="upper left")
    ax[2].set_title("human state occupancy")
    fig.colorbar(h2, ax=ax[2], label="transitions", fraction=.05)
    for a in ax:
        a.axvline(0, color="w", lw=.8, ls=":"); a.axhline(0, color="w", lw=.8, ls=":")
        a.set_xlabel("x  (0 = pad centre)")
    ax[0].set_ylabel("y  (0 = pad height)")
    fig.suptitle("Non-realizability is spread over the whole flight envelope, and the human "
                 "visits states the expert never enters", fontsize=11, y=1.04)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig_satisficing(z, res, out):
    hL, eL = z["h_landed"].astype(bool), z["e_landed"].astype(bool)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    bins = np.linspace(0, .8, 33)
    ax[0].hist(z["h_x"][hL], bins=bins, color=CH_COL, alpha=.75, density=True,
               label=f"human landings (n={hL.sum()})")
    ax[0].hist(z["e_x"][eL], bins=bins, color=CE_COL, alpha=.6, density=True,
               label=f"expert landings (n={eL.sum()})")
    ax[0].axvspan(0, .2, color="0.85", zorder=0)
    ax[0].text(.2, ax[0].get_ylim()[1]*.92, "  landing pad", fontsize=8, color="0.35")
    ax[0].set_xlabel("|x| at rest   (pad half-width $\\approx$ 0.2)")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("Humans come to rest wherever it is safe")

    bins = np.linspace(0, .5, 33)
    ax[1].hist(z["h_ang"][hL], bins=bins, color=CH_COL, alpha=.75, density=True, label="human")
    ax[1].hist(z["e_ang"][eL], bins=bins, color=CE_COL, alpha=.6, density=True, label="expert")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("|$\\theta$| at rest (rad)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("...and at whatever tilt they end up with\n(log density — note the human tail)")

    bins = np.linspace(-400, 330, 60)
    ax[2].hist(z["h_ret"], bins=bins, color=CH_COL, alpha=.75, label="human (all 699)")
    ax[2].hist(z["e_ret"], bins=bins, color=CE_COL, alpha=.6, label="expert (150)")
    ax[2].axvline(200, color="k", ls="--", lw=1)
    ax[2].text(190, ax[2].get_ylim()[1]*.55, '"solved" (200)', fontsize=8, rotation=90,
               ha="right", va="center")
    ax[2].set_xlabel("episode return"); ax[2].set_ylabel("demos")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    ax[2].set_title("Bimodal: good-enough landings + crashes")

    s = res["satisficing"]["human"]; e = res["satisficing"]["expert"]
    fig.suptitle(
        f"Satisficing: {s['landed']}/{s['n']} human demos come to rest safely, "
        f"{100*s['landed_offpad_frac']:.0f}% of those OFF the pad (|x|>0.2)\n"
        f"and they still score {s['landed_mean_return']:.0f} vs the expert's "
        f"{e['landed_mean_return']:.0f} — LunarLander pays +100 for resting anywhere with |x|<1",
        fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/home/marzii/IRL3/figures/ext700/multimodality_stats.json")
    ap.add_argument("--npz", default="/home/marzii/IRL3/figures/ext700/multimodality_arrays.npz")
    ap.add_argument("--subdir", default="ext700")
    a = ap.parse_args()
    res = json.load(open(a.json)); z = np.load(a.npz)
    fig_bc_ceiling(res, out_path(a.subdir, "multimodality_bc_ceiling.png"))
    fig_state_cells(z, res, out_path(a.subdir, "multimodality_state_cells.png"))
    fig_error_map(z, res, out_path(a.subdir, "multimodality_error_map.png"))
    fig_satisficing(z, res, out_path(a.subdir, "satisficing_landing_sites.png"))


if __name__ == "__main__":
    main()
