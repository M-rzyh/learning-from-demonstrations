"""Trajectory features from Shah et al., "Imitation Learning via Focused Satisficing"
(IJCAI 2025, arXiv:2505.14820) computed for each demo in our 699 LunarLander pool.

The paper (Table 5 / App. E.2) defines ADDITIVE trajectory cost features for LunarLander --
per-timestep values summed over the trajectory, f_k(xi) = sum_t f_k(s_t, a_t):

    (x-position)^2, (y-position)^2, (x-velocity)^2, (y-velocity)^2,
    (angle)^2, (angular velocity)^2, and control cost = sum_t ||a_t||^2.

This script computes exactly those sums (states 0..T-1, each paired with its action, the same
obs[:len(acts)] convention demo_features.py uses) and writes them to a SEPARATE CSV keyed by the
same demo_id. It does NOT touch demo_features.py or its CSV, and deliberately does NOT do the
paper's quadratic outer-product expansion, nor any normalization / ranking / filtering /
subdominance -- base features only.

Control cost, discrete vs continuous
------------------------------------
The paper uses LunarLanderContinuous (a_t = (main, lateral) in [-1,1]^2) so control cost is
sum ||a_t||^2. Our demos are DISCRETE (0 = noop, 1 = left engine, 2 = main, 3 = right).
Squaring action IDs would be meaningless, so we map each discrete action to the minimal-norm
continuous action that produces the same engine behaviour and take its squared norm:

    noop        -> (0, 0)   ||.||^2 = 0     (main fires only for a[0] > 0, so "off" = 0)
    main fire   -> (1, 0)   ||.||^2 = 1     (discrete main runs at FULL power, m_power = 1.0)
    side fire   -> (0, +-1) ||.||^2 = 1     (discrete side engines also run at full power)

    control_cost = sum_t ||a_t^cont||^2 = number of engine-firing timesteps.

As a documented companion (NOT in the paper) we also record the env's own fuel economics,
matching LunarLander's reward penalties of 0.30/frame (main) and 0.03/frame (side):

    control_cost_fuel = 0.30 * (#main firings) + 0.03 * (#side firings)

Safety
------
The paper defines NO safety feature for LunarLander (safety-style features appear in this
subdominance line of work in driving domains). Included here at our request as the natural
LunarLander analogue -- an unsafe-termination indicator:

    safety = 1.0 if the episode ended in a crash (game_over / out of bounds), else 0.0

using the same outcome verdict as demo_features.traj_features.

Outcome-based padding (--pad)
-----------------------------
Additive sums under-count short episodes: a crash at t=100 stops accruing cost and looks
CHEAP -- the degeneracy Shah et al. describe in App. E.3 and repair by padding short
trajectories to a fixed horizon with a fixed cost vector f_pad (value unspecified in the
paper). Length-only padding would also punish fast landings (our best demos), so we make the
pad OUTCOME-DEPENDENT, in the spirit of absorbing-state handling in adversarial IL
(Kostrikov et al., ICLR 2019, arXiv:1809.02925), with horizon h = 1000 (the env step cap):

    landed  -> pad cost 0 per missing step (a settled lander accrues no cost; equivalently,
               its padded sums equal the raw sums)
    crashed -> each of the (h - T) missing steps is charged f_pad, the POOL-MEAN per-step
               cost per feature: f_pad[k] = (sum of feature k over all real timesteps of all
               699 demos) / (total real timesteps). Computed once from the full pool and
               frozen -- NOT from any training subset -- so feature values stay comparable
               across subsets/experiments. Real steps only; padding never feeds its own mean.
    timeout -> no padding (a timeout IS h steps; also censored, per Pardo et al. 2018).

`safety` is never padded (it is an indicator, not an additive cost). NOTE: crash-padding and
`safety` encode the same event -- use one or the other downstream, not both.

--pad writes a SEPARATE CSV (raw + padded columns + pad bookkeeping) and a separate
histogram figure; the raw CSV and raw figure are only written in the default (no --pad) run.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/features_satisficing_paper.py
    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/features_satisficing_paper.py --pad
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_features import traj_features                      # read-only reuse of the outcome verdict

POOL = ("/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/"
        "lunarlander/session_3_ext700")
HORIZON = 1000            # env step cap; padding target h

# additive cost features (padded); safety is an indicator and is never padded
PAD_FEATS = ["sum_x2", "sum_y2", "sum_vx2", "sum_vy2", "sum_theta2", "sum_omega2",
             "control_cost", "control_cost_fuel"]

# (csv column, panel subtitle)
PANELS = [
    ("return",            "total episode reward (reference)"),
    ("sum_x2",            "sum of (x-position)^2"),
    ("sum_y2",            "sum of (y-position)^2"),
    ("sum_vx2",           "sum of (x-velocity)^2"),
    ("sum_vy2",           "sum of (y-velocity)^2"),
    ("sum_theta2",        "sum of (angle)^2"),
    ("sum_omega2",        "sum of (angular velocity)^2"),
    ("control_cost",      "sum ||a_t||^2, discrete mapping"),
    ("control_cost_fuel", "0.30/main + 0.03/side firing (ours)"),
    ("safety",            "1 = crashed (ours; not in paper)"),
]

# discrete action -> ||a^cont||^2 under the minimal-norm full-power mapping documented above
ACT_SQNORM = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}


def satisficing_features(ep):
    obs = np.asarray(ep["obs"], float)
    acts = np.asarray(ep["acts"]).astype(int)
    S = obs[:len(acts)]                                       # state paired with each action
    x, y, vx, vy, th, om = (S[:, i] for i in range(6))
    outcome = traj_features(ep)["outcome"]
    n_main = int((acts == 2).sum())
    n_side = int(((acts == 1) | (acts == 3)).sum())
    return dict(
        outcome=outcome,
        ep_len=int(len(acts)),
        ret=float(np.asarray(ep["rews"], float).sum()),
        sum_x2=float((x ** 2).sum()),
        sum_y2=float((y ** 2).sum()),
        sum_vx2=float((vx ** 2).sum()),
        sum_vy2=float((vy ** 2).sum()),
        sum_theta2=float((th ** 2).sum()),
        sum_omega2=float((om ** 2).sum()),
        control_cost=float(sum(ACT_SQNORM[a] for a in acts)),   # = n_main + n_side
        control_cost_fuel=0.30 * n_main + 0.03 * n_side,
        safety=float(outcome == "crashed"),
    )


def compute_f_pad(rows):
    """Pool-mean per-step cost per feature: total pool sum / total real timesteps."""
    total_steps = sum(r["ep_len"] for r in rows)
    return {f: sum(r[f] for r in rows) / total_steps for f in PAD_FEATS}


def add_padding(rows, f_pad, horizon=HORIZON):
    """Outcome-based padding. Adds pad_steps and <feature>_padded to each row in place."""
    for r in rows:
        pad = max(horizon - r["ep_len"], 0) if r["outcome"] == "crashed" else 0
        r["pad_steps"] = pad
        for f in PAD_FEATS:
            r[f + "_padded"] = r[f] + pad * f_pad[f]


def hist_figure(rows, panels, out_png, suptitle):
    outc = np.array([r["outcome"] for r in rows])
    fig, axes = plt.subplots(2, 5, figsize=(21, 8))
    axes = axes.ravel()
    for ax, (f, sub) in zip(axes, panels):
        v = np.array([float(r[f]) for r in rows], float)
        bins = np.histogram_bin_edges(v, bins=30)
        ax.hist(v[outc == "landed"], bins=bins, color="#2ca02c", alpha=.65, label="landed")
        ax.hist(v[outc != "landed"], bins=bins, color="#d62728", alpha=.55, label="crashed")
        m = v.mean()
        ax.axvline(m, color="k", lw=1.8)
        txt = f"mean {m:+.0f}" if f == "return" else f"mean {m:.3g}"
        ax.annotate(txt, (0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                    fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_title(f"{f}\n({sub})", fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("value", fontsize=8); ax.set_ylabel("count (# demos)", fontsize=8)
    axes[0].legend(fontsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=140); print("saved", out_png)


def stats_table(rows, feats):
    outc = np.array([r["outcome"] for r in rows])
    ret = np.array([float(r["return"]) for r in rows])
    print(f"\n{'feature':>25} | {'mean':>10} | {'std':>10} | {'landed':>10} | {'crashed':>10} | corr(return)")
    print("-" * 95)
    for f in feats:
        v = np.array([float(r[f]) for r in rows], float)
        print(f"{f:>25} | {v.mean():>10.3f} | {v.std():>10.3f} | "
              f"{v[outc=='landed'].mean():>10.3f} | {v[outc!='landed'].mean():>10.3f} | "
              f"{np.corrcoef(v, ret)[0,1]:>+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--pad", action="store_true",
                    help="write the outcome-padded CSV/figure instead of the raw ones")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--out_csv", default=os.path.join(os.path.dirname(POOL),
                    "session_3_ext700_features_satisficing.csv"))
    ap.add_argument("--out_png", default="/home/marzii/IRL3/figures/ext700/"
                                         "satisficing_feature_hists.png")
    ap.add_argument("--out_csv_padded", default=os.path.join(os.path.dirname(POOL),
                    "session_3_ext700_features_satisficing_padded.csv"))
    ap.add_argument("--out_png_padded", default="/home/marzii/IRL3/figures/ext700/"
                                                "satisficing_feature_hists_padded.png")
    a = ap.parse_args()

    import datasets; datasets.disable_progress_bar()
    ds = datasets.load_from_disk(a.pool)
    rows = []
    for i in range(len(ds)):
        f = satisficing_features(ds[i])
        f["demo_id"] = i
        f["return"] = f.pop("ret")
        rows.append(f)

    if not a.pad:
        cols = ["demo_id", "outcome", "return", "sum_x2", "sum_y2", "sum_vx2", "sum_vy2",
                "sum_theta2", "sum_omega2", "control_cost", "control_cost_fuel", "safety"]
        with open(a.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in cols})
        print(f"wrote {a.out_csv}  ({len(rows)} demos)")
        outc = np.array([r["outcome"] for r in rows])
        hist_figure(rows, PANELS, a.out_png,
                    f"Shah et al. (2025) satisficing-paper trajectory features over {len(rows)} human "
                    f"demos ({int((outc=='landed').sum())} landed / {int((outc!='landed').sum())} crashed)"
                    "  --  additive sums over timesteps; solid black = pool mean")
        stats_table(rows, [f for f, _ in PANELS[1:]])
        return

    # ---- padded mode: separate outputs only; the raw CSV/figure are never rewritten ----
    f_pad = compute_f_pad(rows)
    total_steps = sum(r["ep_len"] for r in rows)
    print(f"f_pad (pool-mean per-step cost over {total_steps} real timesteps, horizon h={a.horizon}):")
    for f in PAD_FEATS:
        print(f"    {f:>18}: {f_pad[f]:.6f}")
    add_padding(rows, f_pad, a.horizon)

    cols = (["demo_id", "outcome", "return", "ep_len", "pad_steps"]
            + PAD_FEATS + [f + "_padded" for f in PAD_FEATS] + ["safety"])
    with open(a.out_csv_padded, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"wrote {a.out_csv_padded}  ({len(rows)} demos)")

    outc = np.array([r["outcome"] for r in rows])
    # safety panel intentionally dropped here: padding already encodes the crash event
    panels = [("return", "total episode reward (reference)")] + \
             [(f + "_padded", sub + ", padded") for (f, sub) in PANELS[1:-1]]
    hist_figure(rows, panels, a.out_png_padded,
                f"Satisficing-paper features with OUTCOME-BASED padding to h={a.horizon} "
                f"({len(rows)} demos: {int((outc=='landed').sum())} landed pad 0 / "
                f"{int((outc=='crashed').sum())} crashed pad f_pad)  --  solid black = pool mean")
    stats_table(rows, [f + "_padded" for f in PAD_FEATS])


if __name__ == "__main__":
    main()
