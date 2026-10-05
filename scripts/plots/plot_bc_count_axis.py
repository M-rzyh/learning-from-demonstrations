"""BC: final performance vs number of demonstrations, expert vs human.

Same format as the GAIL and PT count-axis figures. Metric = the 50-episode
DETERMINISTIC evaluation each BC run wrote (eval_data/agent_alignment.json ->
ep_reward_mean), scored with the TRUE environment reward on a fresh env at the
1000-step cap, env seeded reset(seed=SEED*1000+ep) -- the same protocol as GAIL.

Arms (resolved from the BC index CSVs, so the grouping is explicit):
  expert  demos 0..N-1 of expert/lunarlander/4615187/rollouts/perframe_demos_500_holdK1
          -- the exact demos GAIL E4 used (SHUFFLE=0, identical for every seed)
  human   session_3 shuffled with the run's seed, first N -- the exact demos
          GAIL H6 used at that (N, seed)

Seed semantics differ between the arms and this matters for the error bars:
  expert : seed varies training only (same demos every seed)
  human  : seed varies the demo subset AND training

Usage:
    python scripts/plots/plot_bc_count_axis.py
    python scripts/plots/plot_bc_count_axis.py --xscale log --err sd
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms

RUNS = Path("/scratch/marzii/imitation_runs/bc/lunarlander")
IDX = Path("/home/marzii/IRL3/experiments/BC")
OUT_DEFAULT = Path("/home/marzii/IRL3/figures/bc_count_axis_expert_vs_human.png")

# two-sided t_{0.975} by degrees of freedom, for 95% CIs from few seeds
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
       27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def spread(sd, n, kind):
    """sd, standard error, or the half-width of a 95% CI over n seeds.

    sd and n may be scalars or arrays (one entry per budget).
    """
    sd, n = np.asarray(sd, dtype=float), np.asarray(n, dtype=float)
    if kind == "sd":
        return sd
    se = sd / np.sqrt(n)
    if kind != "ci95":
        return se
    t = np.vectorize(lambda k: T95.get(int(k) - 1, 1.96))(n)
    return se * t

STYLE = {
    "expert": dict(color="tab:blue", marker="o", ls="-",
                   label="expert demos (matched to GAIL E4)"),
    "human_ranked": dict(color="tab:orange", marker="s", ls="--",
                         label="human demos, return-ranked (matched to GAIL C1)"),
    "human_random": dict(color="tab:green", marker="^", ls=":",
                         label="human demos, random (matched to GAIL H6)"),
    "human_rand200": dict(color="tab:orange", marker="s", ls="--",
                          label="human demos, random from the return>200 pool"),
    "expert_rand": dict(color="tab:blue", marker="o", ls="-",
                        label="expert demos, random per seed"),
    "human_fixed200": dict(color="tab:orange", marker="s", ls="--",
                           label="human demos, fixed first N of the return>200 pool"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", type=Path, default=RUNS)
    p.add_argument("--expert_index", type=Path,
                   default=IDX / "bc_e4_matched_2026-09-17.csv")
    p.add_argument("--human_random_index", type=Path,
                   default=IDX / "bc_h6_matched_shuffle_2026-09-17.csv")
    p.add_argument("--human_ranked_index", type=Path,
                   default=IDX / "bc_c1_matched_2026-09-18.csv")
    p.add_argument("--human_rand200_index", type=Path,
                   default=IDX / "bc_rand_gt200_2026-09-22.csv")
    p.add_argument("--expert_rand_index", type=Path,
                   default=IDX / "bc_rand_expert_2026-09-23.csv")
    p.add_argument("--human_fixed200_index", type=Path,
                   default=IDX / "bc_fixed_gt200_2026-09-24.csv")
    p.add_argument("--arms", nargs="+", default=["expert", "human_ranked"],
                   choices=["expert", "human_ranked", "human_random", "human_rand200",
                            "expert_rand", "human_fixed200"],
                   help="arms to plot (human_random, matched to GAIL H6, excluded by default)")
    p.add_argument("--err", choices=["se", "sd", "ci95"], default="se",
                   help="error bars: standard error, standard deviation, or "
                        "95% CI over seeds (t-based)")
    p.add_argument("--xscale", choices=["linear", "log"], default="linear")
    p.add_argument("--budgets", type=int, nargs="*", default=None,
                   help="only plot these demo counts (default: every N found)")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--title", default="BC on LunarLander: expert vs human demonstrations")
    p.add_argument("--aaai", action="store_true",
                   help="AAAI-27 single-column export: 3.3in wide, 9pt, no title, "
                        "Type-42 fonts, tight bbox, writes .pdf and 300-dpi .png")
    p.add_argument("--xlabel", default=None)
    p.add_argument("--ylabel", default=None)
    p.add_argument("--xticks", type=int, nargs="+", default=None,
                   help="labelled x ticks (default: every budget)")
    p.add_argument("--legend_loc", default=None)
    p.add_argument("--legend_bbox", type=float, nargs=2, default=None)
    p.add_argument("--baseline_csv", type=Path, default=None,
                   help="per-seed true-reward PPO final evals (column mean_return); "
                        "drawn as a dashed gray line with a mean +- err band")
    p.add_argument("--dump_json", type=Path, default=None,
                   help="also write the plotted series, baseline and axis settings as JSON "
                        "(used to build the combined multi-panel figure)")
    return p.parse_args()


def load_index(path: Path):
    """{N: [job_ids]} from a BC index CSV."""
    by_n = defaultdict(list)
    if not path.exists():
        print(f"  (missing index) {path}")
        return by_n
    for r in csv.DictReader(open(path)):
        job = r.get("slurm_job_id", "").strip()
        if job:
            by_n[int(r["N"])].append(job)
    return by_n


def final_eval(run_dir: Path):
    f = run_dir / "eval_data" / "agent_alignment.json"
    if not f.exists():
        return None
    try:
        d = json.load(open(f))
        return float(d["ep_reward_mean"]), int(d.get("n_episodes", -1))
    except (KeyError, json.JSONDecodeError):
        return None


def main():
    a = parse_args()
    arms = {"expert": load_index(a.expert_index),
            "human_ranked": load_index(a.human_ranked_index),
            "human_random": load_index(a.human_random_index),
            "human_rand200": load_index(a.human_rand200_index),
            "expert_rand": load_index(a.expert_rand_index),
            "human_fixed200": load_index(a.human_fixed200_index)}

    if a.aaai:
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                             "axes.labelsize": 9, "xtick.labelsize": 9,
                             "ytick.labelsize": 9, "legend.fontsize": 9})
    lw, cap, msz = (1.2, 2, 4) if a.aaai else (2, 3, 6)
    short = {"expert": "expert demos", "human_ranked": "human demos",
             "human_random": "human demos (random)", "human_rand200": "human demos (random)",
             "expert_rand": "expert demos (random)", "human_fixed200": "human demos (fixed)"}
    dump = {"series": [], "baseline": None}
    fig, ax = plt.subplots(figsize=(3.3, 2.4) if a.aaai else (7.0, 4.4))
    all_x = set()

    for name in a.arms:
        xs, means, errs, ns = [], [], [], []
        print(f"--- {name}")
        for n in sorted(arms[name]):
            if a.budgets and n not in a.budgets:
                continue
            vals, eps = [], set()
            for job in arms[name][n]:
                r = final_eval(a.runs_root / job)
                if r is not None:
                    vals.append(r[0])
                    eps.add(r[1])
            if not vals:
                print(f"  (no eval) N={n}")
                continue
            v = np.asarray(vals)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            xs.append(n)
            means.append(v.mean())
            errs.append(spread(sd, len(v), a.err))
            ns.append(len(v))
            all_x.add(n)
            print(f"  N={n:<5} seeds={len(v):<3} mean={v.mean():7.1f} sd={sd:6.1f} "
                  f"episodes={sorted(eps)}")
        if not xs:
            continue
        st = STYLE[name]
        ax.errorbar(xs, means, yerr=errs, color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=lw, ms=msz, capsize=cap,
                    label=(f"{short[name]}, {max(ns)} seeds" if a.aaai
                           else f"{st['label']} (n={max(ns)} seeds)"))
        dump["series"].append(dict(x=[float(v) for v in xs], mean=[float(v) for v in means],
                                   err=[float(v) for v in errs], n=[int(v) for v in ns],
                                   color=st["color"], marker=st["marker"], ls=st["ls"]))

    if a.baseline_csv is not None:
        b = np.loadtxt(a.baseline_csv, delimiter=",", skiprows=1, usecols=2, ndmin=1)
        bsd = b.std(ddof=1) if len(b) > 1 else 0.0
        be = spread(bsd, len(b), a.err)
        ax.axhline(b.mean(), color="#555555", ls="--", lw=1.0 if a.aaai else 1.4, zorder=1)
        ax.axhspan(b.mean() - be, b.mean() + be, color="#555555", alpha=0.15, zorder=0)
        dump["baseline"] = dict(mean=float(b.mean()), err=float(be), n=int(len(b)), color="#555555")
        print(f"baseline: n={len(b)} mean={b.mean():.1f} sd={bsd:.1f}")

    xticks = sorted(all_x)
    if a.xscale == "log":
        ax.set_xscale("log")
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        ax.set_xlim(0, max(xticks) * 1.04)
    ax.set_xticks(xticks)
    if a.xticks:
        ax.set_xticks(a.xticks)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(a.xlabel or "number of demonstrations")
    ax.set_ylabel(a.ylabel or "final return\n(50 deterministic eval episodes)")
    if not a.aaai:
        ax.set_title(a.title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    if a.aaai:
        ax.legend(fontsize=9, loc=a.legend_loc or "best", bbox_to_anchor=a.legend_bbox,
                  frameon=False, handlelength=2.2, borderaxespad=0.1)
    else:
        ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()

    if a.dump_json:
        for s, lab in zip(dump["series"], ax.get_legend_handles_labels()[1]):
            s["label"] = lab
        dump.update(xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), xscale=ax.get_xscale(),
                    xticks=[float(t) for t in ax.get_xticks()],
                    xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
        a.dump_json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dump, open(a.dump_json, "w"), indent=1)
        print(f"wrote {a.dump_json}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.aaai:
        # tight bbox plus a small uneven margin (left, bottom, right, top; inches)
        # so every text box sits fully inside the page
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        bbox = matplotlib.transforms.Bbox.from_extents(bb.x0 - 0.12, bb.y0 - 0.10,
                                                       bb.x1 + 0.04, bb.y1 + 0.04)
        for ext in (".pdf", ".png"):
            o = a.out.with_suffix(ext)
            fig.savefig(o, dpi=300, bbox_inches=bbox)
            print(f"\nwrote {o}")
    else:
        fig.savefig(a.out, dpi=200)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
