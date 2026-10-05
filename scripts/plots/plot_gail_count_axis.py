"""GAIL: final performance vs number of demonstrations, three arms.

Same format as the PT count-axis figure: linear x-axis anchored at 0, mean +/- s.e.
over seeds, metric = the 50-episode DETERMINISTIC final evaluation that each run
already wrote (eval_data/agent_alignment.json -> ep_reward_mean, n_episodes=50,
scored with the TRUE environment reward on a fresh unwrapped env).

Arms, resolved from the experiment index CSVs so the grouping is explicit:
  E4  expert demos, cap 1000               (10 seeds)
  C1  human demos, return-ranked top-K     (10 seeds; 5 at K=699)
  H6  human demos, random K from session_3 (5 seeds)

CAVEAT worth repeating wherever this figure is shown: E4 and C1 use SHUFFLE=0, so
their seeds vary training only (fixed demo set). H6 uses SHUFFLE=1, so its seeds
vary the demo subset AND training -- its error bars are not the same quantity.

Usage:
    python scripts/plots/plot_gail_count_axis.py
    python scripts/plots/plot_gail_count_axis.py --err sd --xscale log
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

RUNS = Path("/scratch/marzii/imitation_runs/gail/lunarlander")
IDX = Path("/home/marzii/IRL3/experiments/GAIL")
OUT_DEFAULT = Path("/home/marzii/IRL3/figures/gail_count_axis_E4_C1_H6.png")

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

ARM_STYLE = {
    "E4": dict(color="tab:blue", marker="o", ls="-", label="E4: expert demos"),
    "C1": dict(color="tab:orange", marker="s", ls="--",
               label="C1: human demos, return-ranked top-K"),
    "H6": dict(color="tab:green", marker="^", ls=":",
               label="H6: human demos, random K"),
    "R2": dict(color="tab:green", marker="s", ls="--",
               label="human demos, random K from the return>200 pool"),
    "R3": dict(color="tab:blue", marker="o", ls="-",
               label="expert demos, random K per seed"),
    "F2": dict(color="tab:green", marker="s", ls="--",
               label="human demos, fixed first K of the return>200 pool"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", type=Path, default=RUNS)
    p.add_argument("--index_dir", type=Path, default=IDX)
    p.add_argument("--err", choices=["se", "sd", "ci95"], default="se",
                   help="error bars: standard error, standard deviation, or "
                        "95% CI over seeds (t-based)")
    p.add_argument("--xscale", choices=["linear", "log"], default="linear")
    p.add_argument("--budgets", type=int, nargs="*", default=None,
                   help="only plot these demo counts (default: every N found)")
    p.add_argument("--arms", nargs="+", default=["E4", "C1"], choices=["E4", "C1", "H6", "R2", "R3", "F2"],
                   help="arms to plot (H6 excluded by default)")
    p.add_argument("--drop", type=int, nargs="*", default=[],
                   help="demo counts to leave out, e.g. --drop 699")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--title", default="GAIL on LunarLander: expert vs human demonstrations")
    p.add_argument("--aaai", action="store_true",
                   help="AAAI-27 single-column export: 3.3in wide, 9pt, no title, "
                        "Type-42 fonts, tight bbox, writes .pdf and 300-dpi .png")
    p.add_argument("--xlabel", default=None)
    p.add_argument("--ylabel", default=None)
    p.add_argument("--xticks", type=int, nargs="+", default=None,
                   help="labelled x ticks (default: every budget)")
    p.add_argument("--legend_loc", default=None)
    p.add_argument("--legend_bbox", type=float, nargs=2, default=None)
    p.add_argument("--dump_json", type=Path, default=None,
                   help="also write the plotted series, baseline and axis settings as JSON "
                        "(used to build the combined multi-panel figure)")
    p.add_argument("--baseline_csv", type=Path, default=None,
                   help="per-seed true-reward PPO final evals (column mean_return); "
                        "drawn as a dashed gray line with a mean +- err band")
    p.add_argument("--human_color", default=None,
                   help="override the C1 colour (e.g. to differ from the BC figure)")
    return p.parse_args()


def rows(path: Path):
    if not path.exists():
        print(f"  (missing index) {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def build_arms(index_dir: Path):
    arms = {"E4": defaultdict(list), "C1": defaultdict(list), "H6": defaultdict(list),
            "R2": defaultdict(list), "R3": defaultdict(list), "F2": defaultdict(list)}

    for pattern, arm_name, key in (("gail_rand_gt200_*.csv", "rand_gt200", "R2"),
                                   ("gail_rand_expert_*.csv", "rand_expert", "R3"),
                                   ("gail_fixed_gt200_*.csv", "fixed_gt200", "F2")):
        for f in sorted(index_dir.glob(pattern)):
            for r in rows(f):
                job = r.get("slurm_job_id", "").strip()
                if r.get("arm") == arm_name and job:
                    arms[key][int(r["N"])].append(job)

    for r in rows(index_dir / "gail_countaxis_kappa1000_2026-09-07.csv"):
        job = r.get("slurm_job_id", "").strip()
        if not job:
            continue
        if r["arm"] == "expert":
            arms["E4"][int(r["N"])].append(job)
        elif r["arm"] == "human":
            arms["C1"][int(r["N"])].append(job)

    for r in rows(index_dir / "gail_ret_only_countaxis_2026-09-03.csv"):
        if r.get("condition") == "ret_only" and r.get("slurm_job_id", "").strip():
            arms["C1"][int(r["K"])].append(r["slurm_job_id"].strip())

    for r in rows(index_dir / "gail_2feat_rankings_top10_2026-08-28.csv"):
        if r.get("ranking") == "ret_only" and r.get("slurm_job_id", "").strip():
            arms["C1"][int(r["K"])].append(r["slurm_job_id"].strip())

    for r in rows(index_dir / "gail_session3_curation_2026-08-05.csv"):
        cond = r.get("condition", "")
        if cond.startswith("rand") and r.get("slurm_job_id", "").strip():
            arms["H6"][int(cond[4:])].append(r["slurm_job_id"].strip())

    return arms


def final_eval(run_dir: Path):
    """(mean_return, n_episodes) from the run's 50-episode deterministic eval."""
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
    arms = build_arms(a.index_dir)

    if a.aaai:
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                             "axes.labelsize": 9, "xtick.labelsize": 9,
                             "ytick.labelsize": 9, "legend.fontsize": 9})
    lw, cap, msz = (1.2, 2, 4) if a.aaai else (2, 3, 6)
    short = {"E4": "expert demos", "C1": "human demos", "H6": "human demos (random)",
             "R2": "human demos (random)", "R3": "expert demos (random)",
             "F2": "human demos (fixed)"}
    dump = {"series": [], "baseline": None}
    fig, ax = plt.subplots(figsize=(3.3, 2.4) if a.aaai else (7.0, 4.4))
    all_x = set()

    for arm in a.arms:
        xs, means, errs, ns = [], [], [], []
        print(f"--- {arm}")
        for n in sorted(arms[arm]):
            if a.budgets and n not in a.budgets:
                continue
            if n in a.drop:
                continue
            vals, eps = [], set()
            for job in arms[arm][n]:
                r = final_eval(a.runs_root / job)
                if r is not None:
                    vals.append(r[0])
                    eps.add(r[1])
            if not vals:
                print(f"  (no final eval) N={n}")
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
        st = dict(ARM_STYLE[arm])
        if arm == "C1" and a.human_color:
            st["color"] = a.human_color
        ax.errorbar(xs, means, yerr=errs, color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=lw, ms=msz, capsize=cap,
                    label=(f"{short[arm]}, {max(ns)} seeds" if a.aaai
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
