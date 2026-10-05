"""Human supervision on LunarLander -- eval reward vs HUMAN TIME, for GAIL demo rankings
AND the PT preference count axis (before and after the label-encoding fix).

Same format as plot_human_time_gail_rankings.py, with two PT series added.

x, GAIL : 10.8 s per demo (300 demos in 54 min) x K demos  -- an ESTIMATE, constant per demo.
x, PT   : the MEASURED sum of `time_sec` over exactly the preference pairs each seed was
          given, recovered from labels_n1000_clean.pkl. Not an estimate.
y, GAIL : mean of each run's 50-episode deterministic post-training eval.
y, PT   : mean of each run's `last10_eval_reward` (the last 10 of 200 evals x 10 episodes).
band    : +/- 1 std over seeds.  Point labels = K demos / N preferences.

PT uses seeds 0-9 for BOTH old and new so the two are directly comparable (the old runs
have 30 seeds available, the fixed re-runs only 10).

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_human_time_gail_pt.py
"""
import argparse, json, os, pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_2feat_learning_curves import discover, LABEL, GA

DEMO_SEC = 54 * 60 / 300                      # 10.8 s per demo
SERIES   = ["only", "ad", "adxy", "satpad", "satad", "satad8"]
POOL     = "/scratch/marzii/PT/lunarlander/human_pref_clean/labels_n1000_clean.pkl"
LBL_OLD  = "/scratch/marzii/PT/human_label/human"          # raw web codes (broken)
LBL_NEW  = "/scratch/marzii/PT/human_label/human_fixed"    # translated (correct)
PT_RUNS  = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
SEEDS    = range(10)


def eval_stats(jobs):
    v = [json.load(open(f))["ep_reward_mean"]
         for j in jobs
         if os.path.exists(f := f"{GA}/{j}/eval_data/agent_alignment.json")]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else None


def pt_label_seconds(lbl_tree, N, seed, pool_time):
    """Measured human seconds for the exact pairs this (N, seed) was trained on."""
    d = f"{lbl_tree}/lunarlander-human-count-N{N}-s{seed}"
    fa, fb = f"{d}/indices_num{N}_q100", f"{d}/indices_2_num{N}_q100"
    if not os.path.exists(fa):
        return None
    A = pickle.load(open(fa, "rb")); B = pickle.load(open(fb, "rb"))
    return float(sum(pool_time[(int(x), int(y))] for x, y in zip(A, B)))


def pt_series(cond_suffix, lbl_tree, Ns, pool_time):
    """-> (minutes, mean eval, std eval, N) arrays over the Ns that have runs."""
    xs, ys, es, ks = [], [], [], []
    for N in Ns:
        cond = f"lunarlander-human-count-N{N}{cond_suffix}"
        vals, secs = [], []
        for s in SEEDS:
            p = f"{PT_RUNS}/{cond}/seed_{s}/eval_summary.json"
            if not os.path.exists(p):
                continue
            vals.append(float(json.load(open(p))["last10_eval_reward"]))
            t = pt_label_seconds(lbl_tree, N, s, pool_time)
            if t is not None:
                secs.append(t)
        if not vals or not secs:
            continue
        xs.append(np.mean(secs) / 60.0); ys.append(np.mean(vals))
        es.append(np.std(vals)); ks.append(N)
    return np.array(xs), np.array(ys), np.array(es), ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/marzii/IRL3/figures/ext700/human_time_gail_pt.png")
    a = ap.parse_args()

    w = pickle.load(open(POOL, "rb"))
    pa, pb, ts = w["pair_starts_a"], w["pair_starts_b"], w["time_sec"]
    pool_time = {(int(pa[i]), int(pb[i])): float(ts[i]) for i in range(len(pa))}

    runs = discover()
    fig, ax = plt.subplots(figsize=(12, 6.5))

    print(f"{'series':>18} | K/N -> eval (n seeds)")
    for tag, marker in zip(SERIES, "os^Dv*"):
        cells = {K: st for (t, K), jb in runs.items() if t == tag
                 and (st := eval_stats(jb.values())) is not None}
        if not cells:
            continue
        Ks = sorted(cells)
        x = np.array([K * DEMO_SEC / 60 for K in Ks])
        y = np.array([cells[K][0] for K in Ks]); e = np.array([cells[K][1] for K in Ks])
        line, = ax.plot(x, y, marker + "-", lw=2.0, ms=7,
                        label="GAIL demos — " + LABEL.get(tag, tag), zorder=4)
        ax.fill_between(x, y - e, y + e, color=line.get_color(), alpha=0.12, lw=0, zorder=1)
        for K, xi, yi in zip(Ks, x, y):
            ax.annotate(f"{K}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=line.get_color())
        print(f"{'GAIL '+tag:>18} | " + "  ".join(f"{K}:{cells[K][0]:+.0f}({cells[K][2]})" for K in Ks))

    for suffix, tree, Ns, lab, col, mk in [
        ("",             LBL_OLD, [10, 100, 250, 750, 1000], "PT prefs — OLD (broken labels)", "#7f7f7f", "X"),
        ("-fixedlabels", LBL_NEW, [10, 50, 100, 750, 1000],  "PT prefs — NEW (fixed labels)",  "#000000", "P"),
    ]:
        x, y, e, ks = pt_series(suffix, tree, Ns, pool_time)
        if not len(x):
            continue
        ax.plot(x, y, mk + "--", lw=2.4, ms=9, color=col, label=lab, zorder=5)
        ax.fill_between(x, y - e, y + e, color=col, alpha=0.10, lw=0, zorder=1)
        for K, xi, yi in zip(ks, x, y):
            ax.annotate(f"{K}", (xi, yi), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7.5, color=col, fontweight="bold")
        print(f"{lab:>18} | " + "  ".join(f"{k}:{v:+.0f}@{xi:.1f}min" for k, v, xi in zip(ks, y, x)))

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.axhline(200, color="gray", ls=":", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Human time (min, log scale)  —  GAIL: 10.8 s/demo (estimate) · "
                  "PT: measured per-preference decision time; labels = K demos / N preferences")
    ax.set_ylabel("Mean eval reward (true env)")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax.set_title("Human supervision on LunarLander — eval reward vs human time\n"
                 "GAIL on top-K demo rankings vs PT on N preferences (before/after the label fix); "
                 "band = ±1 std over seeds", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150); print("\nsaved", a.out)


if __name__ == "__main__":
    main()
