"""Count axis on HUMAN supervision: PT preferences vs GAIL return-ranked demos.

Style matches PT/scripts/plots/plot_grid_pt_vs_gail.py (the oracle count-axis figure):
GAIL = green squares "-s", PT = blue circles "-o", errorbar caps, linear x, N labels.

Difference from the oracle version: human time here is MEASURED, not a flat rate.
  GAIL x : sum of per-episode duration_sec over exactly the top-K demos the return
           ranking selected  (timing.csv from the 3 source sessions; mean 11.43 s/demo)
  PT   x : sum of per-pair time_sec over exactly the pairs each seed was shown
y both  : 50-episode deterministic post-training eval / last10_eval_reward.

    python scripts/curation/plot_count_axis_human_pt_vs_gail_retonly.py
"""
import csv, glob, json, os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DR   = "/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos"
GA   = "/scratch/marzii/imitation_runs/gail/lunarlander"
LOGS = "/scratch/marzii/imitation_runs/_slurm_logs/gail/lunarlander"
PT_RUNS = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
POOL = "/scratch/marzii/PT/lunarlander/human_pref_clean/labels_n1000_clean.pkl"
LBL  = "/scratch/marzii/PT/human_label/human"
RANK = f"{DR}/lunarlander/rankings_2feat/rank_return_only.csv"
SESSIONS = [("lunarlander/session_3", 200), ("lunarlander/session_4", 100),
            ("lunarlander_batch5/session_1", 399)]
GAIL_C, PT_C = "#2ca02c", "#1f77b4"


def demo_durations():
    dur = []
    for sub, n in SESSIONS:
        rows = list(csv.DictReader(open(f"{DR}/{sub}/timing.csv")))
        assert len(rows) == n
        dur += [float(r["duration_sec"]) for r in rows]
    return np.array(dur)


def gail_runs_by_tag():
    out = {}
    for f in glob.glob(f"{LOGS}/*.out"):
        tag = seed = nd = None
        with open(f, errors="ignore") as fh:
            for line in fh:
                if line.startswith("Demo path:"):  tag = line.strip().split("/")[-1]
                elif line.startswith("N demos:"):  nd = int(line.split()[2])
                elif line.startswith("Seed:"):     seed = int(line.split()[1]); break
        if tag and seed is not None:
            job = os.path.basename(f).rsplit("_", 1)[1][:-4]
            key = (tag, nd)
            prev = out.setdefault(key, {}).get(seed)
            if prev is None or int(job) > int(prev):
                out[key][seed] = job
    return out


def evals(jobs):
    v = [float(np.mean(json.load(open(p))["ep_rewards"])) for j in jobs
         if os.path.exists(p := f"{GA}/{j}/eval_data/agent_alignment.json")]
    return np.array(v)


def main():
    dur = demo_durations()
    rank = sorted(csv.DictReader(open(RANK)), key=lambda r: int(r["rank"]))
    order = [int(r["demo_id"]) for r in rank]
    runs = gail_runs_by_tag()

    fig, ax = plt.subplots(figsize=(11, 6))

    # ---------- GAIL, return-only ranking ----------
    print("GAIL return-only ranking (human demos, measured time):")
    gx, gy, ge, gn = [], [], [], []
    for K in [10, 50, 100, 250, 699]:
        tag = "session_3_ext700" if K == 699 else f"session_3_ext700_ret_only_top{K}"
        jb = runs.get((tag, K), {})
        v = evals(jb.values())
        if not len(v): continue
        t = dur[order[:K]].sum() / 60.0
        gx.append(t); gy.append(v.mean()); ge.append(v.std()); gn.append((K, len(v)))
        print(f"  K={K:>4d}  n={len(v):2d}  mean={v.mean():+7.2f} ±{v.std():5.2f}   human={t:6.2f} min")
    ax.errorbar(gx, gy, yerr=ge, fmt="-s", color=GAIL_C, markersize=7, capsize=4, linewidth=1.6,
                label="GAIL count axis (10 seeds, return-ranked human demos, measured time)")
    for (K, _), x, y in zip(gn, gx, gy):
        ax.annotate(f"N={K}", (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color=GAIL_C)

    # ---------- PT, human preferences (v2) ----------
    print("\nPT human preferences (v2, fixed labels, measured time):")
    w = pickle.load(open(POOL, "rb"))
    pa, pb, ts = w["pair_starts_a"], w["pair_starts_b"], w["time_sec"]
    ptime = {(int(pa[i]), int(pb[i])): float(ts[i]) for i in range(len(pa))}
    px, py, pe, pn = [], [], [], []
    for N in [10, 50, 100, 750, 1000]:
        vals, secs = [], []
        for s in range(30):
            p = f"{PT_RUNS}/lunarlander-human-count-v2-N{N}/seed_{s}/eval_summary.json"
            if not os.path.exists(p): continue
            vals.append(float(json.load(open(p))["last10_eval_reward"]))
            d = f"{LBL}/lunarlander-human-count-v2-N{N}-s{s}"
            A = pickle.load(open(f"{d}/indices_num{N}_q100", "rb"))
            B = pickle.load(open(f"{d}/indices_2_num{N}_q100", "rb"))
            secs.append(sum(ptime[(int(x), int(y))] for x, y in zip(A, B)))
        if not vals: continue
        v = np.array(vals); t = np.mean(secs) / 60.0
        px.append(t); py.append(v.mean()); pe.append(v.std()); pn.append((N, len(v)))
        print(f"  N={N:>4d}  n={len(v):2d}  mean={v.mean():+7.2f} ±{v.std():5.2f}   human={t:6.2f} min")
    ax.errorbar(px, py, yerr=pe, fmt="-o", color=PT_C, markersize=8, capsize=4, linewidth=1.8,
                label="PT count axis (30 seeds, human preferences, measured time)")
    for (N, _), x, y in zip(pn, px, py):
        ax.annotate(f"N={N}", (x, y), textcoords="offset points", xytext=(6, 8),
                    fontsize=8, color=PT_C, fontweight="bold")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Human Time (min)")
    ax.set_ylabel("Mean Eval Reward (mean over seeds)")
    ax.set_title("Count axis — PT vs GAIL on HUMAN supervision (measured human time)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "/home/marzii/IRL3/figures/count_axis_human_pt_vs_gail_retonly.png"
    fig.savefig(out, dpi=150)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
