"""Count axis on HUMAN supervision: PT preferences vs GAIL demos, x = ACTUAL human time.

x  PT   : measured sum of per-pair decision time (`time_sec`) over exactly the pairs each
          seed was shown -- recovered from labels_n1000_clean.pkl.
x  GAIL : measured sum of per-episode `duration_sec` over exactly the demos each ranking
          selected -- recovered from the source sessions' timing.csv. Falls back to the
          pool mean (11.43 s/demo) only if the pool-order mapping fails its check.
y  PT   : last10_eval_reward.       y GAIL : 50-episode deterministic post-training eval.
band    : +/-1 std over seeds.      labels : N preferences / K demos.

NOTE ON "RETURN-RANKED": the pure return-only ranking exists at K=10 ONLY, so it is drawn
as a single point. The nearest return-based ranking with a real count axis is `ra`
(return + action divergence, 50/50), shown across K = 5..500.

    python scripts/curation/plot_count_axis_pt_vs_gail_human.py
"""
import argparse, csv, glob, json, os, pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DR = "/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos"
GA = "/scratch/marzii/imitation_runs/gail/lunarlander"
LOGS = "/scratch/marzii/imitation_runs/_slurm_logs/gail/lunarlander"
PT_RUNS = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
POOL = "/scratch/marzii/PT/lunarlander/human_pref_clean/labels_n1000_clean.pkl"
LBL = "/scratch/marzii/PT/human_label/human"
FEAT = f"{DR}/lunarlander/session_3_ext700_features.csv"
SESSIONS = [("lunarlander/session_3", 200), ("lunarlander/session_4", 100),
            ("lunarlander_batch5/session_1", 399)]


def demo_durations():
    """pool index -> recorded duration_sec, verified against the features CSV returns."""
    dur, rew = [], []
    for sub, n in SESSIONS:
        rows = list(csv.DictReader(open(f"{DR}/{sub}/timing.csv")))
        assert len(rows) == n, f"{sub}: {len(rows)} != {n}"
        dur += [float(r["duration_sec"]) for r in rows]
        rew += [float(r["ep_reward"]) for r in rows]
    dur, rew = np.array(dur), np.array(rew)
    feat = list(csv.DictReader(open(FEAT)))
    fret = np.array([float(r["return"]) for r in feat])
    ok = len(fret) == len(rew) and np.allclose(fret, rew, atol=1.0)
    print(f"  [timing] pool-order check vs features CSV returns: {'PASS' if ok else 'FAIL'}"
          f"  (n={len(dur)}, total={dur.sum()/60:.1f} min, mean={dur.mean():.2f} s/demo)")
    return dur, ok


def gail_runs():
    """demo-dir basename -> {seed: jobid}"""
    out = {}
    for f in glob.glob(f"{LOGS}/*.out"):
        tag = seed = None
        with open(f, errors="ignore") as fh:
            for line in fh:
                if line.startswith("Demo path:"):
                    tag = line.strip().split("/")[-1]
                elif line.startswith("Seed:"):
                    seed = int(line.split()[1]); break
        if tag and seed is not None:
            job = os.path.basename(f).rsplit("_", 1)[1][:-4]
            prev = out.setdefault(tag, {}).get(seed)
            if prev is None or int(job) > int(prev):
                out[tag][seed] = job
    return out


def gail_eval(jobs):
    v = [json.load(open(p))["ep_reward_mean"] for j in jobs
         if os.path.exists(p := f"{GA}/{j}/eval_data/agent_alignment.json")]
    return (np.mean(v), np.std(v), len(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/marzii/IRL3/figures/count_axis_pt_vs_gail_human.png")
    ap.add_argument("--pt_cond", default="lunarlander-human-count-v2-N{N}")
    a = ap.parse_args()

    dur, ok = demo_durations()
    mean_s = dur.mean()
    rank = np.argsort(-np.array([float(r["rank_score"]) for r in csv.DictReader(open(FEAT))])) \
        if "rank_score" in open(FEAT).readline() else None

    runs = gail_runs()
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # ---- GAIL series -------------------------------------------------------
    print(f"\n{'series':>26} | K -> eval (n)")
    for tag, pat, col, mk, lab in [
        ("ra",       "session_3_ext700_ra_top{K}",       "#d62728", "s",
         "GAIL demos — return + action divergence"),
        ("ret_only", "session_3_ext700_ret_only_top{K}", "#8c564b", "o",
         "GAIL demos — return ONLY (single K available)"),
    ]:
        pts = []
        for K in [5, 10, 30, 50, 100, 200, 300, 500]:
            jb = runs.get(pat.format(K=K))
            if not jb: continue
            st = gail_eval(jb.values())
            if st: pts.append((K, *st))
        if not pts: continue
        Ks = [p[0] for p in pts]
        x = np.array([K * mean_s / 60 for K in Ks])
        y = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
        ax.plot(x, y, mk + "-", color=col, lw=2.0, ms=8, label=lab, zorder=4)
        if len(x) > 1:
            ax.fill_between(x, y - e, y + e, color=col, alpha=0.12, lw=0, zorder=1)
        else:
            ax.errorbar(x, y, yerr=e, color=col, capsize=4, lw=2, zorder=4)
        for K, xi, yi in zip(Ks, x, y):
            ax.annotate(f"{K}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, color=col)
        print(f"{tag:>26} | " + "  ".join(f"{p[0]}:{p[1]:+.0f}({p[3]})" for p in pts))

    # ---- PT series ---------------------------------------------------------
    w = pickle.load(open(POOL, "rb"))
    pa, pb, ts = w["pair_starts_a"], w["pair_starts_b"], w["time_sec"]
    ptime = {(int(pa[i]), int(pb[i])): float(ts[i]) for i in range(len(pa))}
    xs, ys, es, ns = [], [], [], []
    for N in [10, 50, 100, 750, 1000]:
        vals, secs = [], []
        for s in range(30):
            p = f"{PT_RUNS}/{a.pt_cond.format(N=N)}/seed_{s}/eval_summary.json"
            if not os.path.exists(p): continue
            vals.append(float(json.load(open(p))["last10_eval_reward"]))
            d = f"{LBL}/lunarlander-human-count-v2-N{N}-s{s}"
            A = pickle.load(open(f"{d}/indices_num{N}_q100", "rb"))
            B = pickle.load(open(f"{d}/indices_2_num{N}_q100", "rb"))
            secs.append(sum(ptime[(int(x), int(y))] for x, y in zip(A, B)))
        if not vals: continue
        xs.append(np.mean(secs) / 60); ys.append(np.mean(vals)); es.append(np.std(vals)); ns.append((N, len(vals)))
    if xs:
        x, y, e = np.array(xs), np.array(ys), np.array(es)
        ax.plot(x, y, "P--", color="#000000", lw=2.4, ms=10, zorder=5,
                label="PT preferences — human (v2, fixed labels)")
        ax.fill_between(x, y - e, y + e, color="#000000", alpha=0.10, lw=0, zorder=1)
        for (N, n), xi, yi in zip(ns, x, y):
            ax.annotate(f"{N}", (xi, yi), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8, fontweight="bold")
        print(f"{'PT human v2':>26} | " + "  ".join(f"{N}:{yi:+.0f}({n})@{xi:.1f}min"
                                                   for (N, n), yi, xi in zip(ns, y, x)))

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.axhline(200, color="gray", ls=":", alpha=0.3)
    ax.set_xscale("log")
    src = "measured per-demo / per-preference time" if ok else "PT measured; GAIL = pool mean"
    ax.set_xlabel(f"Human time (min, log scale) — {src}; labels = N preferences / K demos")
    ax.set_ylabel("Mean eval reward (true env)")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    ax.set_title("Count axis on HUMAN supervision — PT preferences vs GAIL demos\n"
                 "eval reward vs actual human time; band = ±1 std over seeds", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150); print("\nsaved", a.out)


if __name__ == "__main__":
    main()
