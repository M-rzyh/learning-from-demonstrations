"""Count/human-time axis (LINEAR x) -- post-training EVAL reward vs true recorded human time.

True recorded rates (project handoff, not estimates):
  PT    : 1000 preferences took 151 min -> 9.06 s / preference
  demos : 300 demos took 54 min        -> 10.8 s / demo   (GAIL + BC warm-start)

Series:
  PT human                       PrefTransformer, N prefs (fixedlabels set)
  GAIL human (return)            return-only top-K (K=10/50/100/250)
  GAIL human (return+AD)         return + action_divergence, RA K-sweep
  GAIL (satisficing+AD)          satad8 top-K
  BC warm-start                  BC(top-K 9-feat) -> PPO true reward

y = mean over seeds of each run's 50-episode deterministic eval; band = +/-1 std.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_count_eval_5methods.py
"""
import csv, glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_2feat_learning_curves import discover, GA

BW = "/scratch/marzii/imitation_runs/bc_warmstart/lunarlander"
PT = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
EXP = "/home/marzii/IRL3/experiments"
# MEASURED human time (validated 2026-09-04 with a live 10-demo session: 19.97 steps/s):
#   demos : per-episode duration_sec from the 3 sessions' timing.csv, summed over the EXACT
#           demos each ranking selects (demo_id-aligned; session_3 + session_4 + batch5)
#   PT    : per-pair time_sec from labels_n1000_clean.pkl, summed over the pairs each seed saw
DR = "/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos"
RANKDIR = f"{DR}/lunarlander/rankings_2feat"
POOL_PKL = "/scratch/marzii/PT/lunarlander/human_pref_clean/labels_n1000_clean.pkl"
LBL_FIXED = "/scratch/marzii/PT/human_label/human_fixed"
SESSIONS = [("lunarlander/session_3", 200), ("lunarlander/session_4", 100),
            ("lunarlander_batch5/session_1", 399)]


def demo_durations():
    dur = []
    for sub, n in SESSIONS:
        rows = list(csv.DictReader(open(f"{DR}/{sub}/timing.csv")))
        assert len(rows) == n
        dur += [float(r["duration_sec"]) for r in rows]
    return np.array(dur)


def ranking_minutes(dur, rank_csv):
    """{K: measured minutes to record that ranking's top-K}; demo_id indexes dur."""
    ids = [int(r["demo_id"]) for r in
           sorted(csv.DictReader(open(rank_csv)), key=lambda r: int(r["rank"]))]
    return lambda K: (dur.sum() if K == 699 else dur[ids[:K]].sum()) / 60.0


def gail_eval(jobs):
    v = [json.load(open(f))["ep_reward_mean"] for j in jobs
         if os.path.exists(f := f"{GA}/{j}/eval_data/agent_alignment.json")]
    return (np.mean(v), np.std(v), len(v)) if v else None


def gail_ksweep(idx, kcol="K", jcol="slurm_job_id"):
    per = {}
    for r in csv.DictReader(open(idx)):
        per.setdefault(int(r[kcol]), []).append(r[jcol].strip())
    return {K: s for K in sorted(per) if (s := gail_eval(per[K]))}


def fullpool_jobs():
    """K=699 = the whole pool -- one shared cell for every ranking. RA-sweep seeds 0-4
    plus the 2026-09-03 top-up seeds 5-9 (gail_satad8_ext index, condition 'fullpool')."""
    jobs = [r["slurm_job_id"].strip()
            for r in csv.DictReader(open(f"{EXP}/GAIL/gail_session3_ext700_RA_ksweep_2026-08-13.csv"))
            if r["K"] == "699"]
    fx = f"{EXP}/GAIL/gail_satad8_ext_2026-09-03.csv"
    if os.path.exists(fx):
        jobs += [r["slurm_job_id"].strip() for r in csv.DictReader(open(fx))
                 if r["condition"] == "fullpool"]
    return jobs


def main():
    import pickle
    runs = discover()
    fp699 = gail_eval(fullpool_jobs())
    dur = demo_durations()
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # ---- PT human (fixedlabels), x = measured per-pair seconds of the pairs each seed saw ----
    w = pickle.load(open(POOL_PKL, "rb"))
    ptime = {(int(a), int(b)): float(t) for a, b, t in
             zip(w["pair_starts_a"], w["pair_starts_b"], w["time_sec"])}
    pt, ptx = {}, {}
    for N in [10, 50, 100, 750, 1000]:
        vals, secs = [], []
        for f in glob.glob(f"{PT}/lunarlander-human-count-N{N}-fixedlabels/seed_*/eval_summary.json"):
            vals.append(json.load(open(f))["last10_eval_reward"])
            seed = f.split("seed_")[1].split("/")[0]
            d = f"{LBL_FIXED}/lunarlander-human-count-N{N}-s{seed}"
            A = pickle.load(open(f"{d}/indices_num{N}_q100", "rb"))
            B = pickle.load(open(f"{d}/indices_2_num{N}_q100", "rb"))
            secs.append(sum(ptime[(int(x), int(y))] for x, y in zip(A, B)))
        if vals:
            pt[N] = (np.mean(vals), np.std(vals), len(vals))
            ptx[N] = np.mean(secs) / 60.0
    draw(ax, pt, ptx, "PT human (preferences)", "o")

    # ---- GAIL return-only (K sweep, incl. the 2026-09-03 countaxis batch) ----
    ro = {K: s for (t, K) in sorted(runs) if t == "only"
          and (s := gail_eval(runs[(t, K)].values()))}
    if fp699: ro[699] = fp699

    xro = ranking_minutes(dur, f"{RANKDIR}/rank_return_only.csv")
    draw(ax, ro, {K: xro(K) for K in ro}, "GAIL human (return)", "v")

    # ---- GAIL return+AD: RA K-sweep ----
    # discover() sees every ra_top*/ret_ad_top* run incl. seed top-ups (tag 'ad'); the
    # Aug-13 CSV is no longer needed as a source
    ra = {K: s for (t, K) in sorted(runs) if t == "ad" and K != 5
          and (s := gail_eval(runs[(t, K)].values()))}
    if fp699: ra[699] = fp699                # shared full-pool endpoint (same runs on all GAIL curves)
    xra = ranking_minutes(dur, f"{RANKDIR}/rank_return_ad.csv")
    draw(ax, ra, {K: xra(K) for K in ra}, "GAIL human (return+AD)", "s")

    # ---- GAIL satisficing+AD (satad8) ----
    sp = {K: s for K in (10, 50, 100, 200, 300)
          if (s := gail_eval(runs.get(("satpad", K), {}).values()))}
    if fp699: sp[699] = fp699
    xsp = ranking_minutes(dur, f"{RANKDIR}/rank_satpad.csv")
    draw(ax, sp, {K: xsp(K) for K in sp}, "GAIL (satisficing only)", "P")

    sa = {K: s for K in (10, 50, 100, 200, 300)
          if (s := gail_eval(runs.get(("satad8", K), {}).values()))}
    if fp699: sa[699] = fp699

    xsa = ranking_minutes(dur, f"{RANKDIR}/rank_satad8.csv")
    draw(ax, sa, {K: xsa(K) for K in sa}, "GAIL (satisficing+AD)", "*")

    # ---- BC warm-start ----
    bc = {}
    per = {}
    for f in ("bc_warmstart_ext700_2026-08-13.csv", "bc_warmstart_topup_2026-09-03.csv"):
        if not os.path.exists(f"{EXP}/BC/{f}"):
            continue
        for r in csv.DictReader(open(f"{EXP}/BC/{f}")):
            per.setdefault(int(r["K"]), []).append(r["jobid"].strip())
    for K, js in per.items():
        v = [json.load(open(f))["ep_reward_mean"] for j in sorted(set(js))
             if os.path.exists(f := f"{BW}/{j}/eval_data/meta.json")]
        if v: bc[K] = (np.mean(v), np.std(v), len(v))
    # BC used the 9-feature ranking's top-K (session_3_ext700_top{K})
    feat = sorted(csv.DictReader(open(f"{DR}/lunarlander/session_3_ext700_features.csv")),
                  key=lambda r: int(r["rank"]))
    bids = [int(r["demo_id"]) for r in feat]
    xbc = {K: (dur.sum() if K == 699 else dur[bids[:K]].sum()) / 60.0 for K in bc}
    draw(ax, dict(sorted(bc.items())), xbc, "BC warm-start", "D")

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.axhline(200, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("Human time (min)  --  MEASURED per-pair / per-demo durations; "
                  "point labels = N prefs / K demos")
    ax.set_ylabel("Mean eval reward (true env, 50-episode deterministic)")
    ax.grid(True, alpha=0.25); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Human supervision on LunarLander -- eval reward vs true recorded human time",
                 fontsize=12)
    fig.tight_layout()
    out = "/home/marzii/IRL3/figures/ext700/count_eval_5methods.png"
    fig.savefig(out, dpi=150); print("saved", out)


def draw(ax, d, xmap, label, marker):
    if not d:
        print(f"  {label}: no data"); return
    Ks = sorted(d)
    x = np.array([xmap[k] for k in Ks])
    y = np.array([d[k][0] for k in Ks]); e = np.array([d[k][1] for k in Ks])
    line, = ax.plot(x, y, marker + "-", lw=2.0, ms=8, label=label, zorder=4)
    ax.fill_between(x, y - e, y + e, color=line.get_color(), alpha=0.12, lw=0, zorder=1)
    for k, xi, yi in zip(Ks, x, y):
        ax.annotate(f"{k}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=line.get_color())
    print(f"  {label}: " + "  ".join(f"{k}:{d[k][0]:+.0f}@{xmap[k]:.1f}min" for k in Ks))


if __name__ == "__main__":
    main()
