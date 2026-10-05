"""Gather the post-training eval data for every GAIL run in the 2-feature ranking study.

Runs are found the same way plot_2feat_learning_curves.py finds them (Slurm log -> demo
path -> ranking tag + seed), so nothing needs to be registered by hand.

Writes two CSVs:
  <out>_runs.csv      one row per (ranking, seed): mean/std/median return, landed & crash
                      fractions, action alignment vs the reference expert, job id, paths
  <out>_episodes.csv  one row per evaluated episode: return, length, terminal
                      (8 rankings x 10 seeds x 50 episodes = 4000 rows)

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/export_2feat_eval_data.py
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_2feat_learning_curves import discover, LABEL, ORDER, GA

LAND_THRESH = 200.0     # conventional LunarLander "solved" episode
CRASH_THRESH = 0.0


def episodes(job):
    """Per-episode (return, length, terminal) from the 50-episode post-training eval."""
    f = f"{GA}/{job}/eval_data/agent_rollouts.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    rets = np.array([float(e.sum()) for e in d["rews"]])
    lens = np.array([len(e) for e in d["rews"]], int)
    term = np.asarray(d["terminal"], bool)
    return rets, lens, term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/marzii/IRL3/experiments/GAIL/gail_2feat_eval")
    ap.add_argument("--tags", default="", help="comma list; default = all found")
    ap.add_argument("--topk", type=int, default=0,
                    help="demo count to export; 0 = every K found")
    a = ap.parse_args()

    runs = discover(topk=a.topk or None)
    if a.topk:                                   # discover() keyed by tag alone in this mode
        runs = {(t, a.topk): v for t, v in runs.items()}
    sel = a.tags.split(",") if a.tags else None
    want = sorted((k for k in runs if sel is None or k[0] in sel),
                  key=lambda x: (ORDER.index(x[0]) if x[0] in ORDER else 99, x[0], x[1]))

    run_rows, ep_rows = [], []
    for t, K in want:
        for seed in sorted(runs[(t, K)]):
            job = runs[(t, K)][seed]
            e = episodes(job)
            if e is None:
                print(f"  skip {t} K={K} seed {seed} (job {job}): no eval yet")
                continue
            rets, lens, term = e
            align = None
            af = f"{GA}/{job}/eval_data/agent_alignment.json"
            if os.path.exists(af):
                align = json.load(open(af)).get("alignment_fraction")
            run_rows.append(dict(
                ranking=t, label=LABEL.get(t, t), n_demos=K, seed=seed, job_id=job,
                n_episodes=len(rets),
                eval_mean=round(float(rets.mean()), 3),
                eval_std=round(float(rets.std()), 3),
                eval_median=round(float(np.median(rets)), 3),
                eval_min=round(float(rets.min()), 3),
                eval_max=round(float(rets.max()), 3),
                frac_landed=round(float((rets >= LAND_THRESH).mean()), 4),
                frac_crashed=round(float((rets < CRASH_THRESH).mean()), 4),
                mean_ep_len=round(float(lens.mean()), 1),
                frac_terminal=round(float(term.mean()), 4),
                alignment=None if align is None else round(float(align), 4),
                rollouts_npz=f"{GA}/{job}/eval_data/agent_rollouts.npz",
            ))
            for i, (r, l, tm) in enumerate(zip(rets, lens, term)):
                ep_rows.append(dict(ranking=t, n_demos=K, seed=seed, job_id=job, episode=i,
                                    ep_return=round(float(r), 3), ep_len=int(l),
                                    terminal=bool(tm)))

    for path, rows in [(f"{a.out}_runs.csv", run_rows), (f"{a.out}_episodes.csv", ep_rows)]:
        if not rows:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {path}  ({len(rows)} rows)")

    print(f"\n{'ranking':>26} |  K  | seeds | eval mean +/- std | landed% | crash% | align")
    print("-" * 94)
    for t, K in want:
        rs = [r for r in run_rows if r["ranking"] == t and r["n_demos"] == K]
        if not rs:
            continue
        m = np.array([r["eval_mean"] for r in rs])
        print(f"{LABEL.get(t, t):>26} | {K:3d} | {len(rs):5d} | {m.mean():+7.1f} +/- {m.std():5.1f} |"
              f" {100*np.mean([r['frac_landed'] for r in rs]):6.1f}% |"
              f" {100*np.mean([r['frac_crashed'] for r in rs]):5.1f}% |"
              f" {np.mean([r['alignment'] for r in rs if r['alignment'] is not None]):.3f}")


if __name__ == "__main__":
    main()
