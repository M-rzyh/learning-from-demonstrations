"""Does curation lower the REALIZABILITY ceiling, or does it only remove crashes?

GAIL on the ranked top-K subsets shows an inverted U (peak ~+178 at K=50-100, down to +97 at
K=699). Two candidate explanations:
  (a) curation removes the multi-modality -> the demos become realizable by one policy;
  (b) curation only removes crashes, and the surviving landings are just as inconsistent.

Measure the irreducible BC error (leave-one-demo-out kNN plurality error, see
multimodality_analysis.py) on each top-K subset. To keep the estimate comparable across K, every
subset is subsampled to the SAME number of demos -- only *which* demos changes, not how many.

    python scripts/curation/curation_bc_ceiling.py
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multimodality_analysis import load_flat, knn_action_stats, n_demos, EXPERT  # noqa: E402

ROOT = "/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander"
BASE = "session_3_ext700"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="30,50,100,200,300,500,699")
    ap.add_argument("--n_take", type=int, default=30, help="demos sampled from every subset")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out_json", default="/home/marzii/IRL3/figures/ext700/curation_bc_ceiling.json")
    a = ap.parse_args()

    # pooled scaler over the FULL human pool + expert, so every K uses one shared metric
    full = load_flat(os.path.join(ROOT, BASE))
    exp = load_flat(EXPERT)
    P = np.vstack([full["S"], exp["S"]])
    scaler = (P.mean(0), P.std(0) + 1e-8)

    res = {"n_take": a.n_take, "k": a.k, "seeds": a.seeds, "by_K": {}}
    for K in [int(v) for v in a.ks.split(",")]:
        path = os.path.join(ROOT, BASE if K >= 699 else f"{BASE}_top{K}")
        if not os.path.isdir(path):
            print(f"  K={K}: missing {path}, skipped"); continue
        N = n_demos(path)
        errs, divs, rets, land = [], [], [], []
        for s in range(a.seeds):
            rng = np.random.default_rng(1000 + s)
            ids = rng.choice(N, size=min(a.n_take, N), replace=False)
            d = load_flat(path, ids)
            st = knn_action_stats(d, scaler, k=a.k)
            m = st["used"] > 0
            errs.append(float(st["err"][m].mean())); divs.append(float(st["div"][m].mean()))
            rets.append(float(d["rets"].mean())); land.append(float((d["lasts"] >= 99).mean()))
        res["by_K"][str(K)] = dict(
            n_in_subset=int(N),
            err=dict(mean=float(np.mean(errs)), std=float(np.std(errs))),
            div=dict(mean=float(np.mean(divs)), std=float(np.std(divs))),
            demo_return=float(np.mean(rets)), landed_frac=float(np.mean(land)))
        r = res["by_K"][str(K)]
        print(f"  K={K:4d} (n={N:3d})  BCerr {r['err']['mean']:.3f}+-{r['err']['std']:.3f}"
              f"   div {r['div']['mean']:.3f}   demo return {r['demo_return']:+7.1f}"
              f"   landed {100*r['landed_frac']:.0f}%", flush=True)

    # expert reference at the same demo count
    e_err = []
    for s in range(a.seeds):
        rng = np.random.default_rng(2000 + s)
        d = load_flat(EXPERT, rng.choice(exp["n_demos"], size=a.n_take, replace=False))
        st = knn_action_stats(d, scaler, k=a.k); m = st["used"] > 0
        e_err.append(float(st["err"][m].mean()))
    res["expert_reference"] = dict(mean=float(np.mean(e_err)), std=float(np.std(e_err)))
    print(f"  expert reference ({a.n_take} demos): {np.mean(e_err):.3f}+-{np.std(e_err):.3f}")

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(res, open(a.out_json, "w"), indent=2)
    print("wrote", a.out_json)


if __name__ == "__main__":
    main()
