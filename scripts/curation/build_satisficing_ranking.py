"""Rank the 699 demos by the satisficing-paper cost features alone -- NO return.

Uses the OUTCOME-PADDED features (raw sums under-count early crashes; see the padding notes in
features_satisficing_paper.py) and exactly the paper's 7 LunarLander features: the six squared
state sums + control_cost. Our additions (control_cost_fuel, safety) are excluded.

All 7 are costs (lower = better), so no per-feature direction handling is needed:
    score = mean over the 7 features of (1 - percentile(value));  rank 1 = best.
Same percentile-mean scheme as demo_features.add_ranking / build_2feat_rankings, minus return.

    python scripts/curation/build_satisficing_ranking.py --make_subsets
"""
import argparse, csv, os
import numpy as np
from scipy.stats import rankdata

L = "/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander"
FEATS = ["sum_x2_padded", "sum_y2_padded", "sum_vx2_padded", "sum_vy2_padded",
         "sum_theta2_padded", "sum_omega2_padded", "control_cost_padded"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default=f"{L}/session_3_ext700_features_satisficing_padded.csv")
    ap.add_argument("--pool", default=f"{L}/session_3_ext700")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--topks", default="10,50,100")
    ap.add_argument("--ad_mode", default="none", choices=("none", "block", "vote8"),
                    help="combine with action_divergence: 'block' = (satpad_score + ad_pct)/2, "
                         "'vote8' = ad joins the mean as an 8th feature; ad read from the "
                         "demo-features v2 CSV, GOOD_DIR -1 (lower better)")
    ap.add_argument("--make_subsets", action="store_true")
    a = ap.parse_args()

    stem = {"none": "satpad", "block": "satad", "vote8": "satad8"}[a.ad_mode]
    a.out_csv = a.out_csv or f"{L}/rankings_2feat/rank_{stem}.csv"
    rows = list(csv.DictReader(open(a.features_csv)))
    n = len(rows)
    did = np.array([int(r["demo_id"]) for r in rows])
    ret = np.array([float(r["return"]) for r in rows])          # reference column only
    outc = np.array([r["outcome"] for r in rows]); landed = outc == "landed"

    pct = [(rankdata(np.array([float(r[f]) for r in rows])) - 1) / (n - 1) for f in FEATS]
    score = np.mean([1 - p for p in pct], axis=0)               # all costs: lower value = better
    if a.ad_mode != "none":
        v2 = {int(r["demo_id"]): float(r["action_divergence"])
              for r in csv.DictReader(open(f"{L}/session_3_ext700_features_v2.csv"))}
        ad_score = 1 - (rankdata(np.array([v2[i] for i in did])) - 1) / (n - 1)
        if a.ad_mode == "block":                                # equal weight: costs | consistency
            score = (score + ad_score) / 2
        else:                                                   # vote8: ad = 1 vote in 8
            score = np.mean([1 - p for p in pct] + [ad_score], axis=0)
    order = np.argsort(-score)
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)

    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "demo_id", "rank_score", "outcome", "return"] + FEATS)
        for i in order:
            w.writerow([rank[i], did[i], f"{score[i]:.6f}", outc[i], f"{ret[i]:.3f}"]
                       + [rows[i][f] for f in FEATS])
    auc = (rankdata(score)[landed].mean() - (landed.sum() + 1) / 2) / (~landed).sum()
    print(f"ranking '{stem}' (ad_mode={a.ad_mode}, no return): "
          f"AUC(landed over crashed) = {auc:.3f}")
    for K in [int(x) for x in a.topks.split(",")]:
        top = rank <= K
        print(f"  top-{K:<4} landed {int(landed[top].sum())}/{K}   "
              f"mean return of top-{K}: {ret[top].mean():+.1f}")
        if a.make_subsets:
            import datasets; datasets.disable_progress_bar()
            ds = datasets.load_from_disk(a.pool)
            ids = [int(did[i]) for i in np.argsort(rank)[:K]]
            sub = f"{os.path.dirname(a.pool)}/session_3_ext700_{stem}_top{K}"
            ds.select(ids).save_to_disk(sub)
            print(f"          -> {sub}")


if __name__ == "__main__":
    main()
