"""Build one 2-feature ranking per consistency metric: return (higher better) + that metric
(lower better), equal-weighted mean of oriented percentiles -- the same scheme as
demo_features.add_ranking, just restricted to 2 features.

Writes one ranking CSV per metric, optionally materializes the top-K demo subset for training,
and reports how well each ranking separates landed from crashed demos.

    python scripts/curation/build_2feat_rankings.py --features_csv <f.csv> [--topk 10] [--make_subsets]
"""
import argparse, csv, os
import numpy as np
from scipy.stats import rankdata

METRICS = ["action_divergence", "action_entropy",
           "self_action_divergence", "self_action_entropy",
           "cross_demo_action_divergence", "cross_demo_action_entropy",
           "action_divergence_xy", "action_entropy_xy"]
SHORT = {"action_divergence": "ad", "action_entropy": "ae",
         "self_action_divergence": "sad", "self_action_entropy": "sae",
         "cross_demo_action_divergence": "cdad", "cross_demo_action_entropy": "cdae",
         "action_divergence_xy": "adxy", "action_entropy_xy": "aexy"}


def rank_two(ret, met):
    """mean of (percentile of return, percentile of -metric); rank 1 = best."""
    n = len(ret)
    p_ret = (rankdata(ret) - 1) / (n - 1)              # higher return = better
    p_met = 1 - (rankdata(met) - 1) / (n - 1)          # lower metric = better
    score = (p_ret + p_met) / 2
    order = np.argsort(-score)                         # best first
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    return score, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--out_dir", default="/scratch/marzii/imitation_runs/demos/human_baseline/"
                                         "record_human_demos/lunarlander/rankings_2feat")
    ap.add_argument("--pool", default="/scratch/marzii/imitation_runs/demos/human_baseline/"
                                      "record_human_demos/lunarlander/session_3_ext700")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--make_subsets", action="store_true")
    ap.add_argument("--metrics", default="", help="comma list; default = all in METRICS")
    ap.add_argument("--tag_suffix", default="", help="appended to the short tag, e.g. k50")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.features_csv)))
    n = len(rows)
    did = np.array([int(r["demo_id"]) for r in rows])
    ret = np.array([float(r["return"]) for r in rows])
    outc = np.array([r["outcome"] for r in rows])
    landed = outc == "landed"
    os.makedirs(a.out_dir, exist_ok=True)

    print(f"pool: {n} demos, {landed.sum()} landed / {(~landed).sum()} not\n")
    hdr = (f"{'ranking (return + X)':>34} | {'top-K landed':>12} | {'AUC':>5} | "
           f"{'mean rank L':>11} | {'mean rank C':>11} | perfect?")
    print(hdr); print("-" * len(hdr))

    for m in (a.metrics.split(",") if a.metrics else METRICS):
        met = np.array([float(r[m]) for r in rows])
        score, rank = rank_two(ret, met)

        tag = SHORT[m] + a.tag_suffix
        out = os.path.join(a.out_dir, f"rank_return_{tag}.csv")
        order = np.argsort(rank)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "demo_id", "rank_score", "outcome", "return", m])
            for i in order:
                w.writerow([rank[i], did[i], f"{score[i]:.6f}", outc[i],
                            f"{ret[i]:.3f}", f"{met[i]:.6f}"])

        top = rank <= a.topk
        # AUC = P(a random landed demo ranks better than a random crashed one)
        auc = (rankdata(score)[landed].mean() - (landed.sum() + 1) / 2) / (~landed).sum()
        perfect = rank[landed].max() < rank[~landed].min()
        print(f"{'return + ' + m:>34} | {int(landed[top].sum()):>7}/{a.topk:<4} | {auc:>5.3f} | "
              f"{rank[landed].mean():>11.1f} | {rank[~landed].mean():>11.1f} | {perfect}")

        if a.make_subsets:
            import datasets; datasets.disable_progress_bar()
            ds = datasets.load_from_disk(a.pool)
            ids = [int(did[i]) for i in np.argsort(rank)[:a.topk]]
            sub = os.path.join(os.path.dirname(a.pool),
                               f"session_3_ext700_ret_{tag}_top{a.topk}")
            ds.select(ids).save_to_disk(sub)

    print(f"\nranking CSVs -> {a.out_dir}")
    if a.make_subsets:
        print(f"top-{a.topk} subsets -> {os.path.dirname(a.pool)}/session_3_ext700_ret_<tag>_top{a.topk}")
    print("\nAUC = P(random landed demo outranks a random crashed one); 0.5 = no signal, 1.0 = perfect")
    print("perfect? = every landed demo ranked above every crashed demo")


if __name__ == "__main__":
    main()
