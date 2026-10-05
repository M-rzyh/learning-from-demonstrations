"""Emit a minimal features CSV with action_divergence computed at a chosen kNN neighbour count.

Only the columns build_2feat_rankings needs (demo_id, outcome, return, action_divergence), so
it skips the expensive self_*/cross_demo_* metrics. The neighbour selection reproduces
demo_features.add_action_divergence exactly: same 6-D pool-normalized state, same k+40 candidate
window, same "first k OTHER-demo neighbours, multi-vote allowed" rule -- verified against
rank_return_ad.csv at k=15.

    python scripts/curation/make_divergence_k_features.py --ks 15,50,100
"""
import argparse, csv, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_features as F

POOL = ("/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/"
        "lunarlander/session_3_ext700")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--ks", default="15,50,100", help="comma list of kNN neighbour counts")
    ap.add_argument("--dims", type=int, default=6, choices=(2, 6),
                    help="state dims for the kNN: 6 = action_divergence, 2 = action_divergence_xy")
    ap.add_argument("--out_dir", default=os.path.dirname(POOL))
    a = ap.parse_args()

    import datasets; datasets.disable_progress_bar()
    ds = datasets.load_from_disk(a.pool)
    ks = [int(x) for x in a.ks.split(",")]

    S, A, DID = F._flatten(ds)
    S = S[:, :a.dims]                                  # 2 -> (x, y) only, matching add_action_divergence_X_Y
    col = "action_divergence" if a.dims == 6 else "action_divergence_xy"
    stem = "ad" if a.dims == 6 else "adxy"
    Sn = F._normalize(S)
    tree = cKDTree(Sn)
    print(f"{len(ds)} demos, {len(S)} states")

    base = [dict(demo_id=i,
                 outcome=F.traj_features(ds[i])["outcome"],
                 ret=float(np.sum(ds[i]["rews"]))) for i in range(len(ds))]

    for k in ks:
        # One query per k with EXACTLY the window add_action_divergence uses. Querying once at a
        # wide k and slicing is not equivalent: the pool has duplicate states (seeded resets), and
        # cKDTree's ordering among equidistant neighbours depends on the query width.
        kq = min(k + 40, len(Sn))
        _, idx = tree.query(Sn, k=kq)
        neigh = idx[:, 1:]                             # drop self
        io = DID[neigh] != DID[:, None]
        disag = A[neigh] != A[:, None]
        Wk = neigh.shape[1]
        take = io & (np.cumsum(io, 1) <= k)            # first k OTHER-demo neighbours
        cnt = take.sum(1)
        per_state = np.where(cnt > 0, (take & disag).sum(1) / np.maximum(cnt, 1), 0.0)
        div = np.array([per_state[DID == i].mean() for i in range(len(ds))])
        short = (cnt < k).mean()

        out = os.path.join(a.out_dir, f"session_3_ext700_features_{stem}k{k}.csv")
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["demo_id", "outcome", "return", col])
            for r, d in zip(base, div):
                w.writerow([r["demo_id"], r["outcome"], f"{r['ret']:.3f}", f"{d:.6f}"])
        print(f"  k={k:3d}  mean div={div.mean():.4f}  window={kq:3d}  "
              f"states short of k: {100*short:.1f}%  ->  {os.path.basename(out)}")


if __name__ == "__main__":
    main()
