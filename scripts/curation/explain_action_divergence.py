#!/usr/bin/env python3
"""Explain / reproduce demo_features.add_action_divergence, step by step.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/explain_action_divergence.py --toy

  --toy                     tiny 3-demo example, prints every intermediate array
  --session <dir> [--demos N] [--k K]   run on real demos, print per-demo results
  --verify   <dir> [--demos N]          prove this reproduces the real function exactly
"""
import argparse, sys
import numpy as np
from scipy.spatial import cKDTree
np.set_printoptions(precision=3, suppress=True)


def divergence(obs_list, act_list, k=15, verbose=False):
    """obs_list[i], act_list[i] = one demo. Returns (per_state_div, per_demo_div, S, A, DID)."""
    # STEP 1 -- pool all (state, action) pairs, remembering which demo each came from
    print(f"Computing action divergence: {len(obs_list)} demos, {k} neighbors")
    S, A, DID = [], [], []
    for i, (obs, acts) in enumerate(zip(obs_list, act_list)):
        obs = np.asarray(obs, float); acts = np.asarray(acts).astype(int)
        print(f"demo {i}: {len(obs)} states, {len(acts)} actions")
        # print(f"  obs: {obs}, acts: {acts}")
        S.append(obs[:len(acts), :6])       # FIRST 6 DIMS ONLY: x,y,vx,vy,theta,omega
        # print(f"  S: {S[-1]}")
        A.append(acts)                      # (leg_L, leg_R are dropped)
        # print(f"  A: {A[-1]}")
        DID.append(np.full(len(acts), i))   # demo id per row
        # print(f"  DID: {DID[-1]}")
    S = np.vstack(S); A = np.concatenate(A); DID = np.concatenate(DID)

    # STEP 2 -- z-score each dim so no dimension dominates the distance
    Sn = (S - S.mean(0)) / (S.std(0) + 1e-8)
    print(f"  Sn: {Sn}")
    
    # STEP 3 -- k+40 nearest neighbours for every state (over-fetch, filtered next)
    tree = cKDTree(Sn)
    kq = min(k + 40, len(Sn)) # The extra 40 are fetched because some nearest neighbors may belong to the same demo and will later be filtered out.
    _, idx = tree.query(Sn, k=kq)

    # STEP 4 -- per state: of the k nearest states FROM OTHER DEMOS, what fraction
    #           took a different action?
    div = np.empty(len(S))
    for i in range(len(S)):
        neigh = idx[i][1:]                          # drop self
        other = neigh[DID[neigh] != DID[i]][:k]     # other demos only, nearest k
        div[i] = float(np.mean(A[other] != A[i])) if len(other) else 0.0
        if verbose:
            print(f"  row {i}: demo={DID[i]} state0={S[i,0]:+.2f} action={A[i]}  "
                  f"| neighbours {list(other)} actions {list(A[other])} "
                  f"-> {int((A[other]!=A[i]).sum())}/{len(other)} differ = {div[i]:.3f}")

    # STEP 5 -- average over each demo's own rows
    per_demo = np.array([div[DID == i].mean() for i in range(len(obs_list))])
    return div, per_demo, S, A, DID


def load(session, n=None):
    import datasets; datasets.disable_progress_bar()
    ds = datasets.load_from_disk(session)
    n = len(ds) if n is None else min(n, len(ds))
    return [ds[i]["obs"] for i in range(n)], [ds[i]["acts"] for i in range(n)], ds, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--session"); ap.add_argument("--verify")
    ap.add_argument("--demos", type=int, default=None)
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()

    if a.toy:
        obs = [[[0.0,0,0,0,0,0],[1.0,0,0,0,0,0]],
               [[0.1,0,0,0,0,0],[1.1,0,0,0,0,0]],
               [[0.2,0,0,0,0,0],[1.2,0,0,0,0,0]]]
        act = [[0,1],[0,2],[3,1]]
        print("TOY: 3 demos x 2 steps, only dim0 varies, k=2\n")
        div, per_demo, S, A, DID = divergence(obs, act, k=2, verbose=True)
        print("\nper-demo action_divergence:")
        for i, v in enumerate(per_demo):
            print(f"  demo {i}: mean{div[DID==i]} = {v:.3f}")
        return

    if a.session:
        obs, act, ds, n = load(a.session, a.demos)
        div, per_demo, S, A, DID = divergence(obs, act, k=a.k)
        print(f"{n} demos, {len(S):,} states, k={a.k}")
        print(f"POOL MEAN action_divergence = {per_demo.mean():.4f}")
        order = np.argsort(per_demo)
        print("\n most consistent demos:", [f"{i}:{per_demo[i]:.3f}" for i in order[:5]])
        print(" most divergent demos :", [f"{i}:{per_demo[i]:.3f}" for i in order[-5:]])
        return

    if a.verify:
        sys.path.insert(0, "/home/marzii/IRL3/scripts/curation")
        from demo_features import add_action_divergence
        obs, act, ds, n = load(a.verify, a.demos)
        _, mine, _, _, _ = divergence(obs, act, k=15)
        rows = [{"demo_id": i} for i in range(n)]
        add_action_divergence(rows, ds.select(range(n)))     # the REAL function
        theirs = np.array([r["action_divergence"] for r in rows])
        print(f"max abs difference vs demo_features.add_action_divergence: {np.abs(mine-theirs).max():.2e}")
        print("IDENTICAL" if np.allclose(mine, theirs) else "MISMATCH")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
