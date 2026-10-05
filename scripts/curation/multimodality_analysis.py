"""Characterize the *multi-modality* of human LunarLander demos (not just summarize it).

`demo_features.add_action_divergence` gives ONE number per demo (mean cross-demo action
disagreement). This script opens that number up:

  1. IRREDUCIBLE BC ERROR — leave-one-demo-out kNN action prediction. The error of the best
     nonparametric Markov policy pi(a|s) on the demo set itself: a lower bound on what ANY
     BC/GAIL policy can achieve on 0-1 action loss. Realizability, in error units.
  2. STATE-CONDITIONAL ACTION DISTRIBUTIONS — KMeans cells on pooled normalized state; per-cell
     p(a|s) for human vs expert. Shows the modes instead of averaging them away.
  3. WHERE the multi-modality lives — normalized entropy of p(a|s) over the (x, y) plane.
  4. IS IT PER-STEP NOISE OR A PER-DEMO MODE? — within a state cell, do individual demos commit
     to one action (trajectory-level latent mode) or resample every visit (per-step noise)?
     Observed within-demo purity vs a within-cell permutation baseline.
  5. SATISFICING — where the humans actually come to rest, and how much reward that still pays.

All comparisons are DEMO-COUNT MATCHED (human subsampled to the expert's n, several seeds), since
divergence estimates depend on how densely the state space is covered.

    python scripts/curation/multimodality_analysis.py --out_json <f.json> [--quick]
"""
import argparse, json, os
import numpy as np

HUMAN = "/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander/session_3_ext700"
EXPERT = "/scratch/marzii/imitation_runs/demos/noisy_demos/lunarlander/expert_4615187/n100_p0_clean"
NA = 4                     # LunarLander-v2 discrete actions: noop, left, main, right
ACT_NAMES = ["noop", "left engine", "main engine", "right engine"]
STATE_DIM = 6              # x, y, vx, vy, angle, angvel  (leg-contact flags excluded)


# ---------------------------------------------------------------- loading

def load_flat(path, demo_ids=None):
    """-> dict of flat per-transition arrays S (n,6), A (n,), DID (n,), T (n,) + per-demo info."""
    import datasets; datasets.disable_progress_bar()
    ds = datasets.load_from_disk(path)
    ids = range(len(ds)) if demo_ids is None else demo_ids
    S, A, DID, T, finals, rets, lasts = [], [], [], [], [], [], []
    for j, i in enumerate(ids):
        ep = ds[int(i)]
        obs = np.asarray(ep["obs"], float); acts = np.asarray(ep["acts"]).astype(int)
        rews = np.asarray(ep["rews"], float)
        S.append(obs[:len(acts), :STATE_DIM]); A.append(acts)
        DID.append(np.full(len(acts), j)); T.append(np.arange(len(acts)))
        finals.append(obs[-1]); rets.append(float(rews.sum()))
        lasts.append(float(rews[-1]) if len(rews) else 0.0)
    return dict(S=np.vstack(S), A=np.concatenate(A), DID=np.concatenate(DID),
                T=np.concatenate(T), finals=np.array(finals), rets=np.array(rets),
                lasts=np.array(lasts), n_demos=len(finals))


def n_demos(path):
    import datasets; datasets.disable_progress_bar()
    return len(datasets.load_from_disk(path))


# ---------------------------------------------------------------- 1. kNN realizability

def knn_action_stats(d, scaler, k=15, extra=60):
    """Leave-one-DEMO-out kNN over cross-demo neighbours only.

    Returns per-transition arrays:
      div     - fraction of the k cross-demo neighbours taking a different action
                (identical definition to demo_features.add_action_divergence)
      err     - 1 if this transition's action != the plurality action of its k neighbours.
                Mean(err) = leave-one-demo-out kNN classification error = the error rate of the
                best nonparametric Markov policy => an estimate of the IRREDUCIBLE BC error.
      ent     - normalized entropy (base NA) of the neighbour action distribution
      p       - (n, NA) neighbour action distribution
    """
    from scipy.spatial import cKDTree
    S, A, DID = d["S"], d["A"], d["DID"]
    Sn = (S - scaler[0]) / scaler[1]
    tree = cKDTree(Sn)
    kq = min(k + extra, len(Sn))
    _, idx = tree.query(Sn, k=kq, workers=-1)
    n = len(S)
    div = np.zeros(n); err = np.zeros(n); ent = np.zeros(n); p = np.zeros((n, NA))
    used = np.zeros(n, int)
    for i in range(n):
        neigh = idx[i][1:]
        other = neigh[DID[neigh] != DID[i]][:k]
        used[i] = len(other)
        if len(other) == 0:
            continue
        a = A[other]
        cnt = np.bincount(a, minlength=NA).astype(float)
        pi = cnt / cnt.sum()
        p[i] = pi
        div[i] = float(np.mean(a != A[i]))
        err[i] = float(A[i] != int(np.argmax(cnt)))
        nz = pi[pi > 0]
        ent[i] = float(-(nz * np.log(nz)).sum() / np.log(NA))
    return dict(div=div, err=err, ent=ent, p=p, used=used)


def matched_summary(path, n_take, scaler, k, seeds, rng_base=0):
    """Subsample `n_take` demos `seeds` times; report mean/std of the three headline numbers."""
    N = n_demos(path)
    out = {"div": [], "err": [], "ent": []}
    for s in range(seeds):
        rng = np.random.default_rng(rng_base + s)
        ids = rng.choice(N, size=min(n_take, N), replace=False) if n_take < N else np.arange(N)
        d = load_flat(path, ids)
        st = knn_action_stats(d, scaler, k=k)
        m = st["used"] > 0
        for key in out:
            out[key].append(float(st[key][m].mean()))
    return {key: dict(mean=float(np.mean(v)), std=float(np.std(v)), runs=v) for key, v in out.items()}


# ---------------------------------------------------------------- 2/4. cells

def fit_cells(Sn_pool, n_cells, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_cells, random_state=seed, n_init=10, batch_size=4096)
    km.fit(Sn_pool)
    return km


def cell_action_dist(cells, A, n_cells):
    """-> counts (n_cells, NA)."""
    C = np.zeros((n_cells, NA))
    np.add.at(C, (cells, A), 1.0)
    return C


def demo_mode_test(cells, A, DID, T, n_cells, min_visits=2, min_gap=20, n_perm=500, seed=0):
    """Is the multi-modality PER-STEP noise or a PER-DEMO latent mode?

    Inside each state cell, keep each demo's visits that are >= `min_gap` env steps apart (so a
    high purity cannot come from the action simply being held across consecutive frames). For
    each (cell, demo) with >= min_visits kept visits, purity = max_a count_a / n_visits.
    Baseline: permute the action labels among all kept visits WITHIN the cell (destroys the
    demo->action association, preserves the cell's action marginal and the visit counts).
    observed >> permuted  =>  each demo commits to a mode  =>  trajectory-level multi-modality.
    """
    rng = np.random.default_rng(seed)
    order = np.lexsort((T, DID, cells))
    c, a, dd, t = cells[order], A[order], DID[order], T[order]
    obs_pur, perm_mat, n_groups, n_cells_used = [], [], 0, 0
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    ends = np.r_[starts[1:], len(c)]
    for s, e in zip(starts, ends):
        ca, cd, ct = a[s:e], dd[s:e], t[s:e]
        keep = []
        last_d, last_t = -1, -10**9
        for j in range(len(ca)):                      # already sorted by (demo, t)
            if cd[j] != last_d:
                last_d, last_t = cd[j], ct[j]; keep.append(j); continue
            if ct[j] - last_t >= min_gap:
                last_t = ct[j]; keep.append(j)
        if len(keep) < 2 * min_visits:
            continue
        keep = np.array(keep)
        ka, kd = ca[keep], cd[keep]
        uq, inv, cnt = np.unique(kd, return_inverse=True, return_counts=True)
        ok = cnt >= min_visits
        if ok.sum() < 2:                               # need >= 2 demos to compare against
            continue
        n_cells_used += 1

        gidx = [np.flatnonzero(inv == gi) for gi in np.flatnonzero(ok)]

        def purity(labels):
            vals = [np.bincount(labels[g], minlength=NA).max() / len(g) for g in gidx]
            return float(np.mean(vals)), len(vals)

        po, ng = purity(ka); obs_pur.append(po); n_groups += ng
        # keep the FULL permutation vector per cell so the null can be aggregated coherently
        perm_mat.append([purity(rng.permutation(ka))[0] for _ in range(n_perm)])
    if not obs_pur:
        return None
    obs = float(np.mean(obs_pur))
    null = np.mean(np.array(perm_mat), axis=0)      # (n_perm,) null draws of the SAME aggregate
    sd = float(null.std())
    return dict(observed=obs, permuted_mean=float(null.mean()), permuted_sd=sd,
                z=float((obs - null.mean()) / sd) if sd > 0 else float("nan"),
                p_two_sided=float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()),
                excess_purity=obs - float(null.mean()),
                n_cells=n_cells_used, n_demo_cell_groups=n_groups, n_perm=n_perm,
                min_gap=min_gap, min_visits=min_visits)


# ---------------------------------------------------------------- 5. satisficing

def satisficing_stats(d):
    fin, rets, lasts = d["finals"], d["rets"], d["lasts"]
    landed = lasts >= 99
    crashed = lasts <= -99
    x = np.abs(fin[:, 0])
    return dict(
        n=int(d["n_demos"]),
        landed=int(landed.sum()), crashed=int(crashed.sum()),
        other=int((~landed & ~crashed).sum()),
        mean_return=float(rets.mean()), median_return=float(np.median(rets)),
        landed_mean_return=float(rets[landed].mean()) if landed.any() else float("nan"),
        landed_x_mean=float(x[landed].mean()) if landed.any() else float("nan"),
        landed_x_median=float(np.median(x[landed])) if landed.any() else float("nan"),
        landed_offpad_frac=float((x[landed] > 0.2).mean()) if landed.any() else float("nan"),
        landed_far_frac=float((x[landed] > 0.4).mean()) if landed.any() else float("nan"),
        landed_touchdown_vy=float(np.abs(fin[landed, 3]).mean()) if landed.any() else float("nan"),
        landed_angle=float(np.abs(fin[landed, 4]).mean()) if landed.any() else float("nan"),
        x_all=x.tolist(), landed_mask=landed.tolist(), returns=rets.tolist(),
        touchdown_vy_all=np.abs(fin[:, 3]).tolist(), land_angle_all=np.abs(fin[:, 4]).tolist(),
    )


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=HUMAN)
    ap.add_argument("--expert", default=EXPERT)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--k_sweep", default="5,15,30,50")
    ap.add_argument("--seeds", type=int, default=5, help="matched-subsample repeats")
    ap.add_argument("--n_cells", type=int, default=600)
    ap.add_argument("--min_cell", type=int, default=50,
                    help="min transitions per cell in BOTH sets to count it")
    ap.add_argument("--out_json", default="/home/marzii/IRL3/figures/ext700/multimodality_stats.json")
    ap.add_argument("--npz", default="/home/marzii/IRL3/figures/ext700/multimodality_arrays.npz")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.k_sweep, args.n_cells = 2, "15", 60

    print("[load] full pools ...", flush=True)
    H = load_flat(args.human)
    E = load_flat(args.expert)
    print(f"  human {H['n_demos']} demos / {len(H['S'])} transitions; "
          f"expert {E['n_demos']} demos / {len(E['S'])} transitions", flush=True)

    # POOLED scaler: identical metric for both sets, so distances mean the same thing.
    P = np.vstack([H["S"], E["S"]])
    scaler = (P.mean(0), P.std(0) + 1e-8)
    res = {"config": vars(args) | {"human_n": H["n_demos"], "expert_n": E["n_demos"],
                                   "human_transitions": int(len(H["S"])),
                                   "expert_transitions": int(len(E["S"]))}}

    # ---- sanity: reproduce demo_features' per-set-normalized action_divergence at k=15
    # demo_features.add_action_divergence uses per-SET normalization, a k+40 query window and
    # reports the mean over DEMOS of per-demo means. Reproduce it exactly as a pipeline check.
    print("[sanity] reproducing demo_features.add_action_divergence ...", flush=True)
    res["sanity_action_divergence_perset"] = {}
    for name, d in (("human", H), ("expert", E)):
        sc = (d["S"].mean(0), d["S"].std(0) + 1e-8)
        st = knn_action_stats(d, sc, k=15, extra=40)
        dm = np.array([st["div"][d["DID"] == i].mean() for i in range(d["n_demos"])])
        res["sanity_action_divergence_perset"][name] = dict(
            demo_mean=float(dm.mean()), step_weighted=float(st["div"][st["used"] > 0].mean()))
        print(f"   {name}: demo-mean {dm.mean():.4f}  step-weighted "
              f"{st['div'][st['used'] > 0].mean():.4f}", flush=True)

    # ---- 1. irreducible BC error, demo-count matched, k sweep
    ks = [int(v) for v in args.k_sweep.split(",")]
    n_take = E["n_demos"]
    res["matched"] = {"n_take": int(n_take), "by_k": {}}
    for k in ks:
        print(f"[matched] k={k} (human subsampled to {n_take} demos x {args.seeds} seeds) ...", flush=True)
        hs = matched_summary(args.human, n_take, scaler, k, args.seeds)
        es = matched_summary(args.expert, n_take, scaler, k, 1)
        res["matched"]["by_k"][str(k)] = {"human": hs, "expert": es}
        print(f"   irreducible BC err  human {hs['err']['mean']:.3f}+-{hs['err']['std']:.3f}"
              f"   expert {es['err']['mean']:.3f}", flush=True)
        print(f"   norm. entropy H(a|s) human {hs['ent']['mean']:.3f}"
              f"   expert {es['ent']['mean']:.3f}", flush=True)

    # ---- full-pool stats (all 699 human demos) at the headline k
    print("[full] all-demo stats at k=%d ..." % args.k, flush=True)
    stH = knn_action_stats(H, scaler, k=args.k)
    stE = knn_action_stats(E, scaler, k=args.k)
    res["full_pool"] = {
        n: dict(irreducible_bc_err=float(s["err"][s["used"] > 0].mean()),
                action_divergence=float(s["div"][s["used"] > 0].mean()),
                norm_entropy=float(s["ent"][s["used"] > 0].mean()),
                frac_states_multimodal=float((s["ent"][s["used"] > 0] > 0.5).mean()))
        for n, s in (("human", stH), ("expert", stE))}
    print("   ", json.dumps(res["full_pool"], indent=2), flush=True)

    # ---- 2/3/4. state cells
    print("[cells] KMeans on pooled normalized state ...", flush=True)
    Pn = (P - scaler[0]) / scaler[1]
    km = fit_cells(Pn, args.n_cells)
    cH = km.predict((H["S"] - scaler[0]) / scaler[1])
    cE = km.predict((E["S"] - scaler[0]) / scaler[1])
    CH = cell_action_dist(cH, H["A"], args.n_cells)
    CE = cell_action_dist(cE, E["A"], args.n_cells)

    def cell_ent(C):
        tot = C.sum(1, keepdims=True)
        p = np.divide(C, np.maximum(tot, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(p > 0, np.log(p), 0.0)
        return -(p * lg).sum(1) / np.log(NA), tot.ravel()
    eH, nH = cell_ent(CH); eE, nE = cell_ent(CE)
    shared = (nH >= args.min_cell) & (nE >= args.min_cell)
    res["cells"] = dict(
        n_cells=args.n_cells, min_cell=args.min_cell, n_shared=int(shared.sum()),
        mean_cell_entropy_human=float(np.average(eH[shared], weights=nH[shared])),
        mean_cell_entropy_expert=float(np.average(eE[shared], weights=nE[shared])),
        # per-cell irreducible error under the cell partition (best constant action per cell)
        cell_bc_err_human=float(1 - CH[shared].max(1).sum() / CH[shared].sum()),
        cell_bc_err_expert=float(1 - CE[shared].max(1).sum() / CE[shared].sum()),
    )
    # Transition-count matched version: max-count concentration is upward-biased in small samples,
    # and the human pool has ~4x the transitions, so subsample it to the expert's count.
    rng = np.random.default_rng(0)
    sub = rng.choice(len(cH), size=len(cE), replace=False)
    CHs = cell_action_dist(cH[sub], H["A"][sub], args.n_cells)
    sh2 = (CHs.sum(1) >= args.min_cell) & (nE >= args.min_cell)
    res["cells"]["matched_transitions"] = dict(
        n_shared=int(sh2.sum()), n_transitions=int(len(cE)),
        cell_bc_err_human=float(1 - CHs[sh2].max(1).sum() / CHs[sh2].sum()),
        cell_bc_err_expert=float(1 - CE[sh2].max(1).sum() / CE[sh2].sum()),
    )
    print("   ", json.dumps(res["cells"], indent=2), flush=True)

    print("[modes] per-demo latent mode vs per-step noise ...", flush=True)
    res["demo_mode_test"] = {
        "human": demo_mode_test(cH, H["A"], H["DID"], H["T"], args.n_cells),
        "expert": demo_mode_test(cE, E["A"], E["DID"], E["T"], args.n_cells),
    }
    print("   ", json.dumps(res["demo_mode_test"], indent=2), flush=True)

    # ---- 5. satisficing
    print("[satisficing] ...", flush=True)
    sH, sE = satisficing_stats(H), satisficing_stats(E)
    res["satisficing"] = {"human": {k: v for k, v in sH.items() if not isinstance(v, list)},
                          "expert": {k: v for k, v in sE.items() if not isinstance(v, list)}}
    print("   ", json.dumps(res["satisficing"], indent=2), flush=True)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(
        args.npz,
        h_xy=H["S"][:, :2], e_xy=E["S"][:, :2], h_ent=stH["ent"], e_ent=stE["ent"],
        h_err=stH["err"], e_err=stE["err"],
        h_used=stH["used"], e_used=stE["used"], h_cells=cH, e_cells=cE,
        CH=CH, CE=CE, cell_centers=km.cluster_centers_,
        h_x=np.array(sH["x_all"]), e_x=np.array(sE["x_all"]),
        h_landed=np.array(sH["landed_mask"]), e_landed=np.array(sE["landed_mask"]),
        h_ret=np.array(sH["returns"]), e_ret=np.array(sE["returns"]),
        h_tvy=np.array(sH["touchdown_vy_all"]), e_tvy=np.array(sE["touchdown_vy_all"]),
        h_ang=np.array(sH["land_angle_all"]), e_ang=np.array(sE["land_angle_all"]),
        scaler_mean=scaler[0], scaler_std=scaler[1],
    )
    print(f"[done] wrote {args.out_json} and {args.npz}")


if __name__ == "__main__":
    main()
