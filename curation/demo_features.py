"""Per-trajectory feature extraction + visualization for demonstration curation.

`traj_features(ep)` is the SINGLE reusable extractor (Phase 0 offline AND Phase B online use it,
so thresholds mean the same thing live as in calibration). `GOOD_DIR[f]` = +1 if higher is
better, -1 if lower is better — used for the "good half" percentile filter.

    # build the features CSV + histograms (landed vs crashed overlay, median marked):
    python scripts/curation/demo_features.py --session <session_dir> --out_csv f.csv --plot_dir <dir>
    # demo the AND filter on chosen features at a percentile:
    python scripts/curation/demo_features.py --session <dir> --filter return,angle_mean_abs,x_wander --pct 50 --plot_dir <dir>
"""
import argparse, os
import numpy as np

# feature -> +1 (higher is better) / -1 (lower is better)
GOOD_DIR = {
    "return": +1, "n_steps": -1, "legs_down": +1,
    "angle_mean_abs": -1, "angle_max_abs": -1, "angle_std": -1,
    "angvel_mean_abs": -1,                                  # mean |omega| — rotational stability
    "x_wander": -1, "x_std": -1, "vx_mean_abs": -1,
    "descent_rate": -1,                                     # mean downward speed
    "land_x_offset": -1, "touchdown_vy": -1, "touchdown_vx": -1, "land_angle": -1,
    # Measured at FIRST LEG CONTACT, not at the final frame -- see the note on touchdown_vy in
    # traj_features(). These are the features that actually measure landing softness.
    "contact_vy": -1, "contact_vx": -1, "contact_angle": -1,
    "action_divergence": -1,                                # cross-demo per-state action disagreement
    "action_entropy": -1,                                   # entropy of the SAME neighbour set
    # Action-consistency / multimodality metrics (all lower = more consistent). Same 6-D
    # pool-normalized state as action_divergence; only the NEIGHBOUR SET differs:
    #   self_*        -> neighbours from this demo ONLY  (within-person consistency)
    #   cross_demo_*  -> the k closest DISTINCT other demos, one vote each (across-person)
    # *_divergence asks "was THIS action unusual?"; *_entropy ignores the taken action and asks
    # "is this state ambiguous?".  Recorded only -- NOT in ACTIVE_FEATS, so ranking is unchanged.
    "self_action_divergence": -1,
    "self_action_entropy": -1,
    "cross_demo_action_divergence": -1,
    "cross_demo_action_entropy": -1,
    "action_divergence_xy": -1,
    "action_entropy_xy": -1,
}
FEATURES = list(GOOD_DIR)
# ACTIVE_FEATS = the ON features (user-selected) used for ranking + curation filtering.
# Every other feature stays defined/computed (still in the CSV and histograms) but is OFF for
# selection — nothing is deleted, just toggled off. Edit this list to turn features on/off.
ACTIVE_FEATS = [
    "return",             # outcome / quality
    "descent_rate",       # gentle descent (lower better)
    "land_x_offset",      # landed near the pad (landing quality)
    "touchdown_vy",       # soft landing — vertical (landing quality)
    "touchdown_vx",       # soft landing — horizontal (landing quality)
    "angvel_mean_abs",    # rotational stability
    "vx_mean_abs",        # lateral stability
    "angle_mean_abs",     # uprightness / tilt (stability)
    "action_divergence",  # cross-demo consistency (realizability)
]
RANK_FEATS = ACTIVE_FEATS  # ranking/filter use only the ON features


def add_ranking(rows, feats=None):
    """Composite goodness rank across features. Each feature -> percentile in [0,1] oriented so
    1 = best (accounts for good-direction); score = mean percentile across features; rank 1 = best.
    Graded alternative to the hard AND filter: robust to being just-barely on the wrong side of
    one threshold, and lets you take an exact top-k."""
    from scipy.stats import rankdata
    feats = feats or RANK_FEATS
    n = len(rows)
    pr = {}
    for f in feats:
        v = np.array([r[f] for r in rows], float)
        p = (rankdata(v) - 1) / (n - 1)          # 0..1 by value (ties averaged)
        pr[f] = (1 - p) if GOOD_DIR[f] < 0 else p  # flip so 1 = best
    score = np.mean([pr[f] for f in feats], axis=0)
    order = np.argsort(-score)                    # best first
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    for i, r in enumerate(rows):
        r["rank_score"] = float(score[i]); r["rank"] = int(rank[i])


def traj_features(ep):
    """ep: dict with obs (T+1,8), acts (T,), rews (T,), terminal (bool). -> feature dict."""
    obs = np.asarray(ep["obs"], float); acts = np.asarray(ep["acts"]).astype(int)
    rews = np.asarray(ep["rews"], float); term = bool(np.asarray(ep["terminal"]))
    x, vx, vy, angle, angvel = obs[:, 0], obs[:, 2], obs[:, 3], obs[:, 4], obs[:, 5]
    fin = obs[-1]
    legs_down = int(fin[6] > 0.5 and fin[7] > 0.5)
    ret = float(rews.sum())
    # Match human_demo.py's live verdict: LunarLander pays +100 on a successful landing and -100
    # on a crash at the FINAL step, so the last-step reward is the reliable success signal (the
    # final leg-contact flags are not — the last stored frame often isn't the settled pose).
    last = float(rews[-1]) if len(rews) else 0.0
    if not term:        outcome = "timeout"
    elif last >= 99:    outcome = "landed"
    elif last <= -99:   outcome = "crashed"
    else:               outcome = "ended"
    # NOTE: action_divergence is NOT here — it needs the whole demo set; add_action_divergence()
    # fills it in. Everything else is computable from a single trajectory (usable online too).
    # WARNING -- touchdown_vy / touchdown_vx are measured on the FINAL stored frame, and
    # LunarLander only terminates a successful landing once the lander is ASLEEP (at rest), so
    # both are EXACTLY 0.0 for 100% of successful landings (verified on all 412 human + 149
    # expert landings). They are therefore crash INDICATORS, not landing-softness measures:
    # rho(touchdown_vy, landed) = -0.955 on the ext700 pool. Kept for backwards compatibility --
    # every existing ranking/subset used them -- but use contact_* below to measure how hard the
    # demonstrator actually touched down.
    ci = np.flatnonzero((obs[:, 6] > 0.5) | (obs[:, 7] > 0.5))
    c = obs[ci[0]] if len(ci) else fin      # no leg ever touched (out of bounds / timeout) -> final
    return dict(
        n_steps=len(acts), return_=ret, legs_down=legs_down, outcome=outcome,
        contact_vy=float(abs(c[3])), contact_vx=float(abs(c[2])),
        contact_angle=float(abs(c[4])),
        angle_mean_abs=float(np.mean(np.abs(angle))), angle_max_abs=float(np.max(np.abs(angle))),
        angle_std=float(np.std(angle)),
        angvel_mean_abs=float(np.mean(np.abs(angvel))),               # rotational stability
        x_wander=float(np.sum(np.abs(np.diff(x)))), x_std=float(np.std(x)),
        vx_mean_abs=float(np.mean(np.abs(vx))),
        descent_rate=float(np.mean(np.clip(-vy, 0, None))),           # mean downward speed
        land_x_offset=float(abs(fin[0])), touchdown_vy=float(abs(fin[3])),
        touchdown_vx=float(abs(fin[2])), land_angle=float(abs(fin[4])),
    )


def _flatten(ds):
    """All demos -> one flat table: S (n,6) states, A (n,) actions, DID (n,) source demo id.
    Only the first 6 obs dims are kept (x, y, vx, vy, angle, angvel); the two leg-contact flags
    are dropped because a 0/1 flag would dominate a Euclidean distance. obs has T+1 rows and
    acts has T, so the final state (which has no action) is dropped."""
    S, A, DID = [], [], []
    for i in range(len(ds)):
        obs = np.asarray(ds[i]["obs"], float); acts = np.asarray(ds[i]["acts"]).astype(int)
        S.append(obs[:len(acts), :6]); A.append(acts); DID.append(np.full(len(acts), i))
    return np.vstack(S), np.concatenate(A), np.concatenate(DID)


def _normalize(S):
    """z-score each dim over the whole pool, so no dimension dominates the distance."""
    return (S - S.mean(0)) / (S.std(0) + 1e-8)


def _n_actions(A, n_actions=None):
    return max(2, int(A.max()) + 1) if n_actions is None else max(2, int(n_actions))


def _norm_entropy(acts, na):
    """Normalized entropy of an action multiset: -sum p log p / log(na), in [0, 1]."""
    if len(acts) == 0:
        return 0.0
    p = np.bincount(acts, minlength=na).astype(float)
    p = p[p > 0] / len(acts)                       # drop zero-probability actions (no 0*log0)
    return float(-(p * np.log(p)).sum() / np.log(na))


def add_action_divergence(rows, ds, k=15, n_actions=None):
    """Per-demo action divergence: for each state in a demo, how much its action disagrees with
    what OTHER demos do at nearby states (kNN in normalized 6-D continuous state). Higher = this
    demo is inconsistent with the rest -> exactly the realizability signal that breaks BC/GAIL.

    NOTE: neighbours are picked by distance alone, so ONE nearby demo can supply all k votes.
    See add_cross_demo_distinct_metrics for the one-vote-per-demo variant.

    Also writes `action_entropy`: the normalized entropy of the SAME neighbour set. The divergence
    asks "was a_t unusual here?"; the entropy ignores a_t and asks "is this state ambiguous?"."""
    from scipy.spatial import cKDTree
    S, A, DID = _flatten(ds)
    Sn = _normalize(S)
    na = _n_actions(A, n_actions)
    ent = np.zeros(len(S))
    tree = cKDTree(Sn)
    kq = min(k + 40, len(Sn))
    _, idx = tree.query(Sn, k=kq)
    div = np.empty(len(S))
    for i in range(len(S)):
        neigh = idx[i][1:]                                   # drop self
        other = neigh[DID[neigh] != DID[i]][:k]              # only OTHER demos' states
        div[i] = float(np.mean(A[other] != A[i])) if len(other) else 0.0
        ent[i] = _norm_entropy(A[other], na)                 # same neighbours, ignores A[i]
    for r in rows:
        m = DID == r["demo_id"]
        r["action_divergence"] = float(div[m].mean()) if m.any() else 0.0
        r["action_entropy"] = float(ent[m].mean()) if m.any() else 0.0

def add_action_divergence_X_Y(rows, ds, k=15, n_actions=None):
    """Per-demo action divergence: for each state in a demo, how much its action disagrees with
    what OTHER demos do at nearby states (kNN in normalized 6-D continuous state). Higher = this
    demo is inconsistent with the rest -> exactly the realizability signal that breaks BC/GAIL.

    NOTE: neighbours are picked by distance alone, so ONE nearby demo can supply all k votes.
    See add_cross_demo_distinct_metrics for the one-vote-per-demo variant.

    Also writes `action_entropy`: the normalized entropy of the SAME neighbour set. The divergence
    asks "was a_t unusual here?"; the entropy ignores a_t and asks "is this state ambiguous?"."""
    from scipy.spatial import cKDTree
    S, A, DID = [], [], []
    for i in range(len(ds)):
        obs = np.asarray(ds[i]["obs"], float); acts = np.asarray(ds[i]["acts"]).astype(int)
        S.append(obs[:len(acts), :2]); A.append(acts); DID.append(np.full(len(acts), i))
    S,A,DID = np.vstack(S), np.concatenate(A), np.concatenate(DID)
    Sn = _normalize(S)
    na = _n_actions(A, n_actions)
    ent = np.zeros(len(S))
    tree = cKDTree(Sn)
    kq = min(k + 40, len(Sn))
    _, idx = tree.query(Sn, k=kq)
    div = np.empty(len(S))
    for i in range(len(S)):
        neigh = idx[i][1:]                                   # drop self
        other = neigh[DID[neigh] != DID[i]][:k]              # only OTHER demos' states
        div[i] = float(np.mean(A[other] != A[i])) if len(other) else 0.0
        ent[i] = _norm_entropy(A[other], na)                 # same neighbours, ignores A[i]
    for r in rows:
        m = DID == r["demo_id"]
        r["action_divergence_xy"] = float(div[m].mean()) if m.any() else 0.0
        r["action_entropy_xy"] = float(ent[m].mean()) if m.any() else 0.0

def add_self_action_metrics(rows, ds, k=15, n_actions=None):
    """WITHIN-person consistency. For each state, look only at the k nearest states from the SAME
    demo (self excluded) and ask:
      self_action_divergence -- fraction of those neighbours whose action differs from a_t, i.e.
                                1 - P_self(a_t | s_t). "Is this demonstrator's current action
                                unusual compared with what they themselves usually do here?"
      self_action_entropy    -- normalized entropy of the neighbours' action distribution. Ignores
                                a_t entirely: "does this demonstrator use several different
                                actions around this state?"
    Both averaged over the demo's own states."""
    from scipy.spatial import cKDTree
    S, A, DID = _flatten(ds)
    Sn = _normalize(S)                       # pool-level normalization, same metric space
    na = _n_actions(A, n_actions)
    per = {}
    for j in np.unique(DID):
        m = np.flatnonzero(DID == j)
        Sj, Aj = Sn[m], A[m]
        kk = min(k, len(m) - 1)              # a demo can offer at most len-1 self-neighbours
        if kk < 1:                           # single-state demo -> no neighbours at all
            per[int(j)] = (0.0, 0.0); continue
        _, idx = cKDTree(Sj).query(Sj, k=kk + 1, workers=-1)
        idx = np.atleast_2d(idx)[:, 1:]      # drop self (always the nearest, distance 0)
        dv = np.mean(Aj[idx] != Aj[:, None], axis=1)
        en = np.array([_norm_entropy(Aj[row], na) for row in idx])
        per[int(j)] = (float(dv.mean()), float(en.mean()))
    for r in rows:
        d, e = per.get(int(r["demo_id"]), (0.0, 0.0))
        r["self_action_divergence"] = d
        r["self_action_entropy"] = e


def add_cross_demo_distinct_metrics(rows, ds, k=15, n_actions=None):
    """ACROSS-person consistency, one vote per demo. For each state we take the k closest DISTINCT
    other demos, each contributing only its single nearest state, and ask:
      cross_demo_action_divergence -- fraction of those demos whose action differs from a_t.
      cross_demo_action_entropy    -- normalized entropy of their action distribution (ignores
                                      a_t): "do different people disagree around this state?"

    Unlike add_action_divergence, a single spatially-adjacent demo cannot dominate the vote.
    Computed EXACTLY: for every demo j we 1-NN-query all states against demo j's own tree and keep
    a running best-k. The k+40 window used by add_action_divergence is NOT usable here -- on the
    699-demo pool it yields a median of only 7 distinct demos."""
    from scipy.spatial import cKDTree
    S, A, DID = _flatten(ds)
    Sn = _normalize(S)
    na = _n_actions(A, n_actions)
    n, demos = len(Sn), np.unique(DID)
    kk = min(k, len(demos) - 1)              # at most (n_demos - 1) distinct OTHER demos
    if kk < 1:
        for r in rows:
            r["cross_demo_action_divergence"] = 0.0
            r["cross_demo_action_entropy"] = 0.0
        return
    best_d = np.full((n, kk), np.inf)
    best_a = np.full((n, kk), -1, dtype=np.int64)
    for j in demos:
        m = DID == j
        d1, i1 = cKDTree(Sn[m]).query(Sn, k=1, workers=-1)     # nearest state IN demo j, for all
        d1 = np.asarray(d1, float).copy(); a1 = A[m][np.asarray(i1).ravel()]
        d1[m] = np.inf                                         # a demo never votes on itself
        cd = np.column_stack([best_d, d1])                     # (n, kk+1)
        ca = np.column_stack([best_a, a1])
        o = np.argsort(cd, axis=1, kind="stable")[:, :kk]      # keep the kk closest demos
        best_d = np.take_along_axis(cd, o, axis=1)
        best_a = np.take_along_axis(ca, o, axis=1)
    valid = np.isfinite(best_d)                                # guards pools with < kk+1 demos
    dv = np.zeros(n); en = np.zeros(n)
    for i in range(n):
        acts = best_a[i][valid[i]]
        if len(acts) == 0:
            continue
        dv[i] = float(np.mean(acts != A[i]))
        en[i] = _norm_entropy(acts, na)
    for r in rows:
        m = DID == r["demo_id"]
        r["cross_demo_action_divergence"] = float(dv[m].mean()) if m.any() else 0.0
        r["cross_demo_action_entropy"] = float(en[m].mean()) if m.any() else 0.0


def load_rows(session, self_k=15, cross_k=15, div_k=15):
    import datasets; datasets.disable_progress_bar()
    ds = datasets.load_from_disk(session)
    rows = []
    for i in range(len(ds)):
        f = traj_features(ds[i]); f["demo_id"] = i
        f["return"] = f.pop("return_")   # csv-friendly name
        rows.append(f)
    add_action_divergence(rows, ds, k=div_k)   # cross-demo feature, needs the whole set
    add_action_divergence_X_Y(rows, ds)      # cross-demo feature, needs the whole set
    add_self_action_metrics(rows, ds, k=self_k)              # within-person consistency
    add_cross_demo_distinct_metrics(rows, ds, k=cross_k)     # across-person, one vote per demo
    add_ranking(rows)                     # composite goodness rank across RANK_FEATS
    return rows


def good_mask(rows, feats, pct):
    """Keep demos in the good `pct`% of EACH feature (AND). pct=50 -> good half."""
    vals = {f: np.array([r[f] for r in rows], float) for f in feats}
    mask = np.ones(len(rows), bool)
    thr = {}
    for f in feats:
        d = GOOD_DIR[f]
        # good side = top pct% when d=+1, bottom pct% when d=-1
        q = np.percentile(vals[f], (100 - pct) if d > 0 else pct)
        thr[f] = q
        mask &= (vals[f] >= q) if d > 0 else (vals[f] <= q)
    return mask, thr


def plot_hists(rows, plot_dir, feats=None, mask=None, thr=None, pct=50):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    feats = feats or [f for f in FEATURES if f != "legs_down"]
    outc = np.array([r["outcome"] for r in rows])
    ncol = 4; nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        v = np.array([r[f] for r in rows], float)
        bins = np.histogram_bin_edges(v, bins=25)
        ax.hist(v[outc == "landed"], bins=bins, color="#2ca02c", alpha=.65, label="landed")
        ax.hist(v[outc != "landed"], bins=bins, color="#d62728", alpha=.55, label="crashed/timeout")
        med = np.median(v); ax.axvline(med, color="k", ls="--", lw=1)
        # shade the good side of the (pct) threshold if provided, else the median
        t = (thr or {}).get(f, med); d = GOOD_DIR[f]
        lo, hi = ax.get_xlim()
        ax.axvspan(t, hi, color="#2ca02c", alpha=.10) if d > 0 else ax.axvspan(lo, t, color="#2ca02c", alpha=.10)
        ax.set_title(f"{f}  ({'higher' if d>0 else 'lower'} better)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # counts are whole numbers
        ax.set_xlabel("feature value", fontsize=7); ax.set_ylabel("count (# demos)", fontsize=7)
    for ax in axes[len(feats):]: ax.axis("off")
    axes[0].legend(fontsize=7)
    tag = "filtered" if mask is not None else "all"
    if mask is not None:
        fig.suptitle(f"session_3 features — good-{pct}% AND filter on {feats}: "
                     f"{int(mask.sum())}/{len(rows)} pass (mean return {np.mean([rows[i]['return'] for i in np.where(mask)[0]]):+.0f})",
                     fontsize=11)
    else:
        fig.suptitle("session_3 per-trajectory features (landed=green vs crashed=red; dashed=median)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(plot_dir, exist_ok=True)
    out = os.path.join(plot_dir, f"session3_feature_hists_{tag}.png")
    fig.savefig(out, dpi=130); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True); ap.add_argument("--out_csv")
    ap.add_argument("--plot_dir", required=True); ap.add_argument("--filter", default="")
    ap.add_argument("--pct", type=float, default=50)
    ap.add_argument("--self_k", type=int, default=15, help="neighbours for the self_* metrics")
    ap.add_argument("--cross_k", type=int, default=15,
                    help="distinct other demos for the cross_demo_* metrics")
    ap.add_argument("--div_k", type=int, default=15,
                    help="kNN neighbours for action_divergence / action_entropy")
    a = ap.parse_args()
    rows = load_rows(a.session, self_k=a.self_k, cross_k=a.cross_k, div_k=a.div_k)
    print(f"{len(rows)} trajectories; outcomes:",
          {o: sum(r['outcome'] == o for r in rows) for o in ("landed", "crashed", "timeout")})
    if a.out_csv:
        import csv
        cols = ["demo_id", "rank", "rank_score", "outcome", "return"] + [f for f in FEATURES if f != "return"]
        with open(a.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({c: r[c] for c in cols})
        print("wrote", a.out_csv)
    if a.filter:
        feats = a.filter.split(",")
        mask, thr = good_mask(rows, feats, a.pct)
        print(f"AND good-{a.pct:.0f}% filter on {feats}: {mask.sum()}/{len(rows)} pass")
        plot_hists(rows, a.plot_dir, feats=feats, mask=mask, thr=thr, pct=a.pct)
    else:
        plot_hists(rows, a.plot_dir)


if __name__ == "__main__":
    main()
