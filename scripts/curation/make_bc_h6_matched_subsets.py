"""Materialise BC demo subsets that EXACTLY match Human GAIL H6.

H6 (gail_session3_curation_2026-08-05.csv, conditions rand{5,10,50,100,200}) did:

    ds = load_from_disk(session_3)            # 200 human demos
    ds = ds.shuffle(seed=SHUFFLE_SEED)        # SHUFFLE_SEED == SEED for every row
    ds.save_to_disk(<run>/shuffled_demos)
    ... training then used demos[:N]          # ingredients/demonstrations.py:127

so the demos H6 actually trained on at (N, seed) are the FIRST N entries of that
run's shuffled_demos. This script copies exactly those into BC-owned directories:

    <out_root>/session_3_seed<seed>_N<N>/

Nothing in session_3 or in any GAIL run directory is modified -- the H6 dirs are
only read. A manifest records, for every subset, the session_3 indices it holds,
so the provenance is explicit and checkable later.

Verification performed here (fails loudly rather than writing a wrong subset):
  * every trajectory in the new subset is byte-identical (obs+acts hash) to the
    corresponding H6 trajectory, in the same order;
  * every trajectory maps back to a unique session_3 index;
  * the subset length equals N.

Usage (compute node, imitation-gail env):
    python scripts/curation/make_bc_h6_matched_subsets.py --dry_run
    python scripts/curation/make_bc_h6_matched_subsets.py
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_from_disk
import datasets

datasets.disable_progress_bar()

SESSION3 = Path("/scratch/marzii/imitation_runs/demos/human_baseline/"
                "record_human_demos/lunarlander/session_3")
GAIL_RUNS = Path("/scratch/marzii/imitation_runs/gail/lunarlander")
INDEX = Path("/home/marzii/IRL3/experiments/GAIL/gail_session3_curation_2026-08-05.csv")
OUT_ROOT = Path("/scratch/marzii/imitation_runs/demos/bc_h6_matched")
MANIFEST = Path("/home/marzii/IRL3/experiments/BC/bc_h6_matched_subsets.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 50, 100, 200])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def sig(traj):
    o = np.asarray(traj["obs"], dtype=np.float32)
    a = np.asarray(traj["acts"])
    return hashlib.md5(np.round(o, 5).tobytes() + a.tobytes()).hexdigest()


def main():
    a = parse_args()

    base = load_from_disk(str(SESSION3))
    sig2id, rets = {}, []
    for i, t in enumerate(base):
        sig2id[sig(t)] = i
        rets.append(float(np.sum(t["rews"])))
    rets = np.asarray(rets)
    print(f"session_3: {len(base)} demos, {(rets > 200).sum()} with return > 200")

    # (N, seed) -> H6 job id
    jobs = {}
    for r in csv.DictReader(open(INDEX)):
        cond = r.get("condition", "")
        if not cond.startswith("rand"):
            continue
        jobs[(int(r["N"]), int(r["seed"]))] = r["slurm_job_id"].strip()

    rows = []
    for n in a.budgets:
        for s in a.seeds:
            job = jobs.get((n, s))
            if not job:
                raise SystemExit(f"no H6 run recorded for N={n} seed={s}")
            src = GAIL_RUNS / job / "shuffled_demos"
            if not src.exists():
                raise SystemExit(f"missing H6 artifact: {src}")

            h6 = load_from_disk(str(src))
            if len(h6) < n:
                raise SystemExit(f"{src} holds {len(h6)} trajs, need {n}")
            subset = h6.select(range(n))            # exactly what H6 trained on

            h6_sigs = [sig(t) for t in subset]
            ids = [sig2id.get(x, -1) for x in h6_sigs]
            if -1 in ids:
                raise SystemExit(f"N={n} seed={s}: a trajectory is not in session_3")
            if len(subset) != n:
                raise SystemExit(f"N={n} seed={s}: got {len(subset)} trajectories")

            dst = a.out_root / f"session_3_seed{s}_N{n}"
            n_succ = int((rets[ids] > 200).sum())
            print(f"  N={n:<4} seed={s}  ids[:5]={ids[:5]}  succ>200={n_succ:<3} -> {dst}")

            if not a.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    raise SystemExit(f"REFUSING to overwrite existing {dst}")
                subset.save_to_disk(str(dst))
                # read back and confirm byte-identity with the H6 order
                back = load_from_disk(str(dst))
                if [sig(t) for t in back] != h6_sigs:
                    raise SystemExit(f"VERIFY FAILED: {dst} differs from H6 order")

            rows.append(dict(N=n, seed=s, h6_job=job, n_demos=n,
                             n_success_gt200=n_succ,
                             mean_return=round(float(rets[ids].mean()), 2),
                             subset_path=str(dst),
                             session3_ids=" ".join(map(str, ids))))

    if not a.dry_run:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(a.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote manifest {a.manifest}")
    print(f"{len(rows)} subsets {'planned' if a.dry_run else 'written and verified'}")


if __name__ == "__main__":
    main()
