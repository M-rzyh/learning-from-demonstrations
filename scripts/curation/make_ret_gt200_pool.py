"""Build the 'good human demos' pool: every session_3_ext700 demo with true
episodic return > 200 (403 of 699), in the original order.

This is a pool, not a per-seed subset: the training scripts shuffle it with
ds.shuffle(seed=SHUFFLE_SEED) and take the first N, so each seed gets its own
random draw and BC and GAIL get the SAME draw for the same (N, seed).

Usage (compute node, imitation-gail env):
    python scripts/curation/make_ret_gt200_pool.py [--threshold 200]
"""
import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk

ROOT = Path("/scratch/marzii/imitation_runs/demos/human_baseline/"
            "record_human_demos/lunarlander")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=ROOT / "session_3_ext700")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=200.0)
    a = p.parse_args()
    out = a.out or ROOT / f"session_3_ext700_ret_gt{int(a.threshold)}"

    ds = load_from_disk(str(a.src))
    rets = np.array([float(np.sum(r)) for r in ds["rews"]])
    keep = np.flatnonzero(rets > a.threshold)          # original order
    print(f"{a.src.name}: {len(ds)} demos, {len(keep)} with return > {a.threshold:g} "
          f"(min {rets[keep].min():.1f}, max {rets[keep].max():.1f})")

    if out.exists():
        raise SystemExit(f"refusing to overwrite existing {out}")
    sub = ds.select(keep.tolist())
    sub.save_to_disk(str(out))
    print(f"wrote {out}  ({len(sub)} demos)")

    # index of which original demos went in, for provenance
    idx = out.parent / f"{out.name}_index.csv"
    with open(idx, "w") as f:
        f.write("pool_position,ext700_index,return\n")
        for i, j in enumerate(keep):
            f.write(f"{i},{j},{rets[j]:.2f}\n")
    print(f"wrote {idx}")


if __name__ == "__main__":
    main()
