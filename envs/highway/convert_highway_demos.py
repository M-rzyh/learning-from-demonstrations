"""Convert highway demos from the laptop collector (epNNN.npz) into imitation's dataset format.

Input:  a collector output folder with epNNN.npz files (collect_highway_demos.py or _ui.py).
Output: a dataset folder (data-*.arrow, dataset_info.json, state.json) that run_gail_highway.sh and
        launchers/bc/run_bc_highway.sh read through DEMO_PATH. Same format and same steps as save_demos() in
        envs/lunarlander/human_demo.py: TrajectoryWithRew + imitation.data.serialize.save.

Only obs, acts, rews and terminal go into the dataset. Everything else (seed, flagged, n_applied,
applied_log, flags.json, all_episodes_log.csv) stays in the source folder; a manifest written next to
the dataset (<dst>_manifest.json) records which source files went in.

Run inside a job with the imitation-gail env and the repo's src on PYTHONPATH, e.g.
  python envs/highway/convert_highway_demos.py <src folder> <dst folder> [--flagged-only]
"""
import argparse, glob, json, os, time

import numpy as np
from imitation.data import serialize
from imitation.data.types import TrajectoryWithRew

p = argparse.ArgumentParser()
p.add_argument("src", help="collector output folder with epNNN.npz")
p.add_argument("dst", help="dataset folder to create (must not exist)")
p.add_argument("--flagged-only", action="store_true", help="only episodes saved with F (what GAIL trains on)")
a = p.parse_args()

if os.path.exists(a.dst):
    raise SystemExit(f"refusing to overwrite existing {a.dst}")
files = sorted(glob.glob(os.path.join(a.src, "ep*.npz")))
if not files:
    raise SystemExit(f"no ep*.npz in {a.src}")

trajs, used, skipped = [], [], []
for f in files:
    d = np.load(f)
    if a.flagged_only and not ("flagged" in d and bool(d["flagged"])):
        skipped.append(os.path.basename(f))
        continue
    obs, acts, rews = d["obs"].astype(np.float32), d["acts"].astype(np.int64), d["rews"].astype(np.float64)
    assert len(obs) == len(acts) + 1 == len(rews) + 1, f"{f}: obs/acts/rews lengths don't line up"
    trajs.append(TrajectoryWithRew(obs=obs, acts=acts, rews=rews, infos=None, terminal=bool(d["terminal"])))
    used.append(os.path.basename(f))
if not trajs:
    raise SystemExit("no episodes selected (with --flagged-only: nothing flagged?)")

serialize.save(a.dst, trajs)
meta_path = os.path.join(a.src, "meta.json")
manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "src": os.path.abspath(a.src), "dst": os.path.abspath(a.dst),
            "flagged_only": a.flagged_only, "n_episodes": len(trajs), "n_steps": int(sum(len(t.acts) for t in trajs)),
            "files": used, "skipped_not_flagged": skipped,
            "collector_meta": json.load(open(meta_path)) if os.path.exists(meta_path) else None}
with open(a.dst.rstrip("/") + "_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"{len(trajs)} episodes, {manifest['n_steps']} steps -> {a.dst}"
      + (f"  ({len(skipped)} not flagged, skipped)" if skipped else ""))
