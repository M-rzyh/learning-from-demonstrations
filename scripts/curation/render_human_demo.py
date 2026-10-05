"""Render one recorded human demonstration to an .mp4 by replaying its actions.

The demos store (obs, acts, rews) but no frames, so the episode is reconstructed:
reset LunarLander-v2 with the demo's recording seed (the session convention is
seed = 42 + demo_id, demo_id being the index in session_3_ext700) and step the
recorded actions. The replay is VERIFIED against the stored rewards before the
video is written -- if it does not match, nothing is saved.

Usage (compute node, imitation-gail env):
    python scripts/curation/render_human_demo.py --pool_position 0 --out /path/demo.mp4
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
from datasets import load_from_disk
import datasets; datasets.disable_progress_bar()

DEMOS = Path("/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, default=DEMOS / "session_3_ext700_ret_gt200")
    p.add_argument("--index_csv", type=Path,
                   default=DEMOS / "session_3_ext700_ret_gt200_index.csv")
    p.add_argument("--pool_position", type=int, default=0)
    p.add_argument("--seed_base", type=int, default=42, help="env seed = seed_base + demo_id")
    p.add_argument("--gym_id", default="LunarLander-v2")
    p.add_argument("--max_episode_steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-3, help="max |reward| mismatch allowed")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    a = parse_args()
    rows = list(csv.DictReader(open(a.index_csv)))
    row = next(r for r in rows if int(r["pool_position"]) == a.pool_position)
    demo_id = int(row["ext700_index"])
    stored_return = float(row["return"])

    t = load_from_disk(str(a.pool))[a.pool_position]
    acts = np.asarray(t["acts"]).ravel()
    rews = np.asarray(t["rews"], dtype=float)
    print(f"pool position {a.pool_position} -> ext700 demo {demo_id}: "
          f"{len(acts)} steps, stored return {stored_return:.2f}")

    env = gym.make(a.gym_id, max_episode_steps=a.max_episode_steps, render_mode="rgb_array")
    obs, _ = env.reset(seed=a.seed_base + demo_id)
    frames, replay, done = [env.render()], [], False
    for act in acts:
        obs, r, term, trunc, _ = env.step(int(act))
        replay.append(float(r))
        frames.append(env.render())
        if term or trunc:
            break
    env.close()

    replay = np.asarray(replay)
    n = min(len(replay), len(rews))
    err = float(np.max(np.abs(replay[:n] - rews[:n]))) if n else float("inf")
    print(f"replay: {len(replay)} steps, return {replay.sum():.2f}; "
          f"max per-step reward mismatch {err:.2e}")
    if len(replay) != len(rews) or err > a.tol:
        raise SystemExit("replay does not reproduce the stored demo -- not writing a video "
                         f"(steps {len(replay)} vs {len(rews)}, max err {err:.2e})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.out, np.asarray(frames, dtype=np.uint8), fps=a.fps, macro_block_size=None)
    print(f"wrote {a.out} ({len(frames)} frames, verified)")


if __name__ == "__main__":
    main()
