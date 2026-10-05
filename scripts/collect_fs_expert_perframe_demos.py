#!/usr/bin/env python3
"""Roll out the FS-trained expert in a *non-FS* env, holding each decided action
for 10 underlying env steps but saving EVERY frame's (obs, action, reward) pair.

This produces demos in the same per-frame format as human demos, so they can
be fed into GAIL with FRAME_SKIP=0 without a granularity mismatch.

Usage:
    python collect_fs_expert_perframe_demos.py \\
        --policy /scratch/marzii/imitation_runs/expert/lunarlander/<JOBID>/policies/final/model.zip \\
        --output /scratch/marzii/imitation_runs/expert/lunarlander/<JOBID>/rollouts/perframe_demos \\
        --n-episodes 50 --hold-k 10 --max-frames 400 --seed 0
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from datasets import Dataset, Features, Sequence, Value
from stable_baselines3 import PPO


def collect(policy_path, output_dir, n_episodes, hold_k, max_frames, seed):
    model = PPO.load(policy_path)
    env = gym.make("LunarLander-v2", max_episode_steps=max_frames)

    trajectories = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_obs = [obs.tolist()]
        ep_acts = []
        ep_rews = []
        terminal = False

        held_action = None
        step = 0
        done = False
        while not done:
            if step % hold_k == 0:
                act, _ = model.predict(obs, deterministic=True)
                held_action = int(act)
            obs, reward, terminated, truncated, _ = env.step(held_action)
            ep_obs.append(obs.tolist())
            ep_acts.append(held_action)
            ep_rews.append(float(reward))
            done = bool(terminated or truncated)
            terminal = bool(terminated)
            step += 1

        trajectories.append({
            "obs": ep_obs,
            "acts": ep_acts,
            "rews": ep_rews,
            "terminal": terminal,
            "infos": [{} for _ in ep_acts],
        })
        print(f"  ep {ep+1}/{n_episodes}: len={len(ep_acts)}, reward={sum(ep_rews):.1f}")

    env.close()

    # Match the schema of existing human/expert demos
    features = Features({
        "obs": Sequence(Sequence(Value("float32"))),
        "acts": Sequence(Value("int64")),
        "infos": Sequence(Value("string")),
        "terminal": Value("bool"),
        "rews": Sequence(Value("float64")),
    })
    # Convert info dicts to JSON strings for the schema
    import json as _json
    for t in trajectories:
        t["infos"] = [_json.dumps(i) for i in t["infos"]]
    ds = Dataset.from_list(trajectories, features=features)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"\nSaved {len(trajectories)} trajectories to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, help="Path to PPO model.zip")
    p.add_argument("--output", required=True, help="Output directory (HF arrow dataset)")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--hold-k", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    collect(args.policy, args.output, args.n_episodes, args.hold_k, args.max_frames, args.seed)


if __name__ == "__main__":
    main()
