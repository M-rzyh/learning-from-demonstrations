#!/usr/bin/env python3
"""Roll out the FS-trained expert with ONLINE Q-weighted noise injection.

At each underlying env step:
  - Every `hold_k` frames, query the policy for a new argmax action (held action).
  - With probability `noise_prob`, override with a sample from the policy's
    softmax distribution at the CURRENT state (Q-weighted, not uniform random).
  - Env actually steps with that action; trajectory state evolves accordingly.

The Q-weighted sampling uses PPO's policy.get_distribution(obs).distribution.probs.

Usage:
    python collect_fs_expert_perframe_demos_online_qweighted.py \\
        --policy /scratch/marzii/imitation_runs/expert/lunarlander/4720242/policies/final/model.zip \\
        --output /scratch/marzii/imitation_runs/demos/noisy_demos_online_qweighted/lunarlander/expert_4720242/n100_p25_s0 \\
        --n-episodes 100 --hold-k 10 --max-frames 400 \\
        --noise-prob 0.25 --noise-seed 0 \\
        --optimal-policy /scratch/marzii/imitation_runs/expert/lunarlander/4720242/policies/final/model.zip
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor


def policy_action_probs(model, obs):
    """Return shape-(n_actions,) numpy array of softmax probabilities for current obs."""
    obs_tensor = obs_as_tensor(obs.reshape(1, -1), model.policy.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
    return dist.distribution.probs.cpu().numpy()[0]


def collect(policy_path, output_dir, n_episodes, hold_k, max_frames,
            noise_prob, noise_seed, env_seed=0, n_actions=4):
    model = PPO.load(policy_path)
    env = gym.make("LunarLander-v2", max_episode_steps=max_frames)
    rng = np.random.default_rng(noise_seed)

    trajectories = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=env_seed + ep)
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
            if noise_prob > 0 and rng.random() < noise_prob:
                # Q-weighted sample from current policy distribution
                probs = policy_action_probs(model, obs)
                used_action = int(rng.choice(n_actions, p=probs))
            else:
                used_action = held_action
            obs, reward, terminated, truncated, _ = env.step(used_action)
            ep_obs.append(obs.tolist())
            ep_acts.append(used_action)
            ep_rews.append(float(reward))
            done = bool(terminated or truncated)
            terminal = bool(terminated)
            step += 1

        trajectories.append({
            "obs": ep_obs,
            "acts": ep_acts,
            "rews": ep_rews,
            "terminal": terminal,
            "infos": [json.dumps({}) for _ in ep_acts],
        })

    env.close()

    features = Features({
        "obs": Sequence(Sequence(Value("float32"))),
        "acts": Sequence(Value("int64")),
        "infos": Sequence(Value("string")),
        "terminal": Value("bool"),
        "rews": Sequence(Value("float64")),
    })
    ds = Dataset.from_list(trajectories, features=features)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    return ds


def compute_demo_alignment(ds_path, optimal_policy_path):
    ds = load_from_disk(ds_path)
    model = PPO.load(optimal_policy_path)
    matches = 0
    total = 0
    for ep in ds:
        obs = np.asarray(ep["obs"], dtype=np.float32)
        acts = np.asarray(ep["acts"], dtype=np.int64)
        states = obs[:len(acts)]
        pred_acts, _ = model.predict(states, deterministic=True)
        pred_acts = np.asarray(pred_acts, dtype=np.int64)
        matches += int((pred_acts == acts).sum())
        total += len(acts)
    return matches, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--hold-k", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--noise-prob", type=float, required=True)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--optimal-policy", required=True)
    p.add_argument("--n-actions", type=int, default=4)
    args = p.parse_args()

    ds = collect(args.policy, args.output, args.n_episodes, args.hold_k,
                 args.max_frames, args.noise_prob, args.noise_seed,
                 args.env_seed, args.n_actions)

    matches, total = compute_demo_alignment(args.output, args.optimal_policy)
    alignment = matches / max(1, total)
    meta = {
        "alignment_fraction": alignment,
        "matches": matches,
        "total_transitions": total,
        "optimal_ref": args.optimal_policy,
        "expert_policy_used_for_collection": args.policy,
        "noise_prob": args.noise_prob,
        "noise_seed": args.noise_seed,
        "noise_type": "online_qweighted_softmax",
        "n_episodes": args.n_episodes,
        "hold_k": args.hold_k,
        "n_actions": args.n_actions,
    }
    with open(Path(args.output) / "demo_alignment.json", "w") as f:
        json.dump(meta, f, indent=2)
    rewards = [sum(ep["rews"]) for ep in ds]
    print(f"  alignment = {alignment:.4f} ({matches}/{total}), "
          f"reward mean = {np.mean(rewards):.1f}")


if __name__ == "__main__":
    main()
