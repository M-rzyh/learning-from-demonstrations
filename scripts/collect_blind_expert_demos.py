#!/usr/bin/env python3
"""Collect DEGRADED ("blind") expert demonstrations via ONLINE frame-blanking.

Frame-blanking difficulty, Exp 1 (harder SUPERVISION, normal task): on blanked
frames the expert is fed a ZEROED observation, so it acts "blind"; the env then
steps with that action, so errors compound (physically consistent). The RECORDED
observation is the TRUE 8-D state and the recorded action is the expert's
(possibly blind) action -> standard 8-D demos; the agent still learns the NORMAL
LunarLander. Use with the non-frame-skip expert 4615187.

NEW file -- does NOT modify collect_fs_expert_perframe_demos_online_noise.py.

Usage:
    python scripts/collect_blind_expert_demos.py \\
        --policy   /scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip \\
        --optimal-policy /scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip \\
        --output   $SCRATCH/imitation_runs/frame_blanking/demos/blind_expert/n100_blank50_s0 \\
        --n-episodes 100 --hold-k 1 --max-frames 400 \\
        --blank-mode stochastic --blank-prob 0.5 --blank-seed 0
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from stable_baselines3 import PPO


def _blanked(step, mode, p, k, rng):
    """Is this query-frame blanked? stochastic = coin(p); deterministic = every k-th."""
    if mode == "deterministic":
        return (step % int(k)) == 0
    return rng.random() < float(p)


def collect(policy_path, output_dir, n_episodes, hold_k, max_frames,
            blank_mode, blank_prob, blank_k, blank_seed, env_seed=0, block_len=10):
    model = PPO.load(policy_path)
    env = gym.make("LunarLander-v2", max_episode_steps=max_frames)
    rng = np.random.default_rng(blank_seed)

    trajectories = []
    n_blanked = n_queries = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=env_seed + ep)
        ep_obs = [obs.tolist()]          # TRUE observations
        ep_acts, ep_rews = [], []
        terminal = False
        held_action = None
        step = 0
        done = False
        block_on = False              # streaming block state (per episode; handles variable length)
        while not done:
            if step % hold_k == 0:
                if blank_mode == "block":
                    # every block_len frames, flip one coin for whether the NEXT block is blanked
                    if step % int(block_len) == 0:
                        block_on = rng.random() < float(blank_prob)
                    blanked = block_on
                else:
                    blanked = _blanked(step, blank_mode, blank_prob, blank_k, rng)
                # feed the expert a ZEROED obs on blanked frames -> it acts "blind"
                obs_fed = np.zeros_like(obs) if blanked else obs
                act, _ = model.predict(obs_fed, deterministic=True)
                held_action = int(act)
                n_blanked += int(blanked)
                n_queries += 1
            obs, reward, terminated, truncated, _ = env.step(held_action)
            ep_obs.append(obs.tolist())   # record TRUE next state (physically consistent)
            ep_acts.append(held_action)
            ep_rews.append(float(reward))
            done = bool(terminated or truncated)
            terminal = bool(terminated)
            step += 1

        trajectories.append({
            "obs": ep_obs, "acts": ep_acts, "rews": ep_rews,
            "terminal": terminal, "infos": [json.dumps({}) for _ in ep_acts],
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
    return ds, n_blanked, n_queries


def compute_demo_alignment(ds_path, optimal_policy_path):
    ds = load_from_disk(ds_path)
    model = PPO.load(optimal_policy_path)
    matches = total = 0
    for ep in ds:
        obs = np.asarray(ep["obs"], dtype=np.float32)
        acts = np.asarray(ep["acts"], dtype=np.int64)
        states = obs[:len(acts)]
        pred_acts, _ = model.predict(states, deterministic=True)
        matches += int((np.asarray(pred_acts, dtype=np.int64) == acts).sum())
        total += len(acts)
    return matches, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    p.add_argument("--optimal-policy", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--hold-k", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--blank-mode", choices=["stochastic", "deterministic", "block"], default="stochastic")
    p.add_argument("--blank-prob", type=float, default=0.5)
    p.add_argument("--blank-k", type=int, default=2)
    p.add_argument("--block-len", type=int, default=10,
                   help="block mode: contiguous blackout length in frames (streaming; any episode length).")
    p.add_argument("--blank-seed", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=0)
    args = p.parse_args()

    ds, n_blanked, n_queries = collect(
        args.policy, args.output, args.n_episodes, args.hold_k, args.max_frames,
        args.blank_mode, args.blank_prob, args.blank_k, args.blank_seed, args.env_seed,
        block_len=args.block_len)

    matches, total = compute_demo_alignment(args.output, args.optimal_policy)
    alignment = matches / max(1, total)
    meta = {
        "alignment_fraction": alignment, "matches": matches, "total_transitions": total,
        "optimal_ref": args.optimal_policy, "expert_policy_used_for_collection": args.policy,
        "difficulty": "frame_blank", "blank_mode": args.blank_mode, "blank_prob": args.blank_prob,
        "blank_k": args.blank_k, "block_len": args.block_len, "blank_seed": args.blank_seed,
        "blanked_query_fraction": n_blanked / max(1, n_queries),
        "n_episodes": args.n_episodes, "hold_k": args.hold_k,
        "noise_type": "online_frame_blank_blind_expert",
    }
    with open(Path(args.output) / "demo_alignment.json", "w") as f:
        json.dump(meta, f, indent=2)
    rewards = [sum(ep["rews"]) for ep in ds]
    print(f"  blank={args.blank_prob} align={alignment:.4f} ({matches}/{total}) "
          f"reward_mean={np.mean(rewards):.1f} blanked_frac={n_blanked/max(1,n_queries):.3f}")


if __name__ == "__main__":
    main()
