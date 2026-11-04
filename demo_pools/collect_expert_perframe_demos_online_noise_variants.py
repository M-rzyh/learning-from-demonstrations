#!/usr/bin/env python3
"""Online action-noise demo collector with EXACT-% selection and EXCLUDE-EXPERT
replacement — the GAIL analogs of PT's exact-% and flip label noise.

NEW file; does NOT modify collect_fs_expert_perframe_demos_online_noise.py.
Same rollout structure and save format; two extra knobs:

  --noise-select coin   : per-step Bernoulli(p)          (original behaviour)
                exact   : exactly round(p*max_frames) of the horizon's steps are
                          corrupted (pre-drawn per episode) -> removes the corrupted-
                          COUNT wobble. (Minor for GAIL: N demos = hundreds of
                          transitions, so the coin fraction is already tight.)

  --noise-replace uniform : random action over all n_actions   (original; the random
                            action equals the expert's ~1/n_actions of the time -> a
                            "1/4-correct floor" at 100%)
                  exclude : random action over the (n_actions-1) NON-expert actions
                            -> truly wrong action, removes the 1/4 floor (the analog
                            of PT's flip / anti-signal). Expect a bigger drop at 100%.

Trajectory stays physically consistent (env steps with the realized action; errors
compound). Records true 8-D obs + the realized action. Use the non-frame-skip expert
4615187 to match the existing noise sweep.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from stable_baselines3 import PPO


def collect(policy_path, output_dir, n_episodes, hold_k, max_frames,
            noise_prob, noise_seed, noise_select="coin", noise_replace="uniform",
            env_seed=0, n_actions=4):
    model = PPO.load(policy_path)
    env = gym.make("LunarLander-v2", max_episode_steps=max_frames)
    rng = np.random.default_rng(noise_seed)

    trajectories = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=env_seed + ep)
        # EXACT: pre-draw exactly round(p*max_frames) corrupted step-indices for this
        # episode's horizon (episodes ending early realize a fraction very close to p).
        corrupt_at = set()
        if noise_select == "exact" and noise_prob > 0:
            k = int(round(noise_prob * max_frames))
            if k > 0:
                corrupt_at = set(int(x) for x in rng.choice(max_frames, size=k, replace=False))

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

            if noise_select == "exact":
                corrupt = step in corrupt_at
            else:
                corrupt = noise_prob > 0 and rng.random() < noise_prob

            if corrupt:
                if noise_replace == "exclude":
                    choices = [a for a in range(n_actions) if a != held_action]
                    used_action = int(rng.choice(choices))
                else:
                    used_action = int(rng.integers(0, n_actions))
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
    return ds


def compute_demo_alignment(ds_path, optimal_policy_path):
    ds = load_from_disk(ds_path)
    model = PPO.load(optimal_policy_path)
    matches = total = 0
    for ep in ds:
        obs = np.asarray(ep["obs"], dtype=np.float32)
        acts = np.asarray(ep["acts"], dtype=np.int64)
        pred, _ = model.predict(obs[:len(acts)], deterministic=True)
        matches += int((np.asarray(pred, dtype=np.int64) == acts).sum())
        total += len(acts)
    return matches, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--optimal-policy", required=True)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--hold-k", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--noise-prob", type=float, required=True)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--noise-select", choices=["coin", "exact"], default="coin")
    p.add_argument("--noise-replace", choices=["uniform", "exclude"], default="uniform")
    p.add_argument("--noise-mode",
                   choices=["coin_uniform", "exact_uniform", "coin_exclude", "exact_exclude"],
                   default=None,
                   help="Convenience: pick one of the 4 named modes; sets "
                        "--noise-select + --noise-replace together (overrides them). "
                        "coin_uniform reproduces the original collector.")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--n-actions", type=int, default=4)
    args = p.parse_args()

    _MODES = {"coin_uniform": ("coin", "uniform"), "exact_uniform": ("exact", "uniform"),
              "coin_exclude": ("coin", "exclude"), "exact_exclude": ("exact", "exclude")}
    if args.noise_mode:
        args.noise_select, args.noise_replace = _MODES[args.noise_mode]

    ds = collect(args.policy, args.output, args.n_episodes, args.hold_k,
                 args.max_frames, args.noise_prob, args.noise_seed,
                 args.noise_select, args.noise_replace, args.env_seed, args.n_actions)

    matches, total = compute_demo_alignment(args.output, args.optimal_policy)
    alignment = matches / max(1, total)
    meta = {
        "alignment_fraction": alignment, "matches": matches, "total_transitions": total,
        "optimal_ref": args.optimal_policy, "expert_policy_used_for_collection": args.policy,
        "noise_prob": args.noise_prob, "noise_seed": args.noise_seed,
        "noise_select": args.noise_select, "noise_replace": args.noise_replace,
        "noise_type": "online_rollout_variants", "n_episodes": args.n_episodes,
        "hold_k": args.hold_k, "n_actions": args.n_actions,
    }
    with open(Path(args.output) / "demo_alignment.json", "w") as f:
        json.dump(meta, f, indent=2)
    rewards = [sum(ep["rews"]) for ep in ds]
    print(f"  select={args.noise_select} replace={args.noise_replace} p={args.noise_prob} "
          f"align={alignment:.4f} ({matches}/{total}) reward_mean={np.mean(rewards):.1f}")


if __name__ == "__main__":
    main()
