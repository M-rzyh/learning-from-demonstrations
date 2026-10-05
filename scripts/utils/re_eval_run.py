#!/usr/bin/env python3
"""Re-evaluate a finished GAIL run with N episodes.

Loads the trained agent and the optimal expert, rolls out N episodes,
overwrites <run_dir>/eval_data/{agent_rollouts,optimal_at_agent_states,agent_alignment}.

Usage:
    python re_eval_run.py <run_dir> [--n-eps 100] [--optimal-ref PATH]
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imitation.envs.action_repeat  # noqa: F401
import numpy as np
from stable_baselines3 import PPO


def load_meta(run_dir):
    """Read existing meta.json or build from sacred config."""
    eval_dir = Path(run_dir) / "eval_data"
    meta_path = eval_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    # Fallback: read sacred config to get gym_id, max_ep_steps, seed
    cfg_path = Path(run_dir) / "sacred" / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    return {
        "slurm_job_id": Path(run_dir).name,
        "seed": cfg.get("seed", 0),
        "gym_id": cfg.get("environment", {}).get("gym_id", "LunarLander-v2"),
        "max_episode_steps": cfg.get("environment", {}).get("max_episode_steps", 400),
    }


def run_eval(run_dir, n_eps, optimal_ref):
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    gym_id = meta.get("gym_id", "LunarLander-v2")
    max_ep_steps = int(meta.get("max_episode_steps", 400))
    seed = int(meta.get("seed", 0))

    eval_dir = run_dir / "eval_data"
    eval_dir.mkdir(exist_ok=True)

    # Find agent
    candidates = [
        run_dir / "checkpoints" / "final" / "gen_policy" / "model.zip",
        run_dir / "checkpoints" / "final" / "model.zip",
    ]
    agent_path = next((p for p in candidates if p.exists()), None)
    if agent_path is None:
        print(f"[skip] no agent in {run_dir}")
        return False

    agent = PPO.load(str(agent_path))
    optimal = PPO.load(optimal_ref)
    env = gym.make(gym_id, max_episode_steps=max_ep_steps)

    all_obs, all_acts, all_rews, all_terminal, all_opt_acts = [], [], [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        ep_obs = [obs.copy()]
        ep_act, ep_rew, ep_opt = [], [], []
        done = False
        while not done:
            a_agent, _ = agent.predict(obs, deterministic=True)
            a_opt, _ = optimal.predict(obs, deterministic=True)
            ep_act.append(int(a_agent))
            ep_opt.append(int(a_opt))
            obs, r, term, trunc, _ = env.step(int(a_agent))
            ep_rew.append(float(r))
            ep_obs.append(obs.copy())
            done = bool(term or trunc)
        all_obs.append(np.array(ep_obs, dtype=np.float32))
        all_acts.append(np.array(ep_act, dtype=np.int64))
        all_rews.append(np.array(ep_rew, dtype=np.float64))
        all_opt_acts.append(np.array(ep_opt, dtype=np.int64))
        all_terminal.append(bool(term))
    env.close()

    np.savez_compressed(
        eval_dir / "agent_rollouts.npz",
        obs=np.array(all_obs, dtype=object),
        acts=np.array(all_acts, dtype=object),
        rews=np.array(all_rews, dtype=object),
        terminal=np.array(all_terminal, dtype=bool),
    )
    np.savez_compressed(
        eval_dir / "optimal_at_agent_states.npz",
        opt_acts=np.array(all_opt_acts, dtype=object),
    )

    matches = sum(int((a == o).sum()) for a, o in zip(all_acts, all_opt_acts))
    total = sum(len(a) for a in all_acts)
    alignment = matches / max(1, total)
    ep_rewards = [float(r.sum()) for r in all_rews]
    align_json = {
        "alignment_fraction": alignment,
        "matches": matches,
        "total_steps": total,
        "n_episodes": n_eps,
        "ep_rewards": ep_rewards,
        "ep_reward_mean": float(np.mean(ep_rewards)),
        "ep_reward_std": float(np.std(ep_rewards)),
        "optimal_ref": optimal_ref,
        "agent_policy": str(agent_path),
    }
    with open(eval_dir / "agent_alignment.json", "w") as f:
        json.dump(align_json, f, indent=2)
    print(f"  {run_dir.name}: align={alignment:.3f}, eval_reward={np.mean(ep_rewards):+.1f}±{np.std(ep_rewards):.1f}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--n-eps", type=int, default=100)
    p.add_argument("--optimal-ref",
                   default="/scratch/marzii/imitation_runs/expert/lunarlander/4720242/policies/final/model.zip")
    args = p.parse_args()
    run_eval(args.run_dir, args.n_eps, args.optimal_ref)


if __name__ == "__main__":
    main()
