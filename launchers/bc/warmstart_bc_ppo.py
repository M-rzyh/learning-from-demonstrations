"""BC warm-start -> PPO on the TRUE env reward (Matt's idea #1, variants A/B).

Pipeline for ONE (demo_set, seed):
  1. Build a PPO on LunarLander-v2 with imitation's tuned train_rl config (net_arch=[64,64],
     gamma=0.999, gae_lambda=0.98, ent_coef=0.01, lr=2.5e-4).  <- the config that solves LL on
     the true reward, so this is the right baseline for "do human demos help true-reward RL".
  2. BC-pretrain THAT SAME policy object (bc.BC(policy=ppo.policy)) on the demos -> weights land
     directly in the PPO (no arch mismatch, no state_dict copying).
  3. Eval the BC-only policy (reference point: what cloning alone gets, before any RL).
  4. ppo.learn(total_timesteps) on the true LunarLander-v2 reward, warm-started from step 2.
  5. Eval the final policy -> eval_data/agent_rollouts.npz (SAME format as the GAIL/AIRL eval,
     so it drops straight into the existing plots).

  --outcome landed (with --features_csv) keeps only the landed demos (variant B), using the exact
  same demo_id->outcome verdict as the ranking work, so the 180-count matches what we reported.
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import gymnasium as gym
import imitation.envs.action_repeat  # noqa: F401  (registers FS envs; harmless here)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from imitation.algorithms import bc
from imitation.data import serialize, rollout


def eval_policy(policy, gym_id, max_ep_steps, seed, n_eps):
    """Roll n_eps deterministic episodes in the plain env; return ragged obs/acts/rews arrays."""
    env = gym.make(gym_id, max_episode_steps=max_ep_steps)
    all_obs, all_acts, all_rews, all_term = [], [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        ep_obs, ep_act, ep_rew = [obs.copy()], [], []
        done = False
        while not done:
            a, _ = policy.predict(obs, deterministic=True)
            a = int(a)
            obs, r, term, trunc, _ = env.step(a)
            ep_act.append(a); ep_rew.append(float(r)); ep_obs.append(obs.copy())
            done = bool(term or trunc)
        all_obs.append(np.array(ep_obs, dtype=np.float32))
        all_acts.append(np.array(ep_act, dtype=np.int64))
        all_rews.append(np.array(ep_rew, dtype=np.float64))
        all_term.append(bool(term))
    env.close()
    return all_obs, all_acts, all_rews, all_term


def save_rollouts(path, obs, acts, rews, term):
    np.savez_compressed(
        path,
        obs=np.array(obs, dtype=object), acts=np.array(acts, dtype=object),
        rews=np.array(rews, dtype=object), terminal=np.array(term, dtype=bool),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_demos", type=int, default=0)          # 0 = all
    ap.add_argument("--bc_epochs", type=int, default=50)
    ap.add_argument("--total_timesteps", type=int, default=1_000_000)
    ap.add_argument("--max_ep_steps", type=int, default=1000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--n_eval_episodes", type=int, default=50)
    ap.add_argument("--gym_id", default="LunarLander-v2")
    ap.add_argument("--features_csv", default="")             # for --outcome filtering
    ap.add_argument("--outcome", default="")                  # e.g. "landed" -> variant B
    a = ap.parse_args()

    out = Path(a.out_dir); (out / "eval_data").mkdir(parents=True, exist_ok=True)

    # ---- load demos (with rewards) ----
    try:
        trajs = list(serialize.load_with_rewards(a.demo_path))
    except Exception:
        trajs = list(serialize.load(a.demo_path))
    n_total = len(trajs)

    # ---- optional outcome filter (variant B: landed only), using the ranking CSV's verdict ----
    kept_ids = None
    if a.outcome and a.features_csv:
        rowmap = {int(r["demo_id"]): r["outcome"] for r in csv.DictReader(open(a.features_csv))}
        kept_ids = [i for i in range(n_total) if rowmap.get(i) == a.outcome]
        trajs = [trajs[i] for i in kept_ids]
        print(f"[demos] outcome filter '{a.outcome}': kept {len(trajs)}/{n_total}")
    if a.n_demos and a.n_demos < len(trajs):
        trajs = trajs[: a.n_demos]
    print(f"[demos] using {len(trajs)} trajectories from {a.demo_path}")

    transitions = rollout.flatten_trajectories(trajs)
    print(f"[demos] {len(transitions)} transitions")

    # ---- build PPO (tuned train_rl LunarLander config), true-reward training ----
    venv = sb3_make_vec_env(
        a.gym_id, n_envs=a.n_envs, seed=a.seed,
        env_kwargs=dict(max_episode_steps=a.max_ep_steps),
    )
    ppo = PPO(
        "MlpPolicy", venv, seed=a.seed, verbose=1,
        policy_kwargs=dict(net_arch=[64, 64]),
        n_steps=8192 // a.n_envs, batch_size=64, n_epochs=4,
        gamma=0.999, gae_lambda=0.98, ent_coef=0.01, learning_rate=2.5e-4,
    )

    # ---- BC-pretrain PPO's own policy ----
    rng = np.random.default_rng(a.seed)
    bc_trainer = bc.BC(
        observation_space=venv.observation_space, action_space=venv.action_space,
        rng=rng, demonstrations=transitions, policy=ppo.policy,
    )
    print(f"[bc] pretraining {a.bc_epochs} epochs ...")
    bc_trainer.train(n_epochs=a.bc_epochs)

    # ---- eval BC-only (reference) ----
    o, ac, rw, tm = eval_policy(ppo.policy, a.gym_id, a.max_ep_steps, a.seed, a.n_eval_episodes)
    bc_rewards = [float(r.sum()) for r in rw]
    save_rollouts(out / "eval_data" / "bc_agent_rollouts.npz", o, ac, rw, tm)
    json.dump({"stage": "bc_only", "ep_reward_mean": float(np.mean(bc_rewards)),
               "ep_reward_std": float(np.std(bc_rewards)), "ep_rewards": bc_rewards},
              open(out / "eval_data" / "bc_eval.json", "w"), indent=2)
    print(f"[bc] BC-only mean return = {np.mean(bc_rewards):+.1f} +/- {np.std(bc_rewards):.1f}")

    # ---- RL on true reward, warm-started from BC ----
    print(f"[rl] PPO.learn {a.total_timesteps} steps on true env reward ...")
    ppo.learn(total_timesteps=a.total_timesteps)
    ppo.save(str(out / "final_ppo.zip"))

    # ---- eval final ----
    o, ac, rw, tm = eval_policy(ppo.policy, a.gym_id, a.max_ep_steps, a.seed, a.n_eval_episodes)
    final_rewards = [float(r.sum()) for r in rw]
    save_rollouts(out / "eval_data" / "agent_rollouts.npz", o, ac, rw, tm)
    json.dump({"stage": "bc_warmstart_ppo", "seed": a.seed,
               "demo_path": a.demo_path, "outcome_filter": a.outcome or None,
               "n_demos": len(trajs), "bc_epochs": a.bc_epochs,
               "total_timesteps": a.total_timesteps,
               "gym_id": a.gym_id, "max_episode_steps": a.max_ep_steps,
               "bc_only_reward_mean": float(np.mean(bc_rewards)),
               "ep_reward_mean": float(np.mean(final_rewards)),
               "ep_reward_std": float(np.std(final_rewards)),
               "ep_rewards": final_rewards},
              open(out / "eval_data" / "meta.json", "w"), indent=2)
    print(f"[done] BC-only {np.mean(bc_rewards):+.1f}  ->  BC+RL {np.mean(final_rewards):+.1f} "
          f"+/- {np.std(final_rewards):.1f}  ({len(final_rewards)} eps)")


if __name__ == "__main__":
    main()
