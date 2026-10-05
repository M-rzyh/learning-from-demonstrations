#!/bin/bash
#SBATCH --job-name=ppo-truereward-matched
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/ppo_truereward_matched/lunarlander/%x_%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/ppo_truereward_matched/lunarlander/%x_%j.err
#
# GROUND-TRUTH-REWARD PPO BASELINE, matched to the GAIL generator (E4, cap-1000
# count axis). Same learner, same budget, same evaluation; the ONLY difference is
# that PPO optimizes the true LunarLander reward instead of the discriminator.
#
# Matched by USING THE SAME INGREDIENT DEFAULTS GAIL USES -- deliberately NOT
# `with lunar_lander`, whose named config would override the policy to MlpPolicy
# [64,64] and rl to batch 8192 / lr 2.5e-4 / 4 epochs / gamma 0.999 / lambda 0.98 /
# ent_coef 0.01. Leaving those at the ingredient defaults reproduces the generator:
#   policy : FeedForward32Policy, net_arch [32,32], Tanh, ortho init  (policy.py:23-24)
#   rl     : PPO, batch_size 2048 -> n_steps 256 x 8 envs, minibatch 64,
#            n_epochs 10, lr 3e-4, ent_coef 0.0                        (rl.py:57-65)
#   sb3    : gamma 0.99, gae_lambda 0.95, clip 0.2, vf_coef 0.5,
#            max_grad_norm 0.5, normalize_advantage True, target_kl None
#
# Explicit overrides below: env id / cap 1000 / 8 envs, total_timesteps=999424
# (= 488 x 2048, the exact number of env steps the GAIL generator consumes), and
# normalize_reward=False (train_rl defaults it to True via VecNormalize; GAIL has
# no VecNormalize anywhere).
#
# Outputs (never touches expert/ or ppo_baseline/):
#   /scratch/marzii/imitation_runs/ppo_truereward_matched/lunarlander/<jobid>/
#       policies/final/model.zip   final policy
#       eval_data/                 50-episode deterministic eval, GAIL protocol
#       monitor/, log/, rollouts/
#
# Submit one seed:  sbatch --export=ALL,SEED=0 imitation/run_ppo_truereward_matched.sh

set -euo pipefail

mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/ppo_truereward_matched/lunarlander

# ---- required env setup (identical to train_expert_lunar_lander.sh) ----
module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

# Use the local repo, not the installed package. MUST precede the import check.
export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"

which python
python -c "import imitation; print('imitation', imitation.__version__)"

cd /home/marzii/IRL3/imitation

# ---- matched settings ----
SEED=${SEED:-0}
N_ENVS=${N_ENVS:-8}
ENV_GYM_ID=${ENV_GYM_ID:-LunarLander-v2}
ENV_MAX_EP_STEPS=${ENV_MAX_EP_STEPS:-1000}
# 488 rollouts x 2048 steps. SB3's loop is `while num_timesteps < total_timesteps`
# and each rollout adds n_steps*num_envs = 2048, so an exact multiple yields
# exactly 488 iterations and exactly 999,424 env steps -- no rounding up.
TOTAL_STEPS=${TOTAL_STEPS:-999424}
N_EVAL_EPISODES=${N_EVAL_EPISODES:-50}

LOG_ROOT=${LOG_ROOT:-/scratch/marzii/imitation_runs/ppo_truereward_matched}
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR="$LOG_ROOT/lunarlander/$JOB_ID"
mkdir -p "$RUN_DIR"

echo "=== PPO true-reward baseline (matched to GAIL generator) ==="
echo "seed=$SEED  envs=$N_ENVS  cap=$ENV_MAX_EP_STEPS  total_timesteps=$TOTAL_STEPS"
echo "run dir: $RUN_DIR"

python train_rl_launcher.py \
  train_rl \
  with \
  environment.gym_id="$ENV_GYM_ID" \
  environment.max_episode_steps=$ENV_MAX_EP_STEPS \
  environment.num_vec=$N_ENVS \
  environment.parallel=True \
  total_timesteps=$TOTAL_STEPS \
  normalize_reward=False \
  rollout_save_n_episodes=50 \
  rollout_save_n_timesteps=None \
  policy_save_interval=50000 \
  policy_save_final=True \
  seed=$SEED \
  logging.log_dir="$RUN_DIR" \
  logging.log_format_strs="['tensorboard','stdout']"

# ---- evaluation: identical protocol to run_gail_lunarlander.sh ----
# 50 deterministic episodes, fresh unwrapped env at the same cap, env seeded
# seed*1000+ep, scored with the TRUE environment reward.
echo ""
echo "[eval] 50-episode deterministic evaluation"
python - "$RUN_DIR" "$ENV_GYM_ID" "$ENV_MAX_EP_STEPS" "$SEED" "$N_EVAL_EPISODES" <<'PY'
import json, sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

run_dir, gym_id, max_ep_steps, seed, n_eps = sys.argv[1:6]
max_ep_steps, seed, n_eps = int(max_ep_steps), int(seed), int(n_eps)

agent_path = Path(run_dir) / "policies" / "final" / "model.zip"
if not agent_path.exists():
    print(f"[eval] no final policy at {agent_path}; skipping")
    sys.exit(0)

eval_dir = Path(run_dir) / "eval_data"
eval_dir.mkdir(parents=True, exist_ok=True)

print(f"[eval] loading {agent_path}")
agent = PPO.load(str(agent_path))
env = gym.make(gym_id, max_episode_steps=max_ep_steps)

all_obs, all_acts, all_rews, all_terminal, ep_returns = [], [], [], [], []
for ep in range(n_eps):
    obs, _ = env.reset(seed=seed * 1000 + ep)
    ep_obs, ep_act, ep_rew = [obs.copy()], [], []
    done = False
    while not done:
        a, _ = agent.predict(obs, deterministic=True)
        ep_act.append(int(a))
        obs, r, term, trunc, _ = env.step(int(a))
        ep_rew.append(float(r))
        ep_obs.append(obs.copy())
        done = bool(term or trunc)
    all_obs.append(np.array(ep_obs, dtype=np.float32))
    all_acts.append(np.array(ep_act, dtype=np.int64))
    all_rews.append(np.array(ep_rew, dtype=np.float64))
    all_terminal.append(bool(term))
    ep_returns.append(float(np.sum(ep_rew)))
env.close()

np.savez_compressed(
    eval_dir / "agent_rollouts.npz",
    obs=np.array(all_obs, dtype=object),
    acts=np.array(all_acts, dtype=object),
    rews=np.array(all_rews, dtype=object),
    terminal=np.array(all_terminal, dtype=bool),
)

r = np.asarray(ep_returns)
summary = dict(
    run_dir=str(run_dir),
    condition="ppo_truereward_matched",
    reward_source="ground_truth_env_reward",
    gym_id=gym_id,
    max_episode_steps=max_ep_steps,
    seed=seed,
    n_eval_episodes=n_eps,
    deterministic=True,
    eval_seed_convention="seed*1000+ep",
    ep_rewards=ep_returns,
    mean_return=float(r.mean()),
    sd_return=float(r.std(ddof=1)),
    se_return=float(r.std(ddof=1) / np.sqrt(len(r))),
    frac_return_above_200=float((r > 200).mean()),
    mean_episode_length=float(np.mean([len(a) for a in all_acts])),
)
with open(eval_dir / "agent_alignment.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(eval_dir / "meta.json", "w") as f:
    json.dump({k: v for k, v in summary.items() if k != "ep_rewards"}, f, indent=2)

print(json.dumps({k: v for k, v in summary.items() if k != "ep_rewards"}, indent=2))
print(f"[eval] wrote {eval_dir/'agent_alignment.json'}")
PY

echo "done: seed=$SEED  $RUN_DIR"
