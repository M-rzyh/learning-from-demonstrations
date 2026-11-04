#!/bin/bash
#SBATCH --job-name=highway-ppo
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/expert/highway/%x_%j.out
#SBATCH --error=/scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/expert/highway/%x_%j.err
#
# PPO expert on highway-env highway-v0 (default env config), via the library's train_rl.
# Adapted from train_expert_lunar_lander.sh (unchanged): `highway` named config (PPO settings from
# highway-env's own SB3 PPO example), highway-env 1.9.1 from a side folder, CPU only.
# Saves, like the LunarLander expert:
#   <run>/policies/final/model.zip   the expert (OPTIMAL_REF for GAIL eval)
#   <run>/rollouts/final.npz         N_ROLLOUTS expert episodes = expert demos for BC/GAIL
#   <run>/log/videos/                a few expert episodes as video
#
#   sbatch launchers/rl/train_expert_highway.sh                      # TOTAL_STEPS=200000 by default
#   TOTAL_STEPS=20000 sbatch --time=00:30:00 launchers/rl/train_expert_highway.sh   # short pilot
#   USE_WANDB=1 sbatch launchers/rl/train_expert_highway.sh          # also log to W&B (project expert-highway)
set -euo pipefail
mkdir -p /scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/expert/highway

module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r
source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r
HIGHWAY_PKGS=/scratch/marzii/envs/highway-pkgs
export PYTHONPATH="/home/marzii/learning-from-demonstrations/algorithms/imitation/src:${HIGHWAY_PKGS}:${PYTHONPATH:-}"
export SDL_VIDEODRIVER=offscreen   # no screen; never "dummy" (highway-env then skips drawing)

python -c "import imitation; print('imitation', imitation.__version__)"

ENV_GYM_ID="highway_env:highway-v0"
ENV_MAX_EP_STEPS=${ENV_MAX_EP_STEPS:-40}
TOTAL_STEPS=${TOTAL_STEPS:-200000}
N_ENVS=${N_ENVS:-8}
PARALLEL=${PARALLEL:-True}   # True = one process per env (highway-v0 is CPU-bound: ~13 steps/s with all 8 envs in one process)
N_ROLLOUTS=${N_ROLLOUTS:-50}
N_VIDEOS=${N_VIDEOS:-3}
SEED=${SEED:-}
SEED_ARGS=()
if [[ -n "$SEED" ]]; then SEED_ARGS=("seed=$SEED"); fi

JOB_ID="${SLURM_JOB_ID:-local}"
LOG_ROOT=${LOG_ROOT:-"/scratch/marzii/learning-from-demonstrations/imitation_runs/expert"}
RUN_DIR="$LOG_ROOT/highway/$JOB_ID"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"   # work from the run dir on scratch, so cwd-relative output lands there, never in $HOME
echo "env=$ENV_GYM_ID cap=$ENV_MAX_EP_STEPS  total_steps=$TOTAL_STEPS  n_envs=$N_ENVS parallel=$PARALLEL  rollouts=$N_ROLLOUTS  seed=${SEED:-<random>}"
echo "run dir: $RUN_DIR"

# ---- optional Weights & Biases logging (USE_WANDB=1), same switch as launchers/gail/run_gail_highway.sh.
#      Default 0 keeps the command unchanged. Needs `wandb login` once; compute nodes reach wandb through
#      the cluster proxy. TensorBoard is written either way. ----
USE_WANDB=${USE_WANDB:-0}
LOG_FMT="['tensorboard','stdout']"
WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  LOG_FMT="['tensorboard','stdout','wandb']"
  WANDB_PROJECT=${WANDB_PROJECT:-expert-highway}
  RUN_TAG=${RUN_TAG:-ppo-${TOTAL_STEPS}}
  WANDB_ARGS=(
    logging.wandb.wandb_name_prefix="expert-${RUN_TAG}"
    logging.wandb.wandb_tag="$RUN_TAG"
    logging.wandb.wandb_kwargs.project="$WANDB_PROJECT"
  )
  echo "wandb: ON  project=$WANDB_PROJECT  tag=$RUN_TAG"
fi

# EXTRA_ARGS: extra sacred overrides, space-separated (e.g. for quick tests)
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

python /home/marzii/learning-from-demonstrations/launchers/entrypoints/train_rl_launcher.py \
  train_rl \
  with highway \
  total_timesteps="$TOTAL_STEPS" \
  environment.num_vec="$N_ENVS" \
  environment.parallel="$PARALLEL" \
  environment.gym_id="$ENV_GYM_ID" \
  environment.max_episode_steps="$ENV_MAX_EP_STEPS" \
  rollout_save_n_episodes="$N_ROLLOUTS" \
  rollout_save_n_timesteps=None \
  policy_save_interval=50000 \
  logging.log_dir="$RUN_DIR" \
  logging.log_format_strs="$LOG_FMT" \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
  ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
  ${EXTRA[@]+"${EXTRA[@]}"}

# ---- quick quality check + a few videos ----
python - "$RUN_DIR" "$ENV_GYM_ID" "$ENV_MAX_EP_STEPS" "$N_VIDEOS" <<'PY'
import json, sys, pathlib
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

run_dir, env_id, max_ep, n_videos = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
model = PPO.load(run_dir / "policies" / "final" / "model.zip")

def run(env, n, seed0):
    rets, crashes, speeds = [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep); done, ret, sp = False, 0.0, []
        while not done:
            obs, r, term, trunc, info = env.step(int(model.predict(obs, deterministic=True)[0]))
            ret += r; sp.append(info.get("speed", 0.0)); done = term or trunc
        rets.append(ret); crashes.append(bool(info.get("crashed", False))); speeds.append(float(np.mean(sp)))
    return rets, crashes, speeds

env = gym.make(env_id, max_episode_steps=max_ep)
rets, crashes, speeds = run(env, 20, 10_000)
env.close()
summary = {"n_episodes": 20, "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
           "crash_rate": float(np.mean(crashes)), "mean_speed": float(np.mean(speeds)), "max_return": max_ep}
json.dump(summary, open(run_dir / "expert_quality.json", "w"), indent=2)
print("[expert check]", summary)

if n_videos > 0:
    env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_ep)
    env.unwrapped.configure({"offscreen_rendering": True})   # highway-env 1.9.1 default is False
    env = RecordVideo(env, str(run_dir / "log" / "videos"), episode_trigger=lambda e: e < n_videos)
    run(env, n_videos, 20_000)
    env.close()
    print("videos:", run_dir / "log" / "videos")
PY
echo "expert run $RUN_DIR done"
