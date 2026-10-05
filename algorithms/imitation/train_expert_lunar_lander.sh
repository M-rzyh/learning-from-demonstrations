#!/bin/bash
#SBATCH --job-name=lunarlander-ppo
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/expert/lunarlander/%x_%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/expert/lunarlander/%x_%j.err

set -euo pipefail

mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/expert/lunarlander

# ---- required env setup ----
module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

# Ensure we use the local repo (with lunar_lander config) instead of the installed package.
# MUST come before the `import imitation` check below: under `set -e` the check aborts the job
# in ~1s otherwise. Same fix as run_gail_lunarlander.sh (WORKLOG 2026-06-30).
export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"

which python
python -c "import sys; print(sys.executable)"
python -c "import imitation; print('imitation', imitation.__version__)"

# ---- quick sanity check ----
python - <<'PY'
import gymnasium as gym
env = gym.make("LunarLander-v2")
print("ok", env.spec.id, env.spec.max_episode_steps)
env.close()
PY

# ---- train expert PPO via train_rl ----
export TOTAL_STEPS=1000000
export N_ENVS=8

# Optional live rendering (slow). Example:
#   RENDER_MODE=human sbatch train_expert_lunar_lander.sh
RENDER_MODE=${RENDER_MODE:-}
EXTRA_ENV_ARGS=()
if [[ -n "${RENDER_MODE}" ]]; then
  EXTRA_ENV_ARGS=(
    "environment.num_vec=1"
    "environment.parallel=False"
    "environment.env_make_kwargs.render_mode=${RENDER_MODE}"
  )
fi

# ---- frame-skip (action repeat k=10) toggle ----
# FRAME_SKIP=0 (default): original behavior (LunarLander-v2, max_ep_steps=400)
# FRAME_SKIP=1: use LunarLander-v2-FS10 (policy decides every 10 env steps; 40 decisions/ep)
FRAME_SKIP=${FRAME_SKIP:-0}
ENV_GYM_ID="LunarLander-v2"
# Overridable episode cap. Default 400 = every previous expert run, unchanged.
# Set ENV_MAX_EP_STEPS=1000 to match the BC warm-start / PT pipelines.
ENV_MAX_EP_STEPS=${ENV_MAX_EP_STEPS:-400}
VIDEO_ENV_ID="LunarLander-v2"
if [[ "$FRAME_SKIP" == "1" ]]; then
  ENV_GYM_ID="LunarLander-v2-FS10"
  ENV_MAX_EP_STEPS=40
  VIDEO_ENV_ID="LunarLander-v2-FS10"
fi
echo "Frame skip: $FRAME_SKIP  (gym_id=$ENV_GYM_ID, max_ep_steps=$ENV_MAX_EP_STEPS)"

# Optional explicit seed. Unset = sacred draws a random one (what runs 4615153/70/87 did,
# so their behaviour is unchanged); set SEED=<n> for a reproducible sweep.
SEED=${SEED:-}
SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS=("seed=$SEED")
fi
echo "Seed:       ${SEED:-<random>}"

# pass job id so python can create OUTDIR = base/ENV_NAME/JOB_ID
export JOB_ID="${SLURM_JOB_ID}"

# Env-name-based structure: imitation_runs/<root>/<env_name>/<job_id>
# LOG_ROOT is overridable so a PPO *baseline* sweep can be kept out of expert/,
# which is provenance for the demo pools (4615187 in particular is referenced by
# explicit path from the demo collectors and from run_gail_lunarlander.sh OPTIMAL_REF).
ENV_NAME="lunarlander"
LOG_ROOT=${LOG_ROOT:-"/scratch/marzii/imitation_runs/expert"}
RUN_DIR="$LOG_ROOT/$ENV_NAME/$JOB_ID"
echo "Log root:   $LOG_ROOT"
mkdir -p "$RUN_DIR"

python train_rl_launcher.py \
  train_rl \
  with lunar_lander \
  total_timesteps=$TOTAL_STEPS \
  environment.num_vec=$N_ENVS \
  environment.gym_id="$ENV_GYM_ID" \
  environment.max_episode_steps=$ENV_MAX_EP_STEPS \
  rollout_save_n_episodes=50 \
  rollout_save_n_timesteps=None \
  policy_save_interval=50000 \
  logging.log_dir="$RUN_DIR" \
  logging.log_format_strs="['tensorboard','stdout']" \
  ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
  "${EXTRA_ENV_ARGS[@]}"

# ---- record 5 expert episodes as video ----
echo "Recording 5 expert episodes from: $RUN_DIR"

python - "$RUN_DIR" "$VIDEO_ENV_ID" "$ENV_MAX_EP_STEPS" <<'PY'
import sys, pathlib
import imitation.envs.action_repeat  # noqa: F401  (registers LunarLander-v2-FS10)
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

run_dir = pathlib.Path(sys.argv[1])
env_id = sys.argv[2]
max_ep_steps = int(sys.argv[3])
policy_path = run_dir / "policies" / "final" / "model.zip"
video_dir = run_dir / "log" / "videos"

model = PPO.load(policy_path)
env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_ep_steps)
env = RecordVideo(env, str(video_dir), episode_trigger=lambda e: e < 5)

for ep in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {ep+1}: reward={total_reward:.1f}")

env.close()
print(f"Videos saved to: {video_dir}")
PY

# Original, unchanged:
#   sbatch train_expert_lunar_lander.sh

# With frame skip (k=10):
#   FRAME_SKIP=1 sbatch train_expert_lunar_lander.sh 