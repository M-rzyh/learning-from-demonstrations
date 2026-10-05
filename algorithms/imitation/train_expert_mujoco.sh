#!/bin/bash
#SBATCH --job-name=expert-walker-ppo-mujoco
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/expert/mujoco/%x_%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/expert/mujoco/%x_%j.err

set -euo pipefail

# logs directory for slurm output/error files (your current lines are fine)
mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/expert/mujoco

# ---- required env setup ----
module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

which python
python -c "import sys; print(sys.executable)"
python -c "import imitation; print('imitation', imitation.__version__)"

# [LABEL: MUJOCO_VERSION_SCRIPT] MuJoCo version - no special rendering config needed
# (keeping osmesa config for consistency, but not required for MuJoCo)
export MUJOCO_GL=osmesa
unset DISPLAY

# Import path for scripts
export PYTHONPATH="/scratch/marzii/imitation_runs:${PYTHONPATH:-}"

# ---- quick mujoco sanity check ----
python - <<'PY'
import gymnasium as gym
# [LABEL: ENV_TEST_MUJOCO] Testing standard Gymnasium Walker2d-v4
env=gym.make("Walker2d-v4")
print("ok", env.spec.id, env.spec.max_episode_steps)
env.close()
PY

# ---- train expert PPO ----
export TOTAL_STEPS=12000000
export N_ENVS=8

# pass job id so python can create OUTDIR = base/TOTAL_STEPS/JOBID
export JOB_ID="${SLURM_JOB_ID}"

python /scratch/marzii/imitation_runs/train_expert_walker_mujoco.py
