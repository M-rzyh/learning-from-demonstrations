#!/bin/bash
#SBATCH --job-name=walker-ppo
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/expert/walkerwalk/%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/expert/walkerwalk/%j.err

set -euo pipefail

# logs directory for slurm output/error files (your current lines are fine)
mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/expert/walkerwalk

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

# Headless rendering configuration for DM Control
export MUJOCO_GL=disable
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY

# Import DM Control gymnasium wrapper
export PYTHONPATH="/scratch/marzii/imitation_runs:${PYTHONPATH:-}"

# ---- quick mujoco sanity check ----
python - <<'PY'
import gymnasium as gym
# [LABEL: DM_CONTROL_ENV_TEST_EXPERT] Changed from Walker2d-v4 to DMControl/Walker-walk-v0
import dm_control_gymnasium_wrapper
env=gym.make("DMControl/Walker-walk-v0")
# [OLD_SETTING] env=gym.make("Walker2d-v4")
print("ok", env.spec.id, env.spec.max_episode_steps)
env.close()
PY

# ---- train expert PPO ----
export TOTAL_STEPS=3000000
export N_ENVS=8

# pass job id so python can create OUTDIR = base/TOTAL_STEPS/JOBID
export JOB_ID="${SLURM_JOB_ID}"

python /scratch/marzii/imitation_runs/train_expert_walker.py