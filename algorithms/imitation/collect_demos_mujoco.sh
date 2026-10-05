#!/bin/bash
#SBATCH --job-name=collect-demos-walker-mujoco
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/demos/mujoco/%x_%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/demos/mujoco/%x_%j.err

set -euo pipefail

mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/demos/mujoco

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

# [LABEL: MUJOCO_VERSION_SCRIPT] MuJoCo version
export MUJOCO_GL=osmesa
unset DISPLAY

export PYTHONPATH="/scratch/marzii/imitation_runs:${PYTHONPATH:-}"

# ---- collect expert demonstrations ----
export TOTAL_STEPS=6000000
export N_DEMOS=100

# Optional: specify expert job ID (if not specified, uses latest)
# export EXPERT_JOB_ID="12345"

export JOB_ID="${SLURM_JOB_ID}"

# [LABEL: COLLECT_DEMOS_MUJOCO] Collect demonstrations from MuJoCo expert
python /scratch/marzii/imitation_runs/make_demos_walker_mujoco.py
