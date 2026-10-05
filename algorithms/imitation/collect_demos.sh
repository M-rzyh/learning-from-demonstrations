#!/bin/bash
#SBATCH --job-name=collect-demos-walker
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/demos/walker/%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/demos/walker/%j.err

set -euo pipefail
mkdir -p "output/collecting_demo"

module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

export MUJOCO_GL=egl
unset PYOPENGL_PLATFORM
unset DISPLAY

which python
python -c "import sys; print('python:', sys.executable)"
python -c "import imitation; print('imitation', imitation.__version__)"

# Headless rendering configuration for DM Control
export MUJOCO_GL=disable
export PYOPENGL_PLATFORM=osmesa

# Import DM Control gymnasium wrapper
export PYTHONPATH="/scratch/marzii/imitation_runs:${PYTHONPATH:-}"

# Register environments before running
python - <<'SETUP_PY'
import sys
sys.path.insert(0, '/scratch/marzii/imitation_runs')
import dm_control_gymnasium_wrapper
SETUP_PY

# ---- choose which expert to demo from (CHANGE THESE) ----
export TOTAL_STEPS=3000000
export EXPERT_JOB_ID=4330738     # <-- set this to the expert training job id
export N_DEMOS=100
export SEED=0
export DEVICE=cuda

# demo output location
# [LABEL: COLLECT_DEMOS_DMC] Changed output directory to reflect DM Control Walker environment
export OUT_DIR="$SCRATCH/imitation_runs/demos/demos_walker_dmc/steps_${TOTAL_STEPS}/expert_job_${EXPERT_JOB_ID}"
# [OLD_SETTING] export OUT_DIR="$SCRATCH/imitation_runs/demos/demos_walker2d_v4/steps_${TOTAL_STEPS}/expert_job_${EXPERT_JOB_ID}"
mkdir -p "$OUT_DIR"

# optional: verify expert exists before running
# MODEL_PATH="$SCRATCH/imitation_runs/expert/expert_ppo_walker_${TOTAL_STEPS}/${EXPERT_JOB_ID}/ppo_walker.zip"
MODEL_PATH="$SCRATCH/imitation_runs/expert/${TOTAL_STEPS}/${EXPERT_JOB_ID}/ppo_walker.zip"
echo "Expect expert at: $MODEL_PATH"
ls -lh "$MODEL_PATH"

python /scratch/marzii/imitation_runs/make_demos_walker.py
