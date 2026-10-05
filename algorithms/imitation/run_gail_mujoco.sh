#!/bin/bash
#SBATCH --job-name=gail-walker-mujoco
#SBATCH --account=aip-mtaylor3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/gail/mujoco/%x_%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/gail/mujoco/%x_%j.err

set -euo pipefail

mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/gail/mujoco

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

# ---- GAIL training ----
python -m imitation.scripts.train_adversarial gail \
  with environment = dict( \
           gym_id = "Walker2d-v4", \
       ) \
       demonstrations = dict( \
           trajectory_path = "/scratch/marzii/imitation_runs/demos_walker_mujoco/steps_6000000", \
       ) \
       sample_until = dict( \
           n_timesteps = None, \
       ) \
       trainer_kwargs = dict( \
           disc_opt_kwargs = dict(batch_size=16), \
           gen_train_timesteps = None, \
           n_disc_updates_per_round = 10, \
       ) \
       total_timesteps = 5000000 \
       reward_net_kwargs = dict( \
           normalize_input_layer=None, \
       )
