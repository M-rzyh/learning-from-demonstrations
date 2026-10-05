#!/bin/bash
#SBATCH --job-name=gail-walker2d
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/gail/walker_dmc/%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/gail/walker_dmc/%j.err
#SBATCH --time=02:30:00                   
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# --- your required env setup ---
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

python - <<'PY'
import mujoco, gymnasium as gym
# [LABEL: DM_CONTROL_ENV_TEST] Changed from Walker2d-v4 to DMControl/Walker-walk-v0
# Requires dm-control library and custom wrapper
import dm_control_gymnasium_wrapper
env=gym.make("DMControl/Walker-walk-v0")
# [OLD_SETTING] env=gym.make("Walker2d-v4")
print("ok", env.spec.id, env.spec.max_episode_steps)
env.close()
PY

# --- run GAIL ---
# python -m imitation.scripts.train_adversarial gail \
#   with environment.gym_id=Walker2d-v4 \
#        demonstrations.source=local \
#        demonstrations.path="/scratch/marzii/imitation_runs/demos/demos_walker2d_v4/expert_trajs_50.npz" \
#        demonstrations.n_expert_demos=50 \
#        total_timesteps=20000 \
#        seed=0 \
#        algorithm_kwargs.allow_variable_horizon=True \
#        logging.log_dir="/scratch/marzii/imitation_runs/gail_walker_run_50" \
#        logging.log_format_strs="['tensorboard','stdout']"

#demonstrations.path="/scratch/marzii/imitation_runs/demos/demos_walker2d_v4/steps_${TOTAL_STEPS}/expert_job_${EXPERT_JOB_ID}/expert_trajs_100.npz" \

# [LABEL: GAIL_TRAINING_DMC] Changed environment from Walker2d-v4 to DMControl/Walker-walk-v0
# Set PYTHONPATH to ensure wrapper is importable in all subprocess environments
export PYTHONPATH="/scratch/marzii/imitation_runs:${PYTHONPATH:-}"

# Run training - using 3M expert (performs better than 6M)
python /scratch/marzii/imitation_runs/run_gail_wrapper.py gail \
  with environment.gym_id=DMControl/Walker-walk-v0 \
       demonstrations.source=local \
       demonstrations.path="/scratch/marzii/imitation_runs/demos/demos_walker_dmc/steps_3000000/expert_job_4330738/expert_trajs_100.npz" \
       demonstrations.n_expert_demos=100 \
       total_timesteps=5000000 \
       seed=0 \
       algorithm_kwargs.allow_variable_horizon=True \
       logging.log_root="output/slurm/${SLURM_JOB_ID}" \
       logging.log_format_strs="['tensorboard','stdout']"

# [OLD_SETTING_GAIL] Original GAIL command using Walker2d-v4 (commented out):
# python -m imitation.scripts.train_adversarial gail \
#   with environment.gym_id=Walker2d-v4 \
#        demonstrations.source=local \
#        demonstrations.path="/scratch/marzii/imitation_runs/demos/demos_walker2d_v4/steps_1000000/expert_job_4259452/expert_trajs_100.npz" \
#        demonstrations.n_expert_demos=100 \
#        total_timesteps=5000000 \
#        seed=0 \
#        algorithm_kwargs.allow_variable_horizon=True \
#        logging.log_root="output/slurm/${SLURM_JOB_ID}" \
#        logging.log_format_strs="['tensorboard','stdout']"