#!/bin/bash
#SBATCH --job-name=gail-walker2d
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/gail/walker_dmc/%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/gail/walker_dmc/%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# --- required env setup (Alliance Canada) ---
module --force purge
module load StdEnv/2023 python/3.10.13 glfw/3.4

export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r

# ensure conda base is not active
source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda deactivate || true

# activate clean venv
source /scratch/marzii/envs/imitation-clean/bin/activate
hash -r

export MUJOCO_PATH=/scratch/marzii/mujoco/mujoco-3.3.2
export MUJOCO_PLUGIN_PATH=/scratch/marzii/mujoco/mujoco-3.3.2/bin
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

which python
python -c "import sys; print(sys.executable)"
python -c "import imitation; print('imitation', imitation.__version__)"

# Import DM Control gymnasium wrapper
export PYTHONPATH="/home/marzii/IRL2/imitation:/scratch/marzii/imitation_runs:${PYTHONPATH:-}"

python - <<'PY'
import gymnasium as gym
import dm_control_gymnasium_wrapper
env = gym.make("DMControl/Walker-walk-v0")
print("ok", env.spec.id, env.spec.max_episode_steps)
env.close()
PY

# --- run GAIL ---
export PYTHONPATH="/home/marzii/IRL2/imitation:/scratch/marzii/imitation_runs:${PYTHONPATH:-}"

python /scratch/marzii/imitation_runs/run_gail_wrapper.py gail \
  with environment.gym_id=DMControl/Walker-walk-v0 \
       demonstrations.source=local \
       demonstrations.path="/scratch/marzii/imitation_runs/demos/demos_walker_dmc/steps_12000000/expert_job_4310732/expert_trajs_100.npz" \
       demonstrations.n_expert_demos=5 \
       total_timesteps=5000000 \
       seed=0 \
       algorithm_kwargs.allow_variable_horizon=True \
       logging.log_root="output/slurm/${SLURM_JOB_ID}" \
       logging.log_format_strs="['tensorboard','stdout']"
