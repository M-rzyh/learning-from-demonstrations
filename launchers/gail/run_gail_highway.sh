#!/bin/bash
#SBATCH --job-name=gail-highway
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/gail/highway/%x_%j.out
#SBATCH --error=/scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/gail/highway/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#
# GAIL (or AIRL with ALGO=airl) on highway-env highway-v0, default env config.
# Adapted from run_gail_lunarlander.sh; that file is unchanged. Differences:
#   - env: highway_env:highway-v0, 40-step episodes (named config `highway`), no frame-skip option
#   - highway-env 1.9.1 comes from a side folder on PYTHONPATH (imitation-gail itself is untouched)
#   - CPU only: the 5x5 observation is tiny and the simulator is CPU-bound
#   - DEMO_PATH is required (converted demos: envs/highway/convert_highway_demos.py, or an
#     expert's rollouts/final.npz); TOTAL_TIMESTEPS is overridable
#   - OPTIMAL_REF is optional: without an expert policy the eval still records the agent's
#     returns, it just skips the action-alignment numbers
#
#   DEMO_PATH=<converted demo dir> N_DEMOS=50 SEED=0 sbatch launchers/gail/run_gail_highway.sh

set -euo pipefail

mkdir -p /scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/gail/highway

# ---- required env setup ----
module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

# local imitation source + highway-env 1.9.1 side folder
HIGHWAY_PKGS=/scratch/marzii/envs/highway-pkgs
export PYTHONPATH="/home/marzii/learning-from-demonstrations/algorithms/imitation/src:${HIGHWAY_PKGS}:${PYTHONPATH:-}"
export SDL_VIDEODRIVER=offscreen   # no screen on compute nodes; never "dummy" (highway-env then skips drawing)

which python
python -c "import imitation; print('imitation', imitation.__version__)"

ENV_GYM_ID="highway_env:highway-v0"
ENV_MAX_EP_STEPS=${ENV_MAX_EP_STEPS:-40}

# ---- sanity check ----
python - "$ENV_GYM_ID" <<'PY'
import sys
from importlib.metadata import version
import gymnasium as gym
env = gym.make(sys.argv[1])
c = env.unwrapped.config
print("ok", sys.argv[1], "highway-env", version("highway-env"), "obs=", env.observation_space,
      "act=", env.action_space, "sim/policy freq=", c["simulation_frequency"], c["policy_frequency"])
env.close()
PY

# ---- demonstrations ----
DEMO_SOURCE=${DEMO_SOURCE:-local}
DEMO_PATH=${DEMO_PATH:?set DEMO_PATH to a converted demo dir or an expert rollouts/final.npz}
N_DEMOS=${N_DEMOS:-50}
DEMO_BATCH_SIZE=${DEMO_BATCH_SIZE:-256}   # highway demos are short (<=40 steps); 50 demos <= 2000 transitions
SEED=${SEED:-1}
SHUFFLE=${SHUFFLE:-0}                     # 0 = first N demos, 1 = shuffle first (SHUFFLE_SEED)
SHUFFLE_SEED=${SHUFFLE_SEED:-$SEED}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-200000}
N_ENVS=${N_ENVS:-8}
PARALLEL=${PARALLEL:-True}   # True = one process per env (highway-v0 is CPU-bound: ~13 steps/s with all 8 envs in one process)
DEMO_NOTE=${DEMO_NOTE:-}

ALGO=${ALGO:-gail}
case "$ALGO" in
  airl) ALGO_DIR="AIRL" ;;
  *)    ALGO_DIR="$ALGO" ;;
esac
ENV_NAME="highway"
LOG_ROOT="/scratch/marzii/learning-from-demonstrations/imitation_runs/$ALGO_DIR"
RUN_DIR="$LOG_ROOT/$ENV_NAME/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"   # work from the run dir on scratch, so cwd-relative output lands there, never in $HOME

echo "algorithm:       $ALGO"
echo "env:             $ENV_GYM_ID  (max_ep_steps=$ENV_MAX_EP_STEPS)"
echo "demo path:       $DEMO_PATH  (source=$DEMO_SOURCE)"
echo "N demos:         $N_DEMOS   demo batch size: $DEMO_BATCH_SIZE"
echo "seed:            $SEED   shuffle: $SHUFFLE (shuffle_seed=$SHUFFLE_SEED)"
echo "total timesteps: $TOTAL_TIMESTEPS   n envs: $N_ENVS   parallel: $PARALLEL"
echo "run dir:         $RUN_DIR"
if [[ -n "$DEMO_NOTE" ]]; then echo "demo note:       $DEMO_NOTE"; fi

# ---- optional: shuffle demos reproducibly ----
if [[ "$SHUFFLE" == "1" ]]; then
  SHUFFLED_DIR="$RUN_DIR/shuffled_demos"
  python - "$DEMO_PATH" "$SHUFFLED_DIR" "$SHUFFLE_SEED" <<'PY'
import sys
from datasets import load_from_disk
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = load_from_disk(src).shuffle(seed=seed)
ds.save_to_disk(dst)
print(f"Shuffled {len(ds)} trajectories, saved to {dst}")
PY
  DEMO_PATH="$SHUFFLED_DIR"
fi

# ---- optional Weights & Biases logging (USE_WANDB=1) ----
USE_WANDB=${USE_WANDB:-0}
LOG_FMT="['tensorboard','stdout']"
WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  LOG_FMT="['tensorboard','stdout','wandb']"
  WANDB_PROJECT=${WANDB_PROJECT:-gail-highway}
  RUN_TAG=${RUN_TAG:-$(basename "$DEMO_PATH")}
  WANDB_ARGS=(
    logging.wandb.wandb_name_prefix="${ALGO}-${RUN_TAG}"
    logging.wandb.wandb_tag="$RUN_TAG"
    logging.wandb.wandb_kwargs.project="$WANDB_PROJECT"
  )
fi

# EXTRA_ARGS: extra sacred overrides, space-separated (e.g. for quick tests)
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

python /home/marzii/learning-from-demonstrations/launchers/entrypoints/train_adversarial_launcher.py "$ALGO" \
  with highway \
  demonstrations.source="$DEMO_SOURCE" \
  demonstrations.path="$DEMO_PATH" \
  demonstrations.n_expert_demos="$N_DEMOS" \
  algorithm_kwargs.demo_batch_size="$DEMO_BATCH_SIZE" \
  total_timesteps="$TOTAL_TIMESTEPS" \
  environment.num_vec="$N_ENVS" \
  environment.parallel="$PARALLEL" \
  environment.gym_id="$ENV_GYM_ID" \
  environment.max_episode_steps="$ENV_MAX_EP_STEPS" \
  seed="$SEED" \
  logging.log_dir="$RUN_DIR" \
  logging.log_format_strs="$LOG_FMT" \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
  ${EXTRA[@]+"${EXTRA[@]}"}

# ---- post-training evaluation: rollouts in the true env (+ action alignment if OPTIMAL_REF) ----
SKIP_EVAL=${SKIP_EVAL:-0}
OPTIMAL_REF=${OPTIMAL_REF:-}      # e.g. .../expert/highway/<job>/policies/final/model.zip
N_EVAL_EPISODES=${N_EVAL_EPISODES:-50}
EVAL_DATA_DIR="$RUN_DIR/eval_data"

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo ""
  echo "===== post-training evaluation ($N_EVAL_EPISODES episodes; optimal ref: ${OPTIMAL_REF:-none}) ====="
  mkdir -p "$EVAL_DATA_DIR"
  python - "$RUN_DIR" "$OPTIMAL_REF" "$EVAL_DATA_DIR" "$N_EVAL_EPISODES" "$ENV_GYM_ID" "$ENV_MAX_EP_STEPS" "$SEED" "$DEMO_PATH" "$N_DEMOS" "$DEMO_BATCH_SIZE" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

run_dir, opt_ref, eval_dir, n_eps, gym_id, max_ep_steps, seed, demo_path, n_demos, demo_bs = sys.argv[1:11]
n_eps = int(n_eps); max_ep_steps = int(max_ep_steps); seed = int(seed)
n_demos = int(n_demos); demo_bs = int(demo_bs)

candidates = [Path(run_dir) / "checkpoints" / "final" / "gen_policy" / "model.zip",
              Path(run_dir) / "checkpoints" / "final" / "model.zip"]
agent_path = next((p for p in candidates if p.exists()), None)
if agent_path is None:
    print(f"[eval] no agent policy found in {run_dir}; skipping eval")
    sys.exit(0)
agent = PPO.load(str(agent_path))
optimal = PPO.load(opt_ref) if opt_ref else None

env = gym.make(gym_id, max_episode_steps=max_ep_steps)
all_obs, all_acts, all_rews, all_terminal, all_opt_acts, crashed = [], [], [], [], [], []
for ep in range(n_eps):
    obs, _ = env.reset(seed=seed * 1000 + ep)
    ep_obs, ep_act, ep_rew, ep_opt = [obs.copy()], [], [], []
    done = False
    while not done:
        a_agent, _ = agent.predict(obs, deterministic=True)
        if optimal is not None:
            ep_opt.append(int(optimal.predict(obs, deterministic=True)[0]))
        ep_act.append(int(a_agent))
        obs, r, term, trunc, info = env.step(int(a_agent))
        ep_rew.append(float(r)); ep_obs.append(obs.copy())
        done = bool(term or trunc)
    all_obs.append(np.array(ep_obs, dtype=np.float32)); all_acts.append(np.array(ep_act, dtype=np.int64))
    all_rews.append(np.array(ep_rew, dtype=np.float64)); all_opt_acts.append(np.array(ep_opt, dtype=np.int64))
    all_terminal.append(bool(term)); crashed.append(bool(info.get("crashed", False)))
env.close()

np.savez_compressed(Path(eval_dir) / "agent_rollouts.npz",
    obs=np.array(all_obs, dtype=object), acts=np.array(all_acts, dtype=object),
    rews=np.array(all_rews, dtype=object), terminal=np.array(all_terminal, dtype=bool),
    crashed=np.array(crashed, dtype=bool))
ep_rewards = [float(r.sum()) for r in all_rews]
out = {"n_episodes": n_eps, "ep_rewards": ep_rewards,
       "ep_reward_mean": float(np.mean(ep_rewards)), "ep_reward_std": float(np.std(ep_rewards)),
       "crash_rate": float(np.mean(crashed)), "agent_policy": str(agent_path), "optimal_ref": opt_ref or None}
if optimal is not None:
    np.savez_compressed(Path(eval_dir) / "optimal_at_agent_states.npz", opt_acts=np.array(all_opt_acts, dtype=object))
    matches = sum(int((a == o).sum()) for a, o in zip(all_acts, all_opt_acts))
    total = sum(len(a) for a in all_acts)
    out.update(alignment_fraction=matches / max(1, total), matches=matches, total_steps=total)
with open(Path(eval_dir) / "agent_alignment.json", "w") as f:
    json.dump(out, f, indent=2)
with open(Path(eval_dir) / "meta.json", "w") as f:
    json.dump({"slurm_job_id": Path(run_dir).name, "seed": seed, "demo_path": demo_path, "n_demos": n_demos,
               "demo_batch_size": demo_bs, "gym_id": gym_id, "max_episode_steps": max_ep_steps}, f, indent=2)
print(f"[eval] ep_reward_mean = {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}   crash rate {np.mean(crashed):.2f}"
      + (f"   alignment {out['alignment_fraction']:.3f}" if optimal is not None else ""))
PY
fi
echo "GAIL run $RUN_DIR done"
