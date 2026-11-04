#!/bin/bash
#SBATCH --job-name=bc-highway
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/bc/highway/%x_%j.out
#SBATCH --error=/scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/bc/highway/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#
# Behavior cloning on highway-env highway-v0 (default env config). Adapted from
# run_bc_lunarlander.sh (unchanged): highway env via the `highway` named config, highway-env 1.9.1
# from a side folder on PYTHONPATH, CPU only, DEMO_PATH required, crash rate added to the eval.
#
#   DEMO_PATH=<converted demo dir> N_DEMOS=50 SEED=0 sbatch launchers/bc/run_bc_highway.sh
set -euo pipefail
mkdir -p /scratch/marzii/learning-from-demonstrations/imitation_runs/_slurm_logs/bc/highway
module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r
source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r
HIGHWAY_PKGS=/scratch/marzii/envs/highway-pkgs
export PYTHONPATH="/home/marzii/learning-from-demonstrations/algorithms/imitation/src:${HIGHWAY_PKGS}:${PYTHONPATH:-}"
export SDL_VIDEODRIVER=offscreen

DEMO_SOURCE=${DEMO_SOURCE:-local}
DEMO_PATH=${DEMO_PATH:?set DEMO_PATH to a converted demo dir or an expert rollouts/final.npz}
N_DEMOS=${N_DEMOS:-50}
SEED=${SEED:-0}
ENV_GYM_ID="highway_env:highway-v0"
ENV_MAX_EP_STEPS=${ENV_MAX_EP_STEPS:-40}
N_EVAL_EPISODES=${N_EVAL_EPISODES:-50}
SHUFFLE=${SHUFFLE:-0}
SHUFFLE_SEED=${SHUFFLE_SEED:-$SEED}
LOG_ROOT="/scratch/marzii/learning-from-demonstrations/imitation_runs/bc"
RUN_DIR="$LOG_ROOT/highway/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"   # work from the run dir on scratch, so cwd-relative output lands there, never in $HOME
echo "DEMO_PATH=$DEMO_PATH  N_DEMOS=$N_DEMOS  SEED=$SEED  env=$ENV_GYM_ID cap=$ENV_MAX_EP_STEPS  RUN_DIR=$RUN_DIR"
echo "Shuffle demos:   $SHUFFLE  (shuffle_seed=$SHUFFLE_SEED)"

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

# EXTRA_ARGS: extra sacred overrides, space-separated (e.g. for quick tests)
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

# ---- BC training (library's `bc` command) ----
python /home/marzii/learning-from-demonstrations/launchers/entrypoints/train_imitation_launcher.py bc with highway \
  demonstrations.source="$DEMO_SOURCE" \
  demonstrations.path="$DEMO_PATH" \
  demonstrations.n_expert_demos="$N_DEMOS" \
  environment.gym_id="$ENV_GYM_ID" \
  environment.max_episode_steps="$ENV_MAX_EP_STEPS" \
  seed="$SEED" \
  logging.log_dir="$RUN_DIR" \
  ${EXTRA[@]+"${EXTRA[@]}"}

# ---- post-training eval: rollouts in the true env, save agent_rollouts.npz (env reward) ----
EVAL_DATA_DIR="$RUN_DIR/eval_data"; mkdir -p "$EVAL_DATA_DIR"
python - "$RUN_DIR" "$EVAL_DATA_DIR" "$N_EVAL_EPISODES" "$ENV_GYM_ID" "$ENV_MAX_EP_STEPS" "$SEED" "$DEMO_PATH" "$N_DEMOS" <<'PY'
import json, sys
from pathlib import Path
import numpy as np, gymnasium as gym, torch as th

run_dir, eval_dir, n_eps, gym_id, max_ep, seed, demo_path, n_demos = sys.argv[1:9]
n_eps = int(n_eps); max_ep = int(max_ep); seed = int(seed); n_demos = int(n_demos)
pol_path = Path(run_dir) / "final.th"
if not pol_path.exists():
    print(f"[eval] no policy at {pol_path}; skipping"); sys.exit(0)
# our own checkpoint is trusted; weights_only=False as in run_bc_lunarlander.sh
policy = th.load(str(pol_path), map_location="cpu", weights_only=False)

env = gym.make(gym_id, max_episode_steps=max_ep)
all_obs, all_acts, all_rews, all_terminal, crashed = [], [], [], [], []
for ep in range(n_eps):
    obs, _ = env.reset(seed=seed * 1000 + ep)
    o, a, r, done = [obs.copy()], [], [], False
    while not done:
        act = int(policy.predict(obs, deterministic=True)[0]); a.append(act)
        obs, rew, term, trunc, info = env.step(act)
        r.append(float(rew)); o.append(obs.copy()); done = bool(term or trunc)
    all_obs.append(np.array(o, dtype=np.float32)); all_acts.append(np.array(a, dtype=np.int64))
    all_rews.append(np.array(r, dtype=np.float64)); all_terminal.append(bool(term))
    crashed.append(bool(info.get("crashed", False)))
env.close()

np.savez_compressed(Path(eval_dir) / "agent_rollouts.npz",
    obs=np.array(all_obs, dtype=object), acts=np.array(all_acts, dtype=object),
    rews=np.array(all_rews, dtype=object), terminal=np.array(all_terminal, dtype=bool),
    crashed=np.array(crashed, dtype=bool))
ep_r = [float(x.sum()) for x in all_rews]
json.dump({"n_episodes": n_eps, "ep_reward_mean": float(np.mean(ep_r)), "ep_reward_std": float(np.std(ep_r)),
           "ep_rewards": ep_r, "crash_rate": float(np.mean(crashed)), "demo_path": demo_path, "n_demos": n_demos,
           "seed": seed, "gym_id": gym_id, "max_episode_steps": max_ep, "policy": str(pol_path)},
          open(Path(eval_dir) / "agent_alignment.json", "w"), indent=2)
print(f"[eval] BC mean env return over {n_eps} eps: {np.mean(ep_r):+.2f}   crash rate {np.mean(crashed):.2f}")
PY
echo "BC run $RUN_DIR seed=$SEED done"
