#!/bin/bash
#SBATCH --job-name=bc-lunarlander
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/bc/lunarlander/%x_%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/bc/lunarlander/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#
# Behavior cloning (BC) on LunarLander-v2, as the diagnostic counterpart to GAIL: BC fits
# pi(a|s) by max-likelihood with NO adversary and NO RL, so it measures how imitable a demo
# distribution is. If human-BC caps well below expert-BC, the human->expert gap is realizability
# (the human isn't a single Markov policy), not a GAIL artifact.
#
# BC itself is the vendored library's ready `bc` command (src/imitation/scripts/train_imitation
# .py); this script is just plumbing + the same true-env eval GAIL uses, so results plug into the
# existing plotting (eval_data/agent_rollouts.npz with ground-truth env returns).
#
#   DEMO_PATH=<demo dir> N_DEMOS=15 SEED=0 sbatch run_bc_lunarlander.sh
set -euo pipefail
mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/bc/lunarlander

module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r
source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r
export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"
cd /home/marzii/IRL3/imitation

DEMO_SOURCE=${DEMO_SOURCE:-local}
DEMO_PATH=${DEMO_PATH:-"/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/final.npz"}
N_DEMOS=${N_DEMOS:-15}
SEED=${SEED:-0}
ENV_GYM_ID=${ENV_GYM_ID:-"LunarLander-v2"}
ENV_MAX_EP_STEPS=${ENV_MAX_EP_STEPS:-1000}   # match PT + human collection cap
N_EVAL_EPISODES=${N_EVAL_EPISODES:-50}
# 0 = use the first N demos of DEMO_PATH (unchanged default, every previous run).
# 1 = shuffle DEMO_PATH first, exactly as run_gail_lunarlander.sh:105-120 does, then
#     take the first N. With SHUFFLE_SEED=SEED this reproduces the Human GAIL H6
#     subsets bit-for-bit (verified: ds.shuffle(seed=S) on session_3 gives the same
#     200-demo order as every H6 shuffled_demos artifact, seeds 0-4, datasets 4.6.0).
SHUFFLE=${SHUFFLE:-0}
SHUFFLE_SEED=${SHUFFLE_SEED:-$SEED}

LOG_ROOT="/scratch/marzii/imitation_runs/bc"
RUN_DIR="$LOG_ROOT/lunarlander/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"
echo "DEMO_PATH=$DEMO_PATH  N_DEMOS=$N_DEMOS  SEED=$SEED  cap=$ENV_MAX_EP_STEPS  RUN_DIR=$RUN_DIR"
echo "Shuffle demos:   $SHUFFLE  (shuffle_seed=$SHUFFLE_SEED)"

# ---- optional: shuffle demos reproducibly (same procedure as the GAIL script) ----
if [[ "$SHUFFLE" == "1" ]]; then
  SHUFFLED_DIR="$RUN_DIR/shuffled_demos"
  echo "Shuffling demos from $DEMO_PATH -> $SHUFFLED_DIR (shuffle_seed=$SHUFFLE_SEED)"
  python - "$DEMO_PATH" "$SHUFFLED_DIR" "$SHUFFLE_SEED" <<'PY'
import sys
from datasets import load_from_disk
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = load_from_disk(src)
ds = ds.shuffle(seed=seed)
ds.save_to_disk(dst)
print(f"Shuffled {len(ds)} trajectories, saved to {dst}")
PY
  DEMO_PATH="$SHUFFLED_DIR"
fi

# ---- BC training (library's `bc` command; same demo/env ingredients as GAIL) ----
python bc/train_imitation_launcher.py bc with lunar_lander \
  demonstrations.source="$DEMO_SOURCE" \
  demonstrations.path="$DEMO_PATH" \
  demonstrations.n_expert_demos="$N_DEMOS" \
  environment.gym_id="$ENV_GYM_ID" \
  environment.max_episode_steps=$ENV_MAX_EP_STEPS \
  seed=$SEED \
  logging.log_dir="$RUN_DIR"

# ---- post-training eval: 50 rollouts in the TRUE env, save agent_rollouts.npz (env reward) ----
EVAL_DATA_DIR="$RUN_DIR/eval_data"; mkdir -p "$EVAL_DATA_DIR"
python - "$RUN_DIR" "$EVAL_DATA_DIR" "$N_EVAL_EPISODES" "$ENV_GYM_ID" "$ENV_MAX_EP_STEPS" "$SEED" "$DEMO_PATH" "$N_DEMOS" <<'PY'
import json, sys
from pathlib import Path
import numpy as np, gymnasium as gym, torch as th
import imitation.envs.action_repeat  # noqa: F401

run_dir, eval_dir, n_eps, gym_id, max_ep, seed, demo_path, n_demos = sys.argv[1:9]
n_eps=int(n_eps); max_ep=int(max_ep); seed=int(seed); n_demos=int(n_demos)

pol_path = Path(run_dir) / "final.th"
if not pol_path.exists():
    print(f"[eval] no policy at {pol_path}; skipping"); sys.exit(0)
# torch>=2.6 defaults weights_only=True and rejects imitation's custom policy class; our own
# checkpoint is trusted, so load with weights_only=False (equivalent to bc.reconstruct_policy).
policy = th.load(str(pol_path), map_location="cpu", weights_only=False)

env = gym.make(gym_id, max_episode_steps=max_ep)
all_obs=[]; all_acts=[]; all_rews=[]; all_terminal=[]
for ep in range(n_eps):
    obs,_=env.reset(seed=seed*1000+ep)
    o=[obs.copy()]; a=[]; r=[]; done=False
    while not done:
        act,_=policy.predict(obs, deterministic=True)
        act=int(act); a.append(act)
        obs,rew,term,trunc,_=env.step(act)
        r.append(float(rew)); o.append(obs.copy()); done=bool(term or trunc)
    all_obs.append(np.array(o,dtype=np.float32)); all_acts.append(np.array(a,dtype=np.int64))
    all_rews.append(np.array(r,dtype=np.float64)); all_terminal.append(bool(term))
env.close()

np.savez_compressed(Path(eval_dir)/"agent_rollouts.npz",
    obs=np.array(all_obs,dtype=object), acts=np.array(all_acts,dtype=object),
    rews=np.array(all_rews,dtype=object), terminal=np.array(all_terminal,dtype=bool))
ep_r=[float(x.sum()) for x in all_rews]
json.dump({"n_episodes":n_eps,"ep_reward_mean":float(np.mean(ep_r)),
           "ep_reward_std":float(np.std(ep_r)),"ep_rewards":ep_r,
           "demo_path":demo_path,"n_demos":n_demos,"seed":seed,
           "gym_id":gym_id,"max_episode_steps":max_ep,"policy":str(pol_path)},
          open(Path(eval_dir)/"agent_alignment.json","w"), indent=2)
print(f"[eval] BC mean env return over {n_eps} eps: {np.mean(ep_r):+.1f}")
PY
echo "BC run $RUN_DIR seed=$SEED done"
