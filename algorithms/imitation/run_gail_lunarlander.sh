#!/bin/bash
#SBATCH --job-name=gail-lunarlander
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/gail/lunarlander/%x_%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/gail/lunarlander/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/gail/lunarlander

# ---- required env setup ----
module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

# Use local IRL3 source
export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"

which python
python -c "import sys; print(sys.executable)"
python -c "import imitation; print('imitation', imitation.__version__)"

# ---- sanity check ----
python - <<'PY'
import gymnasium as gym
env = gym.make("LunarLander-v2")
print("ok", env.spec.id, "max_ep_steps=", env.spec.max_episode_steps,
      "obs=", env.observation_space, "act=", env.action_space)
env.close()
PY

# ---- Demonstration source ----
# Option 1: Expert-generated demos (default)
#   DEMO_SOURCE=local  DEMO_PATH=<expert rollout dir>
# Option 2: Human-collected demos
#   DEMO_SOURCE=local  DEMO_PATH=<human demo dir from collect_human_demos_lunarlander.sh>
#   (human demos are saved in the same HuggingFace Arrow format as expert demos)
#
# To collect human demos first, run interactively:
#   bash collect_human_demos_lunarlander.sh

DEMO_SOURCE=${DEMO_SOURCE:-local}
DEMO_PATH=${DEMO_PATH:-"/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/final.npz"}
N_DEMOS=${N_DEMOS:-50} # default: 50
DEMO_BATCH_SIZE=${DEMO_BATCH_SIZE:-1024} # default: 1024
SEED=${SEED:-1}
SHUFFLE=${SHUFFLE:-0}  # 0 = use first N demos (fixed), 1 = shuffle first (reproducible via SHUFFLE_SEED)
SHUFFLE_SEED=${SHUFFLE_SEED:-$SEED}  # defaults to SEED; set separately to get different demo subsets with same training seed

# ---- frame-skip (action repeat k=10) toggle ----
# FRAME_SKIP=0 (default): original behavior (LunarLander-v2, max_ep_steps=400)
# FRAME_SKIP=1: use LunarLander-v2-FS10 (generator policy decides every 10 env steps)
# NOTE: must match the env used to generate the demos (use FS10 expert demos with FRAME_SKIP=1)
FRAME_SKIP=${FRAME_SKIP:-0}
ENV_GYM_ID="LunarLander-v2"
# Episode cap for GAIL training + eval. Overridable (default 400 = original behavior, so every
# existing run is unchanged). Set ENV_MAX_EP_STEPS=1000 to match the human demo collection cap
# and the PT pipeline (used for the cap-sensitivity rerun).
ENV_MAX_EP_STEPS=${ENV_MAX_EP_STEPS:-400}
if [[ "$FRAME_SKIP" == "1" ]]; then
  ENV_GYM_ID="LunarLander-v2-FS10"
  ENV_MAX_EP_STEPS=40   # FS10 always caps at 40 (10x action repeat); overrides the above
fi

DEMO_NOTE=${DEMO_NOTE:-}  # optional free-text annotation about how demos were prepared

echo "Demo source:     $DEMO_SOURCE"
echo "Demo path:       $DEMO_PATH"
echo "N demos:         $N_DEMOS"
echo "Demo batch size: $DEMO_BATCH_SIZE"
echo "Seed:            $SEED"
echo "Shuffle demos:   $SHUFFLE  (shuffle_seed=$SHUFFLE_SEED)"
echo "Frame skip:      $FRAME_SKIP  (gym_id=$ENV_GYM_ID, max_ep_steps=$ENV_MAX_EP_STEPS)"
if [[ -n "$DEMO_NOTE" ]]; then
  echo "Demo note:       $DEMO_NOTE"
fi

# ---- algorithm selection (must precede LOG_ROOT: each algo gets its own output tree) ----
# GAIL by default; ALGO=airl runs AIRL. Same pipeline; the reward net auto-switches to
# BasicShapedRewardNet (see reward.config_hook, which branches on the sacred command name).
ALGO=${ALGO:-gail}
echo "algorithm: $ALGO"

# Output structure: imitation_runs/<ALGO_DIR>/<env_name>/<job_id>
# GAIL keeps its historical lowercase "gail" tree untouched; AIRL gets its own "AIRL" tree so
# the two never mix (they are different algorithms and were previously indistinguishable on disk).
ENV_NAME="lunarlander"
case "$ALGO" in
  airl) ALGO_DIR="AIRL" ;;
  *)    ALGO_DIR="$ALGO" ;;
esac
LOG_ROOT="/scratch/marzii/imitation_runs/$ALGO_DIR"
RUN_DIR="$LOG_ROOT/$ENV_NAME/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"
echo "run dir:         $RUN_DIR"

# ---- optional: shuffle demos reproducibly (using SEED) ----
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

# ---- optional Weights & Biases logging (USE_WANDB=1). Default 0 keeps the command byte-identical
#      to every previous run. Needs `wandb login` once (writes ~/.netrc); compute nodes reach
#      wandb through the prolog-injected http_proxy=http://squid:3128. TensorBoard is written
#      either way, so nothing is lost if wandb is off. ----
USE_WANDB=${USE_WANDB:-0}
LOG_FMT="['tensorboard','stdout']"
WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  LOG_FMT="['tensorboard','stdout','wandb']"
  WANDB_PROJECT=${WANDB_PROJECT:-gail-lunarlander}
  # run name becomes <prefix>-<gym_id>-seed<seed>; RUN_TAG defaults to the demo dir name so
  # runs from different rankings/subsets are distinguishable in the UI.
  RUN_TAG=${RUN_TAG:-$(basename "$DEMO_PATH")}
  WANDB_ARGS=(
    logging.wandb.wandb_name_prefix="${ALGO}-${RUN_TAG}"
    logging.wandb.wandb_tag="$RUN_TAG"
    logging.wandb.wandb_kwargs.project="$WANDB_PROJECT"
  )
  echo "wandb: ON  project=$WANDB_PROJECT  tag=$RUN_TAG"
fi

python train_adversarial_launcher.py "$ALGO" \
  with lunar_lander \
  demonstrations.source="$DEMO_SOURCE" \
  demonstrations.path="$DEMO_PATH" \
  demonstrations.n_expert_demos="$N_DEMOS" \
  algorithm_kwargs.demo_batch_size="$DEMO_BATCH_SIZE" \
  total_timesteps=1000000 \
  environment.num_vec=8 \
  environment.gym_id="$ENV_GYM_ID" \
  environment.max_episode_steps=$ENV_MAX_EP_STEPS \
  seed=$SEED \
  logging.log_dir="$RUN_DIR" \
  logging.log_format_strs="$LOG_FMT" \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}

# ---- post-training evaluation: 50 rollouts + action alignment ----
# Optional: skip with SKIP_EVAL=1.
SKIP_EVAL=${SKIP_EVAL:-0}
# OPTIMAL_REF=${OPTIMAL_REF:-"/scratch/marzii/imitation_runs/expert/lunarlander/4720242/policies/final/model.zip"}
OPTIMAL_REF=${OPTIMAL_REF:-"/scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip"}
EVAL_DATA_DIR="$RUN_DIR/eval_data"
N_EVAL_EPISODES=${N_EVAL_EPISODES:-50}

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo ""
  echo "===== post-training evaluation ====="
  echo "Optimal ref:     $OPTIMAL_REF"
  echo "Eval episodes:   $N_EVAL_EPISODES"
  echo "Eval data dir:   $EVAL_DATA_DIR"
  mkdir -p "$EVAL_DATA_DIR"

  python - "$RUN_DIR" "$OPTIMAL_REF" "$EVAL_DATA_DIR" "$N_EVAL_EPISODES" "$ENV_GYM_ID" "$ENV_MAX_EP_STEPS" "$SEED" "$DEMO_PATH" "$N_DEMOS" "$DEMO_BATCH_SIZE" "$FRAME_SKIP" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import imitation.envs.action_repeat  # noqa: F401
from stable_baselines3 import PPO

run_dir, opt_ref, eval_dir, n_eps, gym_id, max_ep_steps, seed, demo_path, n_demos, demo_bs, frame_skip = sys.argv[1:12]
n_eps = int(n_eps); max_ep_steps = int(max_ep_steps); seed = int(seed)
n_demos = int(n_demos); demo_bs = int(demo_bs); frame_skip = int(frame_skip)

# Find agent policy
candidates = [
    Path(run_dir) / "checkpoints" / "final" / "gen_policy" / "model.zip",
    Path(run_dir) / "checkpoints" / "final" / "model.zip",
]
agent_path = next((p for p in candidates if p.exists()), None)
if agent_path is None:
    print(f"[eval] no agent policy found in {run_dir}; skipping eval")
    sys.exit(0)

print(f"[eval] loading agent: {agent_path}")
agent = PPO.load(str(agent_path))
print(f"[eval] loading optimal ref: {opt_ref}")
optimal = PPO.load(opt_ref)

env = gym.make(gym_id, max_episode_steps=max_ep_steps)

all_obs = []   # list of arrays per ep
all_acts = []  # list of arrays per ep
all_rews = []
all_terminal = []
all_opt_acts = []  # optimal expert's prediction at each agent state

for ep in range(n_eps):
    obs, _ = env.reset(seed=seed * 1000 + ep)
    ep_obs = [obs.copy()]
    ep_act = []
    ep_rew = []
    ep_opt = []
    done = False
    while not done:
        a_agent, _ = agent.predict(obs, deterministic=True)
        a_opt, _ = optimal.predict(obs, deterministic=True)
        ep_act.append(int(a_agent))
        ep_opt.append(int(a_opt))
        obs, r, term, trunc, _ = env.step(int(a_agent))
        ep_rew.append(float(r))
        ep_obs.append(obs.copy())
        done = bool(term or trunc)
    all_obs.append(np.array(ep_obs, dtype=np.float32))
    all_acts.append(np.array(ep_act, dtype=np.int64))
    all_rews.append(np.array(ep_rew, dtype=np.float64))
    all_opt_acts.append(np.array(ep_opt, dtype=np.int64))
    all_terminal.append(bool(term))

env.close()

# Save rollouts (ragged → use object arrays)
np.savez_compressed(
    Path(eval_dir) / "agent_rollouts.npz",
    obs=np.array(all_obs, dtype=object),
    acts=np.array(all_acts, dtype=object),
    rews=np.array(all_rews, dtype=object),
    terminal=np.array(all_terminal, dtype=bool),
)
np.savez_compressed(
    Path(eval_dir) / "optimal_at_agent_states.npz",
    opt_acts=np.array(all_opt_acts, dtype=object),
)

# Action alignment: agent's action vs optimal's action at each state
matches = sum(int((a == o).sum()) for a, o in zip(all_acts, all_opt_acts))
total = sum(len(a) for a in all_acts)
alignment = matches / max(1, total)
ep_rewards = [float(r.sum()) for r in all_rews]

align_json = {
    "alignment_fraction": alignment,
    "matches": matches,
    "total_steps": total,
    "n_episodes": n_eps,
    "ep_rewards": ep_rewards,
    "ep_reward_mean": float(np.mean(ep_rewards)),
    "ep_reward_std": float(np.std(ep_rewards)),
    "optimal_ref": opt_ref,
    "agent_policy": str(agent_path),
}
with open(Path(eval_dir) / "agent_alignment.json", "w") as f:
    json.dump(align_json, f, indent=2)

meta = {
    "slurm_job_id": Path(run_dir).name,
    "seed": seed,
    "demo_path": demo_path,
    "n_demos": n_demos,
    "demo_batch_size": demo_bs,
    "frame_skip": frame_skip,
    "gym_id": gym_id,
    "max_episode_steps": max_ep_steps,
}
with open(Path(eval_dir) / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[eval] saved rollouts + alignment.json")
print(f"  agent_alignment = {alignment:.4f}  ({matches}/{total})")
print(f"  ep_reward_mean = {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
PY
fi
