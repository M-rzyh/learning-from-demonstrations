#!/bin/bash
#SBATCH --job-name=bc-warmstart
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/bc_warmstart/lunarlander/%x_%j.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/bc_warmstart/lunarlander/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#
# BC warm-start -> PPO on TRUE env reward (Matt idea #1, variants A/B).
#   Variant A (all 300):    DEMO_PATH=<session_3_ext300>
#   Variant B (landed 180): DEMO_PATH=<session_3_ext300> OUTCOME=landed FEATURES_CSV=<...features.csv>
# Submit per (variant, seed):
#   DEMO_PATH=... SEED=$s [OUTCOME=landed FEATURES_CSV=...] sbatch run_bc_warmstart_lunarlander.sh

set -euo pipefail
mkdir -p /scratch/marzii/imitation_runs/_slurm_logs/bc_warmstart/lunarlander

module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r
source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r
export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"

DEMO_PATH=${DEMO_PATH:?must set DEMO_PATH}
SEED=${SEED:-0}
N_DEMOS=${N_DEMOS:-0}                 # 0 = all in the dir (after any outcome filter)
BC_EPOCHS=${BC_EPOCHS:-50}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}
MAX_EP_STEPS=${MAX_EP_STEPS:-1000}
N_EVAL_EPISODES=${N_EVAL_EPISODES:-50}
OUTCOME=${OUTCOME:-}                  # e.g. "landed" for variant B
FEATURES_CSV=${FEATURES_CSV:-}

RUN_DIR="/scratch/marzii/imitation_runs/bc_warmstart/lunarlander/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"

echo "Demo path:   $DEMO_PATH"
echo "Seed:        $SEED"
echo "Outcome:     ${OUTCOME:-<none>}   (features_csv=${FEATURES_CSV:-<none>})"
echo "BC epochs:   $BC_EPOCHS   RL steps: $TOTAL_TIMESTEPS   cap: $MAX_EP_STEPS"
echo "Run dir:     $RUN_DIR"

python bc/warmstart_bc_ppo.py \
  --demo_path "$DEMO_PATH" \
  --out_dir "$RUN_DIR" \
  --seed "$SEED" \
  --n_demos "$N_DEMOS" \
  --bc_epochs "$BC_EPOCHS" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --max_ep_steps "$MAX_EP_STEPS" \
  --n_eval_episodes "$N_EVAL_EPISODES" \
  ${OUTCOME:+--outcome "$OUTCOME"} \
  ${FEATURES_CSV:+--features_csv "$FEATURES_CSV"}
