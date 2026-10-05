#!/bin/bash
# Pure BC matched EXACTLY to expert GAIL E4 (cap-1000 count axis).
#
# Verified from the E4 index and run artifacts before writing this:
#   pool    : expert/lunarlander/4615187/rollouts/perframe_demos_500_holdK1
#             (recorded as demo_path in every E4 eval_data/meta.json)
#   budgets : 1 10 50 100 250 400        (10 seeds each)
#   seeds   : 0..9
#   select  : SHUFFLE=0 -> demos[:N] (ingredients/demonstrations.py:127), so NO
#             shuffling happens and every seed at a given N trains on the SAME
#             first N demonstrations. No shuffled_demos dir exists in any E4 run.
#   cap     : ENV_MAX_EP_STEPS=1000
#
# Therefore BC reproduces E4's demonstrations by pointing at the same pool with
# the same N and SHUFFLE=0 -- no copied datasets, nothing to re-rank. The demo
# IDs are 0..N-1 of the pool for every seed, in both arms, by construction.
#
# Evaluation is the BC script's built-in protocol, which already matches GAIL's:
# 50 episodes, deterministic=True, fresh gym.make(gym_id, max_episode_steps=1000),
# env seeded reset(seed=SEED*1000+ep), scored with the true environment reward.
#
# Usage:
#   bash imitation/bc/submit_bc_e4_matched.sh                            # all 60
#   BUDGETS="10" SEEDS="0" bash imitation/bc/submit_bc_e4_matched.sh     # smoke

set -euo pipefail
cd /home/marzii/IRL3

BUDGETS=${BUDGETS:-"1 10 50 100 250 400"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
DEMO_PATH=${DEMO_PATH:-/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/perframe_demos_500_holdK1}
INDEX=${INDEX:-/home/marzii/IRL3/experiments/BC/bc_e4_matched_$(date +%Y-%m-%d).csv}

[[ -d "$DEMO_PATH" ]] || { echo "missing expert pool: $DEMO_PATH" 1>&2; exit 1; }

if [[ ! -f "$INDEX" ]]; then
  echo "arm,N,seed,slurm_job_id,demo_path,status" > "$INDEX"
fi

echo "budgets: $BUDGETS | seeds: $SEEDS"
echo "demos:   $DEMO_PATH (first N, SHUFFLE=0, same as E4)"
echo "index:   $INDEX"

for N in $BUDGETS; do
  for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
      --job-name="bc-e4-N${N}-s${SEED}" \
      --export=ALL,DEMO_PATH="$DEMO_PATH",N_DEMOS="$N",SEED="$SEED",SHUFFLE=0 \
      imitation/bc/run_bc_lunarlander.sh)
    echo "  submitted N=$N seed=$SEED job=$JOB"
    echo "e4_matched,$N,$SEED,$JOB,$DEMO_PATH,submitted" >> "$INDEX"
  done
done

echo "Done. Watch: squeue -u \$USER | grep bc-e4"
