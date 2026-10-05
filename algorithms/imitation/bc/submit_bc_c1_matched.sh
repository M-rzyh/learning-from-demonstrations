#!/bin/bash
# Pure BC matched EXACTLY to human GAIL C1 (return-ranked top-K of session_3_ext700).
#
# Verified from the C1 index CSVs and the run artifacts before writing this:
#   subsets : session_3_ext700_ret_only_top{K}, pre-built directories that are
#             exactly the global top-K of ext700 by true episodic return
#             (checked: each equals the top-K ranking, and they nest:
#              top-10 in top-50 in top-100 in top-250 in top-400 in all 699)
#   budgets : 10 50 100 250 699        (400 excluded on request)
#   seeds   : 0..9   (K=699 has 5 seeds in C1: 0..4)
#   select  : SHUFFLE=0 -> demos[:N] of an already-ranked dir, so every seed sees
#             the SAME demonstrations; no resampling, no re-ranking here either.
#   cap     : ENV_MAX_EP_STEPS=1000
#
# BC therefore reproduces C1's demonstrations by pointing at the same directory
# with the same N and SHUFFLE=0. Nothing is copied and no C1 run is touched.
#
# Evaluation is the BC script's built-in protocol, identical to GAIL's: 50
# episodes, deterministic=True, fresh gym.make(gym_id, max_episode_steps=1000),
# env seeded reset(seed=SEED*1000+ep), true environment reward.
#
# Kept separate from the H6-matched BC runs (own index file).
#
# Usage:
#   bash imitation/bc/submit_bc_c1_matched.sh                          # all 55
#   BUDGETS="10" SEEDS="0" bash imitation/bc/submit_bc_c1_matched.sh   # smoke

set -euo pipefail
cd /home/marzii/IRL3

BUDGETS=${BUDGETS:-"10 50 100 250 699"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
SEEDS_699=${SEEDS_699:-"0 1 2 3 4"}     # C1 only ran 5 seeds at the full pool
DEMO_ROOT=${DEMO_ROOT:-/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander}
INDEX=${INDEX:-/home/marzii/IRL3/experiments/BC/bc_c1_matched_$(date +%Y-%m-%d).csv}

if [[ ! -f "$INDEX" ]]; then
  echo "arm,N,seed,slurm_job_id,demo_path,status" > "$INDEX"
fi

echo "budgets: $BUDGETS | seeds: $SEEDS (K=699: $SEEDS_699)"
echo "index:   $INDEX"

for N in $BUDGETS; do
  if [[ "$N" == "699" ]]; then
    DEMO_PATH="$DEMO_ROOT/session_3_ext700"
    USE_SEEDS="$SEEDS_699"
  elif [[ "$N" -lt 10 ]]; then
    # No ret_only_top1/top5 dirs exist. The ranked dirs are sorted by return,
    # descending, so demos[:N] of ret_only_top10 IS the global top-N (same demos
    # GAIL C1 uses at N=1/5 via imitation/submit_gail_c1_small_n.sh).
    DEMO_PATH="$DEMO_ROOT/session_3_ext700_ret_only_top10"
    USE_SEEDS="$SEEDS"
  else
    DEMO_PATH="$DEMO_ROOT/session_3_ext700_ret_only_top${N}"
    USE_SEEDS="$SEEDS"
  fi
  if [[ ! -d "$DEMO_PATH" ]]; then
    echo "MISSING subset, skipping: $DEMO_PATH" 1>&2
    continue
  fi
  for SEED in $USE_SEEDS; do
    JOB=$(sbatch --parsable \
      --job-name="bc-c1-N${N}-s${SEED}" \
      --export=ALL,DEMO_PATH="$DEMO_PATH",N_DEMOS="$N",SEED="$SEED",SHUFFLE=0 \
      imitation/bc/run_bc_lunarlander.sh)
    echo "  submitted N=$N seed=$SEED job=$JOB"
    echo "c1_matched,$N,$SEED,$JOB,$DEMO_PATH,submitted" >> "$INDEX"
  done
done

echo "Done. Watch: squeue -u \$USER | grep bc-c1"
