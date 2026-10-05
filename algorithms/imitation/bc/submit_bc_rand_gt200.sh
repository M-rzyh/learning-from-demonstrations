#!/bin/bash
# Pure BC matched EXACTLY to the GAIL rand_gt200 arm (see
# imitation/submit_gail_rand_gt200.sh): same 403-demo pool (human demos with
# true return > 200), same SHUFFLE=1 / SHUFFLE_SEED=SEED draw, same budgets and
# seeds. The shuffle helper in both run scripts is identical, so BC and GAIL
# train on the same demonstrations at every (N, seed). Nothing is copied.
#
# Eval: the BC script's built-in protocol, identical to GAIL's -- 50 episodes,
# deterministic, cap 1000, reset(seed=SEED*1000+ep), true environment reward.
set -euo pipefail
cd /home/marzii/IRL3
D=/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander
DEMO="$D/session_3_ext700_ret_gt200"
INDEX=${INDEX:-/home/marzii/IRL3/experiments/BC/bc_rand_gt200_$(date +%Y-%m-%d).csv}
BUDGETS=${BUDGETS:-"1 5 10 50 100 250 400"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

[[ -d "$DEMO" ]] || { echo "missing pool $DEMO" 1>&2; exit 1; }
[[ -f "$INDEX" ]] || echo "arm,N,seed,slurm_job_id,demo_path,status" > "$INDEX"

for N in $BUDGETS; do
  for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
      --job-name="bc-r200-N${N}-s${SEED}" \
      --export=ALL,DEMO_PATH="$DEMO",N_DEMOS="$N",SEED="$SEED",SHUFFLE=1,SHUFFLE_SEED="$SEED" \
      imitation/bc/run_bc_lunarlander.sh)
    echo "rand_gt200,$N,$SEED,$JOB,$DEMO,submitted" >> "$INDEX"
    echo "  BC N=$N seed=$SEED -> $JOB"
  done
done
echo "Index: $INDEX"
