#!/bin/bash
# Pure BC matched EXACTLY to the GAIL fixed_gt200 arm: same 403-demo human pool
# (return > 200), SHUFFLE=0 -> demos[:N] for every seed, same budgets and seeds.
# Seeds vary training only. The random-human BC runs (bc_rand_gt200_*) and the
# fixed/random synthetic runs are untouched.
set -euo pipefail
cd /home/marzii/IRL3
D=/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander
DEMO=${DEMO:-$D/session_3_ext700_ret_gt200}
INDEX=${INDEX:-/home/marzii/IRL3/experiments/BC/bc_fixed_gt200_$(date +%Y-%m-%d).csv}
BUDGETS=${BUDGETS:-"1 5 10 50 100 250 400"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

[[ -d "$DEMO" ]] || { echo "missing pool $DEMO" 1>&2; exit 1; }
[[ -f "$INDEX" ]] || echo "arm,N,seed,slurm_job_id,demo_path,status" > "$INDEX"

for N in $BUDGETS; do
  for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
      --job-name="bc-fix200-N${N}-s${SEED}" \
      --export=ALL,DEMO_PATH="$DEMO",N_DEMOS="$N",SEED="$SEED",SHUFFLE=0 \
      imitation/bc/run_bc_lunarlander.sh)
    echo "fixed_gt200,$N,$SEED,$JOB,$DEMO,submitted" >> "$INDEX"
    echo "  BC N=$N seed=$SEED -> $JOB"
  done
done
echo "Index: $INDEX"
