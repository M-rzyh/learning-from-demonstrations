#!/bin/bash
# Pure BC matched EXACTLY to the GAIL rand_expert arm (imitation/submit_gail_rand_expert.sh):
# same 500-demo expert pool, same SHUFFLE=1 / SHUFFLE_SEED=SEED draw, same budgets
# and seeds, so BC and GAIL train on identical demonstrations at every (N, seed).
# The fixed-synthetic BC runs (SHUFFLE=0, bc_e4_matched_*) are untouched.
set -euo pipefail
cd /home/marzii/IRL3
DEMO=${DEMO:-/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/perframe_demos_500_holdK1}
INDEX=${INDEX:-/home/marzii/IRL3/experiments/BC/bc_rand_expert_$(date +%Y-%m-%d).csv}
BUDGETS=${BUDGETS:-"1 5 10 50 100 250 400"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

[[ -d "$DEMO" ]] || { echo "missing expert pool $DEMO" 1>&2; exit 1; }
[[ -f "$INDEX" ]] || echo "arm,N,seed,slurm_job_id,demo_path,status" > "$INDEX"

for N in $BUDGETS; do
  for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
      --job-name="bc-rexp-N${N}-s${SEED}" \
      --export=ALL,DEMO_PATH="$DEMO",N_DEMOS="$N",SEED="$SEED",SHUFFLE=1,SHUFFLE_SEED="$SEED" \
      imitation/bc/run_bc_lunarlander.sh)
    echo "rand_expert,$N,$SEED,$JOB,$DEMO,submitted" >> "$INDEX"
    echo "  BC N=$N seed=$SEED -> $JOB"
  done
done
echo "Index: $INDEX"
