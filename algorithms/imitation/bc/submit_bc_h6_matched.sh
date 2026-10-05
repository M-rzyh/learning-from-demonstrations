#!/bin/bash
# Pure BC matched EXACTLY to Human GAIL H6 (random-K subsets of session_3).
#
# Same dataset, same shuffle procedure, same seeds as H6 -- so the subsets come
# out identical without copying or curating anything:
#
#   H6  : ds = load_from_disk(session_3); ds.shuffle(seed=SEED); demos[:N]
#   here: run_bc_lunarlander.sh with SHUFFLE=1 SHUFFLE_SEED=SEED, N_DEMOS=N
#
# Verified before use: ds.shuffle(seed=S) on session_3 reproduces the full
# 200-demo order stored in every H6 shuffled_demos artifact, for seeds 0-4
# (datasets 4.6.0). So BC seed s / N=k trains on exactly H6 seed s / N=k demos.
# session_3 itself is only read; H6 run dirs are not touched at all.
#
#   budgets : 5 10 50 100 200      (same as H6)
#   seeds   : 0 1 2 3 4            (same as H6)
#
# BC hyperparameters are whatever run_bc_lunarlander.sh uses (FeedForward32Policy
# 32x32, Adam 4e-4, batch 32, 50k batches, 1000-step cap, 50-episode greedy eval),
# so these runs stay comparable with the existing BC arms.
#
# Usage:
#   bash imitation/bc/submit_bc_h6_matched.sh                          # all 25
#   BUDGETS="5" SEEDS="0" bash imitation/bc/submit_bc_h6_matched.sh    # one smoke run

set -euo pipefail
cd /home/marzii/IRL3

BUDGETS=${BUDGETS:-"5 10 50 100 200"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DEMO_PATH=${DEMO_PATH:-/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander/session_3}
INDEX=${INDEX:-/home/marzii/IRL3/experiments/BC/bc_h6_matched_$(date +%Y-%m-%d).csv}

[[ -d "$DEMO_PATH" ]] || { echo "missing demo set: $DEMO_PATH" 1>&2; exit 1; }

if [[ ! -f "$INDEX" ]]; then
  echo "arm,N,seed,shuffle_seed,slurm_job_id,demo_path,status" > "$INDEX"
fi

echo "budgets: $BUDGETS | seeds: $SEEDS"
echo "demos:   $DEMO_PATH (shuffled per seed, first N taken)"
echo "index:   $INDEX"

for N in $BUDGETS; do
  for SEED in $SEEDS; do
    JOB=$(sbatch --parsable \
      --job-name="bc-h6-N${N}-s${SEED}" \
      --export=ALL,DEMO_PATH="$DEMO_PATH",N_DEMOS="$N",SEED="$SEED",SHUFFLE=1,SHUFFLE_SEED="$SEED" \
      imitation/bc/run_bc_lunarlander.sh)
    echo "  submitted N=$N seed=$SEED job=$JOB"
    echo "h6_matched,$N,$SEED,$SEED,$JOB,$DEMO_PATH,submitted" >> "$INDEX"
  done
done

echo "Done. Watch: squeue -u \$USER | grep bc-h6"
