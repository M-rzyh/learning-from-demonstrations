#!/bin/bash
# GAIL on RANDOMLY RESAMPLED synthetic (expert) demos -- the fixed-synthetic arm's
# twin, matched to the random-human setup.
#
# Pool  : perframe_demos_500_holdK1 -- the SAME 500-demo expert pool the existing
#         fixed-synthetic (E4) runs use. Nothing is copied.
# Draw  : SHUFFLE=1 with SHUFFLE_SEED=SEED, i.e. ds.shuffle(seed) then demos[:N],
#         so each seed trains on its own random subset. The shuffle helper is
#         byte-identical in the GAIL and BC run scripts, so BC and GAIL see the
#         SAME demos at every (N, seed) -- as for the random-human arm.
# Rest  : identical to the E4/C1/rand_gt200 count-axis runs -- cap 1000,
#         50-episode deterministic eval, FRAME_SKIP=0, demo batch 128 at N=1
#         and 256 otherwise.
#
# The existing fixed-synthetic runs (SHUFFLE=0) are untouched; these land in
# their own index so the two can be compared.
set -euo pipefail
DEMO=${DEMO:-/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/perframe_demos_500_holdK1}
INDEX=${INDEX:-/home/marzii/IRL3/experiments/GAIL/gail_rand_expert_$(date +%Y-%m-%d).csv}
BUDGETS=${BUDGETS:-"1 5 10 50 100 250 400"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

[[ -d "$DEMO" ]] || { echo "missing expert pool $DEMO" 1>&2; exit 1; }
[[ -f "$INDEX" ]] || echo "arm,N,seed,kappa,batch,slurm_job_id,demo_path,status" > "$INDEX"

cd /home/marzii/IRL3/imitation
for N in $BUDGETS; do
  B=$([[ "$N" -eq 1 ]] && echo 128 || echo 256)
  for seed in $SEEDS; do
    jid=$(DEMO_PATH="$DEMO" N_DEMOS=$N DEMO_BATCH_SIZE=$B SHUFFLE=1 SHUFFLE_SEED="$seed" \
          SEED="$seed" N_EVAL_EPISODES=50 FRAME_SKIP=0 ENV_MAX_EP_STEPS=1000 \
          DEMO_NOTE="rand_expert: random ${N} of the 500 expert demos, seed=${seed}" \
          sbatch --parsable --time=01:00:00 run_gail_lunarlander.sh)
    echo "rand_expert,${N},${seed},1000,${B},${jid},${DEMO},submitted" >> "$INDEX"
    echo "  GAIL N=$N seed=$seed batch=$B -> $jid"
  done
done
echo "Index: $INDEX"
