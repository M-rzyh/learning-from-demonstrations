#!/bin/bash
# GAIL on the human return>200 pool with a FIXED demo subset -- the SHUFFLE=0
# counterpart of imitation/submit_gail_rand_gt200.sh.
#
# Pool  : session_3_ext700_ret_gt200 -- the same 403 human demos with true return
#         > 200 used by the random-human arm. Nothing is copied.
# Draw  : SHUFFLE=0, so every seed trains on demos[:N] of the pool in its stored
#         order (the ext700 order, filtered). The subset is identical across
#         seeds, exactly like the fixed-synthetic (E4) runs, so seeds vary
#         training only. BC uses the same pool and the same rule, so BC and GAIL
#         train on identical demonstrations at every (N, seed).
# Rest  : as every other count-axis arm -- cap 1000, 50-episode deterministic
#         eval, FRAME_SKIP=0, demo batch 128 at N=1 and 256 otherwise.
set -euo pipefail
D=/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander
DEMO=${DEMO:-$D/session_3_ext700_ret_gt200}
INDEX=${INDEX:-/home/marzii/IRL3/experiments/GAIL/gail_fixed_gt200_$(date +%Y-%m-%d).csv}
BUDGETS=${BUDGETS:-"1 5 10 50 100 250 400"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

[[ -d "$DEMO" ]] || { echo "missing pool $DEMO" 1>&2; exit 1; }
[[ -f "$INDEX" ]] || echo "arm,N,seed,kappa,batch,slurm_job_id,demo_path,status" > "$INDEX"

cd /home/marzii/IRL3/imitation
for N in $BUDGETS; do
  B=$([[ "$N" -eq 1 ]] && echo 128 || echo 256)
  for seed in $SEEDS; do
    jid=$(DEMO_PATH="$DEMO" N_DEMOS=$N DEMO_BATCH_SIZE=$B SHUFFLE=0 \
          SEED="$seed" N_EVAL_EPISODES=50 FRAME_SKIP=0 ENV_MAX_EP_STEPS=1000 \
          DEMO_NOTE="fixed_gt200: first ${N} of the 403 human demos with return>200, seed=${seed}" \
          sbatch --parsable --time=01:00:00 run_gail_lunarlander.sh)
    echo "fixed_gt200,${N},${seed},1000,${B},${jid},${DEMO},submitted" >> "$INDEX"
    echo "  GAIL N=$N seed=$seed batch=$B -> $jid"
  done
done
echo "Index: $INDEX"
