#!/bin/bash
# GAIL on RANDOM human demos drawn from the "good" pool (return > 200).
#
# Pool  : session_3_ext700_ret_gt200 -- the 403 of 699 session_3 demos with true
#         episodic return > 200, built by scripts/curation/make_ret_gt200_pool.py.
# Draw  : SHUFFLE=1 with SHUFFLE_SEED=SEED, i.e. ds.shuffle(seed) then demos[:N].
#         So (i) each seed trains on its own random draw -- demonstrations AND
#         training vary with the seed -- and (ii) the BC runs submitted by
#         imitation/bc/submit_bc_rand_gt200.sh use the byte-identical shuffle
#         code on the same pool, so BC and GAIL see the SAME demos at each
#         (N, seed). Draws are nested in N within a seed.
# Rest  : as the C1/E4 count-axis runs -- cap 1000, 50-episode deterministic
#         eval, FRAME_SKIP=0, demo batch 128 at N=1 and 256 otherwise.
set -euo pipefail
D=/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander
DEMO="$D/session_3_ext700_ret_gt200"
INDEX=${INDEX:-/home/marzii/IRL3/experiments/GAIL/gail_rand_gt200_$(date +%Y-%m-%d).csv}
BUDGETS=${BUDGETS:-"1 5 10 50 100 250 400"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

[[ -d "$DEMO" ]] || { echo "missing pool $DEMO" 1>&2; exit 1; }
[[ -f "$INDEX" ]] || echo "arm,N,seed,kappa,batch,slurm_job_id,demo_path,status" > "$INDEX"

cd /home/marzii/IRL3/imitation
for N in $BUDGETS; do
  B=$([[ "$N" -eq 1 ]] && echo 128 || echo 256)
  for seed in $SEEDS; do
    jid=$(DEMO_PATH="$DEMO" N_DEMOS=$N DEMO_BATCH_SIZE=$B SHUFFLE=1 SHUFFLE_SEED="$seed" \
          SEED="$seed" N_EVAL_EPISODES=50 FRAME_SKIP=0 ENV_MAX_EP_STEPS=1000 \
          DEMO_NOTE="rand_gt200: random ${N} of the 403 human demos with return>200, seed=${seed}" \
          sbatch --parsable --time=01:00:00 run_gail_lunarlander.sh)
    echo "rand_gt200,${N},${seed},1000,${B},${jid},${DEMO},submitted" >> "$INDEX"
    echo "  GAIL N=$N seed=$seed batch=$B -> $jid"
  done
done
echo "Index: $INDEX"
