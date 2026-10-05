#!/bin/bash
# Submit GAIL training on BLIND-EXPERT demos (frame-blanking, Exp 1).
# The demos are degraded (blind expert); the agent trains in the NORMAL 8-D env.
# N_DEMOS=50, 50-episode final eval — matches the human comparison + GAIL defaults.
#   bash scripts/submit_gail_blind_expert.sh [NSEEDS]   (default 5)
#
# Reuses run_gail_lunarlander.sh unchanged (params via env vars). Runs land in the
# standard gail/lunarlander/<job>/ dir (job-keyed, non-destructive); this index CSV
# maps job -> (blank%, seed). Demos live in frame_blanking/.
set -euo pipefail

DATE=$(date +%Y-%m-%d)
INDEX=/home/marzii/IRL3/experiments/GAIL/gail_blind_expert_block_${DATE}.csv
mkdir -p "$(dirname "$INDEX")"
echo "condition_id,N,blank_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"

DEMO_ROOT=/scratch/marzii/imitation_runs/frame_blanking/demos/blind_expert/block
NSEEDS=${1:-5}
LAST=$((NSEEDS - 1))
cd /home/marzii/IRL3/imitation

for blank in 0 25 50 75; do
  cond="blind_expert_block_blank${blank}"
  for seed in $(seq 0 "$LAST"); do
    demo="$DEMO_ROOT/n100_p${blank}_s${seed}"
    [[ -d "$demo" ]] || { echo "MISSING demo dir: $demo" 1>&2; exit 1; }
    jobid=$(DEMO_PATH="$demo" N_DEMOS=50 DEMO_BATCH_SIZE=512 SHUFFLE=1 SEED="$seed" \
            N_EVAL_EPISODES=50 FRAME_SKIP=0 \
            DEMO_NOTE="frame_blank blind_expert blank=${blank}% seed=${seed} N=50" \
            sbatch --parsable run_gail_lunarlander.sh)
    echo "${cond},50,${blank},${seed},${jobid},${demo},submitted" >> "$INDEX"
    echo "  ${cond} seed=${seed} -> job ${jobid}"
  done
done
echo "Index written: $INDEX"; wc -l "$INDEX"
