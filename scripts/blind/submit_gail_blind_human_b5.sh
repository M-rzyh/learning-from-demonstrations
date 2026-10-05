#!/bin/bash
# Submit GAIL training on BLIND-HUMAN demos for the b=5 frame-blanking study.
# The demos are degraded (human played with a blanked screen); the agent trains in
# the NORMAL 8-D env. N_DEMOS=50, 50-episode final eval — matches the b=10 human runs
# (see gail_blind_human_2026-07-03.csv, meta.json: n_demos=50, demo_batch_size=512,
# frame_skip=0) so the seeds pool cleanly.
#
#   bash scripts/submit_gail_blind_human_b5.sh <blank_pct> <seed_lo> <seed_hi>
#
# Examples:
#   bash scripts/submit_gail_blind_human_b5.sh 0  5 9    # top the 0% baseline up to 10 seeds
#   bash scripts/submit_gail_blind_human_b5.sh 50 0 9    # once blank5_p50 demos exist
#
# 0% is blank-agnostic (b is meaningless when nothing is blanked) so it reuses the
# clean human session. Reuses run_gail_lunarlander.sh unchanged (params via env vars).
# Runs land in gail/lunarlander/<job>/ (job-keyed, non-destructive); the index CSV
# maps job -> (blank%, seed) and is APPENDED to, never overwritten.
set -euo pipefail

BLANK=${1:?usage: submit_gail_blind_human_b5.sh <blank_pct> <seed_lo> <seed_hi>}
LO=${2:?missing seed_lo}
HI=${3:?missing seed_hi}

CLEAN_DEMOS=/scratch/marzii/imitation_runs/human_demos/lunarlander/session_2
BLANK_ROOT=/scratch/marzii/imitation_runs/frame_blanking/demos/human

if [[ "$BLANK" == "0" ]]; then
  DEMO="$CLEAN_DEMOS"
else
  DEMO="$BLANK_ROOT/blank5_p${BLANK}/session_1_flagged"
fi
[[ -d "$DEMO" ]] || { echo "MISSING demo dir: $DEMO" 1>&2; exit 1; }

INDEX=/home/marzii/IRL3/experiments/GAIL/gail_blind_human_b5_$(date +%Y-%m-%d).csv
mkdir -p "$(dirname "$INDEX")"
[[ -f "$INDEX" ]] || echo "condition_id,N,blank_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"

cd /home/marzii/IRL3/imitation
cond="blind_human_b5_blank${BLANK}"
for seed in $(seq "$LO" "$HI"); do
  jobid=$(DEMO_PATH="$DEMO" N_DEMOS=50 DEMO_BATCH_SIZE=512 SHUFFLE=1 SEED="$seed" \
          N_EVAL_EPISODES=50 FRAME_SKIP=0 \
          DEMO_NOTE="frame_blank b=5 blind_human blank=${BLANK}% seed=${seed} N=50" \
          sbatch --parsable run_gail_lunarlander.sh)
  echo "${cond},50,${BLANK},${seed},${jobid},${DEMO},submitted" >> "$INDEX"
  echo "  ${cond} seed=${seed} -> job ${jobid}"
done
echo "Index: $INDEX"
