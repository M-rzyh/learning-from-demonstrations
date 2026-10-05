#!/bin/bash
# GAIL C1 (human, return-ranked top-K of session_3_ext700) at the small budgets
# N = 1 and 5, matching the existing C1 count-axis runs exactly.
#
# The ranked subset directories are sorted by return, descending (verified:
# session_3_ext700_ret_only_top10 and _top400 both start 318.9, 314.8, 309.4, ...).
# With SHUFFLE=0 the code takes demos[:N], so pointing at ret_only_top10 with
# N_DEMOS=1 or 5 trains on exactly the global top-1 / top-5 -- no new subset
# files are created. (Top-1 is human demo 131, the same demo H5 used.)
#
# Everything else as in one_demo_expert_vs_human/jobs/submit_countaxis_k1000.sh:
# cap 1000, SHUFFLE=0, 50-episode eval, FRAME_SKIP=0, demo batch 128 at N=1 and
# 256 otherwise. Rows are APPENDED to the same count-axis index (arm=human), so
# the existing plotting scripts pick them up as C1.
set -euo pipefail
D=/scratch/marzii/imitation_runs/demos/human_baseline/record_human_demos/lunarlander
DEMO="$D/session_3_ext700_ret_only_top10"
INDEX=/home/marzii/IRL3/experiments/GAIL/gail_countaxis_kappa1000_2026-09-07.csv
BUDGETS=${BUDGETS:-"1 5"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

[[ -d "$DEMO" ]] || { echo "missing $DEMO"; exit 1; }
[[ -f "$INDEX" ]] || { echo "missing index $INDEX"; exit 1; }

batch_for () { [[ "$1" -eq 1 ]] && echo 128 || echo 256; }

cd /home/marzii/IRL3/imitation
for N in $BUDGETS; do
  B=$(batch_for "$N")
  for seed in $SEEDS; do
    jid=$(DEMO_PATH="$DEMO" N_DEMOS=$N DEMO_BATCH_SIZE=$B SHUFFLE=0 SEED="$seed" \
          N_EVAL_EPISODES=50 FRAME_SKIP=0 ENV_MAX_EP_STEPS=1000 \
          DEMO_NOTE="countaxis k1000: human ret_only top${N} (first ${N} of top10) seed=${seed}" \
          sbatch --parsable --time=01:00:00 run_gail_lunarlander.sh)
    echo "human,${N},${seed},1000,${B},${jid},${DEMO},submitted" >> "$INDEX"
    echo "  human N=$N seed=$seed batch=$B -> $jid"
  done
done
echo "Index: $INDEX"
