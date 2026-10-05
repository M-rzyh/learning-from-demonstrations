#!/bin/bash
# Submit GAIL training on ACTION-DELAY human demos (the GAIL arm of the delay study).
# The human played with laggy controls (each keypress applied K steps late); the agent trains
# in the normal env. GAIL levels are K=0/5/10 (K=20 is PT-only). Budget N=15 (flagged).
#
#   bash scripts/action_delay/submit_gail_delay_human.sh <k> <seed_lo> <seed_hi>
#   ARM=flagged|first15|all   CHECK=1 (pre-flight only)
#
# K=0 REUSES the existing vanish_p0 clean fps-20 session (DIFFICULTY=none, delay_k=0 ⇒ the
# delay baseline). K=5/10 are their own collections under action_delay/.
set -euo pipefail

K=${1:?usage: submit_gail_delay_human.sh <k> <seed_lo> <seed_hi>}
LO=${2:?missing seed_lo}
HI=${3:?missing seed_hi}
CHECK=${CHECK:-0}
ARM=${ARM:-flagged}

case "$K" in 0|5|10) ;; *) echo "ERROR: k must be 0, 5 or 10 for GAIL (got '$K'; k=20 is PT-only)" 1>&2; exit 1 ;; esac
case "$ARM" in flagged|first15|all) ;; *) echo "ERROR: ARM must be flagged|first15|all" 1>&2; exit 1 ;; esac

if [[ "$K" == "0" ]]; then
  ROOT=/scratch/marzii/imitation_runs/lander_vanish/demos/human/vanish_p0
else
  ROOT=/scratch/marzii/imitation_runs/action_delay/demos/human/delay_k${K}
fi
case "$ARM" in
  flagged) SUB=session_1_flagged ;;
  first15) SUB=session_1 ;;
  all)     SUB=session_1 ;;
esac
DEMO="$ROOT/$SUB"

if [[ ! -d "$DEMO" ]]; then
  echo "PRE-FLIGHT FAILED: missing demo dir $DEMO" 1>&2; exit 1
fi
n_have=$(/scratch/marzii/envs/imitation-gail/bin/python - "$DEMO" <<'PY'
import glob, sys
import pyarrow.ipc
n = 0
for f in glob.glob(sys.argv[1] + "/*.arrow"):
    with pyarrow.ipc.open_stream(f) as rd:
        n += rd.read_all().num_rows
print(n)
PY
)
if [[ "$ARM" == "all" ]]; then
  N_DEMOS=$n_have
  (( N_DEMOS >= 1 )) || { echo "PRE-FLIGHT FAILED: $DEMO has no episodes" 1>&2; exit 1; }
else
  N_DEMOS=15
  (( n_have >= N_DEMOS )) || { echo "PRE-FLIGHT FAILED: $DEMO has $n_have, need $N_DEMOS" 1>&2; exit 1; }
fi
echo "PRE-FLIGHT OK: $DEMO ($n_have episodes available, using $N_DEMOS)  [K=$K ARM=$ARM]"
if [[ "$CHECK" == "1" ]]; then echo "CHECK=1 — nothing submitted."; exit 0; fi

INDEX=/home/marzii/IRL3/experiments/GAIL/gail_lander_delay_human_$(date +%Y-%m-%d).csv
mkdir -p "$(dirname "$INDEX")"
[[ -f "$INDEX" ]] || echo "condition_id,N,delay_k,seed,slurm_job_id,demo_path,status,arm" > "$INDEX"

cd /home/marzii/IRL3/imitation
cond="delay_human_k${K}_${ARM}"
for seed in $(seq "$LO" "$HI"); do
  jobid=$(DEMO_PATH="$DEMO" N_DEMOS=$N_DEMOS DEMO_BATCH_SIZE=512 SHUFFLE=0 SEED="$seed" \
          N_EVAL_EPISODES=50 FRAME_SKIP=0 \
          DEMO_NOTE="action_delay human k=${K} seed=${seed} N=${N_DEMOS} arm=${ARM}" \
          sbatch --parsable --time=01:00:00 run_gail_lunarlander.sh)
  echo "${cond},${N_DEMOS},${K},${seed},${jobid},${DEMO},submitted,${ARM}" >> "$INDEX"
  echo "  ${cond} seed=${seed} -> job ${jobid}"
done
echo "Index: $INDEX"
