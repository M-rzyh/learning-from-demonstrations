#!/bin/bash
# Submit BC training for the expert-vs-human realizability diagnostic.
#
#   bash bc/submit_bc_lunarlander.sh <condition> <seed_lo> <seed_hi>
#   CHECK=1 bash bc/submit_bc_lunarlander.sh expert15 0 29   # pre-flight only
#
# Conditions (all N=15, cap 1000, eval on true env reward):
#   expert15        - first 15 of the SAC expert 4615187 rollouts (GAIL turns these into ~+265)
#   human_flagged15 - the 15 flagged (best) human demos from vanish_p0 (~+269 demo return)
#   human_first15   - the first 15 saved human demos from vanish_p0 (uncurated)
#
# expert-BC should roughly recover the expert (~+270). If human-BC caps well below, the
# human->expert gap is realizability, not a GAIL artifact. Runs are job-keyed (non-destructive);
# the index CSV maps job -> (condition, seed) and is appended, never overwritten.
set -euo pipefail

COND=${1:?usage: submit_bc_lunarlander.sh <expert15|human_flagged15|human_first15> <lo> <hi>}
LO=${2:?missing seed_lo}
HI=${3:?missing seed_hi}
CHECK=${CHECK:-0}

EXP=/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/final.npz
VP0=/scratch/marzii/imitation_runs/lander_vanish/demos/human/vanish_p0
case "$COND" in
  expert15)        DEMO="$EXP" ;;
  human_flagged15) DEMO="$VP0/session_1_flagged" ;;
  human_first15)   DEMO="$VP0/session_1" ;;
  *) echo "ERROR: condition must be expert15|human_flagged15|human_first15 (got '$COND')" 1>&2; exit 1 ;;
esac
N_DEMOS=15

# PRE-FLIGHT: demo dir present with >=15 episodes (rows across the arrow files).
[[ -d "$DEMO" ]] || { echo "PRE-FLIGHT FAILED: missing $DEMO" 1>&2; exit 1; }
n_have=$(/scratch/marzii/envs/imitation-gail/bin/python - "$DEMO" <<'PY'
import glob, sys, pyarrow.ipc
n=0
for f in glob.glob(sys.argv[1]+"/*.arrow"):
    with pyarrow.ipc.open_stream(f) as rd: n+=rd.read_all().num_rows
print(n)
PY
)
(( n_have >= N_DEMOS )) || { echo "PRE-FLIGHT FAILED: $DEMO has $n_have demos, need $N_DEMOS" 1>&2; exit 1; }
echo "PRE-FLIGHT OK: $COND -> $DEMO ($n_have available, using $N_DEMOS)"
[[ "$CHECK" == "1" ]] && { echo "CHECK=1 - nothing submitted."; exit 0; }

INDEX=/home/marzii/IRL3/experiments/BC/bc_lunarlander_$(date +%Y-%m-%d).csv
mkdir -p "$(dirname "$INDEX")"
[[ -f "$INDEX" ]] || echo "condition,N,seed,slurm_job_id,demo_path,status" > "$INDEX"

cd /home/marzii/IRL3/imitation
for seed in $(seq "$LO" "$HI"); do
  jobid=$(DEMO_PATH="$DEMO" N_DEMOS=$N_DEMOS SEED="$seed" ENV_MAX_EP_STEPS=1000 N_EVAL_EPISODES=50 \
          sbatch --parsable bc/run_bc_lunarlander.sh)
  echo "${COND},${N_DEMOS},${seed},${jobid},${DEMO},submitted" >> "$INDEX"
  echo "  bc ${COND} seed=${seed} -> job ${jobid}"
done
echo "Index: $INDEX"
