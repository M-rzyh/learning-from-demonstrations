#!/bin/bash
# Submit GAIL training on LANDER-VANISH human demos (the GAIL arm of the vanishing study).
#
# The demos are degraded — the human piloted with the lander itself removed from the display
# during b=5 blocks, terrain still visible — but the agent trains in the NORMAL 8-D env, as
# on every other difficulty axis. Budget N_DEMOS=15, the low-budget pairing validated against
# PT N=100 on the noise axis.
#
#   bash scripts/lander_vanish/submit_gail_vanish_human.sh <pct> <seed_lo> <seed_hi>
#   CHECK=1 bash scripts/lander_vanish/submit_gail_vanish_human.sh 50 0 9   # pre-flight only
#
# Examples:
#   bash scripts/lander_vanish/submit_gail_vanish_human.sh 0  0 9   # clean baseline (fps 20)
#   bash scripts/lander_vanish/submit_gail_vanish_human.sh 50 0 9
#
# WHY 0% IS ITS OWN COLLECTION, NOT session_2
#   The pre-existing clean human sessions were played at fps 25 (0.45x real time) while every
#   vanish session is fps 20 (0.40x). Reusing them would put a speed difference on the 0%
#   point and confound it with the axis — the same mistake that made the b=10 blanking result
#   unusable. Collect the clean baseline with:
#     DIFFICULTY=none FPS=20 FLAG_TARGET=15 \
#     OUTPUT_DIR=$SCRATCH/imitation_runs/lander_vanish/demos/human/vanish_p0 \
#     bash imitation/collect_human_demos_lunarlander.sh
#
# SHUFFLE=0: all 10 seeds train on the SAME 15 demos, so the band measures training variance
# only. With 15 demos a per-seed subsample would instead measure which demos got picked.
#
# Runs land in gail/lunarlander/<job>/ (job-keyed, non-destructive); the index CSV maps
# job -> (vanish%, seed) and is APPENDED to, never overwritten.
set -euo pipefail

PCT=${1:?usage: submit_gail_vanish_human.sh <pct> <seed_lo> <seed_hi>}
LO=${2:?missing seed_lo}
HI=${3:?missing seed_hi}
CHECK=${CHECK:-0}

case "$PCT" in 0|25|50|75) ;; *) echo "ERROR: pct must be 0, 25, 50 or 75 (got '$PCT')" 1>&2; exit 1 ;; esac

ROOT=/scratch/marzii/imitation_runs/lander_vanish/demos/human

# ARM=flagged (default) — the 15 demos the human JUDGED GOOD. Held constant across levels, so
#     the comparison is "same budget, same quality bar, more of the lander hidden".
# ARM=first15 — the FIRST 15 SAVED episodes (session_1, recording order, UNcurated). Same
#     count as flagged (15) but no quality filter, so flagged-vs-first15 isolates CURATION at a
#     fixed count, and first15-vs-all isolates COUNT at fixed (uncurated) quality.
# ARM=all     — every episode the human saved, crashes included. N is NOT constant across
#     levels (28/33/49/74 at 0/25/50/75%) because a harder level takes more attempts to produce
#     15 good flights, so this arm varies demo COUNT and QUALITY together.
ARM=${ARM:-flagged}
case "$ARM" in
  flagged) SUB=session_1_flagged ;;
  first15) SUB=session_1 ;;          # first 15 of all-saved (SHUFFLE=0 takes the first N)
  all)     SUB=session_1 ;;
  *) echo "ERROR: ARM must be 'flagged', 'first15' or 'all' (got '$ARM')" 1>&2; exit 1 ;;
esac
DEMO="$ROOT/vanish_p${PCT}/$SUB"

# --- PRE-FLIGHT: resolve and check everything BEFORE submitting anything, so a missing or
# short demo dir costs an error message instead of 10 queued jobs that all fail. ---
if [[ ! -d "$DEMO" ]]; then
  echo "PRE-FLIGHT FAILED: missing demo dir $DEMO" 1>&2
  echo "  collect it first (see the header of this script)" 1>&2
  exit 1
fi
# Episodes live as ROWS in one .arrow dataset, not one file each, so count rows. Use the
# imitation-gail interpreter explicitly: the login node's python3 has no pyarrow.
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
  N_DEMOS=$n_have          # use everything that was saved
  if (( N_DEMOS < 1 )); then
    echo "PRE-FLIGHT FAILED: $DEMO has no episodes" 1>&2
    exit 1
  fi
else
  N_DEMOS=15               # flagged and first15 both use exactly 15 (SHUFFLE=0 -> first 15)
  if (( n_have < N_DEMOS )); then
    echo "PRE-FLIGHT FAILED: $DEMO has $n_have demos, need $N_DEMOS" 1>&2
    exit 1
  fi
fi
echo "PRE-FLIGHT OK: $DEMO ($n_have episodes available, using $N_DEMOS)  [ARM=$ARM]"
if [[ "$CHECK" == "1" ]]; then echo "CHECK=1 — nothing submitted."; exit 0; fi

INDEX=/home/marzii/IRL3/experiments/GAIL/gail_lander_vanish_human_$(date +%Y-%m-%d).csv
mkdir -p "$(dirname "$INDEX")"
[[ -f "$INDEX" ]] || echo "condition_id,N,vanish_pct,seed,slurm_job_id,demo_path,status,arm" > "$INDEX"

cd /home/marzii/IRL3/imitation
cond="vanish_human_b5_p${PCT}_${ARM}"
for seed in $(seq "$LO" "$HI"); do
  # --time overrides run_gail_lunarlander.sh's 00:30:00 directive rather than editing that
  # shared script: N=15 runs measured ~24 min, and one noise-sweep job already died on that
  # margin. Walltime is not a scientific parameter, so overriding per-submission is safe.
  jobid=$(DEMO_PATH="$DEMO" N_DEMOS=$N_DEMOS DEMO_BATCH_SIZE=512 SHUFFLE=0 SEED="$seed" \
          N_EVAL_EPISODES=50 FRAME_SKIP=0 \
          DEMO_NOTE="lander_vanish b=5 human vanish=${PCT}% seed=${seed} N=${N_DEMOS} fps=20 arm=${ARM}" \
          sbatch --parsable --time=01:00:00 run_gail_lunarlander.sh)
  echo "${cond},${N_DEMOS},${PCT},${seed},${jobid},${DEMO},submitted,${ARM}" >> "$INDEX"
  echo "  ${cond} seed=${seed} -> job ${jobid}"
done
echo "Index: $INDEX"
