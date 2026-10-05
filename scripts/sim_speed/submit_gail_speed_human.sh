#!/bin/bash
# Submit GAIL training on SIMULATION-SPEED human demos (the GAIL arm of the speed study).
# The human played the SAME task at a faster/slower live pace; the agent trains in the normal
# env. Budget N=15 flagged (also first15 / all arms), 30 seeds.
#
#   bash scripts/sim_speed/submit_gail_speed_human.sh <fps> <seed_lo> <seed_hi>
#   ARM=flagged|first15|all   CHECK=1 (pre-flight only)
#
#   fps: 10 (0.2x, easy) | 20 (0.4x, baseline) | 50 (1.0x real time, hard)
#   20 fps REUSES the existing vanish_p0 clean fps-20 session (DIFFICULTY=none, fps 20 ⇒ the
#   speed baseline), so no new 20 fps play. 10 & 50 are their own collections:
#     DIFFICULTY=none FPS={10,50} FLAG_TARGET=15 FEEDBACK=both \
#     OUTPUT_DIR=$SCRATCH/imitation_runs/sim_speed/demos/human/speed_fps{10,50} \
#     bash imitation/collect_human_demos_lunarlander.sh
set -euo pipefail

FPS=${1:?usage: submit_gail_speed_human.sh <fps> <seed_lo> <seed_hi>}
LO=${2:?missing seed_lo}
HI=${3:?missing seed_hi}
CHECK=${CHECK:-0}
ARM=${ARM:-flagged}

case "$FPS" in 10|20|50) ;; *) echo "ERROR: fps must be 10, 20 or 50 (got '$FPS')" 1>&2; exit 1 ;; esac
case "$ARM" in flagged|first15|all) ;; *) echo "ERROR: ARM must be flagged|first15|all" 1>&2; exit 1 ;; esac

# 20 fps baseline reuses the vanish clean session; 10/50 have their own sim_speed dirs.
if [[ "$FPS" == "20" ]]; then
  ROOT=/scratch/marzii/imitation_runs/lander_vanish/demos/human/vanish_p0
else
  ROOT=/scratch/marzii/imitation_runs/sim_speed/demos/human/speed_fps${FPS}
fi
case "$ARM" in
  flagged) SUB=session_1_flagged ;;
  first15) SUB=session_1 ;;
  all)     SUB=session_1 ;;
esac
DEMO="$ROOT/$SUB"

# --- PRE-FLIGHT ---
if [[ ! -d "$DEMO" ]]; then
  echo "PRE-FLIGHT FAILED: missing demo dir $DEMO" 1>&2
  echo "  collect it first (see the header of this script)" 1>&2
  exit 1
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
echo "PRE-FLIGHT OK: $DEMO ($n_have episodes available, using $N_DEMOS)  [FPS=$FPS ARM=$ARM]"
if [[ "$CHECK" == "1" ]]; then echo "CHECK=1 — nothing submitted."; exit 0; fi

INDEX=/home/marzii/IRL3/experiments/GAIL/gail_lander_speed_human_$(date +%Y-%m-%d).csv
mkdir -p "$(dirname "$INDEX")"
[[ -f "$INDEX" ]] || echo "condition_id,N,speed_fps,seed,slurm_job_id,demo_path,status,arm" > "$INDEX"

cd /home/marzii/IRL3/imitation
cond="speed_human_fps${FPS}_${ARM}"
for seed in $(seq "$LO" "$HI"); do
  jobid=$(DEMO_PATH="$DEMO" N_DEMOS=$N_DEMOS DEMO_BATCH_SIZE=512 SHUFFLE=0 SEED="$seed" \
          N_EVAL_EPISODES=50 FRAME_SKIP=0 \
          DEMO_NOTE="sim_speed human fps=${FPS} seed=${seed} N=${N_DEMOS} arm=${ARM}" \
          sbatch --parsable --time=01:00:00 run_gail_lunarlander.sh)
  echo "${cond},${N_DEMOS},${FPS},${seed},${jobid},${DEMO},submitted,${ARM}" >> "$INDEX"
  echo "  ${cond} seed=${seed} -> job ${jobid}"
done
echo "Index: $INDEX"
