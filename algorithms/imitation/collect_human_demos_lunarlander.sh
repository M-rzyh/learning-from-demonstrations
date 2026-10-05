#!/bin/bash
# ============================================================================
# Collect human demonstrations for LunarLander-v2 (GAIL-compatible format)
#
# THE single launcher for human demos. (start_demo_server.sh and
# run_human_demo_lunarlander.sh were deleted 2026-07-09: three generations of the
# same wrapper passing three different --fps values, none able to pass the
# difficulty flags — which is why blank50 had to be a raw typed command.)
#
# This script runs INTERACTIVELY — do NOT sbatch it.
#
# HOW TO USE:
#   1. Get an interactive session:
#        salloc --account=aip-mtaylor3 --cpus-per-task=2 --mem=4G --time=01:00:00
#   2. Run this script (see CONFIGURATION below).
#   3. It auto-finds a free port and prints the SSH tunnel command + URL.
#      Open a NEW terminal, run the tunnel, then open the URL in your browser.
#   4. Controls:
#        W / ↑  main engine     A / ←  left engine     D / →  right engine
#        S / ↓  coast (noop)
#        SPACE  start the episode.  After it ends: KEEP but do NOT flag.
#        F      save AND FLAG  (the flagged subset is what GAIL trains on)
#        X      discard this attempt
#        Q/ESC  quit & save everything collected so far
#      Every non-discarded episode is saved to session_N/; the FLAGGED subset is
#      ALSO written to session_N_flagged/.
#   5. Output: HuggingFace Arrow format + timing.csv, all_episodes_log.csv,
#      flags.json, and session_meta.json (records the fps actually used).
#
# SPEED: LunarLander physics is 50 steps/s, so FPS=20 => 0.40x real time, matching
# the 20fps PT preference clips. Keep it at 20 unless you are deliberately running a
# speed study — the b=10 blank50 demos were collected at --fps 50 (1.0x real time)
# against a 0.45x clean baseline, which confounded blanking with game speed.
#
# CONFIGURATION (env vars):
#   DIFFICULTY=none|blank|vanish|region|sticky|delay   Difficulty technique (default none)
#     PERCEPTION (hide the view):  blank, vanish, region   -> severity from PCT
#       blank  = the WHOLE frame goes black for a block
#       vanish = same block schedule, but only the LANDER is removed and replaced by the
#                terrain behind it; ground, pad and flags stay visible. Costs ~13 ms/frame
#                (27% of the 50 ms budget at FPS=20), so it does not slow the game.
#     CONTROL    (corrupt the key): sticky         -> severity from PCT
#                                   delay          -> severity from K (whole steps)
#   PCT=50                         Severity in PERCENT (blank: % of blocks blacked out;
#                                  region: % of frame AREA; sticky: % chance of repeating
#                                  the previous action). Not used by delay.
#   K=3                            delay mode: apply each action K steps late. At FPS=20 the
#                                  human feels K/FPS s of lag: K=3 -> 150 ms (laggy but
#                                  flyable), K=6 -> 300 ms (past the ~250 ms reaction budget).
#   BLOCK_LEN=5                    blank mode: blackout length, in displayed frames
#   OUTLINE=                       region mode: e.g. 'red' — draw a border inside the masked
#                                  box so it is distinguishable from the black sky (cosmetic).
#   FPS=20                         Env steps per wall-clock second
#   ---- Task/dynamics (change the PHYSICS; orthogonal to DIFFICULTY) ----
#   GRAVITY=                       LunarLander gravity (default env -10). Accepts the full range
#                                  incl. 0 (e.g. -3 easy, 0 = zero-g float, -12 hard). Empty = env default.
#   ENABLE_WIND=0                  1 = turn on wind + turbulence (gymnasium). Default 0 = no wind.
#   WIND_POWER=15                  Max linear wind (0-20) when ENABLE_WIND=1.
#   TURBULENCE_POWER=1.5           Max rotational wind (0-2) when ENABLE_WIND=1 (dominates the felt difficulty).
#     NB: a run with GRAVITY set or ENABLE_WIND=1 writes to a SEPARATE dynamics/ output dir, so it
#     never overwrites the clean human_demos/lunarlander baseline the plot scripts reuse as 0%.
#   FEEDBACK=reward                What the pilot sees about how the flight went:
#                                    reward  = running score + episode total (default)
#                                    outcome = no numbers, just LANDED / CRASHED / OUT OF TIME
#                                    both    = score AND verdict
#                                    none    = nothing
#                                  Only changes the SCREEN. The console log and the saved demo
#                                  always record the full reward, so no data is lost either way.
#   MAX_EPISODES=0                 0 = unlimited (play until you press Q)
#   FLAG_TARGET=50                 How many flagged demos you are aiming for
#   AUTO_QUIT=0                    1 = stop automatically once FLAG_TARGET is hit
#   SEED=42  BLANK_SEED=0  REGION_SEED=0  STICKY_SEED=0
#   PORT=0 (0 = auto)   OUTPUT_DIR=... (default derived from DIFFICULTY)
#
# EXAMPLES:
#   bash collect_human_demos_lunarlander.sh                             # clean 0%
#   DIFFICULTY=blank  PCT=50 BLOCK_LEN=5 bash collect_human_demos_lunarlander.sh
#   DIFFICULTY=vanish PCT=50 BLOCK_LEN=5 FLAG_TARGET=15 bash collect_human_demos_lunarlander.sh
#   DIFFICULTY=region PCT=25 OUTLINE=red bash collect_human_demos_lunarlander.sh
#   DIFFICULTY=sticky PCT=50             bash collect_human_demos_lunarlander.sh
#   DIFFICULTY=delay  K=3                bash collect_human_demos_lunarlander.sh
#   GRAVITY=0                            bash collect_human_demos_lunarlander.sh   # zero-g
#   ENABLE_WIND=1                        bash collect_human_demos_lunarlander.sh   # wind on
# ============================================================================

set -euo pipefail

# `set -u` would abort on an unset $SCRATCH (it is not always exported inside salloc).
SCRATCH="${SCRATCH:-/scratch/$USER}"
[[ -d "$SCRATCH" ]] || { echo "ERROR: SCRATCH=$SCRATCH does not exist." 1>&2; exit 1; }

# ---- Configuration ----
DIFFICULTY=${DIFFICULTY:-none}
PCT=${PCT:-0}
K=${K:-0}
BLOCK_LEN=${BLOCK_LEN:-5}
OUTLINE=${OUTLINE:-}
FPS=${FPS:-20}
FEEDBACK=${FEEDBACK:-reward}
MAX_EPISODES=${MAX_EPISODES:-0}
FLAG_TARGET=${FLAG_TARGET:-50}
AUTO_QUIT=${AUTO_QUIT:-0}
SEED=${SEED:-42}
BLANK_SEED=${BLANK_SEED:-0}
REGION_SEED=${REGION_SEED:-0}
STICKY_SEED=${STICKY_SEED:-0}
# ---- Task/dynamics (physics) options, orthogonal to DIFFICULTY ----
GRAVITY=${GRAVITY:-}
ENABLE_WIND=${ENABLE_WIND:-0}
WIND_POWER=${WIND_POWER:-15}
TURBULENCE_POWER=${TURBULENCE_POWER:-1.5}

# Assemble the dynamics flags + a filename tag. `set -e` forbids `[[ ]] && arr+=(...)`
# (a false test would abort the script), so use explicit if/fi.
DYN_EXTRA=()
DYN_PARTS=()
if [[ -n "$GRAVITY" ]]; then
    DYN_EXTRA+=(--gravity "$GRAVITY")
    GTAG=$(python3 -c "g=$GRAVITY; print(('m' if g<0 else '')+str(abs(g)).replace('.','p'))")
    DYN_PARTS+=("g${GTAG}")
fi
if [[ "$ENABLE_WIND" == "1" ]]; then
    DYN_EXTRA+=(--enable-wind --wind-power "$WIND_POWER" --turbulence-power "$TURBULENCE_POWER")
    DYN_PARTS+=("wind$(python3 -c "print(str($WIND_POWER).replace('.','p'))")")
fi

# ---- Output dir: derived from the technique so demos land where the plot scripts look ----
# (PT/scripts/plots/plot_frame_blanking_human_b5.py reads blank${BLOCK_LEN}_p${PCT}/session_1_flagged)
PCT_INT=$(printf '%.0f' "$PCT")
case "$DIFFICULTY" in
    none)   DEFAULT_OUT="$SCRATCH/imitation_runs/human_demos/lunarlander" ;;
    blank)  DEFAULT_OUT="$SCRATCH/imitation_runs/frame_blanking/demos/human/blank${BLOCK_LEN}_p${PCT_INT}" ;;
    vanish) DEFAULT_OUT="$SCRATCH/imitation_runs/lander_vanish/demos/human/vanish_p${PCT_INT}" ;;
    region) DEFAULT_OUT="$SCRATCH/imitation_runs/region_mask/demos/human/region${PCT_INT}" ;;
    sticky) DEFAULT_OUT="$SCRATCH/imitation_runs/sticky_actions/demos/human/sticky_p${PCT_INT}" ;;
    delay)  DEFAULT_OUT="$SCRATCH/imitation_runs/action_delay/demos/human/delay_k${K}" ;;
    *) echo "ERROR: DIFFICULTY must be none|blank|vanish|region|sticky|delay, got '$DIFFICULTY'" 1>&2; exit 1 ;;
esac
# delay's severity is K (whole steps); every other technique's is PCT.
if [[ "$DIFFICULTY" == "delay" ]]; then
    (( K > 0 )) || { echo "ERROR: DIFFICULTY=delay needs K>0 (e.g. K=3)" 1>&2; exit 1; }
elif [[ "$DIFFICULTY" != "none" ]] && (( PCT_INT <= 0 )); then
    echo "ERROR: DIFFICULTY=$DIFFICULTY needs PCT>0 (e.g. PCT=50)" 1>&2; exit 1
fi
# A dynamics run (gravity/wind) goes to a SEPARATE dir so it never overwrites the clean
# human_demos/lunarlander baseline (which the PT/GAIL plot scripts reuse as the 0% condition).
if (( ${#DYN_PARTS[@]} > 0 )); then
    DYN_JOINED=$(IFS=_; echo "${DYN_PARTS[*]}")
    if [[ "$DIFFICULTY" != "none" ]]; then
        DYN_JOINED="${DYN_JOINED}_${DIFFICULTY}${PCT_INT}"
    fi
    DEFAULT_OUT="$SCRATCH/imitation_runs/dynamics/demos/human/${DYN_JOINED}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUT}"

# Never silently append: human_demo.py would auto-number to session_2, but the plot
# scripts read session_1_flagged.
if [[ -e "$OUTPUT_DIR/session_1" ]]; then
    echo "REFUSING: $OUTPUT_DIR/session_1 already exists." 1>&2
    echo "  Move it aside, or set OUTPUT_DIR=... to collect somewhere else." 1>&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# ---- Find a free port ----
if [ "${PORT:-0}" = "0" ]; then
    PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    echo "Auto-selected free port: $PORT"
fi

# ---- Environment setup ----
module --force purge 2>/dev/null || true
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE 2>/dev/null || true
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"

python -c "import gymnasium as gym; env = gym.make('LunarLander-v2'); print('LunarLander-v2: OK'); env.close()"

# ---- Assemble the difficulty + quit flags ----
# NB: plain `[[ cond ]] && arr+=(...)` would exit 1 when cond is false, and `set -e`
# would kill the script. Use if/fi.
EXTRA=()
case "$DIFFICULTY" in
    blank)
        EXTRA+=(--difficulty blank --difficulty-pct "$PCT"
                --block-len "$BLOCK_LEN" --blank-seed "$BLANK_SEED") ;;
    vanish)
        EXTRA+=(--difficulty vanish --difficulty-pct "$PCT"
                --block-len "$BLOCK_LEN" --blank-seed "$BLANK_SEED") ;;
    region)
        EXTRA+=(--difficulty region --difficulty-pct "$PCT" --region-seed "$REGION_SEED")
        if [[ -n "$OUTLINE" ]]; then EXTRA+=(--region-outline "$OUTLINE"); fi ;;
    sticky)
        EXTRA+=(--difficulty sticky --difficulty-pct "$PCT" --sticky-seed "$STICKY_SEED") ;;
    delay)
        EXTRA+=(--difficulty delay --delay-k "$K") ;;   # delay takes K, not PCT
esac
if [[ "$AUTO_QUIT" == "1" ]]; then
    EXTRA+=(--auto-quit)
fi

HOSTNAME=$(hostname)
RT=$(python3 -c "print(f'{$FPS/50:.2f}')")

echo ""
echo "============================================================"
echo "  Human Demo Collection — LunarLander-v2"
echo "============================================================"
echo ""
case "$DIFFICULTY" in
    none)   DESC="none (clean)" ;;
    blank)  DESC="blank ${PCT}% (block_len=${BLOCK_LEN})" ;;
    vanish) DESC="lander vanish ${PCT}% (block_len=${BLOCK_LEN}, terrain stays visible)" ;;
    region) DESC="region ${PCT}% of frame area$([[ -n $OUTLINE ]] && echo ", ${OUTLINE} outline")" ;;
    sticky) DESC="sticky actions, p=$(python3 -c "print($PCT/100)")" ;;
    delay)  DESC="action delay, k=${K} steps ($(python3 -c "print(f'{$K*1000/50:.0f}')") ms game time, $(python3 -c "print(f'{$K*1000/$FPS:.0f}')") ms felt at ${FPS} fps)" ;;
esac
echo "  Difficulty:  ${DESC}"
if (( ${#DYN_PARTS[@]} > 0 )); then
    DYN_DESC="gravity=${GRAVITY:-default}"
    if [[ "$ENABLE_WIND" == "1" ]]; then DYN_DESC="${DYN_DESC}, wind ON (power=${WIND_POWER}, turbulence=${TURBULENCE_POWER})"; else DYN_DESC="${DYN_DESC}, no wind"; fi
    echo "  Dynamics:    ${DYN_DESC}"
fi
echo "  FPS:         ${FPS}   (physics 50 steps/s => ${RT}x real time)"
echo "  Episodes:    $([[ $MAX_EPISODES -eq 0 ]] && echo 'unlimited (press Q to finish)' || echo "$MAX_EPISODES")"
echo "  Flag target: ${FLAG_TARGET}   auto-quit: $([[ $AUTO_QUIT == 1 ]] && echo yes || echo 'no (press Q)')"
echo "  Output:      ${OUTPUT_DIR}"
echo ""
echo "  STEP 1 — Open a NEW terminal and run this SSH tunnel:"
echo ""
echo "    ssh -L ${PORT}:localhost:${PORT} ${HOSTNAME}"
echo ""
echo "  STEP 2 — Open this URL in your browser:"
echo ""
echo "    http://localhost:${PORT}"
echo ""
echo "  STEP 3 — Click the game area, then use W/A/D/S to play."
echo "           SPACE starts an episode. After it ends:"
echo "             F = save & FLAG     SPACE = keep (not flagged)"
echo "             X = discard         Q     = quit & save"
echo ""
echo "============================================================"
echo ""
echo "Starting server in 5 seconds..."
sleep 5

# ---- Collect demos ----
cd /home/marzii/IRL3/imitation || exit 1

python human_demo.py \
    --env LunarLander-v2 \
    --output "$OUTPUT_DIR" \
    --max_episodes "$MAX_EPISODES" \
    --flag-target "$FLAG_TARGET" \
    --fps "$FPS" \
    --feedback "$FEEDBACK" \
    --seed "$SEED" \
    --mode web \
    --port "$PORT" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    ${DYN_EXTRA[@]+"${DYN_EXTRA[@]}"}   # dynamics flags; empty-array-safe under `set -u`

echo ""
echo "============================================================"
echo "  Done! Demos saved to: ${OUTPUT_DIR}"
echo "  Speed provenance:     ${OUTPUT_DIR}/session_1/session_meta.json"
echo ""
echo "  To train GAIL on the FLAGGED subset:"
echo "    DEMO_PATH=\"${OUTPUT_DIR}/session_1_flagged\" N_DEMOS=${FLAG_TARGET} sbatch run_gail_lunarlander.sh"
echo "============================================================"
