#!/bin/bash
# UNIFIED GAIL noise-axis submitter. 
# Supersedes submit_gail_grid_noise{,_N50,_N}.sh and per-MODE) submit_gail_noise_variants.sh. One launcher for the whole 0-100% noise axis --> deleted!
# at any demo budget N, over the online/offline × FS/nonFS × variant-MODE matrix.
#
# ┌─ PRE-FLIGHT (the point of this script) ────────────────────────────────────────────┐
# │ Before submitting ANYTHING it resolves and checks EVERY pool it would use. If even   │
# │ one is missing it prints the full missing list and exits WITHOUT submitting a single │
# │ job — so you can never half-submit a sweep against nonexistent pools (the bug that   │
# │ cost the N15 0% jobs). Run CHECK=1 to do only the pre-flight and submit nothing.     │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# OPTIONS (env vars):
#   N=15            demo budget per run (subsample from each pool)
#   LEVELS="10 20 30 40 50 60 70 80 90 100"   noise % axis (default excludes 0 — the
#                    0% point comes from submit_gail_count.sh as count_N{N}; see NOTE below.
#                    You *can* pass a LEVELS containing 0 and it routes to the clean pool,
#                    but then it lands in the NOISE csv as noise_N{N}_p0, not count_N{N}.)
#   PIPELINE=online   online (noise injected during rollout) | offline (label-flip)
#   FS=0            0 = nonFS expert 4615187 | 1 = FS expert 4720242
#   MODE=coin_uniform   coin_uniform | exact_uniform | coin_exclude | exact_exclude
#                       (variants are ONLINE + nonFS only; offline/FS => coin_uniform only)
#   NSEEDS=30       seeds 0..NSEEDS-1
#   BS=            demo_batch_size; empty = derive from N (<=15:128, <=50:512, else 1024)
#   INDEX=         output CSV; empty = experiments/gail_grid_noise_N${N}_${DATE}.csv
#   CHECK=0        1 = pre-flight only (report pool existence, submit nothing)
#
# POOL ROUTING (mirrors pool_generators/ so submitter and generator always agree):
#   level 0 (clean): demos/noisy_demos/lunarlander/expert_${EXPERT}/n100_p0_clean   (ONE pool,
#                    subsampled per seed via SHUFFLE_SEED — clean is not per-seed)
#   level>0:
#     online nonFS coin_uniform -> demos/noisy_demos_online_nonFS/lunarlander/expert_4615187/n100_p{P}_s{S}
#     online nonFS variant      -> demos/noise_variants/demos/${MODE}/n100_p{P}_s{S}
#     online FS  coin_uniform   -> demos/noisy_demos_online_FS/lunarlander/expert_4720242/n100_p{P}_s{S}
#     offline    coin_uniform   -> demos/noisy_demos/lunarlander/expert_${EXPERT}/n100_p{P}_s{S}
#
# NOTE (0% convention): the 0% GAIL point is the SAME experiment as count_N{N} (clean demos
# at budget N). To avoid computing it twice, this script's default LEVELS omits 0 — get the
# 0% point from submit_gail_count.sh (recorded as count_N{N}), which is what the plots read.
#
# EXAMPLES:
#   N=15 bash submit_gail_noise.sh                         # noise 10-100%, N=15, online nonFS
#   CHECK=1 N=50 bash submit_gail_noise.sh                 # just verify the N=50 pools exist
#   MODE=exact_exclude N=100 LEVELS="10 50 100" bash submit_gail_noise.sh
#   PIPELINE=offline N=100 bash submit_gail_noise.sh
set -euo pipefail

N=${N:-15}
LEVELS=${LEVELS:-"10 20 30 40 50 60 70 80 90 100"}
PIPELINE=${PIPELINE:-online}
FS=${FS:-0}
MODE=${MODE:-coin_uniform}
NSEEDS=${NSEEDS:-30}
BS=${BS:-}
CHECK=${CHECK:-0}
DATE=$(date +%Y-%m-%d)
INDEX=${INDEX:-/home/marzii/IRL3/experiments/GAIL/gail_grid_noise_N${N}_${DATE}.csv}

RUNROOT=/scratch/marzii/imitation_runs

# ---- validate the option combination ----
case "$PIPELINE" in online|offline) ;; *) echo "ERROR: PIPELINE must be online|offline" 1>&2; exit 1;; esac
case "$MODE" in coin_uniform|exact_uniform|coin_exclude|exact_exclude) ;;
  *) echo "ERROR: bad MODE '$MODE'" 1>&2; exit 1;; esac
if [[ "$MODE" != "coin_uniform" ]]; then
  [[ "$PIPELINE" == "online" && "$FS" == "0" ]] || {
    echo "ERROR: MODE=$MODE (a variant) exists ONLY for PIPELINE=online, FS=0." 1>&2; exit 1; }
fi
[[ "$PIPELINE" == "offline" && "$MODE" != "coin_uniform" ]] && {
  echo "ERROR: offline has no variants; use MODE=coin_uniform." 1>&2; exit 1; }

EXPERT=$([[ "$FS" == "1" ]] && echo 4720242 || echo 4615187)

# demo_batch_size: high-noise demos are short, so it scales with N (128/512/1024).
if [[ -z "$BS" ]]; then
  if   [[ "$N" -le 15 ]]; then BS=128
  elif [[ "$N" -le 50 ]]; then BS=512
  else BS=1024; fi
fi

# ---- pool path for a (level, seed) ----
# clean (level 0) is one shared pool; noise levels are per-seed pools whose root depends
# on (PIPELINE, FS, MODE).
CLEAN_POOL="$RUNROOT/demos/noisy_demos/lunarlander/expert_${EXPERT}/n100_p0_clean"
if [[ "$PIPELINE" == "offline" ]]; then
  NOISE_ROOT="$RUNROOT/demos/noisy_demos/lunarlander/expert_${EXPERT}"
elif [[ "$MODE" != "coin_uniform" ]]; then
  NOISE_ROOT="$RUNROOT/demos/noise_variants/demos/${MODE}"
elif [[ "$FS" == "1" ]]; then
  NOISE_ROOT="$RUNROOT/demos/noisy_demos_online_FS/lunarlander/expert_4720242"
else
  NOISE_ROOT="$RUNROOT/demos/noisy_demos_online_nonFS/lunarlander/expert_4615187"
fi

pool_for() {  # $1=level $2=seed  ->  echoes the pool dir
  if [[ "$1" == "0" ]]; then echo "$CLEAN_POOL"; else echo "$NOISE_ROOT/n100_p${1}_s${2}"; fi
}

echo "=== GAIL noise submitter ==="
echo "  N=$N  PIPELINE=$PIPELINE  FS=$FS (expert $EXPERT)  MODE=$MODE  bs=$BS"
echo "  levels: $LEVELS   seeds: 0..$((NSEEDS-1))"
echo "  noise root: $NOISE_ROOT"
echo "  clean pool: $CLEAN_POOL"
echo "  index: $INDEX"
echo

# ---- PRE-FLIGHT: verify every pool BEFORE submitting anything ----
missing=()
checked=0
for level in $LEVELS; do
  if [[ "$level" == "0" ]]; then
    checked=$((checked+1))
    [[ -e "$CLEAN_POOL/dataset_info.json" ]] || missing+=("$CLEAN_POOL   (level 0 clean)")
  else
    for seed in $(seq 0 $((NSEEDS-1))); do
      checked=$((checked+1))
      d="$NOISE_ROOT/n100_p${level}_s${seed}"
      [[ -e "$d/dataset_info.json" ]] || missing+=("$d")
    done
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "PRE-FLIGHT FAILED: ${#missing[@]} of $checked pools missing. Submitting nothing." 1>&2
  printf '  MISSING: %s\n' "${missing[@]}" 1>&2
  echo 1>&2
  echo "  Generate them first, e.g.:" 1>&2
  if [[ "$level" == "0" || " $LEVELS " == *" 0 "* ]]; then
    echo "    sbatch scripts/pool_generators/generate_clean_demo_pools.sh $([[ $FS == 1 ]] && echo FS=1)" 1>&2
  fi
  echo "    ${MODE:+MODE=$MODE }$([[ $FS == 1 ]] && echo 'FS=1 ')sbatch scripts/pool_generators/generate_online_noisy_demo_pools.sh" 1>&2
  exit 1
fi
echo "PRE-FLIGHT OK: all $checked pools present."

if [[ "$CHECK" == "1" ]]; then
  echo "CHECK=1 -> not submitting."
  exit 0
fi

# ---- submit (index appends; header only if new) ----
mkdir -p "$(dirname "$INDEX")"
[ -s "$INDEX" ] || echo "condition_id,N,noise_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"
cd /home/marzii/IRL3/imitation

submitted=0
for level in $LEVELS; do
  for seed in $(seq 0 $((NSEEDS-1))); do
    demo=$(pool_for "$level" "$seed")
    # clean (single pool) always shuffles per seed; per-seed noise pool uses the whole pool
    # when N>=100, else subsamples.
    if [[ "$level" == "0" || "$N" -lt 100 ]]; then shuffle=1; else shuffle=0; fi
    jobid=$(DEMO_PATH="$demo" N_DEMOS="$N" DEMO_BATCH_SIZE="$BS" \
            SHUFFLE="$shuffle" SHUFFLE_SEED="$seed" SEED="$seed" FRAME_SKIP=0 \
            DEMO_NOTE="grid cond=noise_N${N}_p${level} N=${N} noise=${level}% seed=${seed}" \
            sbatch --parsable run_gail_lunarlander.sh)
    echo "noise_N${N}_p${level},${N},${level},${seed},${jobid},${demo},submitted" >> "$INDEX"
    submitted=$((submitted+1))
    printf "  N=%s p=%s%% seed=%s -> job %s\n" "$N" "$level" "$seed" "$jobid"
    # clean pool is single: only one seed's worth of dir, but we still submit NSEEDS runs
    # (each a different SHUFFLE_SEED subset) — so no break here.
  done
done
echo
echo "Submitted $submitted jobs -> $INDEX"
