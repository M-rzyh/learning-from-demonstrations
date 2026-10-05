#!/bin/bash
# UNIFIED GAIL count-axis submitter — the demo-BUDGET axis at 0% noise.
# Supersedes submit_gail_grid_count.sh (and the count half of submit_gail_grid.sh).
# Sweeps N ∈ {1,5,10,50,100} (or any list), each subsampling the ONE clean pool
# n100_p0_clean per seed. This is the sibling of submit_gail_noise.sh (which sweeps
# noise at fixed N); here noise is fixed at 0 and N is swept.
#
# ┌─ PRE-FLIGHT (same guarantee as submit_gail_noise.sh) ───────────────────────────────┐
# │ Checks the clean pool exists BEFORE submitting anything; if it's missing it prints   │
# │ where to make it and exits without submitting a single job. CHECK=1 = check only.    │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# OPTIONS (env vars):
#   N_LIST="1 5 10 50 100"   demo budgets to sweep (each subsampled from the clean pool)
#   FS=0            0 = nonFS expert 4615187 | 1 = FS expert 4720242
#   NSEEDS=30       seeds 0..NSEEDS-1
#   BS=            demo_batch_size; empty = tier by N (<=1:128, <=5:512, else 1024)
#   INDEX=         output CSV; empty = experiments/gail_grid_count{_FS}_${DATE}.csv
#   CHECK=0        1 = pre-flight only (report, submit nothing)
#
# POOL: demos/noisy_demos/lunarlander/expert_${EXPERT}/n100_p0_clean (single pool, 150 eps).
#       Each (N, seed) draws a different N-subset via SHUFFLE_SEED=seed. Make it with
#       pool_generators/generate_clean_demo_pools.sh.
#
# EXAMPLES:
#   bash submit_gail_count.sh                              # 1,5,10,50,100 nonFS
#   CHECK=1 bash submit_gail_count.sh                      # just verify the clean pool
#   FS=1 bash submit_gail_count.sh                         # FS expert 4720242
#   N_LIST="10 50 100" bash submit_gail_count.sh
set -euo pipefail

N_LIST=${N_LIST:-"1 5 10 50 100"}
FS=${FS:-0}
NSEEDS=${NSEEDS:-30}
BS=${BS:-}
CHECK=${CHECK:-0}
DATE=$(date +%Y-%m-%d)

RUNROOT=/scratch/marzii/imitation_runs
case "$FS" in 0|1) ;; *) echo "ERROR: FS must be 0|1" 1>&2; exit 1;; esac
EXPERT=$([[ "$FS" == "1" ]] && echo 4720242 || echo 4615187)
FSTAG=$([[ "$FS" == "1" ]] && echo "_FS" || echo "")
INDEX=${INDEX:-/home/marzii/IRL3/experiments/GAIL/gail_grid_count${FSTAG}_${DATE}.csv}

CLEAN_POOL="$RUNROOT/demos/noisy_demos/lunarlander/expert_${EXPERT}/n100_p0_clean"

# demo_batch_size: small N = few transitions, so it scales with N (128/512/1024).
bs_for() {  # $1=N
  if [[ -n "$BS" ]]; then echo "$BS"
  elif [[ "$1" -le 1 ]]; then echo 128
  elif [[ "$1" -le 5 ]]; then echo 512
  else echo 1024; fi
}

echo "=== GAIL count submitter ==="
echo "  N_LIST: $N_LIST   FS=$FS (expert $EXPERT)   seeds: 0..$((NSEEDS-1))"
echo "  clean pool: $CLEAN_POOL"
echo "  index: $INDEX"
echo

# ---- PRE-FLIGHT: the one pool every run needs ----
if [[ ! -e "$CLEAN_POOL/dataset_info.json" ]]; then
  echo "PRE-FLIGHT FAILED: clean pool missing. Submitting nothing." 1>&2
  echo "  MISSING: $CLEAN_POOL" 1>&2
  echo "  Make it: $([[ $FS == 1 ]] && echo 'FS=1 ')sbatch scripts/pool_generators/generate_clean_demo_pools.sh" 1>&2
  exit 1
fi
echo "PRE-FLIGHT OK: clean pool present."

if [[ "$CHECK" == "1" ]]; then
  echo "CHECK=1 -> not submitting."
  exit 0
fi

# ---- submit (index appends; header only if new) ----
mkdir -p "$(dirname "$INDEX")"
[ -s "$INDEX" ] || echo "condition_id,N,noise_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"
cd /home/marzii/IRL3/imitation

submitted=0
for N in $N_LIST; do
  bs=$(bs_for "$N")
  for seed in $(seq 0 $((NSEEDS-1))); do
    jobid=$(DEMO_PATH="$CLEAN_POOL" N_DEMOS="$N" DEMO_BATCH_SIZE="$bs" \
            SHUFFLE=1 SHUFFLE_SEED="$seed" SEED="$seed" FRAME_SKIP=0 \
            DEMO_NOTE="grid cond=count_N${N} N=${N} noise=0% seed=${seed}" \
            sbatch --parsable run_gail_lunarlander.sh)
    echo "count_N${N},${N},0,${seed},${jobid},${CLEAN_POOL},submitted" >> "$INDEX"
    submitted=$((submitted+1))
    printf "  count N=%s seed=%s -> job %s\n" "$N" "$seed" "$jobid"
  done
done
echo
echo "Submitted $submitted jobs -> $INDEX"
