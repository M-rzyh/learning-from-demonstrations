#!/bin/bash
#SBATCH --job-name=gen-clean-pool
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/marzii/imitation_runs/gen_clean_pool_%j.log
#
# CLEAN demo-pool generator (the 0% point). Rolls out the expert with NO action noise
# (noise-prob 0) into a single pool `n100_p0_clean`, which submit_gail_clean_N.sh /
# submit_gail_grid_count.sh subsample per training seed (via SHUFFLE_SEED). This is the
# clean sibling of generate_online_noisy_demo_pools.sh — clean is ONE shared pool, not
# per-seed and not per-level, so there is no LEVELS/MODE/NSEEDS loop.
#
# A clean pool is pipeline-agnostic (noise-prob 0 corrupts nothing), so this online-rolled
# clean pool is interchangeable with the offline one and valid to pair with either the
# online or offline noise sweeps.
#
# OPTIONS (env vars):
#   FS=0            0 = nonFS expert 4615187 (default) | 1 = FS expert 4720242
#   NEPISODES=150  episodes in the pool (default 150, matching the existing clean pool;
#                  gives headroom for subsampling any N)
#   CLEAN_NAME=n100_p0_clean   output dir name (kept = what the submitters read)
#   FORCE=0        1 = regenerate even if the pool already exists (default: skip)
#
# OUTPUT (where the submitters already look):
#   nonFS -> demos/noisy_demos/lunarlander/expert_4615187/n100_p0_clean
#   FS    -> demos/noisy_demos/lunarlander/expert_4720242/n100_p0_clean
#
# EXAMPLES (run from the repo root):
#   sbatch scripts/pool_generators/generate_clean_demo_pools.sh            # nonFS clean, skip if present
#   FORCE=1 sbatch scripts/pool_generators/generate_clean_demo_pools.sh    # regenerate the nonFS clean pool
#   FS=1 sbatch scripts/pool_generators/generate_clean_demo_pools.sh       # FS clean pool
set -euo pipefail

PYTHON=/scratch/marzii/envs/imitation-gail/bin/python
COLLECT=/home/marzii/IRL3/scripts/collect_fs_expert_perframe_demos_online_noise.py

FS=${FS:-0}
NEPISODES=${NEPISODES:-150}
CLEAN_NAME=${CLEAN_NAME:-n100_p0_clean}
FORCE=${FORCE:-0}

if [[ "$FS" == "1" ]]; then
  POLICY=/scratch/marzii/imitation_runs/expert/lunarlander/4720242/policies/final/model.zip
  OUTROOT=/scratch/marzii/imitation_runs/demos/noisy_demos/lunarlander/expert_4720242
else
  POLICY=/scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip
  OUTROOT=/scratch/marzii/imitation_runs/demos/noisy_demos/lunarlander/expert_4615187
fi
[[ -f "$POLICY" ]] || { echo "ERROR: policy not found: $POLICY" 1>&2; exit 1; }
mkdir -p "$OUTROOT"

OUT="$OUTROOT/$CLEAN_NAME"
echo "=== clean demo-pool generation ==="
echo "  FS=$FS  expert=$(basename "$(dirname "$(dirname "$(dirname "$POLICY")")")")"
echo "  episodes: $NEPISODES   ->   $OUT"

if [[ "$FORCE" != "1" && -e "$OUT/dataset_info.json" ]]; then
  echo "  already exists (skip; FORCE=1 to regenerate)."
  exit 0
fi

OPENBLAS_NUM_THREADS=1 $PYTHON "$COLLECT" \
  --policy      "$POLICY" \
  --output      "$OUT" \
  --optimal-policy "$POLICY" \
  --noise-prob  0.0 \
  --noise-seed  0 \
  --n-episodes  "$NEPISODES" \
  --hold-k      1 \
  --max-frames  400 \
  --n-actions   4

echo "Done -> $OUT"
