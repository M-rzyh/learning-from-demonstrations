#!/bin/bash
# Generate all noisy demo datasets for the GAIL grid experiment.
# Source: perframe_demos_150 from FS expert 4720242.
# Output: 4 noise levels × 10 seeds + 1 clean = 41 datasets.

# experts: FS: 4720242 (lunarlander) NonFS: 4615187 (lunarlander)
set -euo pipefail

PYTHON=/scratch/marzii/envs/imitation-gail/bin/python
SCRIPT=/home/marzii/IRL3/scripts/add_action_noise.py
INPUT=/scratch/marzii/imitation_runs/expert/lunarlander/4615187/rollouts/perframe_demos_150_holdK1
# INPUTFS=/scratch/marzii/imitation_runs/expert/lunarlander/4720242/rollouts/perframe_demos_150
OPTIMAL=/scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip
# OPTIMAL_FS=/scratch/marzii/imitation_runs/expert/lunarlander/4720242/policies/final/model.zip
OUTROOT=/scratch/marzii/imitation_runs/demos/noisy_demos/lunarlander/expert_4615187
# OUTROOT_FS=/scratch/marzii/imitation_runs/demos/noisy_demos/lunarlander/expert_4720242

mkdir -p "$OUTROOT"

# Clean baseline (noise=0, just compute alignment of the original demos)
echo "=== Clean baseline (noise=0) ==="
CLEAN_OUT="$OUTROOT/n100_p0_clean"
OPENBLAS_NUM_THREADS=4 $PYTHON $SCRIPT \
  --input "$INPUT" \
  --output "$CLEAN_OUT" \
  --noise-prob 0.0 \
  --seed 0 \
  --optimal-policy "$OPTIMAL"

# 4 noise levels × 10 seeds = 40 noisy datasets
declare -A PROB=( [10]=0.1 [20]=0.2 [25]=0.25 [30]=0.30 [40]=0.40 [50]=0.50 [60]=0.60 [70]=0.70 [75]=0.75 [80]=0.80 [90]=0.90 [100]=1.0)
for noise in 10 20 25 30 40 50 60 70 75 80 90 100; do
  for seed in {0..29}; do
    OUT="$OUTROOT/n100_p${noise}_s${seed}"
    if [ -d "$OUT" ]; then
      echo "[skip] already exists: $OUT"
      continue
    fi
    echo "=== noise=${noise}%, seed=${seed} ==="
    OPENBLAS_NUM_THREADS=4 $PYTHON $SCRIPT \
      --input "$INPUT" \
      --output "$OUT" \
      --noise-prob "${PROB[$noise]}" \
      --seed "$seed" \
      --optimal-policy "$OPTIMAL"
  done
done

echo ""
echo "Done. 41 datasets generated under $OUTROOT"
