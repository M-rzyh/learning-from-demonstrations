#!/bin/bash
#SBATCH --job-name=gen-blind-expert
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/marzii/imitation_runs/frame_blanking/demos/blind_expert/generate_%j.log
#
# Generate BLIND-EXPERT demos (frame-blanking difficulty, Exp 1) with the
# non-frame-skip expert 4615187. On blanked frames the expert is fed a zeroed
# obs -> acts blind -> degraded, physically-consistent 8-D demos. New file; does
# not touch the online-noise collector. Outputs isolated in frame_blanking/.
#   sbatch scripts/pool_generators/generate_blind_expert_demo_pools.sh [NSEEDS]   (default 5)
set -euo pipefail

PYTHON=/scratch/marzii/envs/imitation-gail/bin/python
SCRIPT=/home/marzii/IRL3/scripts/collect_blind_expert_demos.py
POLICY=/scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip
# block b=10 demos live in a separate subdir; the earlier stochastic pools are left intact.
OUTROOT=/scratch/marzii/imitation_runs/frame_blanking/demos/blind_expert/block
mkdir -p "$OUTROOT"

declare -A PROB=( [0]=0.0 [25]=0.25 [50]=0.5 [75]=0.75 )
NSEEDS=${1:-5}

for blank in 0 25 50 75; do
  for seed in $(seq 0 $((NSEEDS - 1))); do
    OUT="$OUTROOT/n100_p${blank}_s${seed}"
    echo "=== block b=10 blank=${blank}% seed=${seed} -> $OUT ==="
    OPENBLAS_NUM_THREADS=1 $PYTHON $SCRIPT \
      --policy "$POLICY" --optimal-policy "$POLICY" \
      --output "$OUT" --n-episodes 100 --hold-k 1 --max-frames 400 \
      --blank-mode block --block-len 10 --blank-prob "${PROB[$blank]}" --blank-seed "$seed"
  done
done
echo "All blind-expert demos generated (blank {0,25,50,75}%, seeds 0-$((NSEEDS-1)))."
