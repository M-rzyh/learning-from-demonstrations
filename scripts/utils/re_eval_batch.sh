#!/bin/bash
#SBATCH --job-name=re-eval-gail
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/imitation_runs/_slurm_logs/re_eval_%A_%a.out
#SBATCH --error=/scratch/marzii/imitation_runs/_slurm_logs/re_eval_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Re-evaluate all GAIL runs with N_EPS episodes (default 100).
# Submit as array: sbatch --array=0-9 re_eval_batch.sh
# Each task processes a slice of the run-dir list.

set -euo pipefail

RUN_LIST=/home/marzii/IRL3/experiments/eval_run_list.txt
N_EPS=${N_EPS:-100}
N_TASKS=${N_TASKS:-10}

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$(wc -l < "$RUN_LIST")
PER_TASK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$(( TASK_ID * PER_TASK + 1 ))
END=$(( START + PER_TASK - 1 ))
[ $END -gt $TOTAL ] && END=$TOTAL

echo "Task $TASK_ID: processing lines $START..$END of $TOTAL ($N_EPS eps each)"

while IFS= read -r RUN_DIR; do
  [ -z "$RUN_DIR" ] && continue
  python /home/marzii/IRL3/scripts/utils/re_eval_run.py "$RUN_DIR" --n-eps $N_EPS
done < <(sed -n "${START},${END}p" "$RUN_LIST")
