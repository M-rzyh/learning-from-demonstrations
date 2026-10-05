#!/bin/bash
#SBATCH --job-name=gen-online-noise
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/marzii/imitation_runs/gen_online_noise_%j.log
#
# Unified ONLINE action-noise demo-pool generator.
#
# OPTIONS (env vars):
# Noise injection(online/offline --> default = online): The only option is online(used to have offline as well). Noise is injected DURING the rollout — the env actually steps with the corrupted action, so the trajectory genuinely diverges.
# FS(nonFS/FS --> default = 0): 0 = nonFS expert 4615187 (default) | 1 = FS expert 4720242
# MODE(coin_uniform/exact_uniform/coin_exclude/exact_exclude --> default = coin_uniform):
#   noise variant = <selection>_<replacement>:
#                     selection  : coin  = Bernoulli(p) per step (count wobbles)
#                                  exact = exactly round(p*frames) steps corrupted
#                     replacement: uniform = random over ALL 4 actions (~1/4 land back on
#                                            the expert -> built-in "info-destruction" floor)
#                                  exclude = random over the 3 NON-expert actions (every
#                                            corruption is wrong -> anti-signal)
#                     -> {coin_uniform (default; = your current N15/N50/N100 pools),
#                         exact_uniform, coin_exclude, exact_exclude}
#                     NB: FS supports ONLY coin_uniform (the variants collector is nonFS-only).
# LEVELS="10 20 25 30 40 50 60 70 75 80 90 100"   noise % levels
# NSEEDS=30      seeds 0..NSEEDS-1
# NEPISODES=100  episodes per pool (dir prefix n${NEPISODES}_)
# FORCE=0        1 = regenerate even if the output dir already exists (default: skip existing)
#
# OUTPUT (kept where the existing pools + submitters expect them):
#   coin_uniform + nonFS -> demos/noisy_demos_online_nonFS/lunarlander/expert_4615187/n100_p{P}_s{S}
#   coin_uniform + FS    -> demos/noisy_demos_online_FS/lunarlander/expert_4720242/n100_p{P}_s{S}
#   other variant (nonFS)-> demos/noise_variants/demos/${MODE}/n100_p{P}_s{S}
#
# EXAMPLES (run from the repo root):
#   sbatch scripts/pool_generators/generate_online_noisy_demo_pools.sh            # nonFS coin_uniform
#   MODE=exact_exclude sbatch scripts/pool_generators/generate_online_noisy_demo_pools.sh  # nonFS anti-signal
#   FS=1 sbatch scripts/pool_generators/generate_online_noisy_demo_pools.sh       # FS coin_uniform

set -euo pipefail

PYTHON=/scratch/marzii/envs/imitation-gail/bin/python
ORIG=/home/marzii/IRL3/scripts/collect_fs_expert_perframe_demos_online_noise.py
VARIANTS=/home/marzii/IRL3/scripts/collect_expert_perframe_demos_online_noise_variants.py

FS=${FS:-0}
MODE=${MODE:-coin_uniform}
NSEEDS=${NSEEDS:-30}
NEPISODES=${NEPISODES:-100}
FORCE=${FORCE:-0}
LEVELS=${LEVELS:-"10 20 25 30 40 50 60 70 75 80 90 100"}

case "$MODE" in
  coin_uniform|exact_uniform|coin_exclude|exact_exclude) ;;
  *) echo "ERROR: MODE must be coin_uniform|exact_uniform|coin_exclude|exact_exclude, got '$MODE'" 1>&2; exit 1 ;;
esac

# --- expert + collector + output root, per (FS, MODE) ---
if [[ "$FS" == "1" ]]; then
  [[ "$MODE" == "coin_uniform" ]] || { echo "ERROR: FS=1 supports only MODE=coin_uniform (variants are nonFS-only)." 1>&2; exit 1; }
  POLICY=/scratch/marzii/imitation_runs/expert/lunarlander/4720242/policies/final/model.zip
  OUTROOT=/scratch/marzii/imitation_runs/demos/noisy_demos_online_FS/lunarlander/expert_4720242
  SCRIPT=$ORIG; MODEARG=()
else
  POLICY=/scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip
  if [[ "$MODE" == "coin_uniform" ]]; then
    # original collector reproduces the EXACT existing nonFS pools (byte-identical default)
    OUTROOT=/scratch/marzii/imitation_runs/demos/noisy_demos_online_nonFS/lunarlander/expert_4615187
    SCRIPT=$ORIG; MODEARG=()
  else
    OUTROOT=/scratch/marzii/imitation_runs/demos/noise_variants/demos/${MODE}
    SCRIPT=$VARIANTS; MODEARG=(--noise-mode "$MODE")
  fi
fi
[[ -f "$POLICY" ]] || { echo "ERROR: policy not found: $POLICY" 1>&2; exit 1; }
mkdir -p "$OUTROOT"

echo "=== online noise-pool generation ==="
echo "  FS=$FS  MODE=$MODE  expert=$(basename "$(dirname "$(dirname "$(dirname "$POLICY")")")")"
echo "  collector: $(basename "$SCRIPT")"
echo "  outroot:   $OUTROOT"
echo "  levels:    $LEVELS   seeds: 0..$((NSEEDS-1))   episodes/pool: $NEPISODES"
echo

# noise % -> fraction (100 -> 1.0)
prob() { python3 -c "print($1/100)"; }

n_levels=$(echo $LEVELS | wc -w)
total=0; made=0; skipped=0
for noise in $LEVELS; do
  for seed in $(seq 0 $((NSEEDS-1))); do
    total=$((total+1))
    OUT="$OUTROOT/n${NEPISODES}_p${noise}_s${seed}"
    if [[ "$FORCE" != "1" && -e "$OUT/dataset_info.json" ]]; then
      skipped=$((skipped+1)); continue
    fi
    echo "=== noise=${noise}% seed=${seed} -> $OUT ==="
    OPENBLAS_NUM_THREADS=1 $PYTHON "$SCRIPT" \
      --policy      "$POLICY" \
      --output      "$OUT" \
      --optimal-policy "$POLICY" \
      --noise-prob  "$(prob "$noise")" \
      --noise-seed  "$seed" \
      --n-episodes  "$NEPISODES" \
      --hold-k      1 \
      --max-frames  400 \
      --n-actions   4 \
      "${MODEARG[@]}"
    made=$((made+1))
    echo "[done $made made / $skipped skipped of $total seen; target $((n_levels*NSEEDS))]"
  done
done

echo
echo "Done. $made generated, $skipped skipped (already existed) -> $OUTROOT"
