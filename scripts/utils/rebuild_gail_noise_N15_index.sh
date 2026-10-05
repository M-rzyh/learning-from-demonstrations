#!/bin/bash
# Rebuild the GAIL N=15 NOISE index (levels 10-100) from what ACTUALLY completed.
#
# Maps each job to (noise, seed) via its log DEMO_NOTE ("cond=noise_N15_p{P} ... seed={S}",
# filename = job id), keeps only jobs whose eval_data/agent_rollouts.npz exists, and writes
# ONE row per (level, seed) — the highest job id wins, so a re-run supersedes the original.
# Cancelled / failed / duplicate jobs are dropped.
#
# Clean (0%) is NOT handled here — it lives in count_N15 (see submit_gail_count.sh). This
# rebuild covers noise levels 10-100 only. Idempotent: re-run any time; safe once the sweep
# is done (expect 300 rows = 10 levels x 30 seeds).
#
#   bash scripts/utils/rebuild_gail_noise_N15_index.sh
set -euo pipefail

DATE=$(date +%Y-%m-%d)
OUT=/home/marzii/IRL3/experiments/GAIL/gail_grid_noise_N15_${DATE}.csv
LOGDIR=/scratch/marzii/imitation_runs/_slurm_logs/gail/lunarlander
GAIL=/scratch/marzii/imitation_runs/gail/lunarlander
NONFS=/scratch/marzii/imitation_runs/demos/noisy_demos_online_nonFS/lunarlander/expert_4615187

python3 - "$LOGDIR" "$GAIL" "$NONFS" "$OUT" <<'EOF'
import glob, os, re, sys
LOGDIR, GAIL, NONFS, OUT = sys.argv[1:5]

best = {}   # (level, seed) -> jobid (highest with eval)
for f in glob.glob(os.path.join(LOGDIR, "gail-lunarlander_*.out")):
    m = re.search(r"cond=noise_N15_p\d+ N=15 noise=(\d+)% seed=(\d+)", open(f, errors="ignore").read())
    if not m:
        continue
    lvl, seed = int(m.group(1)), int(m.group(2))
    if lvl == 0:                       # clean lives in count_N15, not here
        continue
    jid = os.path.basename(f).split("_")[-1].replace(".out", "")
    if not os.path.exists(f"{GAIL}/{jid}/eval_data/agent_rollouts.npz"):
        continue                       # only completed jobs
    prev = best.get((lvl, seed))
    if prev is None or int(jid) > int(prev):   # re-run (higher job id) wins
        best[(lvl, seed)] = jid

rows = sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1]))
with open(OUT, "w") as g:
    g.write("condition_id,N,noise_pct,seed,slurm_job_id,demo_path,status\n")
    for (lvl, seed), jid in rows:
        demo = f"{NONFS}/n100_p{lvl}_s{seed}"
        g.write(f"noise_N15_p{lvl},15,{lvl},{seed},{jid},{demo},done\n")

from collections import Counter
c = Counter(lvl for (lvl, _) in best)
print(f"Wrote {OUT}")
print(f"  completed (level,seed) points: {len(best)}/300")
print("  per level:", {k: c[k] for k in sorted(c)})
missing = [(l, s) for l in range(10, 101, 10) for s in range(30) if (l, s) not in best]
if missing:
    print(f"  still missing {len(missing)}: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
EOF
