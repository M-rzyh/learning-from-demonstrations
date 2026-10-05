"""Learning curves for GAIL trained on the top-10 demos of each 2-feature ranking.

One line per ranking = mean over its 10 seeds, shaded band = +/- 1 std across seeds.

Runs are discovered from the Slurm logs by their demo path, so a ranking that was
submitted later (or is still running) is picked up automatically -- no index CSV to
maintain. Partial runs are drawn as far as they have got.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_2feat_ranking_curves.py
"""
import argparse, glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

GA   = "/scratch/marzii/imitation_runs/gail/lunarlander"
LOGS = "/scratch/marzii/imitation_runs/_slurm_logs/gail/lunarlander"
TAG  = "mean/gen/rollout/ep_rew_mean"          # true env return, NOT ep_rew_wrapped_mean
TOTAL_TIMESTEPS = 1_000_000
ROUNDS_FULL     = 488                          # rounds a completed 1M-step run does
# Two naming conventions produce the same rankings:
#   session_3_ext700_ret_<tag>_top<K>   the 2-feature rankings (tag may carry a kNN k, e.g. adk50)
#   session_3_ext700_ra_top<K>          the Aug-13 sweep; "ra" == return + action_divergence at
#                                       k=15, verified byte-identical to ret_ad_top<K>
# optional "ret_" prefix; tag aliases fold equivalent namings together ("ra" == ret_ad at k=15)
DEMO_RE = re.compile(r"session_3_ext700_(?:ret_)?([a-z0-9]+)_top(\d+)\s*$")
TAG_ALIAS = {"ra": "ad"}

ORDER = ["only", "sad", "sae", "ae", "cdad", "cdae", "adxy", "ad", "satad", "satpad", "satad8"]
LABEL = {"only": "return only (baseline)", "ad": "+ action divergence",
         "adxy": "+ action divergence (x,y)", "ae": "+ action entropy",
         "sad": "+ self action divergence", "sae": "+ self action entropy",
         "cdad": "+ cross-demo divergence", "cdae": "+ cross-demo entropy",
         "adk50": "+ action divergence (k=50)", "adk100": "+ action divergence (k=100)",
         "satpad": "satisficing feats, padded (no return)",
         "adcdae": "+ ad + cross-demo entropy (3 votes)",
         "satad": "satisficing + ad, 50/50 (no return)",
         "satad8": "satisficing + ad as 8th vote (no return)",
         "adxyk50": "+ action divergence x,y (k=50)",
         "adxyk100": "+ action divergence x,y (k=100)"}


def discover(topk=None):
    """slurm .out -> {tag: {seed: job_id}} using the demo path each job printed.

    topk=None keeps every demo count and keys by (tag, K); topk=<int> keeps only that demo
    count and keys by tag alone (the shape the single-K plots expect).

    "ra_top<K>" is folded into tag "ad" -- it is the same subset. Where both namings ran the
    same seed, the HIGHEST job id wins, so the newer batch is used deterministically rather
    than depending on glob order.
    """
    runs = {}
    for f in glob.glob(f"{LOGS}/*.out"):
        tag = seed = K = None
        with open(f, errors="ignore") as fh:
            for line in fh:
                if line.startswith("Demo path:"):
                    m = DEMO_RE.search(line.strip())
                    if not m:
                        break
                    tag = TAG_ALIAS.get(m.group(1), m.group(1))
                    K = int(m.group(2))
                elif line.startswith("Seed:"):
                    seed = int(line.split()[1]); break
        if tag is None or seed is None:
            continue
        if topk is not None and K != topk:
            continue
        job = os.path.basename(f).rsplit("_", 1)[1][:-4]
        key = tag if topk is not None else (tag, K)
        prev = runs.setdefault(key, {}).get(seed)
        if prev is None or int(job) > int(prev):       # newest submission wins, deterministically
            runs[key][seed] = job
    return runs


def curve(job):
    """(rounds,) array of ep_rew_mean for one job, or None."""
    ev = sorted(glob.glob(f"{GA}/{job}/log/events*"))
    if not ev:
        return None
    ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); ea.Reload()
    if TAG not in ea.Tags()["scalars"]:
        return None
    return np.array([s.value for s in ea.Scalars(TAG)], float)


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/marzii/IRL3/figures/ext700/gail_2feat_ranking_curves.png")
    ap.add_argument("--window", type=int, default=15, help="rolling-mean window in rounds")
    ap.add_argument("--min_rounds", type=int, default=20, help="skip seeds shorter than this")
    ap.add_argument("--tags", default="", help="comma list; default = all found")
    ap.add_argument("--topk", type=int, default=10, help="demo count to plot")
    a = ap.parse_args()

    runs = discover(topk=a.topk)
    want = a.tags.split(",") if a.tags else [t for t in ORDER if t in runs]
    print(f"{'ranking':>26} | seeds | rounds | final mean +/- std")
    print("-" * 70)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    summary = []
    for t in want:
        ys = []
        for seed in sorted(runs.get(t, {})):
            y = curve(runs[t][seed])
            if y is not None and len(y) >= a.min_rounds:
                ys.append(y)
        if not ys:
            print(f"{LABEL.get(t, t):>26} | no usable runs"); continue

        n = min(len(y) for y in ys)                    # honest truncation to the shortest seed
        Y = np.vstack([smooth(y[:n], a.window) for y in ys])
        x = np.arange(Y.shape[1]) * (TOTAL_TIMESTEPS / ROUNDS_FULL)
        m, sd = Y.mean(0), Y.std(0)
        partial = n < ROUNDS_FULL
        line, = ax.plot(x, m, lw=2.0, ls="--" if t == "only" else "-", zorder=3,
                        label=f"{LABEL.get(t, t)}  ({len(ys)} seeds{', partial' if partial else ''})")
        c = line.get_color()
        ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.15, lw=0, zorder=2)
        ax.scatter([x[-1]], [m[-1]], color=c, s=28, zorder=4, edgecolor="white", linewidth=0.6)
        summary.append((t, m[-1]))
        print(f"{LABEL.get(t, t):>26} | {len(ys):5d} | {n:6d} | {m[-1]:+7.1f} +/- {sd[-1]:.1f}"
              + ("   (still running)" if partial else ""))

    ax.axhline(0, color="gray", ls=":", alpha=0.5, zorder=1)
    ax.set_xlabel("steps")
    ax.set_ylabel(f"Mean episode return, true env  (rolling {a.window} rounds)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92, ncol=2)
    ax.set_title("GAIL learning curves — top-10 demos", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
