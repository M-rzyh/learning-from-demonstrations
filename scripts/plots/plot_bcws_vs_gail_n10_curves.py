"""BC warm-start vs GAIL learning curves at N=10 demos.

Same visual format as scripts/curation/plot_2feat_learning_curves.py: one line per
arm = mean over its 10 seeds, shaded band = +/- 1 std across seeds, rolling mean,
x-axis in true env steps.

Four arms, all 1M env steps:
  GAIL,  10 expert-oracle demos (4615187 clean pool)  -- count_N10, TB scalars      (10 seeds)
  GAIL,  10 human demos (session_3_ext700_top10)      -- gail_ext700 K=10, TB       (10 seeds)
  BC->PPO warm-start, the SAME 10 human demos         -- bcws_ext700 K=10, stdout   (10 seeds)
  Pure PPO on true reward, NO demos at all            -- ppo_baseline, TB           (10 seeds)

The pure-PPO arm is the no-demo ablation of the BC warm-start arm: identical PPO
(net_arch [64,64], n_steps 1024 x 8 envs, batch 64, n_epochs 4, gamma .999,
gae_lambda .98, ent_coef .01, lr 2.5e-4) for the same 1M steps, with the BC
pretraining removed, at the same 1000-step episode cap.

It deliberately does NOT reuse the two old runs under expert/lunarlander/ (4615170,
4615187). 4615187 is the policy that generated every expert-oracle demo AND is
run_gail_lunarlander.sh's OPTIMAL_REF, and it was the better of the two survivors --
so putting it in a baseline arm would bias that arm upward with a post-hoc-selected
run. Those two also had random sacred seeds and a 400-step cap. These ten are
seeds 0..9, cap 1000, in their own tree, leaving expert/ untouched.

The BC warm-start runs write no TensorBoard log (warmstart_bc_ppo.py calls ppo.learn()
with no tensorboard_log and no callback), so their curve is parsed from the SB3
verbose=1 rollout tables in the Slurm .out, which carry an exact `total_timesteps`
alongside each `ep_rew_mean`. Nothing is inferred -- the two sources are put on the
same env-step axis and smoothed over a matched step span, not a matched point count.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_bcws_vs_gail_n10_curves.py
"""
import argparse, glob, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures_dir import fig_path

GA       = "/scratch/marzii/imitation_runs/gail/lunarlander"
PPO_ROOT = "/scratch/marzii/imitation_runs/ppo_baseline/lunarlander"
BCWS_LOG = "/scratch/marzii/imitation_runs/_slurm_logs/bc_warmstart/lunarlander"
TAG      = "mean/gen/rollout/ep_rew_mean"      # true env return, NOT ep_rew_wrapped_mean
PPO_TAG  = "rollout/ep_rew_mean"               # SB3's own tag in the train_rl runs
TOTAL_TIMESTEPS = 1_000_000
ROUNDS_FULL     = 488                          # GAIL rounds in a completed 1M-step run

# seed -> slurm job id, seeds 0..9 in order. Fixed sets, taken from the index CSVs:
#   experiments/GAIL/gail_grid_count_2026-06-18.csv        (count_N10, first 10 of 30 seeds)
#   experiments/GAIL/gail_session3_ext700_ksweep_*.csv     (gail_ext700, K=10)
#   experiments/BC/bc_warmstart_ext700_*.csv               (bcws_ext700, K=10)
ARMS = [
    ("gail_expert", "GAIL - 10 expert-oracle demos", "tb",
     [5292419, 5292420, 5292421, 5292422, 5292423,
      5292424, 5292425, 5292426, 5292427, 5292428]),
    ("gail_human", "GAIL - 10 human demos (ext700 top10)", "tb",
     [483860, 483862, 483864, 483866, 483868,
      756965, 756966, 756967, 756968, 756969]),
    ("bcws_human", "BC warm-start (BC$\\to$PPO) - same 10 human demos", "stdout",
     [483941, 483942, 483943, 483944, 483945,
      756954, 756955, 756956, 756957, 756958]),
    # No-demo ablation: seeds 0..9, experiments/ppo_baseline_2026-09-02.csv.
    ("ppo_only", "Pure PPO, no demos", "ppo", list(range(767009, 767019))),
    # Top-10 human demos ranked by RETURN ALONE (not the 9-feature composite that
    # `gail_human` uses). experiments/GAIL/gail_2feat_rankings_top10_2026-08-28.csv.
    ("gail_human_ret", "GAIL - 10 human demos (top10 by return only)", "tb",
     [698263, 698294, 698295, 698296, 698297, 698298, 698299, 698300, 698301, 698302]),
    # satad = satisficing feats + action divergence, 50/50, NO return term.
    # experiments/GAIL/gail_satad_2026-09-02.csv, K=10.
    ("gail_human_satad", "GAIL - 10 human demos (satad: satisficing + action div, no return)", "tb",
     list(range(769616, 769626))),
    # satad8 = satisficing feats + action divergence as the 8th vote, NO return term.
    # Distinct ranking from `satad` (which is a 50/50 blend); different demo dir + jobs.
    ("gail_human_satad8", "GAIL - 10 human demos (satad8: satisficing + action div as 8th vote)", "tb",
     list(range(769646, 769656))),
]
DASHED = {"gail_expert", "ppo_only"}           # reference arms, drawn dashed


def _tb(path_glob, tag, real_steps):
    """(steps, returns) from a TensorBoard event file, or None.

    real_steps=True uses the logged step values (PPO logs true env steps); False
    converts the GAIL round index into env steps, since GAIL logs round 1..488.
    """
    ev = sorted(glob.glob(path_glob))
    if not ev:
        return None
    ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    sc = ea.Scalars(tag)
    y = np.array([s.value for s in sc], float)
    x = (np.array([s.step for s in sc], float) if real_steps
         else (np.arange(len(y)) + 1) * (TOTAL_TIMESTEPS / ROUNDS_FULL))
    return x, y


def curve_tb(job):
    """One GAIL job: true env return per round, round index rescaled to env steps."""
    return _tb(f"{GA}/{job}/log/events*", TAG, real_steps=False)


def curve_ppo(job):
    """One pure-PPO baseline job: SB3 logs rollout/ep_rew_mean against real env steps."""
    return _tb(f"{PPO_ROOT}/{job}/log/events*", PPO_TAG, real_steps=True)


ROW_RE = re.compile(r"^\|\s+(ep_rew_mean|total_timesteps)\s+\|\s+(\S+)\s+\|")


def curve_stdout(job):
    """(steps, returns) for one BC warm-start job from the SB3 verbose=1 tables.

    Only the PPO phase is read -- parsing starts at the '[rl] PPO.learn' marker so the
    BC pretraining tables (neglogp / prob_true_act / samples_so_far) cannot leak in.
    """
    f = f"{BCWS_LOG}/bc-warmstart_{job}.out"
    if not os.path.exists(f):
        return None
    steps, rews, rew, started = [], [], None, False
    with open(f, errors="ignore") as fh:
        for line in fh:
            if not started:
                started = line.startswith("[rl] PPO.learn")
                continue
            m = ROW_RE.match(line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            if k == "ep_rew_mean":
                rew = float(v)
            elif k == "total_timesteps" and rew is not None:
                steps.append(float(v)); rews.append(rew); rew = None
    if not steps:
        return None
    return np.array(steps), np.array(rews)


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y, 0
    return np.convolve(y, np.ones(w) / w, mode="valid"), (w - 1) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bcws_vs_gail_n10_curves.png")
    ap.add_argument("--window", type=int, default=15,
                    help="rolling-mean window in GAIL rounds; other arms use the "
                         "window covering the same number of env steps")
    ap.add_argument("--min_points", type=int, default=20, help="skip seeds shorter than this")
    ap.add_argument("--arms", default="", help="comma list of arm keys; default = all")
    a = ap.parse_args()

    gail_step = TOTAL_TIMESTEPS / ROUNDS_FULL
    win_steps = a.window * gail_step               # smooth over a matched step span

    print(f"{'arm':>44} | seeds | pts | win | final mean +/- std")
    print("-" * 88)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    want = a.arms.split(",") if a.arms else [k for k, *_ in ARMS]
    for key, label, src, jobs in ARMS:
        if key not in want:
            continue
        get = {"tb": curve_tb, "stdout": curve_stdout, "ppo": curve_ppo}[src]
        cs = [c for c in (get(j) for j in jobs)
              if c is not None and len(c[1]) >= a.min_points]
        if not cs:
            print(f"{label:>44} | no usable runs"); continue

        n = min(len(y) for _, y in cs)              # honest truncation to the shortest seed
        x_raw = cs[0][0][:n]
        dx = float(np.median(np.diff(x_raw)))       # env steps between consecutive points
        w = max(1, int(round(win_steps / dx)))
        Y = np.vstack([smooth(y[:n], w)[0] for _, y in cs])
        off = smooth(cs[0][1][:n], w)[1]
        x = x_raw[off:off + Y.shape[1]]
        m, sd = Y.mean(0), Y.std(0)

        line, = ax.plot(x, m, lw=2.0, ls="--" if key in DASHED else "-", zorder=3,
                        label=f"{label}  ({len(cs)} seeds)")
        c = line.get_color()
        ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.15, lw=0, zorder=2)
        ax.scatter([x[-1]], [m[-1]], color=c, s=28, zorder=4, edgecolor="white", linewidth=0.6)
        print(f"{label:>44} | {len(cs):5d} | {n:3d} | {w:3d} | {m[-1]:+7.1f} +/- {sd[-1]:.1f}")

    ax.axhline(0, color="gray", ls=":", alpha=0.5, zorder=1)
    ax.set_xlabel("steps")
    ax.set_ylabel(f"Mean episode return, true env  (rolling {win_steps/1000:.0f}k steps)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92, ncol=2)
    ax.set_title("BC warm-start vs GAIL learning curves — 10 demos", fontsize=11)
    fig.tight_layout()
    out = fig_path(a.out)
    fig.savefig(out, dpi=150)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
