"""Final-policy eval return per arm at N=10 demos — the companion to the learning-curve figure.

Same five arms, same seeds, but the FINAL POLICY evaluated for 50 deterministic episodes
instead of the training-rollout curve. The two differ systematically (training rollouts are
under exploration), so quoting one against the other is a category error -- hence this figure.

Every arm's number is produced the same way: policy.predict(deterministic=True), 50 episodes,
at that arm's own training episode cap. The PPO baseline had no such eval (its rollouts/ dir
holds stochastic training rollouts), so it was generated separately -- see
experiments/ppo_baseline_eval_2026-09-02.json.

Dot = one seed's 50-episode mean. Large dot = mean over seeds. Bar = +/- 1 sd across seeds.

    python scripts/plots/plot_n10_final_eval.py
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures_dir import fig_path

GA   = "/scratch/marzii/imitation_runs/gail/lunarlander"
BCWS = "/scratch/marzii/imitation_runs/bc_warmstart/lunarlander"
PPO_EVAL = "/home/marzii/IRL3/experiments/ppo_baseline_eval_2026-09-02.json"

def gail(j):     # eval_data/agent_alignment.json carries the 50 episode returns
    p = f"{GA}/{j}/eval_data/agent_alignment.json"
    return float(np.mean(json.load(open(p))["ep_rewards"])) if os.path.exists(p) else None

def bcws(j):
    p = f"{BCWS}/{j}/eval_data/meta.json"
    return float(np.mean(json.load(open(p))["ep_rewards"])) if os.path.exists(p) else None

def ppo(_):
    return None  # loaded in bulk below

# colours match the learning-curve figure so the two can be read side by side
ARMS = [
    ("BC warm-start (BC$\\to$PPO)\n10 human demos",            "C1", bcws,
     [483941,483942,483943,483944,483945,756954,756955,756956,756957,756958]),
    ("Pure PPO\nno demos",                                     "C2", ppo, []),
    ("GAIL\n10 expert-oracle demos",                           "C0", gail, list(range(5292419,5292429))),
    ("GAIL — satad8\n10 human demos",                          "C4", gail, list(range(769646,769656))),
    ("GAIL — top10 by return only\n10 human demos",            "C3", gail,
     [698263,698294,698295,698296,698297,698298,698299,698300,698301,698302]),
]

def main():
    rows = []
    for name, c, fn, jobs in ARMS:
        if fn is ppo:
            vals = [float(np.mean(v)) for v in json.load(open(PPO_EVAL)).values()]
        else:
            vals = [v for v in (fn(j) for j in jobs) if v is not None]
        rows.append((name, c, np.array(vals)))

    rows.sort(key=lambda r: r[2].mean())          # best at top
    fig, ax = plt.subplots(figsize=(11, 5.6))
    rng = np.random.default_rng(0)

    print(f"{'arm':>46} {'seeds':>5} {'mean':>8} {'sd':>7} {'min':>8} {'max':>8}")
    for y, (name, c, v) in enumerate(rows):
        m, sd = v.mean(), v.std(ddof=1)
        ax.hlines(y, m - sd, m + sd, color=c, lw=3, alpha=0.45, zorder=3)
        ax.scatter(v, y + rng.uniform(-.13, .13, len(v)), s=26, color=c,
                   alpha=0.5, zorder=2, linewidths=0)
        ax.scatter([m], [y], s=130, color=c, zorder=4, edgecolor="white", linewidth=1.4)
        ax.annotate(f"{m:+.1f} ± {sd:.1f}", (m, y), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=9.5, fontweight="bold", color="#333")
        print(f"{name.replace(chr(10),' '):>46} {len(v):5d} {m:8.1f} {sd:7.1f} {v.min():8.1f} {v.max():8.1f}")

    ax.axvline(0, color="gray", ls=":", alpha=0.5, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlabel("Final-policy return, true env  (50 deterministic episodes)")
    ax.set_title("Final policy after 1M steps — 10 demos\n"
                 "dot = one seed's 50-episode mean · bar = ±1 sd across 10 seeds", fontsize=11)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    out = fig_path("n10_final_eval.png")
    fig.savefig(out, dpi=150)
    print("\nsaved", out)

if __name__ == "__main__":
    main()
