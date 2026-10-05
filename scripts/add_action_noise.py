#!/usr/bin/env python3
## the offline noise injection: Exp 7
"""Make a noisy copy of a per-frame demo dataset by re-sampling actions.

With probability `--noise-prob`, each action in each trajectory is replaced
with a uniformly random action from the discrete action space.

Also writes demo_alignment.json — fraction of (s, a) pairs in the resulting
noisy dataset where a == optimal_policy.predict(s).

Usage:
    python add_action_noise.py \\
        --input  /scratch/marzii/imitation_runs/expert/lunarlander/4720242/rollouts/perframe_demos_150 \\
        --output /scratch/marzii/imitation_runs/demos/noisy_demos/lunarlander/expert_4720242/n100_p25_s0 \\
        --noise-prob 0.25 \\
        --seed 0 \\
        --optimal-policy /scratch/marzii/imitation_runs/expert/lunarlander/4720242/policies/final/model.zip \\
        --n-actions 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from stable_baselines3 import PPO


def add_noise_to_dataset(src_path, dst_path, noise_prob, seed, n_actions=4):
    ds = load_from_disk(src_path)
    rng = np.random.default_rng(seed)
    new_rows = []
    for ep in ds:
        acts = np.asarray(ep["acts"], dtype=np.int64).copy()
        if noise_prob > 0:
            mask = rng.random(len(acts)) < noise_prob
            replacements = rng.integers(0, n_actions, size=len(acts))
            acts = np.where(mask, replacements, acts)
        new_rows.append({
            "obs": ep["obs"],
            "acts": acts.tolist(),
            "rews": ep["rews"],
            "terminal": ep["terminal"],
            "infos": ep["infos"],
        })

    features = Features({
        "obs": Sequence(Sequence(Value("float32"))),
        "acts": Sequence(Value("int64")),
        "infos": Sequence(Value("string")),
        "terminal": Value("bool"),
        "rews": Sequence(Value("float64")),
    })
    out = Dataset.from_list(new_rows, features=features)
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(dst_path))
    return out


def compute_demo_alignment(ds_path, optimal_policy_path):
    ds = load_from_disk(ds_path)
    model = PPO.load(optimal_policy_path)
    matches = 0
    total = 0
    for ep in ds:
        obs = np.asarray(ep["obs"], dtype=np.float32)
        acts = np.asarray(ep["acts"], dtype=np.int64)
        # obs has T+1 entries; acts has T entries. Predict on obs[:T].
        states = obs[:len(acts)]
        # SB3 supports batched predict
        pred_acts, _ = model.predict(states, deterministic=True)
        pred_acts = np.asarray(pred_acts, dtype=np.int64)
        matches += int((pred_acts == acts).sum())
        total += len(acts)
    return matches, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--noise-prob", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimal-policy", required=True)
    p.add_argument("--n-actions", type=int, default=4)
    args = p.parse_args()

    print(f"Creating noisy dataset: noise_prob={args.noise_prob}, seed={args.seed}")
    add_noise_to_dataset(args.input, args.output, args.noise_prob, args.seed, args.n_actions)

    print(f"Computing demo alignment vs {args.optimal_policy}")
    matches, total = compute_demo_alignment(args.output, args.optimal_policy)
    alignment = matches / max(1, total)
    meta = {
        "alignment_fraction": alignment,
        "matches": matches,
        "total_transitions": total,
        "optimal_ref": args.optimal_policy,
        "input_dataset": args.input,
        "noise_prob": args.noise_prob,
        "noise_seed": args.seed,
        "noise_type": "resample",
        "n_actions": args.n_actions,
    }
    out_json = Path(args.output) / "demo_alignment.json"
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  alignment = {alignment:.4f} ({matches}/{total})")
    print(f"  saved: {out_json}")


if __name__ == "__main__":
    main()
