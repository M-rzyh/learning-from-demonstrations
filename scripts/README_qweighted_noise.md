# Q-weighted noise for GAIL — implementation plan (deferred)

Status: **NOT yet implemented at the right temperature.** Plan saved here for later.

## Background

The May-11 meeting flagged "weighted random action based on Q-values" as an
alternative to uniform random noise. Idea: simulate a sloppy-but-trained
teacher who picks suboptimal actions sometimes, but never the absolute worst
unless they have to.

## What "Q-weighted" means concretely

When noise fires (with probability `p`), instead of replacing the argmax action
with a uniform random action, replace it with a sample from a Q-weighted
distribution:

```
weights = softmax(logits / τ)
action ~ Categorical(weights)
```

PPO does not output explicit Q values — but for discrete actions its policy
network outputs `logits` (one per action). Relative magnitudes of those logits
encode the same preference structure as Q values (up to a temperature). So we
use `logits / τ` and sample.

### Example (matches user's intended behavior)

Q values for 4 actions: A1=10, A2=5, A3=5, A4=0.

| τ | A1 prob | A2 prob | A3 prob | A4 prob |
|---|---|---|---|---|
| 1 | 99% | <1% | <1% | <1% (essentially argmax) |
| 2 | 79% | 7% | 7% | 1% |
| 5 | 51% | 19% | 19% | 11% |
| 8 | 42% | 21% | 21% | 16% |

Higher τ → flatter distribution → more "noise" with worse actions getting more weight.

## Empirical calibration (already done)

Sampled 500 states from the FS expert (`4720242`) on `LunarLander-v2`:
- Logit value range: [-24.3, 0]
- Per-state spread (max − min): mean=11.85, median=10.74, p95=18.83

Resulting average ranked-probability per τ (top action, 2nd, 3rd, 4th):

| τ | Top | 2nd | 3rd | 4th |
|---|---|---|---|---|
| 1 | 70% | 25% | 5% | 0% |
| 2 | 59% | 31% | 11% | 0% |
| 3 | 52% | 32% | 15% | 2% |
| **5** | **44%** | **32%** | **19%** | **5%** |
| 8 | 38% | 31% | 22% | 9% |
| 12 | 34% | 30% | 24% | 13% |

**Recommendation: τ=5**. Clear preference for the top action (~44%) but the
worst action still gets ~5% probability. Best match to the "sloppy teacher"
intuition.

## What's already on disk

There are 120 GAIL runs at `condition_id=noise_online_qweighted_pXX` in the
index CSV (`/home/marzii/IRL3/experiments/gail_grid_2026-05-12.csv`). These
were collected with τ=1 (softmax of trained logits, essentially argmax) — too
concentrated to be meaningful "noise". They are still useful as a baseline
"stochastic optimal teacher" data point, but they are **not** what we want for
the Q-weighted noise experiment proper.

The collector for those runs is at:
`/home/marzii/IRL3/scripts/collect_fs_expert_perframe_demos_online_qweighted.py`

## Implementation steps (when ready to run)

### 1. Modify the existing collector

Edit `collect_fs_expert_perframe_demos_online_qweighted.py`:

- Add `--temperature` argument (default 5.0)
- Replace this block:
  ```python
  probs = policy_action_probs(model, obs)
  used_action = int(rng.choice(n_actions, p=probs))
  ```
  with:
  ```python
  obs_t = obs_as_tensor(obs.reshape(1, -1), model.policy.device)
  with torch.no_grad():
      logits = model.policy.get_distribution(obs_t).distribution.logits.cpu().numpy()[0]
  scaled = (logits - logits.max()) / temperature  # numerical stability
  probs = np.exp(scaled); probs /= probs.sum()
  used_action = int(rng.choice(n_actions, p=probs))
  ```
- Save `noise_type` as `"online_qweighted_t{temperature}"` in `demo_alignment.json`

### 2. Generate 120 new datasets

Same noise levels (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100) and same
10 seeds. Save under:
`/scratch/marzii/imitation_runs/noisy_demos_online_qweighted_t5/lunarlander/expert_4720242/`

(Choose a different directory than the τ=1 datasets so both are preserved.)

Quick smoke test first: generate 5 episodes at p=50% and check that
`alignment_fraction` lands between 0.4 and 0.6 — if too high (>0.7), τ is too
low; if too low (<0.3), τ is too high.

### 3. Submit 120 GAIL jobs

Mirror the existing `submit_gail_grid.sh` pattern. Use condition IDs
`noise_online_qweighted_t5_pXX`. Default GAIL config: N=100, FRAME_SKIP=0,
DEMO_BATCH_SIZE=1024, 10 seeds.

### 4. Re-scrape and re-plot

Run `scrape_gail_grid_results.py` on the index CSV, then add a fourth series
to the noise comparison plots:
- Post-hoc uniform (existing)
- Online uniform (existing)
- Online Q-weighted τ=1 (existing — labeled "near-argmax stochastic")
- **Online Q-weighted τ=5 (NEW — the proper Q-weighted variant)**

## Optional follow-ups

- Try **τ=8** for a more spread-out noisy teacher (50% noise → ~31% top action)
- Try **post-hoc Q-weighted** version (action-flip on saved demos using softmax
  sampling at the SAVED state). Gives an apples-to-apples comparison with
  post-hoc uniform.
- Auto-calibrate τ per state so that the entropy of the sampling distribution
  is some target fraction of max entropy (rather than a fixed τ across states).

## Compute estimate

- Dataset generation: ~30-40 min for 120 datasets (CPU-only, runs locally)
- 120 GAIL jobs × ~22 min wall-clock each, depending on cluster availability
- Post-eval re-scoring: trivial (~5 min)

## Why this was deferred

The user wanted to focus elsewhere; the current 120 τ=1 runs are an interesting
secondary data point on their own. The proper τ=5 experiment can be added later
with the steps above — it's all incremental on top of existing infrastructure.
