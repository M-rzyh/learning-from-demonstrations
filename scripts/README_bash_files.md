# IRL3 `scripts/` — what each `.sh` is for

Pipeline: 
**generate demo pools** → **submit GAIL training** → (util) rebuild/re-eval.

**Two noise pipelines** share `noise_p{n}` naming but are different data:
- **online** (`noisy_demos_online_nonFS/`) — noise injected *during* the rollout, so the lander genuinely misflies (short, crashing episodes).
- **offline** (`noisy_demos/`, via `add_action_noise`) — a clean recorded trajectory with its action *labels* flipped; the states stay the perfect flight.

## Generate demo pools — in `pool_generators/` (run once; produce the datasets everything else subsamples)
Run from the repo root, e.g. --> `sbatch scripts/pool_generators/generate_clean_demo_pools.sh`.
| Script | Use it to… |
1. | **`generate_clean_demo_pools.sh`** | **Clean (0%) pool** — expert rolled out with `--noise-prob 0` into one shared pool `n100_p0_clean` (not per-seed; the submitters subsample it via `SHUFFLE_SEED`). `FS` (default nonFS 4615187), `NEPISODES` (default 150), `FORCE`. Writes `noisy_demos/…/expert_4615187/n100_p0_clean` — where `submit_gail_count.sh` (and `submit_gail_noise.sh` at level 0) read it. Skips if present unless `FORCE=1`. |
2. | **`generate_online_noisy_demo_pools.sh`** | **Online noise pools** — `FS` (default nonFS 4615187), `MODE` (default `coin_uniform` = your N15/N50/N100 pools; also `exact_uniform`/`coin_exclude`/`exact_exclude`), `LEVELS`/`NSEEDS`/`FORCE`. Default → `noisy_demos_online_nonFS/…/expert_4615187/n100_p{P}_s{S}` (what the noise sweeps subsample); variants → `noise_variants/demos/{MODE}/`. FS supports `coin_uniform` only. Skips existing unless `FORCE=1`. |
3. | **`generate_offline_noisy_demo_pools.sh`** | **Offline noise pools** — `add_action_noise` flips action labels on a clean recorded trajectory (states unchanged). Read by `submit_gail_noise.sh PIPELINE=offline`; not interchangeable with the online pools despite the shared `noise_p{n}` names. ⚠️ header wrongly says "FS 4720242 / 41 datasets" — code is 4615187 / 361. | --> basically useless.
4. | **`generate_blind_expert_demo_pools.sh`** | **Blind-expert** frame-blanking demos — expert fed zeroed obs on blanked frames (block b=10, blank% {0,25,50,75}), expert 4615187. | --> if some day we decide to do agent POMDP/Control variations.

Clean is pipeline-agnostic (noise-prob 0 corrupts nothing), so the online-rolled clean pool pairs with either the online or offline noise sweeps.

## Submit GAIL training

Two unified submitters in **`noise_count/`** (the noise & demo-budget axes) and the frame-blanking
ones in **`blind/`**. Both unified submitters do a **PRE-FLIGHT**: they resolve and check every pool
they'd use and submit *nothing* if any is missing (`CHECK=1` = check only). All append to their index.

### `noise_count/` — noise & budget axes
| Script | Use it to… |
1. | **`submit_gail_noise.sh`** | **Noise axis at a chosen N** — default `LEVELS` = 10–100% (0% excluded, see note). Options `N`, `LEVELS`, `PIPELINE` (online/offline), `FS`, `MODE` (`coin_uniform`/`exact_uniform`/`coin_exclude`/`exact_exclude`), `NSEEDS`; `SHUFFLE`/`BS` auto-derive from `N`. Supersedes the old `grid_noise`/`_N50`/`_N`/`noise_variants`. e.g. `N=15 bash noise_count/submit_gail_noise.sh`. |
2. | **`submit_gail_count.sh`** | **Count axis** — sweeps demo budget `N_LIST` (default 1/5/10/50/100) at 0% noise from the clean pool. Options `FS`, `NSEEDS`, `BS` (tiers by N). Supersedes `grid_count` + the count half of `grid`. e.g. `FS=1 bash noise_count/submit_gail_count.sh`. |

> **0% point convention.** The 0%-noise point at budget N *is* the clean count point `count_N{N}`
> — same experiment. To avoid computing it twice, **the 0% point comes from `submit_gail_count.sh`
> (recorded as `count_N{N}`), which is what the plots read**; `submit_gail_noise.sh`'s default
> `LEVELS` omits 0. (You *can* pass a `LEVELS` containing 0 — it routes correctly to the shared
> `n100_p0_clean` pool — but it then lands in the noise CSV as `noise_N{N}_p0`, not `count_N{N}`,
> so only do that for a deliberately self-contained noise sweep.)

### `blind/` — frame-blanking axis
| Script | Use it to… |
1. | `submit_gail_blind_expert.sh` | Train on **blind-expert** demos, **N=50**, blank% {0,25,50,75} × seeds. ⚠️ index **truncates**. |
2. | `submit_gail_blind_human_b5.sh` | Train on **blind-human** demos (b=5), **N=50** — `bash blind/submit_gail_blind_human_b5.sh <blank_pct> <seed_lo> <seed_hi>`. Appends. |

### `lander_vanish/` — lander-vanishing axis
Same block schedule as frame blanking, but only the **lander** is removed and replaced by the
terrain behind it; ground, pad and flags stay visible. Human arm only, low budget (**N=15**
demos against PT's N=100 preferences).

| Script | Use it to… |
1. | `submit_gail_vanish_human.sh` | Train on vanish-human demos (b=5, fps 20), **N=15** — `bash lander_vanish/submit_gail_vanish_human.sh <pct> <seed_lo> <seed_hi>`. Pre-flights the demo dir and refuses to submit anything if it is missing or short; `CHECK=1` checks only. Appends. |

Collect the demos with `imitation/collect_human_demos_lunarlander.sh`:
`DIFFICULTY=vanish PCT={25,50,75} BLOCK_LEN=5 FPS=20 FLAG_TARGET=15`. The **0% baseline needs
its own fps-20 collection** (`DIFFICULTY=none` + `OUTPUT_DIR=.../vanish_p0`) — the older clean
sessions were played at fps 25, and reusing them would confound the baseline with game speed.

## Utilities
| Script | Use it to… |
1. | `rebuild_gail_noise_N15_index.sh` | Rebuild a truncated N15 noise index from job logs (`DEMO_NOTE`) + the clean-rows side-file. Idempotent; re-run after the sweep finishes. |
2. | `re_eval_batch.sh` | Re-evaluate finished run dirs (`re_eval_run.py`) at a chosen episode count. |

## Gotchas (verified against the code)
- **Clean is one pool** (`n100_p0_clean`), *not* per-seed `n100_p0_s{seed}` — never point a 0% run at the per-seed pattern. Make it with `generate_clean_demo_pools.sh`, submit it with `submit_gail_count.sh` (as `count_N{N}`).
- **Match the expert**: 4615187 for every noise/clean plot; only `submit_gail_count.sh FS=1` uses 4720242. Don't mix on one plot.
- **online vs offline noise differ** despite shared `noise_p{n}` names: the noise sweeps pull `noisy_demos_online_nonFS/`; `submit_gail_noise.sh PIPELINE=offline` pulls `noisy_demos/`.
- **Append-safe submitters**: `submit_gail_noise.sh`, `submit_gail_count.sh`, `submit_gail_blind_human_b5.sh` (all append with a header-guard). Only `submit_gail_blind_expert.sh` still `> $INDEX` (truncates) — a same-day re-run wipes prior rows.
- **Stale header, trust the code**: `generate_offline_noisy_demo_pools.sh` (says "FS 4720242 / 41 datasets"; code is 4615187 / 361).
