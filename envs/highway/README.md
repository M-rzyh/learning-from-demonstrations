# Option A — collect highway-env demos on your laptop, train GAIL on Vulcan

Play highway-env in a normal window on your own computer (no lag, no tunnel), save each
episode as a demo, upload the demos to Vulcan, and train GAIL there exactly like LunarLander.

Files for this env (paths from the repo root):

- `envs/highway/` — the two collectors, `diagnose_highway_keys.py` (key-handling check) and
  `convert_highway_demos.py` (collector output → the dataset format GAIL/BC read)
- `launchers/gail/run_gail_highway.sh`, `launchers/bc/run_bc_highway.sh`, `launchers/rl/train_expert_highway.sh`
- the `highway` named configs inside the library: `algorithms/imitation/src/imitation/scripts/config/{train_adversarial,train_imitation,train_rl}.py`

Two collectors (both stock highway-v0, both with the same game UI: box + panel, SPACE start, Q quit):

| File | Edition | How your keys act | Recorded action per step |
|---|---|---|---|
| `collect_highway_demos.py` | **default** (in use) | highway's own manual control: immediately, on any frame, several per step possible | the last one applied in the step |
| `collect_highway_demos_ui.py` | **ours** | held, latest press wins, applied on the first frame of the next step (like the agent) | exactly the one applied |

The default edition also saves `n_applied` (actions per step) and `applied_log`, and prints at quit how many
episodes had a step with 2+ actions — i.e. broke the agent's one-action-per-step rule.

## The one rule

Laptop and cluster must run **the same game**:

| What | Value | Why |
|---|---|---|
| gymnasium | `0.29.1` | our `imitation` fork pins `gymnasium~=0.29` (GAIL env `imitation-gail`) |
| highway-env | `1.9.1` | newest release that accepts gymnasium 0.29 (≥1.10 needs gymnasium 1.x) |
| `policy_frequency` | `1` (env default) | decisions per second; must divide 15, and **training must use the same value** |

`meta.json` (written next to the demos) records the env, config and versions actually used.

## 1. Set up the laptop (once)

The Mac has miniforge (conda) but no `python3.10` command, so make a conda env:

```bash
conda create -n highway-demos python=3.10 -y
conda activate highway-demos
pip install "gymnasium==0.29.1" "highway-env==1.9.1" pygame
```

Every later session: `conda activate highway-demos` first. Prompt should show `(highway-demos)`, not `(base)`.

## 2. Collect demos

Copy the collector from `envs/highway/` to the laptop, then:

```bash
python collect_highway_demos.py    --out ~/highway_demos/session_1      # default edition, stock highway-v0
python collect_highway_demos_ui.py --out ~/highway_demos_ui/session_1   # our edition
```

**Always pass `--out` with a folder OUTSIDE the repo.** Without it the collectors save to
`./highway_demos` (or `./highway_demos_ui`) relative to the folder you start them from, so running them
from inside a checkout would put human-subject recordings into the repo. Recordings must never be
committed; `.gitignore` also lists `highway_demos*/` as a safety net, but do not rely on it.

- **SPACE** starts an episode, **Q** quits (an unfinished episode is not saved). Ctrl+C also works.
- Arrows: Up = lane left (up the screen), Down = lane right, Right = faster, Left = slower.
- Default edition, after each episode (as in LunarLander `human_demo.py`): **F** = save & flag,
  **SPACE** = save (keep, not flagged), **X** = discard. The panel shows saved / flagged counts vs `--flag-target` (50).
- Default edition, `--feedback reward|outcome|both|none` = what you are shown (return, CRASHED/TIME UP, both,
  nothing). The files always record everything.
- Each saved episode → `epNNN.npz` (obs, acts, rews, terminal, seed, + edition-specific logs) and `meta.json`.
  Default edition also writes `flagged` in each file, `flags.json`, and `all_episodes_log.csv` (every attempt,
  including discarded).
- Re-running into the same folder continues the numbering; it never overwrites.
- **Do not use demos recorded with the old `play()`-based collector** (before 2026-11-03): highway-env 1.9.1
  read the arrow keys itself, so the car obeyed you but `play()` recorded mostly IDLE.

What the agent learns from is the 5×5 `Kinematics` table (nearby cars: present, x, y, vx, vy),
not the picture. The picture is only for you.

## 3. Upload to Vulcan

```bash
scp -r ~/highway_demos/session_1 marzii@vulcan.alliancecan.ca:/scratch/marzii/learning-from-demonstrations/imitation_runs/human_demos/highway/
```
(or OnDemand → Files → Upload at https://vulcan.alliancecan.ca)

## 4. Convert to the format GAIL reads (on Vulcan, inside a job, `imitation-gail` env)

Use `envs/highway/convert_highway_demos.py` (from the repo root):

```bash
python envs/highway/convert_highway_demos.py <uploaded folder> <dst folder on scratch> [--flagged-only]
```

It does the same as `save_demos()` in `envs/lunarlander/human_demo.py`:
`TrajectoryWithRew` + `imitation.data.serialize.save`. The core of it, for reference:

```python
import glob, sys
import numpy as np
from imitation.data import serialize
from imitation.data.types import TrajectoryWithRew

src, dst = sys.argv[1], sys.argv[2]
flagged_only = len(sys.argv) > 3 and sys.argv[3] == "--flagged-only"   # what GAIL trains on, as in LunarLander
trajs = []
for f in sorted(glob.glob(f"{src}/ep*.npz")):
    d = np.load(f)
    if flagged_only and not ("flagged" in d and bool(d["flagged"])):
        continue
    trajs.append(TrajectoryWithRew(obs=d["obs"], acts=d["acts"], rews=d["rews"],
                                   infos=None, terminal=bool(d["terminal"])))
serialize.save(dst, trajs)
print(f"{len(trajs)} episodes, {sum(len(t.acts) for t in trajs)} steps -> {dst}")
```

## 5. Train GAIL (sbatch, as for LunarLander)

From the repo root:

```bash
DEMO_PATH=<converted dir> N_DEMOS=50 SEED=0 sbatch launchers/gail/run_gail_highway.sh
DEMO_PATH=<converted dir> N_DEMOS=50 SEED=0 sbatch launchers/bc/run_bc_highway.sh
```

The launchers use the `highway` named config and `environment.gym_id=highway_env:highway-v0`
(`highway_env:` makes gymnasium import highway-env before building the env). A different env config,
e.g. `environment.env_make_kwargs='{"config": {"policy_frequency": 5}}'`, must match what the demos
were recorded with.

## Known quirks (highway-env 1.9.1)

- `offscreen_rendering` defaults to False even with `render_mode="rgb_array"`; then every `env.render()`
  reads the keyboard and applies arrow keys to the car. Both collectors set it to True (display only).
- One `step()` = 15 physics frames (`simulation_frequency` 15 // `policy_frequency` 1). The collectors
  show all 15; that's why the window is smooth at 1 decision per second.

## Cluster setup (done 2026-11-03)

- highway-env 1.9.1 is installed in a side folder, `/scratch/marzii/envs/highway-pkgs`
  (`pip install --no-deps --target`); `imitation-gail` itself is unchanged. The highway launchers put the
  side folder on `PYTHONPATH`, and the `highway` named configs use `gym_id="highway_env:highway-v0"`.
- Smoke-tested end to end (convert → BC → GAIL → PPO expert) in job 2109975; pilot expert in job 2110385
  (~21 env steps/s with 8 parallel envs, so 200k steps ≈ 2.7 h).

## Related

- Journal: `~/journal/projects/vulcan-home/` (collection setup) and `~/journal/projects/learning-from-demonstrations/`
  (cluster side: launchers, expert, BC/GAIL).
