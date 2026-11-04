# learning-from-demonstrations

Research code for **learning from demonstrations**: imitation learning and inverse RL from
expert and human demonstrations. The current focus is LunarLander, comparing human demonstrations
with synthetic (scripted-expert) ones as demonstration count, noise and difficulty vary.

The repository is organised **by function, with the environment as a parameter**: one folder per
job type (launchers, demo pools, studies, curation, plots), shared by all environments, and a small
`envs/<env>/` folder for the pieces that only make sense for one environment. The same study can then
be run on a new environment without copying it.

## Layout

```
learning-from-demonstrations/
├── README.md  WORKLOG.md  .gitignore
├── algorithms/
│   └── imitation/             # vendored HumanCompatibleAI `imitation` library (upstream files only)
├── launchers/                 # Slurm jobs that each run ONE training run, one script per env
│   ├── entrypoints/           #   train_{adversarial,rl,imitation}_launcher.py (register custom envs, start Sacred)
│   ├── gail/                  #   run_gail_<env>.sh (GAIL, or AIRL with ALGO=airl)
│   ├── bc/                    #   run_bc_<env>.sh, BC warm-start -> PPO
│   └── rl/                    #   PPO experts and true-reward PPO baselines
├── envs/                      # env-specific pieces only
│   ├── lunarlander/           #   human-demo collector, lander-vanish masking and its tests
│   └── walker/legacy/         #   old MuJoCo/DMC Walker scripts (not maintained)
├── demo_pools/                # build expert / noisy / blind demonstration pools
├── studies/                   # submitters that loop over budgets/seeds and sbatch launchers (README inside)
├── curation/                  # demonstration features, rankings, curated subsets and their plots
├── plots/                     # results and demonstration figures
├── utils/                     # re-evaluation, index rebuilds, result scraping
└── experiments/               # run indices and results tables, one folder per algorithm (BC/ GAIL/ AIRL/)
```

**LunarLander-only so far:** everything in `demo_pools/`, `studies/`, `curation/`, `plots/` and `utils/`
was written for LunarLander (it points at LunarLander pools and run dirs on scratch). When a study or
plot gains a second env, parameterise it by env rather than copying it.

Launchers call the entry points by absolute path and `cd` into their run dir on `$SCRATCH` before
training, so nothing depends on the folder you submit from and no output ever lands in `$HOME`.
Submit from the repo root, e.g. `sbatch launchers/gail/run_gail_lunarlander.sh` or
`bash studies/noise_count/submit_gail_noise.sh`.

### `algorithms/imitation/`

- Upstream: [HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation), which provides
  implementations of **BC, GAIL, AIRL and DAgger** (`src/imitation/algorithms/`)
- Local patches kept inside the package: the frame-skip env `src/imitation/envs/action_repeat.py`
  (`LunarLander-v2-FS10`), one named config per env (currently `lunar_lander`) in
  `src/imitation/scripts/config/{train_adversarial,train_imitation,train_rl}.py`, and the compute timer
  in `src/imitation/scripts/train_adversarial.py`. Launchers put `algorithms/imitation/src` on `PYTHONPATH`.

### Adding an environment

1. `envs/<env>/`: anything that only makes sense for this env (human-demo collector, converter,
   wrappers, a README on how to collect demos and train).
2. A `<env>` named config in `algorithms/imitation/src/imitation/scripts/config/{train_adversarial,train_imitation,train_rl}.py`
   (plus a registration under `src/imitation/envs/` if it needs a custom wrapper).
3. `launchers/{gail,bc,rl}/run_*_<env>.sh`: copy the closest existing launcher and set the env name,
   named config and scratch run dir (`imitation_runs/<algo>/<env>/<job>`). The entry points are reused as is.
4. Reuse `demo_pools/`, `studies/`, `curation/` and `plots/`, adding the env as a parameter where needed.

### Adding an algorithm

1. Put the implementation under `algorithms/<name>/`. Use a new folder for a standalone codebase, or
   extend `algorithms/imitation/` if it builds on that library.
2. Add its launchers under `launchers/<name>/` and its run indices and results tables in `experiments/<NAME>/`.
3. Reuse `demo_pools/`, `curation/` and `plots/`, so every method trains and is evaluated on the same
   demonstrations.

## Data and outputs

Nothing generated is tracked here. Demonstration pools, trained policies, Sacred/Slurm logs and
figures live on cluster storage, and `.gitignore` keeps them out of the repository.
**Human demonstration recordings are human-subject data and must never be committed.**

### Local data links

Two gitignored symlinks point at `$SCRATCH`, and each fresh clone has to recreate them:

```bash
mkdir -p /scratch/$USER/learning-from-demonstrations/{figures,output}
ln -s /scratch/$USER/learning-from-demonstrations/figures figures
ln -s /scratch/$USER/learning-from-demonstrations/output  algorithms/imitation/output
```

`output/` is the `imitation` library's default output folder (`cwd/output/...`). The current launchers do not
use it: they pass an explicit `logging.log_dir` and run from their scratch run dir. The link is kept because the
scratch folder holds the earlier IRL3 runs (Sacred runs, `train_rl`, Slurm logs), copied from
`~/IRL3/imitation/output`, and older records point there.

Some curation scripts read `figures/ext700/multimodality_arrays.npz` and
`figures/ext700/curation_bc_ceiling.json` back as inputs.

Run data lives in `/scratch/$USER/learning-from-demonstrations/imitation_runs/`. That covers demonstration
pools (`demos/`), expert policies (`expert/`), and per-algorithm runs (`gail/`, `bc/`, `bc_warmstart/`, `AIRL/`,
`ppo_*`). Slurm logs go to its `_slurm_logs/`.

## Provenance

The repository was created on 2026-10-05 by copying the tracked files of `M-rzyh/IRL3` at commit
`48b20ef` (tag `pre-generalization-snapshot`). Git history was not carried over. In the copy,
`imitation/` became `algorithms/imitation/`, and the generated `figures/` were left out.

## License

`algorithms/imitation/` keeps its upstream MIT `LICENSE`. Please cite the `imitation` library and the
original algorithm papers when you use this code.
