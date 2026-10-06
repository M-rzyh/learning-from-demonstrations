# learning-from-demonstrations

Research code for **learning from demonstrations**: imitation learning and inverse RL from
expert and human demonstrations. The current focus is LunarLander, comparing human demonstrations
with synthetic (scripted-expert) ones as demonstration count, noise and difficulty vary.

The repository is organised so that algorithms, shared tooling and results stay separate, and
new methods can be added without restructuring.

## Layout

```
learning-from-demonstrations/
├── README.md
├── .gitignore
├── WORKLOG.md                 # running log of changes and findings
├── algorithms/
│   └── imitation/             # vendored HumanCompatibleAI `imitation` library + local launchers
├── scripts/                   # shared tooling, independent of any one algorithm
│   ├── pool_generators/       #   build expert / noisy / blind demonstration pools
│   ├── curation/              #   demonstration selection and ranking
│   ├── noise_count/  blind/  action_delay/  lander_vanish/  sim_speed/   # per-study submitters
│   ├── plots/                 #   analysis and figures
│   └── utils/
└── experiments/               # run indices and results tables, one folder per algorithm
    ├── BC/
    ├── GAIL/
    └── AIRL/
```

### `algorithms/imitation/`

- Upstream: [HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation), which provides
  implementations of **BC, GAIL, AIRL and DAgger** (`src/imitation/algorithms/`)
- Local additions: human demonstration collection (`human_demo.py`, `collect_human_demos_lunarlander.sh`),
  difficulty manipulations (`lander_vanish.py`, frame blanking, action delay/sticky actions), and the
  GAIL and BC launchers (`run_gail_*.sh`, `bc/`)

### Adding an algorithm

1. Put the implementation under `algorithms/<name>/`. Use a new folder for a standalone codebase, or
   extend `algorithms/imitation/` if it builds on that library.
2. Put its run indices and results tables in `experiments/<NAME>/`.
3. Reuse `scripts/` for demonstration pools, curation and plotting, so every method trains and is
   evaluated on the same demonstrations.

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
