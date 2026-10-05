"""This ingredient provides (expert) demonstrations to learn from.

The demonstrations are either loaded from disk, from the HuggingFace Dataset Hub, or
sampled from the expert policy provided by the expert ingredient.
"""

import logging
import os
from typing import Any, Dict, Optional, Sequence

import datasets
import huggingface_sb3 as hfsb3
import numpy as np
import sacred

from imitation.data import huggingface_utils, rollout, serialize, types
from imitation.scripts.ingredients import environment, expert
from imitation.scripts.ingredients import logging as logging_ingredient

demonstrations_ingredient = sacred.Ingredient(
    "demonstrations",
    ingredients=[
        expert.expert_ingredient,
        logging_ingredient.logging_ingredient,
        environment.environment_ingredient,
    ],
)
logger = logging.getLogger(__name__)


@demonstrations_ingredient.config
def config():
    # Either "local" or "huggingface" or "generated".
    source = "generated"

    # local path or huggingface repo id to load rollouts from.
    path = None

    # passed to `datasets.load_dataset` if `source` is "huggingface"
    loader_kwargs: Dict[str, Any] = dict(
        split="train",
    )

    # Used to deduce HuggingFace repo id if `path` is None
    organization = "HumanCompatibleAI"

    # Used to deduce HuggingFace repo id if `path` is None
    algo_name = "ppo"

    # Num demos used or sampled. None loads every demo possible.
    n_expert_demos = None
    locals()  # quieten flake8


@demonstrations_ingredient.named_config
def fast():
    # Note: we can't pick `n_expert_demos=1` here because for envs with short episodes
    #   that does not generate the minimum number of transitions required for one batch.
    n_expert_demos = 10  # noqa: F841


@demonstrations_ingredient.capture
def get_expert_trajectories(
    source: str,
    path: str,
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations.

    Args:
        source: Can be either `local` to load rollouts from the disk,
            `huggingface` to load from the HuggingFace hub or
            `generated` to generate the expert trajectories.
        path: A path containing a pickled sequence of `sources.Trajectory`.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: if `source` is not in ["local", "huggingface", "generated"].
    """
    if source == "local":
        if path is None:
            raise ValueError(
                "When source is 'local', path must be set.",
            )
        return _constrain_number_of_demos(serialize.load(path))

    if source == "huggingface":
        return _constrain_number_of_demos(_download_expert_rollouts())

    if source == "generated":
        if path is not None:
            logger.warning("Ignoring path when source is 'generated'")
        return _generate_expert_trajs()

    if source == "human":
        if path is None:
            raise ValueError(
                "When source is 'human', path must be set to the demo .pkl file.",
            )
        return _constrain_number_of_demos(_load_human_demos(path))

    raise ValueError(
        "`source` can either be `local`, `huggingface`, `generated`, or `human`.",
    )


@demonstrations_ingredient.capture
def _constrain_number_of_demos(
    demos: Sequence[types.Trajectory],
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Constrains the number of demonstrations to n_expert_demos if it is not None."""
    if n_expert_demos is None:
        return demos
    else:
        if len(demos) < n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(demos)} are available.",
            )
        if len(demos) > n_expert_demos:
            logger.warning(
                f"Using only the first {n_expert_demos} trajectories out of "
                f"{len(demos)} available.",
            )
            return demos[:n_expert_demos]
        else:
            return demos


@demonstrations_ingredient.capture
def _generate_expert_trajs(
    n_expert_demos: Optional[int],
    _rnd: np.random.Generator,
) -> Optional[Sequence[types.Trajectory]]:
    """Generates expert demonstrations.

    Args:
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.
        _rnd: Random number generator provided by Sacred.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: If n_expert_demos is None.
    """
    if n_expert_demos is None:
        raise ValueError("n_expert_demos must be specified when generating demos.")

    logger.info(f"Generating {n_expert_demos} expert trajectories")
    with environment.make_rollout_venv() as env:  # type: ignore[wrong-arg-count]
        return rollout.rollout(
            expert.get_expert_policy(env),
            env,
            rollout.make_sample_until(min_episodes=n_expert_demos),
            rng=_rnd,
        )


@demonstrations_ingredient.capture
def _download_expert_rollouts(
    environment: Dict[str, Any],
    path: Optional[str],
    organization: Optional[str],
    algo_name: Optional[str],
    loader_kwargs: Dict[str, Any],
):
    if path is not None:
        repo_id = path
    else:
        model_name = hfsb3.ModelName(
            algo_name,
            hfsb3.EnvironmentName(environment["gym_id"]),
        )
        repo_id = hfsb3.ModelRepoId(organization, model_name)

    logger.info(f"Loading expert trajectories from {repo_id}")
    dataset = datasets.load_dataset(repo_id, **loader_kwargs)
    return huggingface_utils.TrajectoryDatasetSequence(dataset)


# ---------------------------------------------------------------------------
# Human demo loading (BPref3 format -> imitation Trajectory)
# ---------------------------------------------------------------------------

# Module-level storage for timing metadata (read by train_adversarial.py)
human_demo_timing: Optional[Dict[str, Any]] = None


def _load_human_demos(path: str) -> Sequence[types.Trajectory]:
    """Load human demonstrations from BPref3 format or pre-converted Trajectory pickle.

    Auto-detects format:
      - BPref3 new format: {'demos': [...], 'timing': {...}}
      - BPref3 old format: list of episode dicts with obs/actions/rewards/dones
      - Pre-converted: list of dicts with obs/acts/infos/terminal keys
      - Pre-converted Trajectory objects

    Stores timing metadata in module-level `human_demo_timing` for TB logging.
    """
    global human_demo_timing
    import pickle as _pkl

    with open(path, "rb") as f:
        data = _pkl.load(f)

    # --- Detect format ---
    timing = None

    if isinstance(data, dict) and "demos" in data:
        # BPref3 new format with timing
        timing = data.get("timing")
        data = data["demos"]

    if isinstance(data, list) and len(data) > 0:
        first = data[0]

        if isinstance(first, types.Trajectory):
            # Already Trajectory objects
            logger.info(f"Loaded {len(data)} Trajectory objects from {path}")
            human_demo_timing = timing
            return data

        if isinstance(first, dict):
            if "obs" in first and "acts" in first:
                # Pre-converted dict format from human_demo.py --export_imitation
                trajs = _dicts_to_trajectories(data)
            elif "observations" in first and "actions" in first:
                # Raw BPref3 episode format
                trajs = _bpref3_to_trajectories(data)
            else:
                raise ValueError(
                    f"Unrecognized demo dict keys: {list(first.keys())}"
                )

            logger.info(
                f"Loaded {len(trajs)} human trajectories from {path}"
                + (f" (demo time: {timing['total_time_sec']:.1f}s)" if timing else "")
            )
            human_demo_timing = timing
            return trajs

    # Also check for timing.pkl alongside the demo file
    if timing is None:
        timing_path = path.replace(".pkl", "") + "_timing.pkl"
        if not os.path.exists(timing_path):
            timing_path = os.path.join(os.path.dirname(path), "timing.pkl")
        if os.path.exists(timing_path):
            with open(timing_path, "rb") as f:
                timing = _pkl.load(f)
            logger.info(f"Loaded timing metadata from {timing_path}")

    human_demo_timing = timing
    raise ValueError(f"Could not parse human demos from {path}")


def _dicts_to_trajectories(
    dicts: Sequence[Dict[str, Any]],
) -> Sequence[types.Trajectory]:
    """Convert list of {obs, acts, infos, terminal} dicts to Trajectory objects."""
    trajs = []
    for d in dicts:
        trajs.append(
            types.Trajectory(
                obs=np.asarray(d["obs"]),
                acts=np.asarray(d["acts"]),
                infos=d.get("infos"),
                terminal=d.get("terminal", True),
            )
        )
    return trajs


def _bpref3_to_trajectories(
    episodes: Sequence[Dict[str, Any]],
) -> Sequence[types.Trajectory]:
    """Convert BPref3 episode dicts to Trajectory objects.

    BPref3 format: {observations (T, dim), next_observations, actions, rewards, dones}
    Trajectory: obs (T+1, dim), acts (T, dim), infos=None, terminal=True
    """
    trajs = []
    for i, ep in enumerate(episodes):
        obs = np.asarray(ep["observations"])
        next_obs = np.asarray(ep["next_observations"])
        acts = np.asarray(ep["actions"])

        if len(obs) == 0:
            logger.warning(f"Skipping empty episode {i}")
            continue

        # obs = (T+1, dim): stack observations + last next_obs
        traj_obs = np.vstack([obs, next_obs[-1:]])
        trajs.append(
            types.Trajectory(
                obs=traj_obs,
                acts=acts,
                infos=None,
                terminal=True,
            )
        )
    return trajs
