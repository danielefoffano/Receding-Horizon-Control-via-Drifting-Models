from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drifting_mpc.config import dump_config
from drifting_mpc.control.oracle import OracleLQRController
from drifting_mpc.data.costs import sample_omega, trajectory_cost_np
from drifting_mpc.data.dataset import OfflineTrajectorySplit, TrajectoryNormalizer, save_manifest
from drifting_mpc.data.trajectory import encode_trajectory
from drifting_mpc.envs.mass_spring_damper import MassSpringDamperEnv, build_msd_spec

CONTROLLER_NAMES = {
    0: "oracle_noisy",
    1: "pd_noisy",
    2: "random_smooth_open_loop",
}


@dataclass
class CollectedTrajectory:
    states: np.ndarray
    actions: np.ndarray
    x0: np.ndarray
    omega: np.ndarray
    collection_cost: float
    zeta: np.ndarray
    controller_id: int


def _rollout_policy(
    env: MassSpringDamperEnv,
    x0: np.ndarray,
    policy,
    episode_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = env.reset(initial_state=x0)
    states = [state.copy()]
    actions = []
    for step in range(episode_len):
        action = np.asarray(policy(step, state.copy()), dtype=np.float32).reshape(env.spec.action_dim)
        state = env.step(action)
        actions.append(action.copy())
        states.append(state.copy())
    return np.stack(states), np.stack(actions)


def _sample_pd_policy(config: dict, rng: np.random.Generator):
    pd_cfg = config["dataset"]["pd"]
    kp = float(rng.uniform(*pd_cfg["kp_range"]))
    kd = float(rng.uniform(*pd_cfg["kd_range"]))
    action_noise_std = float(pd_cfg["action_noise_std"])

    def policy(_: int, state: np.ndarray) -> np.ndarray:
        action = np.array([-(kp * state[0] + kd * state[1])], dtype=np.float32)
        if action_noise_std > 0.0:
            action = (action + rng.normal(0.0, action_noise_std, size=action.shape)).astype(np.float32)
        return action

    return policy


def _sample_random_open_loop(spec, config: dict, rng: np.random.Generator) -> np.ndarray:
    random_cfg = config["dataset"]["random_open_loop"]
    actions = np.zeros((spec.episode_len, spec.action_dim), dtype=np.float32)
    smoothing = float(random_cfg["smoothing"])
    action_std = float(random_cfg["action_std"])
    for step in range(spec.episode_len):
        innovation = rng.normal(0.0, action_std, size=(spec.action_dim,)).astype(np.float32)
        previous = actions[step - 1] if step > 0 else np.zeros(spec.action_dim, dtype=np.float32)
        actions[step] = smoothing * previous + (1.0 - smoothing) * innovation
    return actions


def _collect_single_trajectory(
    env: MassSpringDamperEnv,
    oracle: OracleLQRController,
    x0: np.ndarray,
    omega: np.ndarray,
    controller_id: int,
    config: dict,
    rng: np.random.Generator,
) -> CollectedTrajectory:
    if controller_id == 0:
        states, actions = oracle.rollout(
            env=env,
            x0=x0,
            omega=omega,
            action_noise_std=float(config["dataset"]["oracle_action_noise_std"]),
            rng=rng,
        )
    elif controller_id == 1:
        policy = _sample_pd_policy(config, rng)
        states, actions = _rollout_policy(env, x0, policy, env.spec.episode_len)
    elif controller_id == 2:
        open_loop_actions = _sample_random_open_loop(env.spec, config, rng)
        states, actions = env.rollout_open_loop(x0, open_loop_actions)
    else:
        raise ValueError(f"Unknown controller id: {controller_id}")
    zeta = encode_trajectory(states, actions)
    collection_cost = trajectory_cost_np(states, actions, omega)
    return CollectedTrajectory(
        states=states,
        actions=actions,
        x0=x0.astype(np.float32),
        omega=omega.astype(np.float32),
        collection_cost=collection_cost,
        zeta=zeta,
        controller_id=controller_id,
    )


def _stack_trajectories(trajectories: list[CollectedTrajectory]) -> OfflineTrajectorySplit:
    return OfflineTrajectorySplit(
        states=np.stack([item.states for item in trajectories]).astype(np.float32),
        actions=np.stack([item.actions for item in trajectories]).astype(np.float32),
        x0=np.stack([item.x0 for item in trajectories]).astype(np.float32),
        omega=np.stack([item.omega for item in trajectories]).astype(np.float32),
        collection_cost=np.asarray([item.collection_cost for item in trajectories], dtype=np.float32),
        zeta=np.stack([item.zeta for item in trajectories]).astype(np.float32),
        controller_id=np.asarray([item.controller_id for item in trajectories], dtype=np.int64),
    )


def collect_offline_dataset(config: dict, dataset_dir: str | Path | None = None) -> dict[str, Path]:
    """Collect the mixed offline dataset required by Alternative A."""
    dataset_cfg = config["dataset"]
    target_dir = Path(dataset_dir or dataset_cfg["output_dir"]).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    spec = build_msd_spec(config)
    env = MassSpringDamperEnv(spec)
    oracle = OracleLQRController(spec)

    mixture_cfg = dataset_cfg["mixture"]
    controller_probs = np.asarray(
        [
            mixture_cfg["oracle_fraction"],
            mixture_cfg["pd_fraction"],
            mixture_cfg["random_fraction"],
        ],
        dtype=np.float64,
    )
    controller_probs = controller_probs / controller_probs.sum()

    trajectories: list[CollectedTrajectory] = []
    for _ in range(int(dataset_cfg["num_trajectories"])):
        x0 = env.sample_initial_state(rng)
        omega = sample_omega(rng, config)
        controller_id = int(rng.choice(np.arange(len(controller_probs)), p=controller_probs))
        trajectories.append(_collect_single_trajectory(env, oracle, x0, omega, controller_id, config, rng))

    permutation = rng.permutation(len(trajectories))
    shuffled = [trajectories[index] for index in permutation.tolist()]

    split_cfg = dataset_cfg["split_fractions"]
    train_end = int(len(shuffled) * float(split_cfg["train"]))
    val_end = train_end + int(len(shuffled) * float(split_cfg["val"]))
    split_items = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    normalizer = TrajectoryNormalizer.fit(_stack_trajectories(split_items["train"]).zeta)
    split_paths: dict[str, Path] = {}
    split_sizes: dict[str, int] = {}
    for split_name, items in split_items.items():
        split = _stack_trajectories(items)
        split_paths[split_name] = split.save(target_dir / f"{split_name}.npz")
        split_sizes[split_name] = split.size

    normalization_path = normalizer.save(target_dir / "normalization.npz")
    config_path = dump_config(config, target_dir / "resolved_config.yaml")
    manifest_path = save_manifest(
        target_dir / "manifest.json",
        {
            "dataset_dir": str(target_dir),
            "num_trajectories": int(dataset_cfg["num_trajectories"]),
            "split_sizes": split_sizes,
            "controller_names": CONTROLLER_NAMES,
            "trajectory_shape": {
                "states": [spec.episode_len + 1, spec.state_dim],
                "actions": [spec.episode_len, spec.action_dim],
                "zeta_dim": int((spec.state_dim + spec.action_dim) * spec.horizon),
            },
            "files": {
                "train": str(split_paths["train"]),
                "val": str(split_paths["val"]),
                "test": str(split_paths["test"]),
                "normalization": str(normalization_path),
                "resolved_config": str(config_path),
            },
        },
    )
    return {
        "dataset_dir": target_dir,
        "train": split_paths["train"],
        "val": split_paths["val"],
        "test": split_paths["test"],
        "normalization": normalization_path,
        "manifest": manifest_path,
        "resolved_config": config_path,
    }
