from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from drifting_mpc.config import dump_config
from drifting_mpc.control.oracle import OracleLQRController
from drifting_mpc.data.costs import trajectory_cost_np
from drifting_mpc.data.dataset import load_split
from drifting_mpc.envs.mass_spring_damper import MassSpringDamperEnv, build_msd_spec
from drifting_mpc.evaluation.policy import LearnedMPCPlanner, load_planner_from_checkpoint
from drifting_mpc.methods import get_method_spec
from drifting_mpc.utils.plotting import save_cost_histogram, save_cost_scatter, save_multi_cost_histograms, save_multi_cost_scatter, save_multi_method_rollout_plot, save_rollout_plot


def _select_device(config: dict) -> torch.device:
    requested = str(config["experiment"]["device"]).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _rollout_learned(env: MassSpringDamperEnv, planner: LearnedMPCPlanner, x0: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = env.reset(initial_state=x0)
    states = [state.copy()]
    actions = []
    for _ in range(env.spec.episode_len):
        sample = planner.act(state, omega)
        state = env.step(sample.action)
        actions.append(sample.action.copy())
        states.append(state.copy())
    return np.stack(states), np.stack(actions)


def _rollout_oracle(env: MassSpringDamperEnv, oracle: OracleLQRController, x0: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = env.reset(initial_state=x0)
    states = [state.copy()]
    actions = []
    for _ in range(env.spec.episode_len):
        action = oracle.first_action(state, omega, horizon=env.spec.horizon)
        state = env.step(action)
        actions.append(action.copy())
        states.append(state.copy())
    return np.stack(states), np.stack(actions)


def _save_episode_table(rows: list[dict[str, float]], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def evaluate_vs_oracle(
    ckpt_path: str | Path,
    config: dict | None = None,
    dataset_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    device = _select_device(config) if config is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner, ckpt_config = load_planner_from_checkpoint(ckpt_path, device=device)
    if config is None:
        config = ckpt_config
    spec = build_msd_spec(config)
    method = get_method_spec(ckpt_config)
    dataset_root = Path(dataset_dir or config["dataset"]["output_dir"]).expanduser().resolve()
    test_split = load_split(dataset_root / "test.npz")
    eval_dir = Path(output_dir or config["evaluation"]["output_dir"]).expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(config["experiment"]["seed"]) + 123)
    num_episodes = min(int(config["evaluation"]["num_episodes"]), test_split.size)
    episode_indices = rng.choice(test_split.size, size=num_episodes, replace=False)

    learned_env = MassSpringDamperEnv(spec)
    oracle_env = MassSpringDamperEnv(spec)
    oracle = OracleLQRController(spec)

    learned_costs = []
    oracle_costs = []
    regrets = []
    episode_rows: list[dict[str, float]] = []
    rollout_bundle = []

    for episode_id, index in enumerate(episode_indices.tolist()):
        x0 = test_split.x0[index]
        omega = test_split.omega[index]
        learned_states, learned_actions = _rollout_learned(learned_env, planner, x0, omega)
        oracle_states, oracle_actions = _rollout_oracle(oracle_env, oracle, x0, omega)
        learned_cost = trajectory_cost_np(learned_states, learned_actions, omega)
        oracle_cost = trajectory_cost_np(oracle_states, oracle_actions, omega)
        regret = (learned_cost - oracle_cost) / max(abs(oracle_cost), 1e-8)

        learned_costs.append(learned_cost)
        oracle_costs.append(oracle_cost)
        regrets.append(regret)
        episode_rows.append(
            {
                "episode": float(episode_id),
                "x0_position": float(x0[0]),
                "x0_velocity": float(x0[1]),
                "q_pos": float(omega[0]),
                "q_vel": float(omega[1]),
                "r_u": float(omega[2]),
                "learned_cost": float(learned_cost),
                "oracle_cost": float(oracle_cost),
                "normalized_regret": float(regret),
            }
        )
        rollout_bundle.append((episode_id, learned_states, learned_actions, oracle_states, oracle_actions))

    learned_array = np.asarray(learned_costs, dtype=np.float32)
    oracle_array = np.asarray(oracle_costs, dtype=np.float32)
    regret_array = np.asarray(regrets, dtype=np.float32)

    metrics = {
        "method_variant": method.variant,
        "num_episodes": int(num_episodes),
        "learned_mean_cost": float(learned_array.mean()),
        "learned_std_cost": float(learned_array.std()),
        "learned_median_cost": float(np.median(learned_array)),
        "oracle_mean_cost": float(oracle_array.mean()),
        "oracle_std_cost": float(oracle_array.std()),
        "oracle_median_cost": float(np.median(oracle_array)),
        "normalized_regret_mean": float(regret_array.mean()),
        "normalized_regret_median": float(np.median(regret_array)),
        "fraction_within_10pct": float(np.mean(regret_array <= 0.10)),
        "fraction_within_25pct": float(np.mean(regret_array <= 0.25)),
    }

    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    episode_table_path = _save_episode_table(episode_rows, eval_dir / "episode_metrics.csv")
    histogram_path = save_cost_histogram(learned_array, oracle_array, eval_dir / "cost_histogram.png")
    scatter_path = save_cost_scatter(learned_array, oracle_array, eval_dir / "cost_scatter.png")

    rollout_paths: list[Path] = []
    time_axis = np.arange(spec.episode_len + 1)
    for episode_id, learned_states, learned_actions, oracle_states, oracle_actions in rollout_bundle[: int(config["evaluation"]["num_rollout_plots"])]:
        rollout_paths.append(
            save_rollout_plot(
                time_axis=time_axis,
                learned_states=learned_states,
                learned_actions=learned_actions,
                oracle_states=oracle_states,
                oracle_actions=oracle_actions,
                path=eval_dir / f"rollout_{episode_id:03d}.png",
            )
        )

    config_to_dump = dict(config)
    config_to_dump.setdefault("method", {})["variant"] = method.variant
    config_path = dump_config(config_to_dump, eval_dir / "resolved_config.yaml")
    return {
        "eval_dir": eval_dir,
        "metrics": metrics_path,
        "episode_metrics": episode_table_path,
        "histogram": histogram_path,
        "scatter": scatter_path,
        "config": config_path,
        "rollouts": rollout_paths,
    }



def _slugify_label(label: str) -> str:
    slug = ''.join(ch.lower() if ch.isalnum() else '_' for ch in label).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug or 'method'


def _make_unique_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique = []
    for label in labels:
        base = label
        count = counts.get(base, 0)
        if count == 0:
            unique.append(base)
        else:
            unique.append(f"{base}_{count + 1}")
        counts[base] = count + 1
    return unique


def _build_metric_summary(label: str, learned_array: np.ndarray, oracle_array: np.ndarray, regret_array: np.ndarray, variant: str) -> dict[str, float | str | int]:
    return {
        "label": label,
        "method_variant": variant,
        "num_episodes": int(len(learned_array)),
        "learned_mean_cost": float(learned_array.mean()),
        "learned_std_cost": float(learned_array.std()),
        "learned_median_cost": float(np.median(learned_array)),
        "oracle_mean_cost": float(oracle_array.mean()),
        "oracle_std_cost": float(oracle_array.std()),
        "oracle_median_cost": float(np.median(oracle_array)),
        "normalized_regret_mean": float(regret_array.mean()),
        "normalized_regret_median": float(np.median(regret_array)),
        "fraction_within_10pct": float(np.mean(regret_array <= 0.10)),
        "fraction_within_25pct": float(np.mean(regret_array <= 0.25)),
    }


def _save_summary_table(rows: list[dict[str, object]], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must be non-empty.")
    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def evaluate_multiple_vs_oracle(
    ckpt_paths: list[str | Path],
    labels: list[str] | None = None,
    config: dict | None = None,
    dataset_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    m_plan: int | None = None,
) -> dict[str, Path]:
    if not ckpt_paths:
        raise ValueError("ckpt_paths must be non-empty.")

    device = _select_device(config) if config is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    planners: list[tuple[str, object, dict, str]] = []
    inferred_labels: list[str] = []
    for ckpt_path in ckpt_paths:
        planner, ckpt_config = load_planner_from_checkpoint(ckpt_path, device=device, m_plan=m_plan if m_plan is not None else (int(config["evaluation"]["m_plan"]) if config is not None else None))
        variant = get_method_spec(ckpt_config).variant
        inferred_labels.append(variant)
        planners.append((str(ckpt_path), planner, ckpt_config, variant))

    if config is None:
        config = planners[0][2]

    if labels is None:
        labels = inferred_labels
    if len(labels) != len(planners):
        raise ValueError("labels must match the number of checkpoints.")
    labels = _make_unique_labels([str(label) for label in labels])

    spec = build_msd_spec(config)
    dataset_root = Path(dataset_dir or config["dataset"]["output_dir"]).expanduser().resolve()
    test_split = load_split(dataset_root / "test.npz")
    eval_dir = Path(output_dir or config["evaluation"]["output_dir"]).expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(config["experiment"]["seed"]) + 123)
    num_episodes = min(int(config["evaluation"]["num_episodes"]), test_split.size)
    episode_indices = rng.choice(test_split.size, size=num_episodes, replace=False)

    oracle_env = MassSpringDamperEnv(spec)
    oracle = OracleLQRController(spec)
    learned_envs = [MassSpringDamperEnv(spec) for _ in planners]

    oracle_costs: list[float] = []
    method_costs: dict[str, list[float]] = {label: [] for label in labels}
    method_regrets: dict[str, list[float]] = {label: [] for label in labels}
    episode_rows: list[dict[str, float | str]] = []
    rollout_bundle: list[tuple[int, np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]] = []

    for episode_id, index in enumerate(episode_indices.tolist()):
        x0 = test_split.x0[index]
        omega = test_split.omega[index]
        oracle_states, oracle_actions = _rollout_oracle(oracle_env, oracle, x0, omega)
        oracle_cost = trajectory_cost_np(oracle_states, oracle_actions, omega)
        oracle_costs.append(oracle_cost)

        row: dict[str, float | str] = {
            "episode": float(episode_id),
            "x0_position": float(x0[0]),
            "x0_velocity": float(x0[1]),
            "q_pos": float(omega[0]),
            "q_vel": float(omega[1]),
            "r_u": float(omega[2]),
            "oracle_cost": float(oracle_cost),
        }
        rollout_methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for env, label, (_, planner, ckpt_config, variant) in zip(learned_envs, labels, planners):
            learned_states, learned_actions = _rollout_learned(env, planner, x0, omega)
            learned_cost = trajectory_cost_np(learned_states, learned_actions, omega)
            regret = (learned_cost - oracle_cost) / max(abs(oracle_cost), 1e-8)

            method_costs[label].append(learned_cost)
            method_regrets[label].append(regret)
            slug = _slugify_label(label)
            row[f"cost_{slug}"] = float(learned_cost)
            row[f"regret_{slug}"] = float(regret)
            row[f"variant_{slug}"] = variant
            rollout_methods[label] = (learned_states, learned_actions)

        episode_rows.append(row)
        rollout_bundle.append((episode_id, oracle_states, oracle_actions, rollout_methods))

    oracle_array = np.asarray(oracle_costs, dtype=np.float32)
    metrics_by_method: dict[str, dict[str, float | str | int]] = {}
    summary_rows: list[dict[str, object]] = []
    for label, (_, _, ckpt_config, variant) in zip(labels, planners):
        learned_array = np.asarray(method_costs[label], dtype=np.float32)
        regret_array = np.asarray(method_regrets[label], dtype=np.float32)
        metrics = _build_metric_summary(label, learned_array, oracle_array, regret_array, variant)
        metrics_by_method[label] = metrics
        summary_rows.append(metrics)

    metrics_payload = {
        "num_episodes": int(num_episodes),
        "oracle": {
            "mean_cost": float(oracle_array.mean()),
            "std_cost": float(oracle_array.std()),
            "median_cost": float(np.median(oracle_array)),
        },
        "methods": metrics_by_method,
        "checkpoints": [str(path) for path in ckpt_paths],
        "labels": labels,
    }
    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    summary_path = _save_summary_table(summary_rows, eval_dir / "metrics_summary.csv")
    episode_table_path = _save_episode_table(episode_rows, eval_dir / "episode_metrics.csv")
    histogram_path = save_multi_cost_histograms({label: np.asarray(values, dtype=np.float32) for label, values in method_costs.items()}, oracle_array, eval_dir / "cost_histograms.png")
    scatter_path = save_multi_cost_scatter({label: np.asarray(values, dtype=np.float32) for label, values in method_costs.items()}, oracle_array, eval_dir / "cost_scatter.png")

    rollout_paths: list[Path] = []
    time_axis = np.arange(spec.episode_len + 1)
    for episode_id, oracle_states, oracle_actions, rollout_methods in rollout_bundle[: int(config["evaluation"]["num_rollout_plots"])]:
        rollout_paths.append(
            save_multi_method_rollout_plot(
                time_axis=time_axis,
                oracle_states=oracle_states,
                oracle_actions=oracle_actions,
                method_rollouts=rollout_methods,
                path=eval_dir / f"rollout_{episode_id:03d}.png",
            )
        )

    comparison_config = dict(config)
    comparison_config["comparison"] = {
        "labels": labels,
        "checkpoints": [str(path) for path in ckpt_paths],
        "shared_m_plan": int(m_plan) if m_plan is not None else int(config["evaluation"]["m_plan"]),
        "method_variants": {label: variant for label, (_, _, _, variant) in zip(labels, planners)},
    }
    config_path = dump_config(comparison_config, eval_dir / "resolved_config.yaml")

    return {
        "eval_dir": eval_dir,
        "metrics": metrics_path,
        "summary": summary_path,
        "episode_metrics": episode_table_path,
        "histograms": histogram_path,
        "scatter": scatter_path,
        "config": config_path,
        "rollouts": rollout_paths,
    }
