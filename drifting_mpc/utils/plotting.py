from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "figure.titlesize": 22,
})


def save_training_curves(history: dict[str, list[float]], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Alternative A drifting loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def save_cost_histogram(learned: np.ndarray, oracle: np.ndarray, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(oracle, bins=20, alpha=0.6, label="oracle")
    plt.hist(learned, bins=20, alpha=0.6, label="learned")
    plt.xlabel("Cumulative episode cost")
    plt.ylabel("Count")
    plt.title("Cost comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def save_cost_scatter(learned: np.ndarray, oracle: np.ndarray, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    min_cost = float(min(learned.min(), oracle.min()))
    max_cost = float(max(learned.max(), oracle.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(oracle, learned, alpha=0.7)
    plt.plot([min_cost, max_cost], [min_cost, max_cost], linestyle="--", color="black")
    plt.xlabel("Oracle cost")
    plt.ylabel("Learned cost")
    plt.title("Learned vs oracle")
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def save_rollout_plot(
    time_axis: np.ndarray,
    learned_states: np.ndarray,
    learned_actions: np.ndarray,
    oracle_states: np.ndarray,
    oracle_actions: np.ndarray,
    path: str | Path,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(time_axis, learned_states[:, 0], label="learned")
    axes[0].plot(time_axis, oracle_states[:, 0], label="oracle")
    axes[0].set_ylabel("position")
    axes[0].legend()

    axes[1].plot(time_axis, learned_states[:, 1], label="learned")
    axes[1].plot(time_axis, oracle_states[:, 1], label="oracle")
    axes[1].set_ylabel("velocity")

    action_axis = time_axis[:-1]
    axes[2].plot(action_axis, learned_actions[:, 0], label="learned")
    axes[2].plot(action_axis, oracle_actions[:, 0], label="oracle")
    axes[2].set_ylabel("action")
    axes[2].set_xlabel("step")

    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target



def _subplot_grid(num_panels: int) -> tuple[int, int]:
    if num_panels <= 0:
        raise ValueError("num_panels must be positive.")
    cols = min(4, num_panels)
    rows = int(np.ceil(num_panels / cols))
    return rows, cols



def _flatten_axes(axes):
    if isinstance(axes, np.ndarray):
        return axes.reshape(-1)
    return [axes]



def save_multi_cost_histograms(method_costs: dict[str, np.ndarray], oracle: np.ndarray, path: str | Path) -> Path:
    if not method_costs:
        raise ValueError("method_costs must be non-empty.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    labels = list(method_costs.keys())
    rows, cols = _subplot_grid(len(labels))
    all_values = [oracle.astype(np.float32)] + [np.asarray(method_costs[label], dtype=np.float32) for label in labels]
    global_min = float(min(arr.min() for arr in all_values))
    global_max = float(max(arr.max() for arr in all_values))
    if np.isclose(global_min, global_max):
        global_max = global_min + 1.0
    bins = np.linspace(global_min, global_max, 21)

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False, sharex=True, sharey=True)
    flat_axes = _flatten_axes(axes)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(labels), 1)))
    for idx, label in enumerate(labels):
        ax = flat_axes[idx]
        learned = np.asarray(method_costs[label], dtype=np.float32)
        ax.hist(oracle, bins=bins, alpha=0.55, label='oracle', color='0.4')
        if label =="weighted":
            label = "Drifting MPC"
        elif label == "drift_prior":
            label = "Drifting Prior"
        elif label == "diffusion_prior":
            label = "Diffusion"
        elif label == "guided_diffusion":
            label = "Guided Diffusion"
        ax.hist(learned, bins=bins, alpha=0.55, label=label, color=colors[idx % len(colors)])
        ax.set_title(label)
        ax.set_xlabel('Cumulative episode cost')
        ax.set_ylabel('Count')
    for ax in flat_axes[len(labels):]:
        ax.axis('off')
    handles, labels_legend = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper right', ncol=min(5, len(labels) + 1), frameon=False)
    fig.suptitle('Cost histograms by method', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(target)
    plt.close(fig)
    return target



def save_multi_cost_scatter(method_costs: dict[str, np.ndarray], oracle: np.ndarray, path: str | Path) -> Path:
    if not method_costs:
        raise ValueError("method_costs must be non-empty.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    labels = list(method_costs.keys())
    rows, cols = _subplot_grid(len(labels))
    all_values = [oracle.astype(np.float32)] + [np.asarray(method_costs[label], dtype=np.float32) for label in labels]
    global_min = float(min(arr.min() for arr in all_values))
    global_max = float(max(arr.max() for arr in all_values))
    if np.isclose(global_min, global_max):
        global_max = global_min + 1.0

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False, sharex=True, sharey=True)
    flat_axes = _flatten_axes(axes)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(labels), 1)))
    for idx, label in enumerate(labels):
        ax = flat_axes[idx]
        learned = np.asarray(method_costs[label], dtype=np.float32)
        ax.scatter(oracle, learned, alpha=0.75, color=colors[idx % len(colors)], edgecolors='none')
        ax.plot([global_min, global_max], [global_min, global_max], linestyle='--', color='black', linewidth=1.0, label=label)
        if label =="weighted":
            label = "Drifting MPC"
        elif label == "drift_prior":
            label = "Drifting Prior"
        elif label == "diffusion_prior":
            label = "Diffusion"
        elif label == "guided_diffusion":
            label = "Guided Diffusion"
        ax.set_title(label)
        ax.set_xlabel('Oracle cost')
        if label == "Drifting MPC":
            ax.set_ylabel('Learned Method cost')
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_xscale('log')
        ax.set_yscale('log')
    for ax in flat_axes[len(labels):]:
        ax.axis('off')
    fig.suptitle('Horizon 100')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(target)
    plt.close(fig)
    return target



def save_multi_method_rollout_plot(
    time_axis: np.ndarray,
    oracle_states: np.ndarray,
    oracle_actions: np.ndarray,
    method_rollouts: dict[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path,
) -> Path:
    if not method_rollouts:
        raise ValueError("method_rollouts must be non-empty.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    labels = list(method_rollouts.keys())
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(labels), 1)))

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 13.2), sharex=True)
    axes[0].plot(time_axis, oracle_states[:, 0], label='oracle', color='black', linewidth=2.0)
    axes[1].plot(time_axis, oracle_states[:, 1], label='oracle', color='black', linewidth=2.0)
    axes[2].plot(time_axis[:-1], oracle_actions[:, 0], label='oracle', color='black', linewidth=2.0)

    for idx, label in enumerate(labels):
        states, actions = method_rollouts[label]
        color = colors[idx % len(colors)]
        if label =="weighted":
            label = "Drifting MPC"
        elif label == "drift_prior":
            label = "Drifting MPC Prior"
        elif label == "diffusion_prior":
            label = "Diffusion"
        elif label == "guided_diffusion":
            label = "Guided Diffusion"
        axes[0].plot(time_axis, states[:, 0], label=label, color=color)
        axes[1].plot(time_axis, states[:, 1], label=label, color=color)
        axes[2].plot(time_axis[:-1], actions[:, 0], label=label, color=color)

    axes[0].set_ylabel('position')
    axes[1].set_ylabel('velocity')
    axes[2].set_ylabel('action')
    axes[2].set_xlabel('step')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=min(2, len(labels) + 1), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    #fig.suptitle('Horizon 50')
    fig.savefig(target)
    plt.close(fig)
    return target
