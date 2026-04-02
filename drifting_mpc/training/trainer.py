from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from drifting_mpc.config import dump_config
from drifting_mpc.data.dataset import TrajectoryNormalizer, load_split
from drifting_mpc.envs.mass_spring_damper import build_msd_spec
from drifting_mpc.methods import get_method_spec
from drifting_mpc.models.generator import ConditionalTrajectoryGenerator
from drifting_mpc.training.drifting import TrajectoryDriftingObjective, beta_schedule
from drifting_mpc.utils.plotting import save_training_curves


@dataclass
class TrainingArtifacts:
    run_dir: Path
    checkpoint_path: Path
    last_checkpoint_path: Path
    history_path: Path
    plot_path: Path
    config_path: Path
    normalization_path: Path



def _select_device(config: dict) -> torch.device:
    requested = str(config["experiment"]["device"]).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)



def _build_model(config: dict, spec, device: torch.device) -> ConditionalTrajectoryGenerator:
    model_cfg = config["model"]
    method = get_method_spec(config)
    model = ConditionalTrajectoryGenerator(
        horizon=spec.horizon,
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        omega_dim=3 if method.condition_on_omega else 0,
        eps_dim=int(model_cfg["eps_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_blocks=int(model_cfg["num_blocks"]),
    )
    return model.to(device)



def _save_history(history: dict[str, list[float]], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return target


@torch.no_grad()
def _evaluate_epoch(
    objective: TrajectoryDriftingObjective,
    model: torch.nn.Module,
    beta: float,
    num_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x0_batch, omega_batch = objective.sample_context_batch()
        loss, _ = objective.compute_loss(model, x0_batch, omega_batch, beta)
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(len(losses), 1))



def train_drifting_model(
    config: dict,
    dataset_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TrainingArtifacts:
    spec = build_msd_spec(config)
    device = _select_device(config)
    method = get_method_spec(config)

    dataset_root = Path(dataset_dir or config["dataset"]["output_dir"]).expanduser().resolve()
    train_split = load_split(dataset_root / "train.npz")
    val_split = load_split(dataset_root / "val.npz")
    normalizer = TrajectoryNormalizer.load(dataset_root / "normalization.npz")

    training_cfg = config["training"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir or config["training"]["output_dir"]).expanduser().resolve()
    if output_dir is None:
        run_dir = run_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(config, spec, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    train_objective = TrajectoryDriftingObjective(train_split.to_torch(device), normalizer, config, device)
    val_objective = TrajectoryDriftingObjective(val_split.to_torch(device), normalizer, config, device)

    batch_size = int(training_cfg["batch_size"])
    steps_per_epoch = max(1, train_split.size // batch_size)
    val_batches = int(training_cfg.get("validation_batches", max(1, val_split.size // batch_size)))
    total_steps = steps_per_epoch * int(training_cfg["max_epochs"])
    beta_max = float(training_cfg["beta_max"])
    beta_ramp_fraction = float(training_cfg["beta_ramp_fraction"])
    grad_clip_norm = float(training_cfg["grad_clip_norm"])
    patience = int(training_cfg["patience"])

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "beta": [],
        "beta_eval": [],
        "positive_norm": [],
        "negative_norm": [],
    }
    best_val = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"
    global_step = 0
    beta_eval = beta_max if method.use_cost_tilt else 0.0
    latest_val_loss: float | None = None

    with tqdm(
        total=total_steps,
        desc=f"train_{method.variant}",
        unit="step",
        dynamic_ncols=True,
        leave=True,
    ) as progress_bar:
        for epoch in range(int(training_cfg["max_epochs"])):
            model.train()
            train_losses = []
            positive_norms = []
            negative_norms = []
            beta_value = beta_schedule(global_step, total_steps, beta_max, beta_ramp_fraction) if method.use_cost_tilt else 0.0
            progress_bar.set_description(
                f"train_{method.variant} epoch {epoch + 1}/{int(training_cfg['max_epochs'])}",
                refresh=False,
            )
            for _ in range(steps_per_epoch):
                x0_batch, omega_batch = train_objective.sample_context_batch()
                beta_value = beta_schedule(global_step, total_steps, beta_max, beta_ramp_fraction) if method.use_cost_tilt else 0.0
                optimizer.zero_grad(set_to_none=True)
                loss, stats = train_objective.compute_loss(model, x0_batch, omega_batch, beta_value)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                train_losses.append(stats.loss)
                positive_norms.append(stats.positive_norm)
                negative_norms.append(stats.negative_norm)
                global_step += 1

                progress_bar.update(1)
                progress_bar.set_postfix(
                    train_loss=f"{stats.loss:.4f}",
                    val_loss="nan" if latest_val_loss is None else f"{latest_val_loss:.4f}",
                    beta=f"{beta_value:.2f}",
                    refresh=False,
                )

            train_loss = float(sum(train_losses) / max(len(train_losses), 1))
            val_loss = _evaluate_epoch(val_objective, model, beta_eval, val_batches)
            latest_val_loss = val_loss
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["beta"].append(float(beta_value))
            history["beta_eval"].append(float(beta_eval))
            history["positive_norm"].append(float(sum(positive_norms) / max(len(positive_norms), 1)))
            history["negative_norm"].append(float(sum(negative_norms) / max(len(negative_norms), 1)))

            progress_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                beta=f"{beta_value:.2f}",
            )

            checkpoint_payload = {
                "model_state": model.state_dict(),
                "config": config,
                "history": history,
                "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
                "method_variant": method.variant,
                "model_family": method.model_family,
                "train_beta": float(beta_value),
                "val_beta": float(beta_eval),
                "global_step": int(global_step),
            }
            torch.save(
                {**checkpoint_payload, "checkpoint_type": "last", "val_loss": float(val_loss)},
                last_checkpoint_path,
            )

            if val_loss < best_val:
                best_val = val_loss
                epochs_without_improvement = 0
                torch.save(
                    {**checkpoint_payload, "checkpoint_type": "best", "best_val_loss": best_val},
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

    history_path = _save_history(history, run_dir / "history.json")
    plot_path = save_training_curves(history, run_dir / "loss_curves.png")
    config_path = dump_config(config, run_dir / "resolved_config.yaml")
    normalization_path = normalizer.save(run_dir / "normalization.npz")
    return TrainingArtifacts(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        history_path=history_path,
        plot_path=plot_path,
        config_path=config_path,
        normalization_path=normalization_path,
    )


# Backwards-compatible alias.
train_alternative_a = train_drifting_model


def train_model(config: dict, dataset_dir: str | Path | None = None, output_dir: str | Path | None = None) -> TrainingArtifacts:
    method = get_method_spec(config)
    if method.model_family == "drifting":
        return train_drifting_model(config, dataset_dir=dataset_dir, output_dir=output_dir)
    if method.model_family == "diffusion":
        from drifting_mpc.training.diffusion_trainer import train_diffusion_model

        return train_diffusion_model(config, dataset_dir=dataset_dir, output_dir=output_dir)
    raise ValueError(f"Unsupported model family: {method.model_family}")


# Backwards-compatible alias for generic entry points.
train_model_from_config = train_model
