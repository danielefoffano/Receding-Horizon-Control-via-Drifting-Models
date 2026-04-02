from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from drifting_mpc.config import dump_config
from drifting_mpc.data.dataset import TrajectoryNormalizer, load_split
from drifting_mpc.envs.mass_spring_damper import build_msd_spec
from drifting_mpc.methods import get_method_spec
from drifting_mpc.models.diffusion import ConditionalTrajectoryDiffusionModel
from drifting_mpc.training.diffusion import TrajectoryDiffusionObjective, build_transition_tensor_torch
from drifting_mpc.training.trainer import TrainingArtifacts, _save_history, _select_device
from drifting_mpc.utils.plotting import save_training_curves


def _build_model(config: dict, spec, device: torch.device) -> ConditionalTrajectoryDiffusionModel:
    model_cfg = config["model"]
    diffusion_cfg = config.get("diffusion", {})
    return ConditionalTrajectoryDiffusionModel(
        horizon_steps=spec.horizon + 1,
        transition_dim=spec.action_dim + spec.state_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_blocks=int(model_cfg["num_blocks"]),
        time_embed_dim=int(diffusion_cfg.get("time_embed_dim", 64)),
    ).to(device)


def _fit_transition_normalizer(split, device: torch.device | str) -> TrajectoryNormalizer:
    torch_split = split.to_torch(device)
    transition_tensor = build_transition_tensor_torch(torch_split.states, torch_split.actions)
    flat = transition_tensor.reshape(split.size, -1).detach().cpu().numpy().astype("float32")
    return TrajectoryNormalizer.fit(flat)


@torch.no_grad()
def _evaluate_epoch(objective: TrajectoryDiffusionObjective, model: torch.nn.Module, num_batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x0_batch, omega_batch, traj_batch = objective.sample_batch()
        loss, _ = objective.compute_loss(model, x0_batch, omega_batch, traj_batch)
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(len(losses), 1))


def train_diffusion_model(
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
    normalizer = _fit_transition_normalizer(train_split, device)

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

    train_objective = TrajectoryDiffusionObjective(train_split.to_torch(device), normalizer, config, device)
    val_objective = TrajectoryDiffusionObjective(val_split.to_torch(device), normalizer, config, device)

    batch_size = int(training_cfg["batch_size"])
    steps_per_epoch = max(1, train_split.size // batch_size)
    val_batches = int(training_cfg.get("validation_batches", max(1, val_split.size // batch_size)))
    grad_clip_norm = float(training_cfg["grad_clip_norm"])
    patience = int(training_cfg["patience"])
    max_epochs = int(training_cfg["max_epochs"])

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"
    latest_val_loss: float | None = None

    total_steps = max_epochs * steps_per_epoch
    with tqdm(total=total_steps, desc=f"train_{method.variant}", unit="step", dynamic_ncols=True, leave=True) as progress_bar:
        for epoch in range(max_epochs):
            model.train()
            train_losses = []
            progress_bar.set_description(f"train_{method.variant} epoch {epoch + 1}/{max_epochs}", refresh=False)
            for _ in range(steps_per_epoch):
                x0_batch, omega_batch, traj_batch = train_objective.sample_batch()
                optimizer.zero_grad(set_to_none=True)
                loss, stats = train_objective.compute_loss(model, x0_batch, omega_batch, traj_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                train_losses.append(stats.loss)
                progress_bar.update(1)
                progress_bar.set_postfix(
                    train_loss=f"{stats.loss:.4f}",
                    val_loss="nan" if latest_val_loss is None else f"{latest_val_loss:.4f}",
                    refresh=False,
                )

            train_loss = float(sum(train_losses) / max(len(train_losses), 1))
            val_loss = _evaluate_epoch(val_objective, model, val_batches)
            latest_val_loss = val_loss
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            progress_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

            checkpoint_payload = {
                "model_state": model.state_dict(),
                "config": config,
                "history": history,
                "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
                "method_variant": method.variant,
                "model_family": "diffusion",
                "val_loss": float(val_loss),
            }
            torch.save({**checkpoint_payload, "checkpoint_type": "last"}, last_checkpoint_path)
            if val_loss < best_val:
                best_val = val_loss
                epochs_without_improvement = 0
                torch.save({**checkpoint_payload, "checkpoint_type": "best", "best_val_loss": best_val}, checkpoint_path)
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
