from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from drifting_mpc.data.costs import trajectory_cost_torch
from drifting_mpc.data.dataset import TorchOfflineTrajectorySplit, TrajectoryNormalizer
from drifting_mpc.methods import MethodSpec, get_method_spec


@dataclass
class BatchStatistics:
    loss: float
    positive_norm: float
    negative_norm: float
    beta: float


class TrajectoryDriftingObjective:
    """Anti-symmetric drifting objective used by Drift RHC.

    The generator is conditioned on (x0, omega) and the positive field is built
    from cost-tilted neighbours of the query context; the optional dynamics
    consistency term is added on top (see consistency_weight below).
    """

    def __init__(
        self,
        split: TorchOfflineTrajectorySplit,
        normalizer: TrajectoryNormalizer,
        config: dict,
        device: torch.device | str,
    ):
        self.split = split
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.method: MethodSpec = get_method_spec(config)
        from drifting_mpc.envs.registry import build_spec
        self.spec = build_spec(config)
        train_cfg = config["training"]
        self.batch_size = int(train_cfg["batch_size"])
        self.positive_k = int(train_cfg["positives_k"])
        self.negative_m = int(train_cfg["negatives_m"])
        self.context_temperature = float(train_cfg["context_temperature"])
        self.kernel_kind = str(train_cfg["kernel"]).lower()
        self.kernel_temperatures = [float(value) for value in train_cfg["kernel_temperatures"]]
        self.cost_normalization = str(train_cfg.get("cost_normalization", "zscore")).lower()
        # Optional dynamics-consistency regulariser. Generated plans are ranked by
        # their own predicted states, so if those states do not obey the dynamics
        # the best-of-M selection is ranking infeasible trajectories. The model is
        # frozen and used ONLY here at training time -- inference stays one-step
        # and model-free.
        self.consistency_weight = float(train_cfg.get("consistency_weight", 0.0))
        # Scales the drift term itself, so the contribution of drifting can be
        # ablated against a consistency-only control.
        self.drift_weight = float(train_cfg.get("drift_weight", 1.0))
        # "onestep" is teacher-forced and can be satisfied while the rollout still
        # diverges on an unstable plant; "rollout" penalises the multi-step
        # deviation, which is what the planner's cost estimate actually depends on.
        self.consistency_mode = str(train_cfg.get("consistency_mode", "onestep")).lower()
        self.dynamics_model = None
        if self.consistency_weight > 0.0:
            from drifting_mpc.models.dynamics_model import ResidualDynamicsModel
            ck = torch.load(train_cfg["dynamics_ckpt"], map_location=self.device, weights_only=False)
            dyn = ResidualDynamicsModel(ck["state_dim"], ck["action_dim"], ck["hidden_dim"]).to(self.device)
            dyn.load_state_dict(ck["model_state"])
            dyn.eval()
            for prm in dyn.parameters():
                prm.requires_grad_(False)
            self.dynamics_model = dyn
        self.cost_clip = float(train_cfg.get("cost_clip", 5.0))
        self.mean, self.std = normalizer.as_torch(self.device)
        # Per-timestep state-delta scale, so the consistency residual can be
        # expressed in the SAME normalized units as the drift term. Without this
        # lambda is a unit conversion rather than a trade-off weight, and its
        # value swings by ~1000x between systems.
        _blocks = self.std.reshape(self.spec.horizon, self.spec.action_dim + self.spec.state_dim)
        self.state_scale = _blocks[:, self.spec.action_dim:].clamp_min(1e-6)
        self.normalized_split_zeta = (self.split.zeta - self.mean) / self.std
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(config["experiment"]["seed"]))

    def sample_context_batch(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.batch_size if batch_size is None else int(batch_size)
        indices = torch.randint(0, self.split.size, (size,), generator=self.generator, device=self.device)
        return self.split.x0[indices], self.split.omega[indices]

    def _kernel_log_weights(self, squared_distance: torch.Tensor) -> torch.Tensor:
        temperatures = torch.as_tensor(self.kernel_temperatures, dtype=squared_distance.dtype, device=squared_distance.device)
        if self.kernel_kind == "rbf":
            logits = -squared_distance.unsqueeze(-1) / temperatures
        elif self.kernel_kind == "laplacian":
            logits = -torch.sqrt(torch.clamp(squared_distance, min=1e-12)).unsqueeze(-1) / temperatures
        else:
            raise ValueError(f"Unsupported kernel kind: {self.kernel_kind}")
        return torch.logsumexp(logits, dim=-1) - math.log(len(self.kernel_temperatures))

    def _context_knn(self, x0_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distance_sq = torch.cdist(x0_batch, self.split.x0, p=2.0).square()
        k = min(self.positive_k, self.split.size)
        knn_distance_sq, knn_idx = torch.topk(distance_sq, k=k, largest=False, dim=-1)
        alpha_logits = -knn_distance_sq / self.context_temperature
        y_pool = self.normalized_split_zeta[knn_idx]
        states_pool = self.split.states[knn_idx]
        actions_pool = self.split.actions[knn_idx]
        return alpha_logits, y_pool, states_pool, actions_pool

    def _normalized_zeta(self, zeta: torch.Tensor) -> torch.Tensor:
        return (zeta - self.mean) / self.std

    def _normalize_pool_costs(self, relabeled_cost: torch.Tensor) -> torch.Tensor:
        mode = self.cost_normalization
        if mode in {"none", "raw"}:
            return relabeled_cost
        if mode == "zscore":
            mean = relabeled_cost.mean(dim=-1, keepdim=True)
            std = relabeled_cost.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            normalized = (relabeled_cost - mean) / std
        elif mode == "log_zscore":
            # Heavy-tailed costs compress the informative low-cost region to
            # near-zero width under a plain z-score, which flattens the tilt.
            # Taking logs first restores resolution among good trajectories.
            log_cost = torch.log(relabeled_cost.clamp_min(1e-8))
            mean = log_cost.mean(dim=-1, keepdim=True)
            std = log_cost.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            normalized = (log_cost - mean) / std
        elif mode == "rank":
            # Fully scale-free: only the ordering of costs within the pool
            # matters, so beta means the same thing on any cost scale.
            n = relabeled_cost.shape[-1]
            ranks = relabeled_cost.argsort(dim=-1).argsort(dim=-1).to(relabeled_cost.dtype)
            uniform = ranks / max(n - 1, 1)
            normalized = (uniform - 0.5) * (12.0 ** 0.5)  # zero mean, unit variance
        elif mode == "minmax":
            min_cost = relabeled_cost.min(dim=-1, keepdim=True).values
            max_cost = relabeled_cost.max(dim=-1, keepdim=True).values
            scale = (max_cost - min_cost).clamp_min(1e-6)
            normalized = (relabeled_cost - min_cost) / scale
        else:
            raise ValueError(f"Unsupported cost_normalization mode: {mode}")
        if self.cost_clip > 0.0:
            normalized = normalized.clamp(min=-self.cost_clip, max=self.cost_clip)
        return normalized

    def _forward_generator(
        self,
        model: torch.nn.Module,
        eps: torch.Tensor,
        x0: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        if self.method.condition_on_omega:
            return model(eps, x0, omega)
        return model(eps, x0, None)

    def compute_loss(
        self,
        model: torch.nn.Module,
        x0_batch: torch.Tensor,
        omega_batch: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, BatchStatistics]:
        if self.negative_m < 2:
            raise ValueError("Drifting objective requires at least two generated samples per context.")
        batch_size = x0_batch.shape[0]
        eps = torch.randn(
            batch_size * self.negative_m,
            model.eps_dim,
            generator=self.generator,
            device=self.device,
            dtype=torch.float32,
        )
        expanded_x0 = x0_batch.repeat_interleave(self.negative_m, dim=0)
        expanded_omega = omega_batch.repeat_interleave(self.negative_m, dim=0)
        zeta_generated = self._forward_generator(model, eps, expanded_x0, expanded_omega).reshape(batch_size, self.negative_m, -1)
        z_generated = self._normalized_zeta(zeta_generated)

        alpha_logits, y_pool, states_pool, actions_pool = self._context_knn(x0_batch)
        data_distance_sq = torch.sum((z_generated.unsqueeze(2) - y_pool.unsqueeze(1)).square(), dim=-1)
        log_kernel_data = self._kernel_log_weights(data_distance_sq)
        log_positive_weights = alpha_logits.unsqueeze(1) + log_kernel_data
        effective_beta = float(beta) if self.method.use_cost_tilt else 0.0
        if self.method.use_cost_tilt:
            relabeled_cost = trajectory_cost_torch(states_pool, actions_pool, omega_batch.unsqueeze(1))
            normalized_relabeled_cost = self._normalize_pool_costs(relabeled_cost)
            log_positive_weights = log_positive_weights - effective_beta * normalized_relabeled_cost.unsqueeze(1)
        positive_weights = torch.softmax(log_positive_weights, dim=-1)
        positive_mean = torch.sum(positive_weights.unsqueeze(-1) * y_pool.unsqueeze(1), dim=2)
        positive_field = positive_mean - z_generated

        model_distance_sq = torch.sum((z_generated.unsqueeze(2) - z_generated.unsqueeze(1)).square(), dim=-1)
        log_kernel_model = self._kernel_log_weights(model_distance_sq)
        diagonal_mask = torch.eye(self.negative_m, dtype=torch.bool, device=self.device).unsqueeze(0)
        log_kernel_model = log_kernel_model.masked_fill(diagonal_mask, float("-inf"))
        negative_weights = torch.softmax(log_kernel_model, dim=-1)
        negative_mean = torch.matmul(negative_weights, z_generated)
        negative_field = negative_mean - z_generated

        target = (z_generated + positive_field - negative_field).detach()
        loss = self.drift_weight * 0.5 * (z_generated - target).square().sum(dim=-1).mean()

        consistency_value = 0.0
        if self.dynamics_model is not None:
            from drifting_mpc.data.trajectory import decode_trajectory_torch
            states_gen, actions_gen = decode_trajectory_torch(
                expanded_x0,
                zeta_generated.reshape(batch_size * self.negative_m, -1),
                horizon=self.spec.horizon,
                state_dim=self.spec.state_dim,
                action_dim=self.spec.action_dim,
            )
            if self.consistency_mode == "rollout":
                rolled = [states_gen[:, 0]]
                current = states_gen[:, 0]
                for t in range(actions_gen.shape[1]):
                    current = self.dynamics_model(current, actions_gen[:, t])
                    rolled.append(current)
                rolled = torch.stack(rolled, dim=1)
                resid = (states_gen[:, 1:] - rolled[:, 1:]) / self.state_scale
                consistency = resid.square().sum(dim=(-1, -2)).mean()
            else:
                predicted_next = self.dynamics_model(states_gen[:, :-1], actions_gen)
                resid = (states_gen[:, 1:] - predicted_next) / self.state_scale
                consistency = resid.square().sum(dim=(-1, -2)).mean()
            loss = loss + self.consistency_weight * consistency
            consistency_value = float(consistency.detach().cpu())
        stats = BatchStatistics(
            loss=float(loss.detach().cpu()),
            positive_norm=float(positive_field.norm(dim=-1).mean().detach().cpu()),
            negative_norm=float(negative_field.norm(dim=-1).mean().detach().cpu()),
            beta=effective_beta,
        )
        return loss, stats



def beta_schedule(global_step: int, total_steps: int, beta_max: float, ramp_fraction: float) -> float:
    if total_steps <= 0:
        return beta_max
    ramp_steps = max(1, int(total_steps * ramp_fraction))
    progress = min(global_step / float(ramp_steps), 1.0)
    return beta_max * progress
