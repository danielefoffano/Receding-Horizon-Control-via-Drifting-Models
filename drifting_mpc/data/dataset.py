from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrajectoryNormalizer:
    """Dataset statistics for whitening flattened relative trajectories."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, zeta: np.ndarray, min_std: float = 1e-6) -> "TrajectoryNormalizer":
        mean = zeta.mean(axis=0).astype(np.float32)
        std = zeta.std(axis=0).astype(np.float32)
        std = np.maximum(std, min_std).astype(np.float32)
        return cls(mean=mean, std=std)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryNormalizer":
        payload = np.load(Path(path).expanduser())
        return cls(mean=payload["mean"].astype(np.float32), std=payload["std"].astype(np.float32))

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, mean=self.mean, std=self.std)
        return target

    def normalize_np(self, zeta: np.ndarray) -> np.ndarray:
        return (np.asarray(zeta, dtype=np.float32) - self.mean) / self.std

    def denormalize_np(self, normalized: np.ndarray) -> np.ndarray:
        return np.asarray(normalized, dtype=np.float32) * self.std + self.mean

    def as_torch(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self.mean, dtype=torch.float32, device=device),
            torch.as_tensor(self.std, dtype=torch.float32, device=device),
        )

    def normalize_torch(self, zeta: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        mean, std = self.as_torch(zeta.device if device is None else device)
        return (zeta - mean) / std

    def denormalize_torch(self, normalized: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        mean, std = self.as_torch(normalized.device if device is None else device)
        return normalized * std + mean


@dataclass
class OfflineTrajectorySplit:
    """Serialized split format used for training, validation, and evaluation."""

    states: np.ndarray
    actions: np.ndarray
    x0: np.ndarray
    omega: np.ndarray
    collection_cost: np.ndarray
    zeta: np.ndarray
    controller_id: np.ndarray

    @property
    def size(self) -> int:
        return int(self.x0.shape[0])

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            states=self.states,
            actions=self.actions,
            x0=self.x0,
            omega=self.omega,
            collection_cost=self.collection_cost,
            zeta=self.zeta,
            controller_id=self.controller_id,
        )
        return target

    def to_torch(self, device: torch.device | str) -> "TorchOfflineTrajectorySplit":
        return TorchOfflineTrajectorySplit(
            states=torch.as_tensor(self.states, dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions, dtype=torch.float32, device=device),
            x0=torch.as_tensor(self.x0, dtype=torch.float32, device=device),
            omega=torch.as_tensor(self.omega, dtype=torch.float32, device=device),
            collection_cost=torch.as_tensor(self.collection_cost, dtype=torch.float32, device=device),
            zeta=torch.as_tensor(self.zeta, dtype=torch.float32, device=device),
            controller_id=torch.as_tensor(self.controller_id, dtype=torch.long, device=device),
        )


@dataclass
class TorchOfflineTrajectorySplit:
    states: torch.Tensor
    actions: torch.Tensor
    x0: torch.Tensor
    omega: torch.Tensor
    collection_cost: torch.Tensor
    zeta: torch.Tensor
    controller_id: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.x0.shape[0])


def load_split(path: str | Path) -> OfflineTrajectorySplit:
    payload = np.load(Path(path).expanduser())
    return OfflineTrajectorySplit(
        states=payload["states"].astype(np.float32),
        actions=payload["actions"].astype(np.float32),
        x0=payload["x0"].astype(np.float32),
        omega=payload["omega"].astype(np.float32),
        collection_cost=payload["collection_cost"].astype(np.float32),
        zeta=payload["zeta"].astype(np.float32),
        controller_id=payload["controller_id"].astype(np.int64),
    )


def save_manifest(path: str | Path, payload: dict) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return target


def load_manifest(path: str | Path) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)
