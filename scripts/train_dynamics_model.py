"""Train the one-step dynamics model used by the Drift RHC consistency term.

The model is fit on the same offline dataset, then frozen: training a Drift RHC
generator with consistency_weight > 0 loads this checkpoint and never updates
it, and inference stays model-free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from drifting_mpc.config import clone_config, load_config
from drifting_mpc.data.dataset import load_split
from drifting_mpc.envs.registry import build_spec
from drifting_mpc.models.dynamics_model import ResidualDynamicsModel
from drifting_mpc.utils.seeding import set_global_seed


def transitions(split):
    """(N,H+1,dx),(N,H,du) -> flat (M,dx),(M,du),(M,dx)."""
    x = split.states[:, :-1].reshape(-1, split.states.shape[-1])
    u = split.actions.reshape(-1, split.actions.shape[-1])
    xn = split.states[:, 1:].reshape(-1, split.states.shape[-1])
    return (torch.as_tensor(x), torch.as_tensor(u), torch.as_tensor(xn))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    args = p.parse_args()

    config = clone_config(load_config(args.config))
    set_global_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = build_spec(config)
    root = Path(args.dataset_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    xtr, utr, xntr = [t.to(device) for t in transitions(load_split(root / "train.npz"))]
    xva, uva, xnva = [t.to(device) for t in transitions(load_split(root / "val.npz"))]
    print(f"train transitions {xtr.shape[0]:,}   val {xva.shape[0]:,}")

    model = ResidualDynamicsModel(spec.state_dim, spec.action_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    n = xtr.shape[0]
    history = {"train": [], "val": []}
    best = float("inf")

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = loss_fn(model(xtr[idx], utr[idx]), xntr[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xva, uva), xnva))
        history["train"].append(tot / n); history["val"].append(vl)
        if vl < best:   # a genuine held-out prediction error, so argmin is valid here
            best = vl
            torch.save({"model_state": model.state_dict(), "config": config,
                        "state_dim": spec.state_dim, "action_dim": spec.action_dim,
                        "hidden_dim": args.hidden_dim, "val_mse": vl, "epoch": epoch},
                       out / "dynamics.pt")
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch+1:>4}/{args.epochs}  train {tot/n:.3e}  val {vl:.3e}")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val MSE {best:.3e}  ->  {out/'dynamics.pt'}")


if __name__ == "__main__":
    main()
