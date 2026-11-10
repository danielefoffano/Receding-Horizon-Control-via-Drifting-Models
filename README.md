# Drift RHC — Receding Horizon Control via Drifting Models

Reference implementation of **Drift RHC**: a cost-conditioned trajectory
generator `zeta = G(eps, x0, omega)` trained offline with the anti-symmetric
drifting objective. At control time the planner samples `M` plans for the
current state and cost, scores them with the known cost, applies the first
action of the best one and re-plans. Two variants are included:

1. **Drift RHC** — the drifting objective alone.
2. **Drift RHC + consistency** — plus a penalty on the generated states for
   deviating from a frozen one-step dynamics model (`consistency_weight > 0`).
   The dynamics model is used at training time only; inference stays model-free.

Three environments are included (`environment.system`): `msd`
(mass-spring-damper, 2-D state, exact LQR reference), `cartpole` (4-D, iLQR
reference) and `quadrotor` (planar, 6-D, iLQR reference). Baselines and the
figure/table scripts are not part of this package.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.14, NumPy 2.5, SciPy 1.18. A GPU is optional
(`experiment.device: auto | cpu | cuda`). Run every command below from this
directory; the scripts put it on `sys.path` themselves.

## Example commands

Each config names its own dataset, run and evaluation directories, so the
pipeline is four commands. Dataset collection is shared by all configs for the
same system and horizon.

```bash
# 1. offline dataset
python scripts/collect_dataset.py --config configs/cost_aware_logz_H30.yaml

# 2. train Drift RHC
python scripts/train_model.py --config configs/cost_aware_logz_H30.yaml

# 3. closed-loop evaluation against the reference controller
python scripts/eval_model.py --config configs/cost_aware_logz_H30.yaml \
    --ckpt artifacts/runs/cost_aware_logz_H30/<timestamp>/last.pt
```

For the consistency variant, fit the dynamics model first (the configs expect it
at `artifacts/dynamics/<system>_H<horizon>/dynamics.pt`):

```bash
python scripts/train_dynamics_model.py --config configs/msd_nrmcons30_H30.yaml \
    --dataset-dir artifacts/datasets/msd_H30 --output-dir artifacts/dynamics/msd_H30
python scripts/train_model.py --config configs/msd_nrmcons30_H30.yaml
```

Cart-pole and the quadrotor work the same way, e.g.
`configs/cartpole_tuned_logz_b3_H30.yaml` and
`configs/cartpole_tuned_roll10_H30.yaml` (with consistency).

## Outputs

`train_model.py` prints its run directory and writes `last.pt`, `best.pt`,
`history.json`, `loss_curves.png`, `resolved_config.yaml` and
`normalization.npz` into a timestamped sub-directory of `training.output_dir`.
Checkpoints carry their own config and normaliser. `best.pt` is identical to
`last.pt` by design (todo: remove one of them).

`eval_model.py` writes `metrics.json` (mean/median cost, normalised regret,
fraction within 10%/25% of the reference, planning time), `episode_metrics.csv`,
`cost_histogram.pdf`, `cost_scatter.pdf` and a few `rollout_###.pdf` into
`evaluation.output_dir`.
