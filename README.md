# Drifting MPC on Mass-Spring-Damper

This repository implements four offline trajectory-generation baselines for the mass-spring-damper system under unknown dynamics:

1. **cost_aware**: The generator is conditioned on `(x0, omega)` and the positive field uses exponential cost tilting.
2. **behavior_prior**: the generator is conditioned only on `x0`; the query cost `omega` is used only at evaluation time to score sampled trajectories.
3. **diffusion_behavior_prior**: a diffusion baseline conditioned only on `x0`; it does **not** see `omega` during training and uses unguided diffusion sampling at test time before scoring candidates with the query cost.
4. **guided_diffusion_behavior_prior**: the same `x0`-conditioned diffusion prior, but sampled with Diffuser-style reward guidance using the closed-form gradient of the cumulative reward (equivalently, negative cost) with respect to the predicted denoised trajectory.

## Example commands

### Cost-aware drifting

```bash
python3 scripts/collect_dataset.py --config configs/msd_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run
python3 scripts/train_model.py --config configs/msd_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --output-dir artifacts/runs/msd_cost_aware_smoke_run
python3 scripts/eval_model.py --config configs/msd_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --ckpt artifacts/runs/msd_cost_aware_smoke_run/best.pt --output-dir artifacts/eval/msd_cost_aware_smoke_run
```


## Comparing multiple checkpoints

Use `scripts/eval_compare_models.py` to evaluate multiple checkpoints on the same held-out episodes and save combined plots, for example:

```bash
python scripts/eval_compare_models.py \
  --config configs/msd_default.yaml \
  --ckpts path/to/cost_aware.pt path/to/behavior_prior.pt path/to/diffusion_prior.pt \
  --labels cost_aware behavior_prior diffusion_prior \
  --output-dir artifacts/eval/compare_run
```
