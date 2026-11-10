from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drifting_mpc.config import clone_config, load_config
from drifting_mpc.training.trainer import train_model
from drifting_mpc.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Drift RHC generator on an offline dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Optional override for dataset.output_dir.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional fixed output directory for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = clone_config(load_config(args.config))
    set_global_seed(int(config["experiment"]["seed"]))
    if args.dataset_dir is not None:
        config["dataset"]["output_dir"] = args.dataset_dir
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    outputs = train_model(config, dataset_dir=args.dataset_dir, output_dir=args.output_dir)
    print(f"run_dir={outputs.run_dir}")
    print(f"checkpoint={outputs.checkpoint_path}")
    print(f"last_checkpoint={outputs.last_checkpoint_path}")
    print(f"history={outputs.history_path}")
    print(f"plot={outputs.plot_path}")


if __name__ == "__main__":
    main()
