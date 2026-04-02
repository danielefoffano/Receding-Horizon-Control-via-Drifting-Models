from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drifting_mpc.config import clone_config, load_config
from drifting_mpc.data.collection import collect_offline_dataset
from drifting_mpc.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the offline MSD dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Optional override for dataset.output_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = clone_config(load_config(args.config))
    set_global_seed(int(config["experiment"]["seed"]))
    if args.dataset_dir is not None:
        config["dataset"]["output_dir"] = args.dataset_dir
    outputs = collect_offline_dataset(config, dataset_dir=args.dataset_dir)
    print(f"dataset_dir={outputs['dataset_dir']}")
    print(f"train={outputs['train']}")
    print(f"val={outputs['val']}")
    print(f"test={outputs['test']}")
    print(f"manifest={outputs['manifest']}")


if __name__ == "__main__":
    main()
