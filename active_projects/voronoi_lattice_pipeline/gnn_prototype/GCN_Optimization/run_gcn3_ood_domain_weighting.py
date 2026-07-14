from __future__ import annotations

import argparse
from pathlib import Path

from ood_validation_runner import OODExperimentConfig, run_ood_experiment


MODULE_DIR = Path(__file__).resolve().parent
GNN_ROOT = MODULE_DIR.parent
PIPELINE_ROOT = GNN_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare control and domain-weighted GCN-3 on an OOD-style holdout.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--weight-strength", type=float, default=2.0)
    args = parser.parse_args()
    config = OODExperimentConfig(
        variants=("control", "domain_weighted"),
        weight_strength=args.weight_strength,
        resume=not args.fresh,
    )
    run_ood_experiment(
        config,
        train_root=PIPELINE_ROOT / "source_archives" / "lattice_data" / "Randomness_Sweep",
        predict_root=PIPELINE_ROOT / "datasets" / "Lattice_Guess_Prediction_Input_Data",
        output_root=GNN_ROOT / "outputs" / config.output_group,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
