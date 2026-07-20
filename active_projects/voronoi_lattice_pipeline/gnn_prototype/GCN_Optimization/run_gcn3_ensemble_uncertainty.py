from __future__ import annotations

import argparse
from pathlib import Path

from ensemble_uncertainty_runner import EnsembleUncertaintyConfig, run_ensemble_uncertainty_experiment


MODULE_DIR = Path(__file__).resolve().parent
GNN_ROOT = MODULE_DIR.parent
PIPELINE_ROOT = GNN_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final five-member GCN-3 ensemble with confidence and member-deviation reporting."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    config = EnsembleUncertaintyConfig(
        split_seed=args.split_seed,
        resume=not args.fresh,
    )
    run_ensemble_uncertainty_experiment(
        config,
        train_root=PIPELINE_ROOT / "source_archives" / "lattice_data" / "Randomness_Sweep",
        predict_root=PIPELINE_ROOT / "datasets" / "Lattice_Guess_Prediction_Input_Data",
        output_root=GNN_ROOT / "outputs" / config.output_group,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
