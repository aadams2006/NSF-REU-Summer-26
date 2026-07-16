from __future__ import annotations

import argparse
from pathlib import Path

from ood_domain_adaptation_runner import OODDomainAdaptationConfig, run_ood_domain_adaptation_experiment


MODULE_DIR = Path(__file__).resolve().parent
GNN_ROOT = MODULE_DIR.parent
PIPELINE_ROOT = GNN_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune GCN-3 using existing prediction-like source lattices and an untouched OOD holdout."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--adaptation-repeats", type=int, default=4)
    args = parser.parse_args()
    config = OODDomainAdaptationConfig(
        finetune_lr=args.finetune_lr,
        adaptation_repeats=args.adaptation_repeats,
        resume=not args.fresh,
    )
    run_ood_domain_adaptation_experiment(
        config,
        train_root=PIPELINE_ROOT / "source_archives" / "lattice_data" / "Randomness_Sweep",
        predict_root=PIPELINE_ROOT / "datasets" / "Lattice_Guess_Prediction_Input_Data",
        output_root=GNN_ROOT / "outputs" / config.output_group,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
