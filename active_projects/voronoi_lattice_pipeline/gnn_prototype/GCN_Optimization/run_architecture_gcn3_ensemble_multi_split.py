from __future__ import annotations

import argparse
from pathlib import Path

from ensemble_runner import MultiSplitEnsembleConfig, run_multi_split_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resumable GCN-3 multi-split ensemble.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Stable output directory used for checkpoints and resume. A timestamped directory is used if omitted.",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints already present in --run-dir.")
    args = parser.parse_args()

    config = MultiSplitEnsembleConfig(
        architecture_name="gcn3",
        architecture_label="GCN-3",
        member_seeds=(11, 42, 73, 101, 202),
        split_seeds=(11, 42, 73, 101, 202),
        hidden_dim=24,
        dropout=0.10,
        lr_phase1=0.003,
        lr_phase2=0.0005,
        patience=150,
        weight_decay=1e-5,
        checkpoint_interval=50,
        resume=not args.fresh,
        output_group="gcn3_ensemble_multi_split",
    )
    run_multi_split_ensemble(config, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
