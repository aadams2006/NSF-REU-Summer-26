from __future__ import annotations

import argparse
from pathlib import Path

from residual_error_analysis import ResidualAnalysisConfig, run_residual_error_analysis


MODULE_DIR = Path(__file__).resolve().parent
GNN_ROOT = MODULE_DIR.parent
PIPELINE_ROOT = GNN_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze residual errors from a completed GCN-3 ensemble multi-split run.")
    parser.add_argument(
        "--train-root",
        type=Path,
        default=PIPELINE_ROOT / "source_archives" / "lattice_data" / "Randomness_Sweep",
    )
    parser.add_argument(
        "--predict-root",
        type=Path,
        default=PIPELINE_ROOT / "datasets" / "Lattice_Guess_Prediction_Input_Data",
    )
    parser.add_argument(
        "--ensemble-run-dir",
        type=Path,
        default=GNN_ROOT / "outputs" / "gcn3_ensemble_multi_split" / "run_resumable_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=GNN_ROOT / "outputs" / "gcn3_ensemble_residual_analysis",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Use an exact output directory instead of a timestamped run.")
    parser.add_argument("--worst-case-count", type=int, default=25)
    args = parser.parse_args()

    config = ResidualAnalysisConfig(worst_case_count=args.worst_case_count)
    run_residual_error_analysis(
        config,
        train_root=args.train_root,
        predict_root=args.predict_root,
        ensemble_run_dir=args.ensemble_run_dir,
        output_root=args.output_root,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
