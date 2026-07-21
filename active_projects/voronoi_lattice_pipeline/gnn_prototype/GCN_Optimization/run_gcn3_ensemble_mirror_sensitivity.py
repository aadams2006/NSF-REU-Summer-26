from __future__ import annotations

import argparse
from pathlib import Path

from mirror_sensitivity_runner import MirrorSensitivityConfig, run_mirror_sensitivity_experiment


MODULE_DIR = Path(__file__).resolve().parent
GNN_ROOT = MODULE_DIR.parent
PIPELINE_ROOT = GNN_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure final GCN-3 ensemble sensitivity to mirrored lattices.")
    parser.add_argument("--ensemble-run-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.ensemble_run_dir is None:
        latest_path = GNN_ROOT / "outputs" / "gcn3_ensemble_uncertainty" / "latest_run.txt"
        if not latest_path.is_file():
            raise FileNotFoundError("Pass --ensemble-run-dir or provide the final ensemble latest_run.txt file")
        ensemble_run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
        if not ensemble_run_dir.is_absolute():
            ensemble_run_dir = Path.cwd() / ensemble_run_dir
    else:
        ensemble_run_dir = args.ensemble_run_dir
    config = MirrorSensitivityConfig(resume=not args.fresh)
    run_mirror_sensitivity_experiment(
        config,
        ensemble_run_dir=ensemble_run_dir,
        train_root=PIPELINE_ROOT / "source_archives" / "lattice_data" / "Randomness_Sweep",
        predict_root=PIPELINE_ROOT / "datasets" / "Lattice_Guess_Prediction_Input_Data",
        output_root=GNN_ROOT / "outputs" / config.output_group,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
