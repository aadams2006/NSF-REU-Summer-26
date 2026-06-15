"""
Run the full mock GNN workflow from the project root.

Usage:
    python scripts/run_full_workflow.py
"""

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """
    Run the complete pipeline: data generation -> model training -> result output.
    """
    print("=" * 80)
    print("GNN LATTICE STABILITY PREDICTION - QUICK START")
    print("=" * 80)

    print("\n[1/3] Generating synthetic lattice data...")
    print("-" * 80)
    from data.generate_lattice_data import generate_dataset

    generate_dataset(
        num_samples=100,
        lattice_sizes=[2, 3, 4],
        output_dir=str(PROJECT_ROOT / "data"),
    )

    print("\n[2/3] Training GNN model...")
    print("-" * 80)
    original_dir = Path.cwd()
    os.chdir(SRC_DIR)
    from train import main as train_main

    train_main()
    os.chdir(original_dir)

    print("\n[3/3] Training complete!")
    print("-" * 80)
    print("\nResults Summary:")
    print("  - Model saved to: results/best_model.pt")
    print("  - Visualizations saved to the newest results/runs/run_* folder")
    print("  - Dataset saved to: data/lattice_dataset.pkl")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Review README.md and docs/START_HERE.md")
    print("  2. Run scripts/run_inference_examples.py")
    print("  3. Inspect the newest results/runs/run_* directory")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nError: {error}")
        print("\nTroubleshooting:")
        print("  1. Ensure dependencies are installed: pip install -r requirements.txt")
        print("  2. Run this command from the mock_gnn_workflow directory")
        print("  3. Ensure Python 3.8+ is being used")
