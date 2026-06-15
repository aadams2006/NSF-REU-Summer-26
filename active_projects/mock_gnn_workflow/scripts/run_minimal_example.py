"""
Minimal end-to-end example for the mock GNN workflow.
"""

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from data.generate_lattice_data import create_cubic_lattice, compute_stability_label
from gnn_model import GraphConvolutionalNetwork
from train import convert_nx_to_pytorch_geometric


def main():
    print("\n" + "=" * 60)
    print("GNN LATTICE EXAMPLE - END-TO-END WORKFLOW")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("[1/4] Creating a 3x3x3 cubic lattice...")
    lattice = create_cubic_lattice(size=3)
    print(f"      Created lattice with {len(lattice.nodes())} atoms and {len(lattice.edges())} bonds\n")

    print("[2/4] Computing stability label...")
    true_stability = compute_stability_label(lattice)
    print(f"      True stability: {true_stability:.4f}\n")

    print("[3/4] Converting to PyTorch Geometric format...")
    data = convert_nx_to_pytorch_geometric(lattice)
    data.y = torch.tensor([true_stability], dtype=torch.float)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    print(f"      Node features shape: {data.x.shape}")
    print(f"      Edge index shape: {data.edge_index.shape}")
    print(f"      Label: {data.y.item():.4f}\n")

    print("[4/4] Making prediction with random model...")
    model = GraphConvolutionalNetwork(
        node_feature_dim=2,
        hidden_dim=32,
        num_layers=3,
        output_dim=1,
        global_feature_dim=5,
    ).to(device)

    data = data.to(device)
    with torch.no_grad():
        prediction = model(data)

    pred_stability = prediction.item()
    print(f"      Predicted stability (random model): {pred_stability:.4f}")
    print(f"      True stability: {true_stability:.4f}")
    print(f"      Error: {abs(pred_stability - true_stability):.4f}\n")

    print("=" * 60)
    print("Example complete. Train the model for meaningful predictions.")
    print("=" * 60 + "\n")

    print("Next steps:")
    print("  1. Generate dataset: python data/generate_lattice_data.py")
    print("  2. Train model: python src/train.py")
    print("  3. Run inference: python scripts/run_inference_examples.py")


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        print(f"Error: Missing dependency - {error}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
