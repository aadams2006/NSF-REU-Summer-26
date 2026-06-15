"""
Inference examples for the mock lattice-stability GNN.
"""

import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "best_model.pt")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data.generate_lattice_data import create_cubic_lattice
from gnn_model import GraphConvolutionalNetwork
from train import convert_nx_to_pytorch_geometric


def load_trained_model(model_path=DEFAULT_MODEL_PATH, device="cpu"):
    model = GraphConvolutionalNetwork(
        node_feature_dim=2,
        hidden_dim=64,
        num_layers=3,
        output_dim=1,
        dropout=0.2,
        global_feature_dim=5,
    )

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        print("  Please train the model first: python src/train.py")
        return None

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[OK] Model loaded from {model_path}")
    model = model.to(device)
    model.eval()
    return model


def predict_stability(model, nx_graph, device="cpu"):
    data = convert_nx_to_pytorch_geometric(nx_graph)
    data = data.to(device)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        prediction = model(data)

    return float(np.clip(prediction.item(), 0.0, 1.0))


def compare_structures():
    print("\n" + "=" * 80)
    print("LATTICE STRUCTURE STABILITY COMPARISON")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = load_trained_model(device=device)
    if model is None:
        return

    print(f"{'Size':<10}{'Nodes':<10}{'Edges':<10}{'Predicted Stability':<20}{'Quality'}")
    print("-" * 70)

    structures = []
    for size in [2, 3, 4]:
        for _ in range(3):
            lattice = create_cubic_lattice(size)
            stability = predict_stability(model, lattice, device)

            if stability < 0.4:
                quality = "Weak"
            elif stability < 0.6:
                quality = "Moderate"
            elif stability < 0.8:
                quality = "Good"
            else:
                quality = "Excellent"

            print(f"{size}x{size}x{size}  {len(lattice.nodes()):<10}{len(lattice.edges()):<10}{stability:<20.4f}{quality}")
            structures.append(stability)

    print("-" * 70)
    print("\nStatistics:")
    print(f"  Mean stability:   {np.mean(structures):.4f}")
    print(f"  Std stability:    {np.std(structures):.4f}")
    print(f"  Min stability:    {np.min(structures):.4f}")
    print(f"  Max stability:    {np.max(structures):.4f}")


def analyze_single_structure():
    print("\n" + "=" * 80)
    print("DETAILED STRUCTURE ANALYSIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = load_trained_model(device=device)
    if model is None:
        return

    print("Creating a 3x3x3 cubic lattice...")
    lattice = create_cubic_lattice(3)
    print(f"\nStructure Information:")
    print(f"  Number of atoms:     {len(lattice.nodes())}")
    print(f"  Number of bonds:     {len(lattice.edges())}")
    print(f"  Average degree:      {2 * len(lattice.edges()) / len(lattice.nodes()):.2f}")
    print(f"  Is connected:        {True if __import__('networkx').is_connected(lattice) else False}")

    atom_types = [lattice.nodes[n]["atom_type"] for n in lattice.nodes()]
    print(f"  Atom type 0:         {atom_types.count(0)} atoms")
    print(f"  Atom type 1:         {atom_types.count(1)} atoms")

    stability = predict_stability(model, lattice, device)
    print(f"\nPredicted Stability:  {stability:.4f}")
    print(f"Confidence Range:     [{max(0, stability - 0.1):.4f}, {min(1, stability + 0.1):.4f}]")


def batch_prediction():
    print("\n" + "=" * 80)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = load_trained_model(device=device)
    if model is None:
        return

    data_list = []
    metadata = []
    for index in range(10):
        size = np.random.choice([2, 3, 4])
        lattice = create_cubic_lattice(size)
        data_list.append(convert_nx_to_pytorch_geometric(lattice))
        metadata.append((index, size, len(lattice.nodes()), len(lattice.edges())))

    batch_loader = DataLoader(data_list, batch_size=4, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in batch_loader:
            batch = batch.to(device)
            predictions.extend(model(batch).cpu().numpy().flatten())

    print(f"{'ID':<5}{'Size':<10}{'Nodes':<8}{'Edges':<8}{'Predicted Stability':<20}")
    print("-" * 51)
    for pred, (index, size, nodes, edges) in zip(predictions, metadata):
        print(f"{index:<5}{size:<10}{nodes:<8}{edges:<8}{pred:<20.4f}")
    print("-" * 51)
    print(f"\nAverage predicted stability: {np.mean(predictions):.4f}")


def main():
    compare_structures()
    analyze_single_structure()
    batch_prediction()


if __name__ == "__main__":
    main()
