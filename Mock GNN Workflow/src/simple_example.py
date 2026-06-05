"""
Simple example demonstrating the complete GNN workflow in ~50 lines.

Run this after installing dependencies to see a minimal end-to-end example.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np

# Import our modules
from data.generate_lattice_data import create_cubic_lattice, compute_stability_label
from src.gnn_model import GraphConvolutionalNetwork
from src.train import convert_nx_to_pytorch_geometric


def main():
    print("\n" + "="*60)
    print("GNN LATTICE EXAMPLE - End-to-End Workflow")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Step 1: Create a simple lattice structure
    print("[1/4] Creating a 3×3×3 cubic lattice...")
    lattice = create_cubic_lattice(size=3)
    print(f"      Created lattice with {len(lattice.nodes())} atoms and {len(lattice.edges())} bonds\n")
    
    # Step 2: Compute ground truth label
    print("[2/4] Computing stability label...")
    true_stability = compute_stability_label(lattice)
    print(f"      True stability: {true_stability:.4f}\n")
    
    # Step 3: Convert to PyTorch format
    print("[3/4] Converting to PyTorch Geometric format...")
    data = convert_nx_to_pytorch_geometric(lattice)
    data.y = torch.tensor([true_stability], dtype=torch.float)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    print(f"      Node features shape: {data.x.shape}")
    print(f"      Edge index shape: {data.edge_index.shape}")
    print(f"      Label: {data.y.item():.4f}\n")
    
    # Step 4: Make prediction with untrained model
    print("[4/4] Making prediction with random model...")
    model = GraphConvolutionalNetwork(
        node_feature_dim=2,
        hidden_dim=32,
        num_layers=3,
        output_dim=1,
        global_feature_dim=5
    ).to(device)
    
    data = data.to(device)
    with torch.no_grad():
        prediction = model(data)
    
    pred_stability = prediction.item()
    print(f"      Predicted stability (random model): {pred_stability:.4f}")
    print(f"      True stability: {true_stability:.4f}")
    print(f"      Error: {abs(pred_stability - true_stability):.4f}\n")
    
    print("="*60)
    print("Example complete! Model can now be trained to improve predictions.")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("  1. Generate dataset: python data/generate_lattice_data.py")
    print("  2. Train model: cd src && python train.py")
    print("  3. Run inference: python src/inference.py")
    print()


if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
