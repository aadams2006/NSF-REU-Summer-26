"""
Inference script demonstrating how to use the trained GNN model.

This script shows how to:
1. Load a trained model
2. Create new lattice structures
3. Make predictions on new data
4. Interpret the predictions
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch_geometric.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'results', 'best_model.pt')

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from gnn_model import GraphConvolutionalNetwork
from train import convert_nx_to_pytorch_geometric
from data.generate_lattice_data import create_cubic_lattice


def load_trained_model(model_path=DEFAULT_MODEL_PATH, device='cpu'):
    """
    Load a trained GNN model.
    
    Args:
        model_path (str): Path to saved model weights
        device: torch device
        
    Returns:
        model: Loaded model
    """
    model = GraphConvolutionalNetwork(
        node_feature_dim=1,
        hidden_dim=64,
        num_layers=3,
        output_dim=1,
        dropout=0.2
    )
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"✗ Model file not found at {model_path}")
        print("  Please train the model first: python src/train.py")
        return None
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_stability(model, nx_graph, device='cpu'):
    """
    Predict stability of a lattice structure.
    
    Args:
        model: Trained GNN model
        nx_graph: NetworkX graph of the lattice
        device: torch device
        
    Returns:
        stability (float): Predicted stability score (0-1)
    """
    # Convert to PyTorch Geometric format
    data = convert_nx_to_pytorch_geometric(nx_graph)
    data = data.to(device)
    
    # Create batch for single graph
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    
    with torch.no_grad():
        prediction = model(data)
    
    stability = prediction.item()
    return np.clip(stability, 0.0, 1.0)


def compare_structures():
    """
    Compare stability predictions for different lattice structures.
    """
    print("\n" + "=" * 80)
    print("LATTICE STRUCTURE STABILITY COMPARISON")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_trained_model(device=device)
    if model is None:
        return
    
    print("\nGenerating and analyzing different lattice structures...\n")
    print(f"{'Size':<10}{'Nodes':<10}{'Edges':<10}{'Predicted Stability':<20}{'Quality'}")
    print("-" * 70)
    
    structures = []
    
    # Test different lattice sizes
    for size in [2, 3, 4]:
        for trial in range(3):
            lattice = create_cubic_lattice(size)
            stability = predict_stability(model, lattice, device)
            
            # Quality assessment
            if stability < 0.4:
                quality = "Weak"
            elif stability < 0.6:
                quality = "Moderate"
            elif stability < 0.8:
                quality = "Good"
            else:
                quality = "Excellent"
            
            print(f"{size}x{size}x{size}  {len(lattice.nodes()):<10}{len(lattice.edges()):<10}{stability:<20.4f}{quality}")
            
            structures.append({
                'size': size,
                'stability': stability,
                'nodes': len(lattice.nodes()),
                'edges': len(lattice.edges()),
                'graph': lattice
            })
    
    print("-" * 70)
    
    # Statistics
    stabilities = [s['stability'] for s in structures]
    print(f"\nStatistics:")
    print(f"  Mean stability:   {np.mean(stabilities):.4f}")
    print(f"  Std stability:    {np.std(stabilities):.4f}")
    print(f"  Min stability:    {np.min(stabilities):.4f}")
    print(f"  Max stability:    {np.max(stabilities):.4f}")


def analyze_single_structure():
    """
    Detailed analysis of a single lattice structure.
    """
    print("\n" + "=" * 80)
    print("DETAILED STRUCTURE ANALYSIS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_trained_model(device=device)
    if model is None:
        return
    
    # Create a 3x3x3 lattice
    print("Creating a 3×3×3 cubic lattice...")
    lattice = create_cubic_lattice(3)
    
    # Get statistics
    print(f"\nStructure Information:")
    print(f"  Number of atoms:     {len(lattice.nodes())}")
    print(f"  Number of bonds:     {len(lattice.edges())}")
    print(f"  Average degree:      {2 * len(lattice.edges()) / len(lattice.nodes()):.2f}")
    print(f"  Is connected:        {True if __import__('networkx').is_connected(lattice) else False}")
    
    # Get atom type distribution
    atom_types = [lattice.nodes[n]['atom_type'] for n in lattice.nodes()]
    print(f"  Atom type 0:         {atom_types.count(0)} atoms")
    print(f"  Atom type 1:         {atom_types.count(1)} atoms")
    
    # Predict stability
    stability = predict_stability(model, lattice, device)
    
    print(f"\nPredicted Stability:  {stability:.4f}")
    print(f"Confidence Range:     [{max(0, stability-0.1):.4f}, {min(1, stability+0.1):.4f}]")
    
    # Interpretation
    if stability < 0.3:
        interpretation = "Very weak structure - likely to fail"
    elif stability < 0.5:
        interpretation = "Weak structure - needs reinforcement"
    elif stability < 0.7:
        interpretation = "Moderate structure - acceptable for most applications"
    elif stability < 0.85:
        interpretation = "Strong structure - suitable for demanding applications"
    else:
        interpretation = "Very strong structure - excellent properties"
    
    print(f"Interpretation:       {interpretation}")


def batch_prediction():
    """
    Demonstrate batch prediction on multiple structures.
    """
    print("\n" + "=" * 80)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_trained_model(device=device)
    if model is None:
        return
    
    print("Generating 10 random lattice structures and making batch predictions...\n")
    
    # Create batch of structures
    data_list = []
    metadata = []
    
    for i in range(10):
        size = np.random.choice([2, 3, 4])
        lattice = create_cubic_lattice(size)
        data = convert_nx_to_pytorch_geometric(lattice)
        data_list.append(data)
        metadata.append({
            'id': i,
            'size': size,
            'nodes': len(lattice.nodes()),
            'edges': len(lattice.edges())
        })
    
    # Batch prediction
    batch_loader = DataLoader(data_list, batch_size=4, shuffle=False)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(batch_loader):
            batch = batch.to(device)
            predictions = model(batch)
            all_predictions.extend(predictions.cpu().numpy().flatten())
    
    # Print results
    print(f"{'ID':<5}{'Size':<10}{'Nodes':<8}{'Edges':<8}{'Predicted Stability':<20}")
    print("-" * 51)
    
    for pred, meta in zip(all_predictions, metadata):
        print(f"{meta['id']:<5}{meta['size']:<10}{meta['nodes']:<8}{meta['edges']:<8}{pred:<20.4f}")
    
    print("-" * 51)
    print(f"\nAverage predicted stability: {np.mean(all_predictions):.4f}")


def main():
    """
    Run all inference examples.
    """
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "GNN LATTICE STABILITY PREDICTION - INFERENCE EXAMPLES".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run examples
    compare_structures()
    analyze_single_structure()
    batch_prediction()
    
    print("\n" + "=" * 80)
    print("Inference examples complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
