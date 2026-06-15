"""
Generate synthetic lattice structure data for GNN training.

This script creates simplified cubic lattice structures with varying properties
and labels them with a "stability" score based on their structural characteristics.
"""

import numpy as np
import networkx as nx
import pickle
import os
from pathlib import Path


def create_cubic_lattice(size):
    """
    Create a simple cubic lattice of given size.
    
    Args:
        size (int): Number of atoms along each dimension (creates size^3 atoms total)
    
    Returns:
        G (nx.Graph): NetworkX graph representing the lattice
    """
    G = nx.Graph()
    
    positions = {}
    node_id = 0
    
    # Create nodes and positions
    for i in range(size):
        for j in range(size):
            for k in range(size):
                positions[node_id] = np.array([i, j, k], dtype=np.float32)
                # Random node feature: "atom type" (0 or 1)
                atom_type = np.random.choice([0, 1], p=[0.6, 0.4])
                G.add_node(node_id, atom_type=atom_type, pos=positions[node_id])
                node_id += 1
    
    # Create edges (nearest neighbors)
    for n1 in G.nodes():
        pos1 = positions[n1]
        for n2 in G.nodes():
            if n1 < n2:  # Avoid duplicate edges
                pos2 = positions[n2]
                dist = np.linalg.norm(pos1 - pos2)
                
                # Connect nearest neighbors (distance ≈ 1 for cubic lattice)
                if 0.9 < dist < 1.1:
                    # Random bond strength (0-1)
                    bond_strength = np.random.uniform(0.7, 1.0)
                    G.add_edge(n1, n2, bond_strength=bond_strength)
    
    return G


def compute_stability_label(G):
    """
    Compute a stability label for the lattice structure.
    
    Stability is based on:
    - Average node degree (more connections = more stable)
    - Average bond strength
    - Graph connectivity
    
    Args:
        G (nx.Graph): The lattice structure graph
        
    Returns:
        stability (float): Value between 0 and 1
    """
    if len(G) == 0:
        return 0.0
    
    # Average degree
    avg_degree = 2 * len(G.edges()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
    degree_score = min(avg_degree / 6.0, 1.0)  # Normalized by max expected degree
    
    # Average bond strength
    if len(G.edges()) > 0:
        avg_bond_strength = np.mean([data['bond_strength'] for _, _, data in G.edges(data=True)])
    else:
        avg_bond_strength = 0.0
    
    # Connectivity (is graph connected?)
    connectivity_score = 1.0 if nx.is_connected(G) else 0.5
    
    # Combine scores
    stability = (0.4 * degree_score + 0.4 * avg_bond_strength + 0.2 * connectivity_score)
    
    # Add small noise
    stability += np.random.normal(0, 0.05)
    stability = np.clip(stability, 0.0, 1.0)
    
    return stability


def generate_dataset(num_samples=100, lattice_sizes=[2, 3, 4], output_dir='data'):
    """
    Generate a dataset of lattice structures with stability labels.
    
    Args:
        num_samples (int): Total number of samples to generate
        lattice_sizes (list): Possible lattice sizes to randomly sample from
        output_dir (str): Directory to save the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = []
    
    print(f"Generating {num_samples} lattice structures...")
    for i in range(num_samples):
        # Randomly choose lattice size
        size = np.random.choice(lattice_sizes)
        
        # Create lattice
        G = create_cubic_lattice(size)
        
        # Compute stability label
        stability = compute_stability_label(G)
        
        # Store graph and label
        dataset.append({
            'graph': G,
            'stability': stability,
            'size': size,
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges())
        })
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
    
    # Save dataset
    output_path = os.path.join(output_dir, 'lattice_dataset.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Avg nodes per structure: {np.mean([d['num_nodes'] for d in dataset]):.1f}")
    print(f"  Avg edges per structure: {np.mean([d['num_edges'] for d in dataset]):.1f}")
    print(f"  Stability range: [{min(d['stability'] for d in dataset):.3f}, {max(d['stability'] for d in dataset):.3f}]")
    
    return dataset


if __name__ == '__main__':
    # Generate dataset
    dataset = generate_dataset(num_samples=100, lattice_sizes=[2, 3, 4])
