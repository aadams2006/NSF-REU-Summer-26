"""
Training and evaluation pipeline for the lattice stability prediction GNN.

This script handles:
- Data loading and conversion to PyTorch Geometric format
- Model training with validation
- Evaluation metrics and visualization
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from gnn_model import GraphConvolutionalNetwork, EdgeFeatureGCN


def convert_nx_to_pytorch_geometric(nx_graph):
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.
    
    Args:
        nx_graph (nx.Graph): NetworkX graph
        
    Returns:
        data (Data): PyTorch Geometric Data object
    """
    import networkx as nx
    
    # Create node feature matrix (1 feature: atom type)
    node_features = []
    node_mapping = {}
    for i, (node, attr) in enumerate(nx_graph.nodes(data=True)):
        node_mapping[node] = i
        node_features.append([attr.get('atom_type', 0)])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge index
    edge_index_list = []
    edge_features = []
    for u, v, attr in nx_graph.edges(data=True):
        u_mapped = node_mapping[u]
        v_mapped = node_mapping[v]
        edge_index_list.append([u_mapped, v_mapped])
        edge_index_list.append([v_mapped, u_mapped])  # Add reverse edge for undirected graph
        bond_strength = attr.get('bond_strength', 0.5)
        edge_features.append([bond_strength])
        edge_features.append([bond_strength])
    
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def load_dataset(dataset_path):
    """
    Load the lattice dataset and convert to PyTorch Geometric format.
    
    Args:
        dataset_path (str): Path to the pickled dataset
        
    Returns:
        data_list (list): List of Data objects
        labels (list): List of stability labels
    """
    with open(dataset_path, 'rb') as f:
        raw_dataset = pickle.load(f)
    
    data_list = []
    labels = []
    
    for item in raw_dataset:
        data = convert_nx_to_pytorch_geometric(item['graph'])
        data_list.append(data)
        labels.append(item['stability'])
    
    return data_list, np.array(labels)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): The GNN model
        train_loader (DataLoader): Training data loader
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: torch device
        
    Returns:
        avg_loss (float): Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)
        y = batch.y.view(-1, 1)
        
        # Compute loss
        loss = criterion(out, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model (nn.Module): The GNN model
        data_loader (DataLoader): Data loader
        criterion: Loss function
        device: torch device
        
    Returns:
        metrics (dict): Dictionary of evaluation metrics
    """
    model.eval()
    predictions = []
    ground_truth = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch)
            y = batch.y.view(-1, 1)
            
            loss = criterion(out, y)
            total_loss += loss.item()
            num_batches += 1
            
            predictions.extend(out.cpu().numpy().flatten())
            ground_truth.extend(y.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'mse': mean_squared_error(ground_truth, predictions),
        'mae': mean_absolute_error(ground_truth, predictions),
        'rmse': np.sqrt(mean_squared_error(ground_truth, predictions)),
        'r2': r2_score(ground_truth, predictions),
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return metrics


def train_model(model, train_loader, val_loader, test_loader, epochs=50, lr=0.001, 
                device='cpu', model_save_path='model.pt'):
    """
    Full training loop with validation.
    
    Args:
        model (nn.Module): The GNN model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        epochs (int): Number of epochs
        lr (float): Learning rate
        device: torch device
        model_save_path (str): Path to save best model
        
    Returns:
        history (dict): Training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_r2': []
    }
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("Training GNN model...")
    print("-" * 80)
    print(f"{'Epoch':<10}{'Train Loss':<15}{'Val Loss':<15}{'Val MAE':<15}{'Val R²':<15}")
    print("-" * 80)
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        
        print(f"{epoch+1:<10}{train_loss:<15.6f}{val_loss:<15.6f}{val_metrics['mae']:<15.6f}{val_metrics['r2']:<15.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("-" * 80)
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Set Performance:")
    print(f"  MSE:  {test_metrics['mse']:.6f}")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.6f}")
    
    history['test_metrics'] = test_metrics
    
    return model, history


def plot_results(history, output_dir='results'):
    """
    Plot training history and evaluation results.
    
    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Training and validation loss
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE curves
    axes[0, 1].plot(history['val_mae'], label='Val MAE', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # R² curves
    axes[1, 0].plot(history['val_r2'], label='Val R²', color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Validation R² Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Predictions vs Ground Truth
    if 'test_metrics' in history:
        test_metrics = history['test_metrics']
        axes[1, 1].scatter(test_metrics['ground_truth'], test_metrics['predictions'], 
                          alpha=0.6, s=50, edgecolors='k')
        # Add perfect prediction line
        min_val = min(test_metrics['ground_truth'].min(), test_metrics['predictions'].min())
        max_val = max(test_metrics['ground_truth'].max(), test_metrics['predictions'].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 1].set_xlabel('Ground Truth Stability')
        axes[1, 1].set_ylabel('Predicted Stability')
        axes[1, 1].set_title('Predictions vs Ground Truth (Test Set)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'training_results.png')}")
    plt.close()


def main():
    """
    Main training pipeline.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_path = '../data/lattice_dataset.pkl'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run: python data/generate_lattice_data.py")
        return
    
    print(f"Loading dataset from {dataset_path}...")
    data_list, labels = load_dataset(dataset_path)
    print(f"Loaded {len(data_list)} samples")
    
    # Add labels to data objects
    for data, label in zip(data_list, labels):
        data.y = torch.tensor([label], dtype=torch.float)
    
    # Split dataset: 70% train, 15% val, 15% test
    train_indices, temp_indices = train_test_split(
        range(len(data_list)), test_size=0.3, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42
    )
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    
    # Create model
    model = GraphConvolutionalNetwork(
        node_feature_dim=1,
        hidden_dim=64,
        num_layers=3,
        output_dim=1,
        dropout=0.2
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    os.makedirs('results', exist_ok=True)
    model, history = train_model(
        model, train_loader, val_loader, test_loader,
        epochs=100, lr=0.001, device=device,
        model_save_path='results/best_model.pt'
    )
    
    # Plot results
    plot_results(history)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
