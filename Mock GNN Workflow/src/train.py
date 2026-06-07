"""
Training and evaluation pipeline for the lattice stability prediction GNN.

This script handles:
- Data loading and conversion to PyTorch Geometric format
- Model training with validation
- Evaluation metrics and visualization
"""

import os
import csv
import pickle
import random
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from gnn_model import GraphConvolutionalNetwork, EdgeFeatureGCN

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'lattice_dataset.pkl')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
RUNS_DIR = os.path.join(RESULTS_DIR, 'runs')
RUN_REGISTRY_PATH = os.path.join(RESULTS_DIR, 'run_registry.csv')
LATEST_RUN_PATH = os.path.join(RESULTS_DIR, 'latest_run.txt')
MODEL_PATH = os.path.join(RESULTS_DIR, 'best_model.pt')


def convert_nx_to_pytorch_geometric(nx_graph):
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.
    
    Args:
        nx_graph (nx.Graph): NetworkX graph
        
    Returns:
        data (Data): PyTorch Geometric Data object
    """
    import networkx as nx
    
    # Create node feature matrix using atom type and normalized node degree.
    node_features = []
    node_mapping = {}
    for i, (node, attr) in enumerate(nx_graph.nodes(data=True)):
        node_mapping[node] = i
        normalized_degree = nx_graph.degree[node] / 6.0 if len(nx_graph) > 0 else 0.0
        node_features.append([attr.get('atom_type', 0), normalized_degree])
    
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
    
    num_nodes = len(nx_graph.nodes())
    num_edges = len(nx_graph.edges())
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0
    mean_bond_strength = float(edge_attr.mean().item()) if edge_attr.numel() > 0 else 0.0
    graph_features = torch.tensor([[
        num_nodes / 64.0,
        num_edges / 144.0,
        avg_degree / 6.0,
        mean_bond_strength,
        1.0 if nx.is_connected(nx_graph) else 0.0,
    ]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, graph_features=graph_features)
    return data


def create_run_context():
    """
    Create a unique output directory and metadata for this training run.

    Returns:
        dict: Run metadata including run number, label, timestamps, and paths
    """
    os.makedirs(RUNS_DIR, exist_ok=True)

    existing_run_numbers = []
    for entry in os.listdir(RUNS_DIR):
        entry_path = os.path.join(RUNS_DIR, entry)
        if not os.path.isdir(entry_path) or not entry.startswith('run_'):
            continue

        parts = entry.split('_', 2)
        if len(parts) >= 2 and parts[1].isdigit():
            existing_run_numbers.append(int(parts[1]))

    run_number = max(existing_run_numbers, default=0) + 1
    started_at = datetime.now()
    timestamp_label = started_at.strftime('%Y%m%d_%H%M%S')
    run_label = f"run_{run_number:04d}_{timestamp_label}"
    run_dir = os.path.join(RUNS_DIR, run_label)
    os.makedirs(run_dir, exist_ok=True)

    return {
        'run_number': run_number,
        'run_label': run_label,
        'started_at': started_at,
        'started_at_iso': started_at.isoformat(timespec='seconds'),
        'run_dir': run_dir,
        'model_path': os.path.join(run_dir, 'best_model.pt'),
    }


def save_run_metadata(run_context, history, output_dir, config=None):
    """
    Save run metadata inside the run directory.

    Args:
        run_context (dict): Run metadata
        history (dict): Training history
        output_dir (str): Directory to save metadata
    """
    metadata_path = os.path.join(output_dir, 'run_metadata.csv')
    with open(metadata_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['field', 'value'])
        writer.writerow(['run_number', run_context['run_number']])
        writer.writerow(['run_label', run_context['run_label']])
        writer.writerow(['started_at', run_context['started_at_iso']])
        writer.writerow(['completed_at', run_context['completed_at_iso']])
        writer.writerow(['epochs_trained', len(history['train_loss'])])
        writer.writerow(['best_epoch', history.get('best_epoch', '')])
        writer.writerow(['best_val_loss', history.get('best_val_loss', '')])
        writer.writerow(['model_path', run_context['model_path']])
        if config is not None:
            for key in sorted(config):
                writer.writerow([f'config_{key}', config[key]])

    print(f"CSV saved to {metadata_path}")


def update_run_registry(run_context, history):
    """
    Append the current run to the root-level run registry.

    Args:
        run_context (dict): Run metadata
        history (dict): Training history
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    registry_exists = os.path.exists(RUN_REGISTRY_PATH)

    with open(RUN_REGISTRY_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not registry_exists:
            writer.writerow([
                'run_number',
                'run_label',
                'started_at',
                'completed_at',
                'epochs_trained',
                'best_epoch',
                'best_val_loss',
                'test_loss',
                'test_mse',
                'test_mae',
                'test_rmse',
                'test_r2',
                'run_dir',
                'model_path'
            ])

        test_metrics = history.get('test_metrics', {})
        writer.writerow([
            run_context['run_number'],
            run_context['run_label'],
            run_context['started_at_iso'],
            run_context['completed_at_iso'],
            len(history['train_loss']),
            history.get('best_epoch', ''),
            history.get('best_val_loss', ''),
            test_metrics.get('loss', ''),
            test_metrics.get('mse', ''),
            test_metrics.get('mae', ''),
            test_metrics.get('rmse', ''),
            test_metrics.get('r2', ''),
            run_context['run_dir'],
            run_context['model_path']
        ])


def update_latest_run_pointer(run_context):
    """
    Update a root-level pointer to the latest completed run.

    Args:
        run_context (dict): Run metadata
    """
    with open(LATEST_RUN_PATH, 'w', newline='') as latest_file:
        latest_file.write(f"run_label={run_context['run_label']}\n")
        latest_file.write(f"run_number={run_context['run_number']}\n")
        latest_file.write(f"completed_at={run_context['completed_at_iso']}\n")
        latest_file.write(f"run_dir={run_context['run_dir']}\n")
        latest_file.write(f"model_path={run_context['model_path']}\n")


def sync_latest_model(run_context):
    """
    Copy the latest trained model to the canonical root path for inference.

    Args:
        run_context (dict): Run metadata
    """
    shutil.copyfile(run_context['model_path'], MODEL_PATH)
    print(f"Latest model updated at {MODEL_PATH}")


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


def set_random_seed(seed):
    """
    Set random seeds for reproducible training runs.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(data_list, labels, split_random_state=42):
    """
    Split the dataset into train, validation, and test sets.

    Args:
        data_list (list): Graph data objects
        labels (np.ndarray): Regression labels
        split_random_state (int): Random state for the split

    Returns:
        tuple: train_data, val_data, test_data
    """
    labeled_data = []
    for data, label in zip(data_list, labels):
        data_copy = data.clone()
        data_copy.y = torch.tensor([label], dtype=torch.float)
        labeled_data.append(data_copy)

    train_indices, temp_indices = train_test_split(
        range(len(labeled_data)), test_size=0.3, random_state=split_random_state
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=split_random_state
    )

    train_data = [labeled_data[i] for i in train_indices]
    val_data = [labeled_data[i] for i in val_indices]
    test_data = [labeled_data[i] for i in test_indices]
    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, batch_size=8):
    """
    Create data loaders for the dataset split.

    Args:
        train_data (list): Training set
        val_data (list): Validation set
        test_data (list): Test set
        batch_size (int): Batch size

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_feature_dimensions(data_list):
    """
    Infer node and graph feature dimensions from the converted dataset.
    """
    if not data_list:
        raise ValueError("data_list is empty; cannot infer feature dimensions.")

    sample = data_list[0]
    node_feature_dim = sample.x.shape[1]
    global_feature_dim = sample.graph_features.shape[1] if hasattr(sample, 'graph_features') else 0
    return node_feature_dim, global_feature_dim


def build_model(model_name='gcn', node_feature_dim=1, hidden_dim=64, num_layers=3,
                output_dim=1, dropout=0.2, global_feature_dim=0):
    """
    Build a configured GNN model.

    Args:
        model_name (str): Model type
        node_feature_dim (int): Node feature dimension
        hidden_dim (int): Hidden dimension
        num_layers (int): Number of graph convolution layers
        output_dim (int): Output dimension
        dropout (float): Dropout rate

    Returns:
        nn.Module: Configured model
    """
    if model_name == 'gcn':
        return GraphConvolutionalNetwork(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            global_feature_dim=global_feature_dim
        )
    if model_name == 'edge_gcn':
        return EdgeFeatureGCN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            global_feature_dim=global_feature_dim
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


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
                device='cpu', model_save_path='model.pt', weight_decay=1e-5,
                patience=15, verbose=True):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_r2': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    if verbose:
        print("Training GNN model...")
        print("-" * 80)
        print(f"{'Epoch':<10}{'Train Loss':<15}{'Val Loss':<15}{'Val MAE':<15}{'Val R2':<15}")
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
        
        if verbose:
            print(f"{epoch+1:<10}{train_loss:<15.6f}{val_loss:<15.6f}{val_metrics['mae']:<15.6f}{val_metrics['r2']:<15.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    if verbose:
        print("-" * 80)
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    if verbose:
        print(f"\nTest Set Performance:")
        print(f"  MSE:  {test_metrics['mse']:.6f}")
        print(f"  MAE:  {test_metrics['mae']:.6f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        print(f"  R2:   {test_metrics['r2']:.6f}")
    
    history['test_metrics'] = test_metrics
    history['best_val_loss'] = best_val_loss
    history['best_epoch'] = best_epoch
    
    return model, history


def plot_results(history, output_dir=RESULTS_DIR):
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
    
    # R2 curves
    axes[1, 0].plot(history['val_r2'], label='Val R2', color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R2 Score')
    axes[1, 0].set_title('Validation R2 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Predictions vs Ground Truth
    if 'test_metrics' in history:
        test_metrics = history['test_metrics']
        axes[1, 1].scatter(
            test_metrics['ground_truth'],
            test_metrics['predictions'],
            alpha=0.6,
            s=50,
            edgecolors='k',
            label='Model Predictions'
        )
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


def save_results_csv(history, output_dir=RESULTS_DIR, run_context=None, config=None):
    """
    Save training history and evaluation metrics as CSV files.

    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    history_csv_path = os.path.join(output_dir, 'training_history.csv')
    with open(history_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_r2'])
        for epoch, (train_loss, val_loss, val_mae, val_r2) in enumerate(
            zip(
                history['train_loss'],
                history['val_loss'],
                history['val_mae'],
                history['val_r2']
            ),
            start=1
        ):
            writer.writerow([epoch, train_loss, val_loss, val_mae, val_r2])

    metrics_csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    with open(metrics_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'value'])
        if run_context is not None:
            writer.writerow(['run_number', run_context.get('run_number', '')])
            writer.writerow(['run_label', run_context.get('run_label', '')])
            writer.writerow(['started_at', run_context.get('started_at_iso', '')])
            writer.writerow(['completed_at', run_context.get('completed_at_iso', '')])
        if config is not None:
            for key in sorted(config):
                writer.writerow([f'config_{key}', config[key]])

        writer.writerow(['epochs_trained', len(history['train_loss'])])
        writer.writerow(['best_epoch', history.get('best_epoch', '')])
        writer.writerow(['best_val_loss', history.get('best_val_loss', '')])

        test_metrics = history.get('test_metrics', {})
        for metric_name in ['loss', 'mse', 'mae', 'rmse', 'r2']:
            if metric_name in test_metrics:
                writer.writerow([f'test_{metric_name}', test_metrics[metric_name]])

    if 'test_metrics' in history:
        test_metrics = history['test_metrics']
        predictions_csv_path = os.path.join(output_dir, 'test_predictions.csv')
        with open(predictions_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sample_index', 'ground_truth', 'prediction', 'error', 'absolute_error'])
            for index, (ground_truth, prediction) in enumerate(
                zip(test_metrics['ground_truth'], test_metrics['predictions']),
                start=1
            ):
                error = prediction - ground_truth
                writer.writerow([index, ground_truth, prediction, error, abs(error)])

    print(f"CSV saved to {history_csv_path}")
    print(f"CSV saved to {metrics_csv_path}")
    if 'test_metrics' in history:
        print(f"CSV saved to {os.path.join(output_dir, 'test_predictions.csv')}")


def main():
    """
    Main training pipeline.
    """
    training_config = {
        'model_name': 'gcn',
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.2,
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 15,
        'split_random_state': 42,
        'seed': 42,
    }

    set_random_seed(training_config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    run_context = create_run_context()
    print(f"Run label: {run_context['run_label']}")
    print(f"Run directory: {run_context['run_dir']}")
    
    # Load dataset
    dataset_path = DATASET_PATH
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run: python data/generate_lattice_data.py")
        return
    
    print(f"Loading dataset from {dataset_path}...")
    data_list, labels = load_dataset(dataset_path)
    print(f"Loaded {len(data_list)} samples")
    node_feature_dim, global_feature_dim = get_feature_dimensions(data_list)
    print(f"Node feature dim: {node_feature_dim}, Graph feature dim: {global_feature_dim}")

    train_data, val_data, test_data = split_dataset(
        data_list, labels, split_random_state=training_config['split_random_state']
    )
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size=training_config['batch_size']
    )
    
    # Create model
    model = build_model(
        model_name=training_config['model_name'],
        node_feature_dim=node_feature_dim,
        hidden_dim=training_config['hidden_dim'],
        num_layers=training_config['num_layers'],
        output_dim=1,
        dropout=training_config['dropout'],
        global_feature_dim=global_feature_dim
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, history = train_model(
        model, train_loader, val_loader, test_loader,
        epochs=training_config['epochs'],
        lr=training_config['learning_rate'],
        device=device,
        model_save_path=run_context['model_path'],
        weight_decay=training_config['weight_decay'],
        patience=training_config['patience'],
        verbose=True
    )

    completed_at = datetime.now()
    run_context['completed_at'] = completed_at
    run_context['completed_at_iso'] = completed_at.isoformat(timespec='seconds')
    
    # Plot results
    plot_results(history, output_dir=run_context['run_dir'])
    save_results_csv(
        history, output_dir=run_context['run_dir'],
        run_context=run_context, config=training_config
    )
    save_run_metadata(run_context, history, run_context['run_dir'], config=training_config)
    update_run_registry(run_context, history)
    update_latest_run_pointer(run_context)
    sync_latest_model(run_context)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
