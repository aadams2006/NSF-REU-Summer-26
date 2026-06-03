"""
Graph Neural Network model for predicting lattice structure stability.

Uses PyTorch Geometric for efficient GNN computation on graph-structured data.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader


class GraphConvolutionalNetwork(nn.Module):
    """
    A simple GCN-based model for regression on graph-level properties.
    
    Architecture:
    - GCN convolution layers for node feature aggregation
    - Global pooling to create graph-level representation
    - MLP head for prediction
    """
    
    def __init__(self, node_feature_dim=1, hidden_dim=64, num_layers=3, output_dim=1, dropout=0.2):
        """
        Initialize the GCN model.
        
        Args:
            node_feature_dim (int): Dimension of input node features
            hidden_dim (int): Hidden dimension for GCN layers
            num_layers (int): Number of GCN layers
            output_dim (int): Output dimension (1 for regression)
            dropout (float): Dropout rate
        """
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Initial linear layer to project node features to hidden dimension
        self.initial_linear = nn.Linear(node_feature_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # MLP head for graph-level prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, data):
        """
        Forward pass through the GCN model.
        
        Args:
            data (Data): PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment for multiple graphs
                
        Returns:
            out (torch.Tensor): Predictions [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Project node features to hidden dimension
        x = self.initial_linear(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Apply GCN layers with residual connections
        for gcn_layer in self.gcn_layers:
            x_prev = x
            x = gcn_layer(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
            # Simple residual connection (if dimensions match)
            if x.shape == x_prev.shape:
                x = x + 0.1 * x_prev
        
        # Global mean pooling to create graph-level representation
        graph_repr = global_mean_pool(x, batch)
        
        # MLP head
        out = self.mlp(graph_repr)
        
        return out


class EdgeFeatureGCN(nn.Module):
    """
    Extended GCN that also incorporates edge features (bond information).
    
    This demonstrates how to use both node and edge features in a GNN.
    """
    
    def __init__(self, node_feature_dim=1, edge_feature_dim=1, hidden_dim=64, 
                 num_layers=3, output_dim=1, dropout=0.2):
        """
        Initialize the Edge-aware GCN model.
        
        Args:
            node_feature_dim (int): Dimension of input node features
            edge_feature_dim (int): Dimension of input edge features
            hidden_dim (int): Hidden dimension for GCN layers
            num_layers (int): Number of GCN layers
            output_dim (int): Output dimension (1 for regression)
            dropout (float): Dropout rate
        """
        super(EdgeFeatureGCN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        
        # Initial projections
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(p=dropout)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, data):
        """
        Forward pass with edge features.
        
        Args:
            data (Data): PyTorch Geometric Data object with edge_attr for edge features
            
        Returns:
            out (torch.Tensor): Predictions
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Project features
        x = self.node_proj(x)
        x = torch.relu(x)
        
        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            x_prev = x
            x = gcn_layer(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
            if x.shape == x_prev.shape:
                x = x + 0.1 * x_prev
        
        # Global pooling
        graph_repr = global_mean_pool(x, batch)
        
        # MLP head
        out = self.mlp(graph_repr)
        
        return out
