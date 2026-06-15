"""
Graph Neural Network model for predicting lattice structure stability.

Uses PyTorch Geometric for efficient GNN computation on graph-structured data.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphConvolutionalNetwork(nn.Module):
    """
    A simple GCN-based model for regression on graph-level properties.
    
    Architecture:
    - GCN convolution layers for node feature aggregation
    - Global pooling to create graph-level representation
    - MLP head for prediction
    """
    
    def __init__(self, node_feature_dim=1, hidden_dim=64, num_layers=3, output_dim=1,
                 dropout=0.2, global_feature_dim=0):
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
        self.global_feature_dim = global_feature_dim
        
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

        # Let the learned graph embedding model residual structure while
        # graph summary features provide a direct path for the target formula.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        self.global_feature_residual = (
            nn.Linear(global_feature_dim, output_dim) if global_feature_dim > 0 else None
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
        edge_weight = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_weight = data.edge_attr.view(-1)
        
        # Project node features to hidden dimension
        x = self.initial_linear(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Apply GCN layers with residual connections
        for gcn_layer in self.gcn_layers:
            x_prev = x
            x = gcn_layer(x, edge_index, edge_weight=edge_weight)
            x = torch.relu(x)
            x = self.dropout(x)
            # Simple residual connection (if dimensions match)
            if x.shape == x_prev.shape:
                x = x + 0.1 * x_prev
        
        # Global mean pooling to create graph-level representation
        graph_repr = global_mean_pool(x, batch)
        
        global_features = getattr(data, 'graph_features', None)
        if global_features is not None:
            graph_repr = torch.cat([graph_repr, global_features], dim=-1)

        # MLP head
        out = self.mlp(graph_repr)
        if global_features is not None and self.global_feature_residual is not None:
            out = out + self.global_feature_residual(global_features)
        
        return out


class EdgeFeatureGCN(nn.Module):
    """
    Extended GCN that also incorporates edge features (bond information).
    
    This demonstrates how to use both node and edge features in a GNN.
    """
    
    def __init__(self, node_feature_dim=1, edge_feature_dim=1, hidden_dim=64,
                 num_layers=3, output_dim=1, dropout=0.2, global_feature_dim=0):
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
        self.global_feature_dim = global_feature_dim
        
        # Initial projections
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)
        self.edge_weight_proj = nn.Linear(edge_feature_dim, 1)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(p=dropout)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + global_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        self.global_feature_residual = (
            nn.Linear(global_feature_dim, output_dim) if global_feature_dim > 0 else None
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
        edge_attr = getattr(data, 'edge_attr', None)
        
        # Project features
        x = self.node_proj(x)
        x = torch.relu(x)

        edge_repr = None
        edge_weight = None
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_repr = torch.relu(self.edge_proj(edge_attr))
            edge_weight = torch.sigmoid(self.edge_weight_proj(edge_attr)).view(-1)
        
        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            x_prev = x
            x = gcn_layer(x, edge_index, edge_weight=edge_weight)
            x = torch.relu(x)
            x = self.dropout(x)
            if x.shape == x_prev.shape:
                x = x + 0.1 * x_prev
        
        # Global pooling
        graph_repr = global_mean_pool(x, batch)
        if edge_repr is not None:
            edge_batch = batch[edge_index[0]]
            pooled_edge_repr = global_mean_pool(edge_repr, edge_batch)
        else:
            pooled_edge_repr = torch.zeros_like(graph_repr)
        graph_repr = torch.cat([graph_repr, pooled_edge_repr], dim=-1)
        global_features = getattr(data, 'graph_features', None)
        if global_features is not None:
            graph_repr = torch.cat([graph_repr, global_features], dim=-1)
        
        # MLP head
        out = self.mlp(graph_repr)
        if global_features is not None and self.global_feature_residual is not None:
            out = out + self.global_feature_residual(global_features)
        
        return out
