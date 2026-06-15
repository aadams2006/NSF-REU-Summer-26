# GNN Concepts & Implementation Guide

## Table of Contents
1. [What is a Graph Neural Network?](#what-is-a-graph-neural-network)
2. [Graph Representation](#graph-representation)
3. [Message Passing & Convolution](#message-passing--convolution)
4. [Architecture Design](#architecture-design)
5. [Training & Optimization](#training--optimization)
6. [Application to Lattice Structures](#application-to-lattice-structures)

---

## What is a Graph Neural Network?

### Traditional Neural Networks (Limitations)
- Assume **grid-like structure** (images) or **sequential structure** (text)
- Loss of information when flattening irregular structures
- Cannot capture **relational information** between objects

### Graph Neural Networks (Solutions)
- Work directly on **graph-structured data**
- Preserve **topological information**
- Capture **relational dependencies** between entities
- More **parameter-efficient** than fully connected layers

### Key Advantages
```
Input: Irregular structure          → GNN processes naturally
       ↓
       Cannot use CNNs/RNNs directly
       
       GNNs: Work on graphs!
       ✓ Preserve structure
       ✓ Capture relationships
       ✓ Scalable to large graphs
```

---

## Graph Representation

### Elements of a Graph

```
Graph G = (V, E, X, E_attr)

V:       Set of nodes (atoms in lattice)
E:       Set of edges (bonds between atoms)
X:       Node features (atom properties)
E_attr:  Edge features (bond properties)
```

### Example: Cubic Lattice as a Graph

```
Spatial View:                  Graph View:
  
    0---1---2                    0--1--2
    |   |   |                    |  |  |
    3---4---5     →              3--4--5
    |   |   |                    |  |  |
    6---7---8                    6--7--8

Nodes (V):     {0,1,2,3,4,5,6,7,8}
Edges (E):     {(0,1), (1,2), ..., (7,8)}
Node Attr (X): atom_type ∈ {0, 1}
Edge Attr:     bond_strength ∈ [0.7, 1.0]
```

### Adjacency Matrix Representation

```python
# For 3×3 grid (9 nodes)
Adjacency Matrix A:
      0  1  2  3  4  5  6  7  8
0  [  0  1  0  1  0  0  0  0  0 ]
1  [  1  0  1  0  1  0  0  0  0 ]
2  [  0  1  0  0  0  1  0  0  0 ]
3  [  1  0  0  0  1  0  1  0  0 ]
4  [  0  1  0  1  0  1  0  1  0 ]
5  [  0  0  1  0  1  0  0  0  1 ]
...

# Used in sparse form by PyTorch Geometric:
edge_index = [[0,0,1,1,2,...],
              [1,3,0,2,1,...]]
```

### Feature Matrices

```python
# Node Feature Matrix (X): shape [num_nodes, node_features]
X = [[0],      # Node 0: atom_type=0
     [1],      # Node 1: atom_type=1
     [1],      # Node 2: atom_type=1
     [0],      # ...
     ...]

# Edge Feature Matrix (E_attr): shape [num_edges, edge_features]
E_attr = [[0.85],  # Edge (0,1): bond_strength=0.85
          [0.92],  # Edge (0,3): bond_strength=0.92
          [0.78],  # ...
          ...]
```

---

## Message Passing & Convolution

### The Message Passing Framework

The core idea: **Nodes communicate with neighbors to update their representations**

```
Update Rule (per node i):
h_i^(k+1) = γ^(k)(h_i^(k), □_{j ∈ N(i)} φ^(k)(h_i^(k), h_j^(k), e_{ij}))

Where:
  h_i^(k):        Hidden state of node i at layer k
  N(i):           Neighbors of node i
  φ:              Message function
  □:              Aggregation function (e.g., sum, mean, max)
  γ:              Update function
  e_{ij}:         Edge features between i and j
```

### Simplified Version: Graph Convolution (GCN)

```
h_i^(k+1) = σ(W^(k) * (h_i^(k) + Σ_{j ∈ N(i)} h_j^(k) / d_j))

Where:
  σ:      ReLU activation
  W:      Learnable weight matrix
  d_j:    Degree of node j (normalization)
```

### Visualization of Message Passing

```
Layer k:                    Aggregation:              Layer k+1:
  
Node features:             Message passing:           Updated features:

  2  ──→  •2     2          [2,1,3]                        •2
  │       │ │               ─────→  mean/sum    ────→       │
  1  ──→  •1 ←──            Neighbors     ││             •1
  │   ╱        ║             aggregate   ││            ╱
  3  ──→  •3                           update        •3
           ║║                                          
           ↓↓                                          
     (All connected nodes                        (Updated with
      update simultaneously)                   neighborhood info)
```

### Multi-Layer Message Passing

```
Input: Raw node features
  │
  ├─→ GCN Layer 1: Each node sees 1-hop neighbors
  │
  ├─→ GCN Layer 2: Each node sees 2-hop neighbors
  │
  ├─→ GCN Layer 3: Each node sees 3-hop neighbors
  │
  ↓
Output: Node features with global context
```

This allows information to "propagate" through the graph!

---

## Architecture Design

### Our Implemented Architecture

```
INPUT LAYER
    ↓
    X: Raw node features [batch_size × num_nodes, 1]
    edge_index: Connectivity [2, num_edges]
    
    ╔════════════════════════════════╗
    ║  Feature Projection (Linear)   ║
    ║  1 → 64 dimensions             ║
    ╚════════════════════════════════╝
    ↓
    h₀: [batch_size × num_nodes, 64]
    
    ╔════════════════════════════════╗
    ║     GCN Convolution Block      ║  ×3
    ║   (with ReLU + Dropout)        ║
    ║   64 → 64 dimensions           ║
    ╚════════════════════════════════╝
    ↓
    h₃: [batch_size × num_nodes, 64]
    
    ╔════════════════════════════════╗
    ║     Global Mean Pooling        ║
    ║  Aggregate all node features   ║
    ╚════════════════════════════════╝
    ↓
    graph_repr: [batch_size, 64]
    
    ╔════════════════════════════════╗
    ║   MLP Head (for Regression)    ║
    ║  64 → 32 → 16 → 1 dimension   ║
    ╚════════════════════════════════╝
    ↓
OUTPUT LAYER
    Stability: [batch_size, 1] ∈ [0, 1]
```

### Parameter Count Breakdown

```python
Model Architecture:
├── Input Projection (1 → 64):        64 params
├── GCN Layer 1 (64 → 64):           4,224 params
├── GCN Layer 2 (64 → 64):           4,224 params  
├── GCN Layer 3 (64 → 64):           4,224 params
├── MLP Layer 1 (64 → 32):           2,080 params
├── MLP Layer 2 (32 → 16):             528 params
└── MLP Layer 3 (16 → 1):               17 params
                                    ───────────
Total:                             ~15,361 params
```

### Why This Architecture?

1. **Input Projection**: Transforms raw features to higher-dimensional space for richer representations
2. **Multiple GCN Layers**: 
   - Layer 1: 1-hop neighborhood information
   - Layer 2: 2-hop neighborhood information  
   - Layer 3: 3-hop + global context
3. **Global Pooling**: Converts node-level features to graph-level representation
4. **MLP Head**: Learns non-linear relationships in aggregated features

---

## Training & Optimization

### Loss Function: Mean Squared Error (MSE)

```
Loss = (1/n) * Σ(y_true - y_pred)²

For regression tasks (stability prediction):
- Penalizes large errors more heavily
- Smooth gradient landscape (good for optimization)
```

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        predictions = model(batch)
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    val_loss = evaluate(model, val_loader)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
    elif patience_counter >= max_patience:
        break
```

### Optimization Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning Rate | 0.001 | Standard for Adam, good convergence |
| Weight Decay | 1e-5 | Prevents overfitting |
| Batch Size | 8 | Good balance of memory/gradient estimates |
| Dropout Rate | 0.2 | Moderate regularization |
| Max Epochs | 100 | Sufficient for convergence |
| Early Stopping Patience | 15 | Prevent unnecessary training |

### Evaluation Metrics

```
Mean Squared Error (MSE):
    MSE = (1/n) * Σ(y_true - y_pred)²
    - Sensitive to outliers
    - Same units as target squared

Mean Absolute Error (MAE):
    MAE = (1/n) * Σ|y_true - y_pred|
    - Robust to outliers
    - Same units as target
    
Root Mean Squared Error (RMSE):
    RMSE = √MSE
    - Interpretable (same units as target)
    - Good for penalizing large errors

R² Score (Coefficient of Determination):
    R² = 1 - (SS_res / SS_tot)
    - Proportion of variance explained
    - Range: (-∞, 1], with 1 = perfect fit
    - 0.8-0.9 is typically "good"
```

---

## Application to Lattice Structures

### Problem Formulation

**Input**: A cubic lattice structure with:
- Varying atom types (0 or 1)
- Varying bond strengths (0.7-1.0)
- Random connectivity (can vary size and defects)

**Output**: Stability score (0.0-1.0) representing structural integrity

### Why GNNs are Suitable

1. **Natural Graph Structure**: Lattices are naturally graphs (atoms=nodes, bonds=edges)
2. **Variable Size**: GNNs handle variable-size graphs (unlike CNNs)
3. **Relational Reasoning**: Can capture how atoms influence each other through bonds
4. **Property Prediction**: Excellent for material property prediction

### Feature Engineering

```
Node Features (X):
  ├─ atom_type (0 or 1)           # Material composition
  ├─ position (x, y, z)            # Spatial information
  └─ degree                         # Local connectivity

Edge Features (E_attr):
  ├─ bond_strength (0.7-1.0)      # Bond quality
  ├─ bond_type                     # Type of bond
  └─ distance                      # Inter-atomic distance

Graph Features (aggregated):
  ├─ Number of nodes              # System size
  ├─ Number of edges              # Bond density
  ├─ Average degree               # Local coordination
  └─ Connectivity                 # Graph connectivity
```

### Stability Label Computation

```python
stability = 0.4 * avg_degree_score 
          + 0.4 * avg_bond_strength
          + 0.2 * connectivity_score

Where:
  - avg_degree_score ∝ number of bonds per atom
  - avg_bond_strength: average bond quality
  - connectivity_score: 1 if connected, 0.5 if not
```

### Extension to Real Data

For actual lattice strength prediction (NSF-REU project):

1. **Real Features**:
   - DFT-computed electronic structure
   - Elastic constants
   - Atomic forces

2. **Real Labels**:
   - Young's modulus (stiffness)
   - Ultimate tensile strength
   - Fracture toughness

3. **Advanced Models**:
   - Message Passing Neural Networks (MPNN)
   - Graph Attention Networks (GAT)
   - Crystal Graph Convolution Networks (CGCNN)

---

## Common Pitfalls & Solutions

### Problem: Poor Model Performance (R² < 0.5)
**Causes**:
- Insufficient model capacity
- Not enough training data
- Poor feature engineering
- Too much regularization

**Solutions**:
- Increase `hidden_dim` or `num_layers`
- Collect more data
- Add more meaningful features
- Reduce dropout or weight decay

### Problem: Overfitting (train R² >> val R²)
**Causes**:
- Too large model
- Too few samples
- No regularization
- Training too long

**Solutions**:
- Reduce `hidden_dim` or `num_layers`
- Add dropout
- Add weight decay
- Use early stopping

### Problem: Slow Training
**Causes**:
- Large graph sizes
- High batch size
- CPU usage
- Inefficient code

**Solutions**:
- Reduce batch size
- Use GPU (CUDA)
- Preprocess graphs (e.g., sampling)
- Profile code

---

## References

- Kipf & Welling (2017): ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907)
- Gilmer et al. (2017): ["Neural Message Passing for Quantum Chemistry"](https://arxiv.org/abs/1704.01212)
- Hamilton et al. (2017): ["Inductive Representation Learning on Large Graphs"](https://arxiv.org/abs/1706.02216)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

