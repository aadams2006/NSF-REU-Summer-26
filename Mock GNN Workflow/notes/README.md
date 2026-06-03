# Mock GNN Workflow - Simplified Lattice Strength Prediction

A demonstration project showcasing Graph Neural Networks (GNNs) for predicting structural properties of lattice materials. This simplified example models cubic lattice structures and predicts their "stability" based on structural characteristics.

## Project Overview

### Motivation
This project is designed to demonstrate understanding of GNN principles before working on the real NSF-REU research project, which uses GNNs to predict the strength of lattice structures. The key concepts covered:

- **Graph Representation**: How to represent physical structures as graphs
- **Node & Edge Features**: Encoding material properties (atom types, bond strengths)
- **Message Passing**: How GNNs propagate information through network connections
- **Graph-Level Prediction**: Aggregating node information for structure-level predictions
- **Training & Evaluation**: Standard ML pipeline for graph data

### Technical Approach

**Problem**: Given a cubic lattice structure with varying atom types and bond strengths, predict the overall structural stability.

**Solution**: Use a Graph Convolutional Network (GCN) that:
1. Takes node features (atom type) and edge features (bond strength)
2. Aggregates information through graph convolutions
3. Pools node representations to create a graph-level representation
4. Predicts stability score (0-1) via an MLP head

## Project Structure

```
Mock GNN Workflow/
├── data/
│   ├── generate_lattice_data.py      # Synthetic data generation
│   └── lattice_dataset.pkl           # Generated dataset (after running script)
├── src/
│   ├── gnn_model.py                  # GCN model implementations
│   ├── train.py                      # Training and evaluation pipeline
│   └── utils.py                      # Helper functions (optional)
├── results/
│   ├── best_model.pt                 # Saved model weights
│   └── training_results.png          # Performance visualization
└── notes/
    └── README.md                     # This file
```

## Installation & Setup

### Requirements
- Python 3.8+
- PyTorch
- PyTorch Geometric
- NetworkX
- scikit-learn
- matplotlib
- numpy

### Installation Steps

1. **Clone/navigate to the project:**
   ```bash
   cd Mock\ GNN\ Workflow
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Generate Synthetic Data

```bash
python data/generate_lattice_data.py
```

This creates:
- 100 synthetic cubic lattice structures (varying sizes 2×2×2 to 4×4×4)
- Each structure has atoms with types (0 or 1)
- Bonds have varying strengths (0.7-1.0)
- Stability labels computed from degree, bond strength, and connectivity

Output: `data/lattice_dataset.pkl`

**Output Example:**
```
Generating 100 lattice structures...
  Generated 20/100 samples
  ...
  Generated 100/100 samples

Dataset saved to data/lattice_dataset.pkl
Dataset statistics:
  Total samples: 100
  Avg nodes per structure: 30.8
  Avg edges per structure: 71.2
  Stability range: [0.312, 0.887]
```

### Step 2: Train the GNN Model

```bash
cd src
python train.py
```

This:
- Splits data: 70% train, 15% validation, 15% test
- Trains a 3-layer GCN with early stopping
- Evaluates on test set with multiple metrics
- Saves best model and training plots

**Output Example:**
```
Using device: cuda
Loading dataset from ../data/lattice_dataset.pkl...
Loaded 100 samples
Train: 70, Val: 15, Test: 15

Model Architecture:
GraphConvolutionalNetwork(...)
Total parameters: 12,417

Training GNN model...
--------------------------------------------------------------------------------
Epoch      Train Loss     Val Loss       Val MAE        Val R²          
--------------------------------------------------------------------------------
1          0.087234       0.045123       0.167834       0.623445        
2          0.064123       0.038456       0.154234       0.678932        
...
50         0.008234       0.012567       0.089234       0.924567        

Test Set Performance:
  MSE:  0.012567
  MAE:  0.089234
  RMSE: 0.112087
  R²:   0.924567

Plot saved to results/training_results.png
```

### Step 3: Analyze Results

Check `results/training_results.png` for:
- **Top-left**: Training vs validation loss curves (overfitting indicator)
- **Top-right**: Validation MAE over epochs (model refinement)
- **Bottom-left**: Validation R² score (goodness of fit)
- **Bottom-right**: Predictions vs ground truth scatter plot (accuracy visualization)

## Key Concepts Demonstrated

### 1. **Graph Convolution Operations**
```python
# In gnn_model.py
x = self.gcn_layers[i](x, edge_index)  # Message passing
```
- Each node aggregates information from neighbors
- Edge topology determines information flow
- Multiple layers allow multi-hop neighborhoods

### 2. **Graph-Level Pooling**
```python
graph_repr = global_mean_pool(x, batch)  # Aggregate all nodes
```
- Converts node-level features to graph-level representation
- Essential for whole-structure predictions

### 3. **Feature Engineering**
- **Node features**: atom types (one-hot encodable)
- **Edge features**: bond strengths, bond types
- **Graph features**: size, connectivity, composition

### 4. **Loss Functions & Metrics**
- **Loss**: Mean Squared Error (MSE) for regression
- **Metrics**: 
  - MAE: Mean Absolute Error (interpretability)
  - RMSE: Root Mean Squared Error
  - R²: Coefficient of determination (goodness of fit)

## Extending the Project

### Ideas for Enhancement

1. **More Complex Lattices**:
   - Body-centered cubic (BCC)
   - Face-centered cubic (FCC)
   - Crystal structures with defects

2. **Advanced Models**:
   - Graph Attention Networks (GAT)
   - GraphSAGE
   - Message Passing Neural Networks (MPNN)

3. **Multi-Task Learning**:
   - Predict multiple properties simultaneously
   - Use node-level predictions (e.g., local strain)

4. **Real Data Integration**:
   - Load actual crystallographic data
   - Use density functional theory (DFT) computed properties

5. **Interpretability**:
   - Attention weight visualization
   - Feature importance analysis
   - Graph saliency maps

## Architecture Comparison

### Simple GCN (Implemented)
```
Input → Linear Proj → [GCN → ReLU → Dropout]×3 → Global Pool → [MLP]×3 → Output
```

### With Edge Features (EdgeFeatureGCN)
```
[Node Feat + Edge Feat] → [GCN with Edge Attr]×3 → Global Pool → [MLP] → Output
```

## References & Resources

### Key Papers
- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
- Gilmer et al. (2017): "Neural Message Passing for Quantum Chemistry"
- Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs"

### Libraries
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [NetworkX](https://networkx.org/)
- [PyTorch](https://pytorch.org/)

### Related Research
- Materials discovery with ML
- Molecular property prediction
- Crystal structure analysis

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `train.py` (e.g., `batch_size=4`)

### Issue: Poor model performance (R² < 0.5)
**Solution**: 
- Increase model capacity (`hidden_dim=128`)
- Longer training (`epochs=150`)
- Better data features (add more node/edge attributes)

### Issue: Dataset file not found
**Solution**: Run `python data/generate_lattice_data.py` first

## Performance Benchmarks

On typical runs with 100 samples:
- **Training time**: ~30-60 seconds (CPU), ~5-10 seconds (GPU)
- **Expected R² on test set**: 0.85-0.95
- **Expected MAE**: 0.08-0.12

## Author Notes

This project demonstrates:
- ✅ Understanding of graph representation learning
- ✅ Practical GNN implementation with PyTorch Geometric
- ✅ Proper ML workflow (train/val/test split, early stopping)
- ✅ Evaluation of regression models
- ✅ Synthetic data generation for controlled experiments
- ✅ Code organization and documentation

This serves as a foundation for the more complex lattice strength prediction task in the NSF-REU research project.

---

**Last Updated**: June 2026  
**Contact**: [Your Name/Email]
