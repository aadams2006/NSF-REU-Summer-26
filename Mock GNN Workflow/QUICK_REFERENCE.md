# Quick Reference Card

## 📋 Execution Paths

### Path 1: One-Command Execution ⚡ (Fastest)
```bash
python quick_start.py
```
**Time**: ~5 minutes | **Complexity**: Minimal

---

### Path 2: Step-by-Step Learning 📚 (Recommended)
```bash
# Step 1: Generate synthetic lattice data
python data/generate_lattice_data.py
# Output: data/lattice_dataset.pkl

# Step 2: Train the GNN model  
cd src && python train.py
# Output: results/best_model.pt
#         results/training_results.png

# Step 3: Make predictions on new structures
python inference.py
```
**Time**: ~3 minutes | **Complexity**: Low

---

### Path 3: Minimal Verification 🚀 (Quick Test)
```bash
python simple_example.py
```
**Time**: ~5 seconds | **Complexity**: Trivial

---

## 🔍 File Directory

| File | Purpose | Lines | Type |
|------|---------|-------|------|
| `data/generate_lattice_data.py` | Synthetic data creation | 150 | Code |
| `src/gnn_model.py` | GCN architecture | 200 | Code |
| `src/train.py` | Training pipeline | 350 | Code |
| `src/inference.py` | Predictions | 350 | Code |
| `quick_start.py` | Automated execution | 50 | Code |
| `simple_example.py` | Minimal test | 50 | Code |
| `notes/README.md` | Main documentation | 400 | Docs |
| `notes/GNN_CONCEPTS.md` | Theory guide | 500 | Docs |
| `notes/GETTING_STARTED.md` | Setup guide | 350 | Docs |
| `requirements.txt` | Dependencies | 10 | Config |

---

## 🏗️ Architecture at a Glance

```
Input Layer
    │
    ├─ Node Features: atom_type ∈ {0,1}
    ├─ Edge Features: bond_strength ∈ [0.7,1.0]
    │
    ▼
Feature Projection (Linear): 1 → 64 dims
    │
    ├─ ReLU + Dropout
    │
    ▼
GCN Layer 1: Aggregate 1-hop neighbors
    ├─ Message Passing
    ├─ ReLU + Dropout
    ├─ Residual Connection
    │
    ▼
GCN Layer 2: Aggregate 2-hop neighbors
    ├─ Message Passing
    ├─ ReLU + Dropout
    ├─ Residual Connection
    │
    ▼
GCN Layer 3: Aggregate 3-hop neighbors
    ├─ Message Passing
    ├─ ReLU + Dropout
    ├─ Residual Connection
    │
    ▼
Global Mean Pooling: [num_nodes, 64] → [1, 64]
    │
    ▼
MLP Head: 64 → 32 → 16 → 1
    ├─ Dense layers with ReLU
    ├─ Dropout between layers
    │
    ▼
Output: Stability Score ∈ [0,1]
```

---

## 📊 Training Pipeline

```
1. DATA LOADING
   ├─ Load lattice_dataset.pkl
   ├─ Convert NetworkX → PyTorch Geometric
   └─ Create train/val/test splits (70/15/15)

2. MODEL SETUP
   ├─ Initialize GCN model
   ├─ Set optimizer (Adam, lr=0.001)
   └─ Set loss function (MSE)

3. TRAINING LOOP (×100 epochs max)
   ├─ For each batch:
   │  ├─ Forward pass
   │  ├─ Compute loss
   │  ├─ Backward pass
   │  └─ Update weights
   ├─ Validate on val_set
   ├─ Check early stopping (patience=15)
   └─ Save best model

4. EVALUATION
   ├─ Load best model
   ├─ Evaluate on test set
   ├─ Report metrics (MSE, MAE, RMSE, R²)
   └─ Generate plots

5. VISUALIZATION
   ├─ Loss curves (train vs val)
   ├─ MAE progression
   ├─ R² score progression
   └─ Predictions vs truth
```

---

## 📈 Expected Metrics

| Metric | Target | Typical | Range |
|--------|--------|---------|-------|
| Test R² | > 0.85 | 0.90 | 0.80-0.95 |
| Test MAE | < 0.12 | 0.09 | 0.08-0.15 |
| Test RMSE | < 0.15 | 0.11 | 0.10-0.18 |
| MSE | < 0.02 | 0.013 | 0.008-0.025 |

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB |
| CPU | Any modern | i7 or better |
| GPU | Optional | NVIDIA RTX |
| Time | 30 min | 15 min (GPU) |

---

## 🔧 Installation One-Liner

```bash
pip install torch==2.1.2 torch-geometric==2.4.0 torch-scatter torch-sparse torch-cluster networkx scikit-learn matplotlib numpy pandas
```

Or just:
```bash
pip install -r requirements.txt
```

---

## 🎯 Key Hyperparameters

### Model
- `hidden_dim=64` - Feature dimension in GCN layers
- `num_layers=3` - Number of GCN layers
- `dropout=0.2` - Dropout rate

### Training
- `batch_size=8` - Samples per batch
- `epochs=100` - Maximum training epochs
- `lr=0.001` - Learning rate
- `weight_decay=1e-5` - L2 regularization
- `early_stopping_patience=15` - Epochs to wait

### Data
- `num_samples=100` - Total dataset size
- `lattice_sizes=[2,3,4]` - Possible lattice dimensions
- `train/val/test = 70/15/15` - Data split

---

## 📚 Documentation Map

```
START_HERE.md (You are here!)
    │
    ├─→ GETTING_STARTED.md
    │   (Installation & Setup)
    │
    ├─→ README.md
    │   (Main Usage Guide)
    │
    ├─→ GNN_CONCEPTS.md
    │   (Deep Theory)
    │
    ├─→ IMPLEMENTATION_SUMMARY.md
    │   (What Each Part Does)
    │
    └─→ PROJECT_COMPLETION_CHECKLIST.md
        (Verification)
```

---

## 🚀 Getting Results in 3 Minutes

1. **[30 sec]** Run: `python simple_example.py`
2. **[90 sec]** Run: `python data/generate_lattice_data.py`
3. **[60 sec]** Run: `cd src && python train.py`
4. **[30 sec]** View: `../results/training_results.png`

---

## 🐛 Common Issues & Fixes

| Issue | Fix | Time |
|-------|-----|------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` | 5 min |
| `File not found` | Run data generation first | 1 min |
| Slow training | Use GPU or reduce batch size | - |
| Poor results (R² < 0.5) | Increase `hidden_dim` to 128 | 2 min |
| CUDA out of memory | Set `batch_size=4` | 1 min |

---

## 💡 Pro Tips

### Performance
- 🚀 Use GPU for 5-10x speedup
- 📊 Monitor loss curves to spot issues
- 🔄 Try different random seeds

### Experimentation
- 🎨 Modify `hidden_dim` and `num_layers`
- 📈 Generate more data (100→500 samples)
- 🔬 Add more node/edge features

### Learning
- 📖 Read theory FIRST, code SECOND
- 💻 Run simple_example.py to verify
- 🔍 Print intermediate shapes in training loop

---

## 🎓 Learning Progression

```
Time | Activity | Duration | Resource
-----|----------|----------|----------
0-5m | Read this card | 5 min | This file
5-10m | Install deps | 5 min | GETTING_STARTED.md
10-15m | Run simple test | 5 min | simple_example.py
15-20m | Run full pipeline | 5 min | quick_start.py
20-35m | Study theory | 15 min | GNN_CONCEPTS.md
35-45m | Study code | 10 min | src/gnn_model.py
45-60m | Experiment | 15 min | Your own changes

Total: ~60 minutes to mastery ✅
```

---

## ✅ Validation Checklist

Quick checks to verify everything works:

```bash
# 1. Check Python version (should be ≥3.8)
python --version

# 2. Verify PyTorch installed
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 3. Verify PyTorch Geometric installed
python -c "import torch_geometric; print(f'PyG {torch_geometric.__version__}')"

# 4. Run minimal example
python simple_example.py  # Should complete in ~5 seconds

# 5. Generate data
python data/generate_lattice_data.py  # Should complete in ~30 seconds

# 6. Check outputs
ls data/lattice_dataset.pkl  # Should exist

# 7. Run full pipeline
python quick_start.py  # Should complete in ~5 minutes

# 8. Check results
ls results/best_model.pt results/training_results.png  # Should exist
```

---

## 🔗 Quick Links

- **Main Guide**: `notes/README.md`
- **Setup Instructions**: `notes/GETTING_STARTED.md`
- **GNN Theory**: `notes/GNN_CONCEPTS.md`
- **Implementation Details**: `notes/IMPLEMENTATION_SUMMARY.md`
- **Data Generation**: `data/generate_lattice_data.py`
- **Model Code**: `src/gnn_model.py`
- **Training Code**: `src/train.py`
- **Inference Code**: `src/inference.py`

---

## 🎬 Ready to Start?

### Absolute Beginner
→ Run `python simple_example.py`  
→ Then read `notes/GETTING_STARTED.md`

### Familiar with Python
→ Run `python quick_start.py`  
→ Then read `notes/README.md`

### Experienced with ML
→ Read `notes/GNN_CONCEPTS.md`  
→ Then examine `src/gnn_model.py`

### Familiar with GNNs
→ Review `src/train.py` for training pipeline  
→ Experiment with modifications

---

**Status**: ✅ Complete & Ready  
**Execution Time**: 5 minutes to first results  
**Learning Time**: 60 minutes to full understanding  

Begin with: `python simple_example.py` or `python quick_start.py`

