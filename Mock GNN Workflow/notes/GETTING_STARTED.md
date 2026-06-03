# Getting Started - Complete Setup Guide

This guide walks you through setting up and running the Mock GNN Workflow project step-by-step.

## Prerequisites

### System Requirements
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **GPU** (optional): NVIDIA GPU with CUDA support for faster training

### Check Your Python Version
```bash
python --version
# Should output: Python 3.8.x or higher
```

If you have multiple Python versions, you may need to use `python3` instead of `python`.

---

## Installation

### Step 1: Navigate to Project Directory
```bash
cd "c:\Users\alexg\Downloads\NSF-REU-Summer-26\Mock GNN Workflow"
```

### Step 2: Create Virtual Environment (Recommended)

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt after activation.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Installation Time**: 5-10 minutes (depending on internet speed)

#### Troubleshooting Installation

**Problem**: `error: Microsoft Visual C++ 14.0 is required` (Windows)
```bash
# Download and install Visual C++ Build Tools from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Then retry: pip install -r requirements.txt
```

**Problem**: `Failed building wheel for torch-geometric`
```bash
# This is common. The wheels should build, it just takes time.
# Solution: Be patient, it can take 5-15 minutes.
# Or install pre-built wheels (varies by OS):
pip install torch-geometric --only-binary :all:
```

**Problem**: CUDA-related errors
```bash
# If you want to use GPU (optional):
# First install CUDA toolkit from NVIDIA website
# Then install GPU-enabled PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Running the Project

### Option 1: Automated (Easiest)

Run everything in one command:
```bash
python quick_start.py
```

This will:
1. Generate synthetic lattice data (2 minutes)
2. Train the GNN model (1-2 minutes)
3. Create visualizations (30 seconds)
4. Display results

**Total Time**: ~5 minutes

---

### Option 2: Step-by-Step (For Learning)

#### Step 2A: Generate Synthetic Data
```bash
python data/generate_lattice_data.py
```

**Expected Output**:
```
Generating 100 lattice structures...
  Generated 20/100 samples
  Generated 40/100 samples
  Generated 60/100 samples
  Generated 80/100 samples
  Generated 100/100 samples

Dataset saved to data/lattice_dataset.pkl
Dataset statistics:
  Total samples: 100
  Avg nodes per structure: 30.8
  Avg edges per structure: 71.2
  Stability range: [0.312, 0.887]
```

**Output Files**: `data/lattice_dataset.pkl`

**Time**: ~30 seconds

#### Step 2B: Train Model
```bash
cd src
python train.py
```

**Expected Output**:
```
Using device: cpu (or cuda if GPU available)
Loading dataset from ../data/lattice_dataset.pkl...
Loaded 100 samples
Train: 70, Val: 15, Test: 15

Model Architecture:
GraphConvolutionalNetwork(...)
Total parameters: 12,417

Training GNN model...
---[Training progress table]---
Epoch      Train Loss     Val Loss       Val MAE        Val R²          
----
1          0.087234       0.045123       0.167834       0.623445        
2          0.064123       0.038456       0.154234       0.678932        
...
[continues for ~50 epochs until early stopping]

Test Set Performance:
  MSE:  0.012567
  MAE:  0.089234
  RMSE: 0.112087
  R²:   0.924567

Plot saved to results/training_results.png
Training complete!
```

**Output Files**: 
- `results/best_model.pt` (trained model)
- `results/training_results.png` (performance plots)

**Time**: 30-60 seconds (CPU) or 5-10 seconds (GPU)

#### Step 2C: View Results
Open `results/training_results.png` to see:
- Training loss curves
- Validation MAE over time
- Validation R² score
- Predictions vs ground truth

#### Step 2D: Run Inference Examples
```bash
python inference.py
```

**Expected Output**:
```
========================================
LATTICE STRUCTURE STABILITY COMPARISON
========================================
Using device: cpu

Generating and analyzing different lattice structures...

Size       Nodes      Edges      Predicted Stability      Quality
------
2x2x2      8          12         0.4532                   Moderate
2x2x2      8          12         0.5123                   Moderate
...
[continues with multiple structures]

Statistics:
  Mean stability:   0.6234
  Std stability:    0.1456
  Min stability:    0.3421
  Max stability:    0.8765

[Detailed analysis section]
[Batch prediction examples]
```

**Time**: ~10-30 seconds

---

### Option 3: Simple Verification

Quick test to verify everything is working:
```bash
python simple_example.py
```

**Expected Output**:
```
============================================================
GNN LATTICE EXAMPLE - End-to-End Workflow
============================================================

Device: cpu

[1/4] Creating a 3×3×3 cubic lattice...
      Created lattice with 27 atoms and 54 bonds

[2/4] Computing stability label...
      True stability: 0.7234

[3/4] Converting to PyTorch Geometric format...
      Node features shape: torch.Size([27, 1])
      Edge index shape: torch.Size([2, 108])
      Label: 0.7234

[4/4] Making prediction with random model...
      Predicted stability (random model): 0.5123
      True stability: 0.7234
      Error: 0.2111

============================================================
Example complete! Model can now be trained to improve predictions.
============================================================
```

**Time**: ~5 seconds

---

## Understanding the Output

### Training Curves (`training_results.png`)

**Top-Left: Loss Curves**
- Blue line (Train Loss): Should steadily decrease
- Orange line (Val Loss): Should follow similar pattern
- If gap widens: Model is overfitting

**Top-Right: Validation MAE**
- Should monotonically decrease
- Lower is better

**Bottom-Left: Validation R²**
- Should increase toward 1.0
- Good target: R² > 0.85

**Bottom-Right: Predictions vs Truth**
- Points should lie near the diagonal line
- Scatter around line: Prediction uncertainty

### Console Output Metrics

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| Train Loss | Average error during training | Decreasing |
| Val Loss | Average error on validation data | < 0.02 |
| Val MAE | Mean absolute error | < 0.1 |
| Val R² | Variance explained | > 0.8 |
| Test R² | Final model quality | > 0.85 |

---

## Customizing the Project

### Change Number of Samples
Edit `data/generate_lattice_data.py`:
```python
generate_dataset(num_samples=200)  # Instead of 100
```

### Change Lattice Sizes
Edit `data/generate_lattice_data.py`:
```python
generate_dataset(lattice_sizes=[2, 3, 4, 5])  # Add size 5
```

### Change Model Architecture
Edit `src/train.py`:
```python
model = GraphConvolutionalNetwork(
    node_feature_dim=1,
    hidden_dim=128,      # Increase from 64
    num_layers=4,        # Increase from 3
    output_dim=1,
    dropout=0.3          # Increase from 0.2
)
```

### Change Training Parameters
Edit `src/train.py`:
```python
model, history = train_model(
    model, train_loader, val_loader, test_loader,
    epochs=200,          # Increase from 100
    lr=0.0005,           # Decrease learning rate
    ...
)
```

---

## Troubleshooting

### "Module not found" errors

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: 
```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Then reinstall
pip install -r requirements.txt
```

### CUDA/GPU errors

**Problem**: `cuda out of memory` or `CUDA not available`

**Solution**:
```python
# Edit src/train.py, change device setup to:
device = torch.device('cpu')  # Force CPU

# Or reduce batch size in train.py:
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
```

### Dataset not found

**Problem**: `FileNotFoundError: data/lattice_dataset.pkl`

**Solution**:
```bash
python data/generate_lattice_data.py
```

### Model file not found

**Problem**: `FileNotFoundError: results/best_model.pt`

**Solution**: Train the model first:
```bash
cd src && python train.py
```

### Slow training

**Problem**: Training is very slow (>2 minutes per epoch)

**Solution**:
1. Use GPU if available
2. Reduce dataset size:
   ```python
   generate_dataset(num_samples=50)
   ```
3. Reduce model size:
   ```python
   hidden_dim=32  # Reduce from 64
   ```

---

## Performance Tips

### For CPU Training
- Use smaller model: `hidden_dim=32, num_layers=2`
- Smaller dataset: `num_samples=50`
- Patience for longer training time

### For GPU Training
- Install CUDA toolkit from NVIDIA
- Install GPU PyTorch: see installation troubleshooting
- Use larger model: `hidden_dim=128, num_layers=4`
- Larger dataset: `num_samples=500+`

### For Best Results
- Generate more data (100→500 samples)
- Increase model capacity slightly
- Train longer (100→200 epochs)
- Try different random seeds

---

## File Organization

After running the full pipeline, your directory will look like:

```
Mock GNN Workflow/
├── data/
│   ├── generate_lattice_data.py
│   └── lattice_dataset.pkl          ← Created
│
├── src/
│   ├── gnn_model.py
│   ├── train.py
│   └── inference.py
│
├── results/
│   ├── best_model.pt                ← Created
│   └── training_results.png         ← Created
│
└── notes/
    ├── README.md
    ├── GNN_CONCEPTS.md
    ├── IMPLEMENTATION_SUMMARY.md
    └── GETTING_STARTED.md           ← This file
```

---

## What to Do Next

1. **Understand the Theory**: Read `notes/GNN_CONCEPTS.md` (15 min)
2. **Review Implementation**: Read `notes/README.md` (10 min)
3. **Study the Code**: Review `src/gnn_model.py` (15 min)
4. **Run Experiments**: Try modifying hyperparameters
5. **Extend the Project**: Add new features or lattice types
6. **Prepare for Research**: Ready to work with real lattice data!

---

## Testing Everything Works

Run this checklist:

- [ ] Installation complete: `pip list | grep torch`
- [ ] Dataset generated: `ls data/lattice_dataset.pkl`
- [ ] Simple example runs: `python simple_example.py`
- [ ] Full pipeline works: `python quick_start.py`
- [ ] Inference runs: `cd src && python inference.py`
- [ ] Results saved: `ls results/`

If all checks pass, you're ready to go! ✅

---

## Need Help?

1. Check the error message carefully
2. Look for similar issue in Troubleshooting section
3. Check PyTorch Geometric docs: https://pytorch-geometric.readthedocs.io/
4. Check PyTorch docs: https://pytorch.org/docs/

---

**Last Updated**: June 2026  
**Setup Time**: 10-20 minutes  
**Total Project Time**: 30-45 minutes  

