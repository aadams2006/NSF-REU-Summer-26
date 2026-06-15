# Getting Started

This guide assumes you are running from `active_projects/mock_gnn_workflow`.

## Setup

```bash
cd active_projects/mock_gnn_workflow
pip install -r requirements.txt
```

Optional virtual environment:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Verification

Quick smoke test:

```bash
python scripts/run_minimal_example.py
```

Full pipeline:

```bash
python scripts/run_full_workflow.py
```

## Step-By-Step Run

Generate the dataset:

```bash
python data/generate_lattice_data.py
```

Train the model:

```bash
python src/train.py
```

Run inference examples:

```bash
python scripts/run_inference_examples.py
```

Run the curated hyperparameter search:

```bash
python scripts/optimize_hyperparameters.py
```

## Outputs To Check

- `data/lattice_dataset.pkl`
- `results/best_model.pt`
- `results/latest_run.txt`
- `results/run_registry.csv`
- `results/runs/run_*/training_results.png`

## Common Issues

Dataset missing:

```bash
python data/generate_lattice_data.py
```

Imports failing:

- Run commands from the `mock_gnn_workflow` project root, not from `src/` or `scripts/`.

Training too slow:

- Lower `epochs` or `batch_size` in `src/train.py`.
