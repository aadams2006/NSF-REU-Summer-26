# Mock GNN Workflow

This project is the cleaned active version of the synthetic lattice GNN demo. It generates toy lattice graphs, trains a graph neural network to predict a stability score, and stores repeatable run outputs under `results/`.

## Layout

```text
mock_gnn_workflow/
├── data/
│   ├── generate_lattice_data.py
│   └── lattice_dataset.pkl
├── docs/
│   ├── START_HERE.md
│   ├── GETTING_STARTED.md
│   ├── GNN_CONCEPTS.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   └── PROJECT_COMPLETION_CHECKLIST.md
├── results/
├── scripts/
│   ├── run_full_workflow.py
│   ├── run_inference_examples.py
│   ├── run_minimal_example.py
│   └── optimize_hyperparameters.py
├── src/
│   ├── gnn_model.py
│   └── train.py
└── requirements.txt
```

## Setup

```bash
cd active_projects/mock_gnn_workflow
pip install -r requirements.txt
```

## Common Commands

```bash
python scripts/run_minimal_example.py
python data/generate_lattice_data.py
python src/train.py
python scripts/run_inference_examples.py
python scripts/run_full_workflow.py
python scripts/optimize_hyperparameters.py
```

## What The Model Uses

- Node features: atom type and normalized node degree
- Edge features: bond strength
- Global graph features: node count, edge count, average degree, mean bond strength, connectivity
- Target: synthetic stability score in `[0, 1]`

## Outputs

- `results/best_model.pt`
- `results/latest_run.txt`
- `results/run_registry.csv`
- `results/runs/run_*/`
- `results/hyperparameter_searches/search_*/`

## Docs

- Start with `docs/START_HERE.md`
- Use `docs/GETTING_STARTED.md` for setup and run order
- Use `docs/GNN_CONCEPTS.md` for the theory background
