# Start Here

This is the fastest way to understand and run the active mock GNN workflow.

## What Is In This Project

- `data/generate_lattice_data.py`
  - Generates synthetic lattice graphs and a pickled dataset.
- `src/train.py`
  - Trains the GNN, evaluates it, and writes run outputs under `results/runs/`.
- `scripts/run_inference_examples.py`
  - Loads the latest trained model and runs prediction demos.
- `scripts/run_full_workflow.py`
  - Generates data and launches training in one command.
- `scripts/run_minimal_example.py`
  - Small end-to-end smoke test without training.

## Fastest Run Order

```bash
cd active_projects/mock_gnn_workflow
pip install -r requirements.txt
python scripts/run_minimal_example.py
python scripts/run_full_workflow.py
python scripts/run_inference_examples.py
```

## If You Want The Steps Separated

```bash
python data/generate_lattice_data.py
python src/train.py
python scripts/run_inference_examples.py
```

## Expected Outputs

- `data/lattice_dataset.pkl`
- `results/best_model.pt`
- `results/latest_run.txt`
- `results/run_registry.csv`
- `results/runs/run_*/training_results.png`

## Recommended Reading Order

1. `README.md`
2. `GETTING_STARTED.md`
3. `GNN_CONCEPTS.md`
4. `IMPLEMENTATION_SUMMARY.md`
