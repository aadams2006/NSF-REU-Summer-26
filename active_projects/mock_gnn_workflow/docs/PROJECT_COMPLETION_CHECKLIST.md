# Project Checklist

## Active File Layout

- [x] `README.md`
- [x] `requirements.txt`
- [x] `data/generate_lattice_data.py`
- [x] `src/gnn_model.py`
- [x] `src/train.py`
- [x] `scripts/run_full_workflow.py`
- [x] `scripts/run_inference_examples.py`
- [x] `scripts/run_minimal_example.py`
- [x] `scripts/optimize_hyperparameters.py`
- [x] `docs/START_HERE.md`
- [x] `docs/GETTING_STARTED.md`
- [x] `docs/GNN_CONCEPTS.md`
- [x] `docs/IMPLEMENTATION_SUMMARY.md`

## Run Checklist

- [ ] `python scripts/run_minimal_example.py`
- [ ] `python data/generate_lattice_data.py`
- [ ] `python src/train.py`
- [ ] `python scripts/run_inference_examples.py`
- [ ] `python scripts/run_full_workflow.py`

## Output Checklist

- [ ] `data/lattice_dataset.pkl`
- [ ] `results/best_model.pt`
- [ ] `results/latest_run.txt`
- [ ] `results/run_registry.csv`
- [ ] `results/runs/run_*/training_results.png`

## Organization Goals Met

- [x] Active code separated from the legacy archive
- [x] Runnable entrypoints grouped under `scripts/`
- [x] Core reusable code grouped under `src/`
- [x] Project docs grouped under `docs/`
- [x] Generated outputs grouped under `results/`
