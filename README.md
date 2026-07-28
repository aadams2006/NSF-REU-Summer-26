# NSF-REU-Summer-26

This repository is now split into active projects, reference documents, and a preserved legacy archive.

## Repository Layout

- `active_projects/mock_gnn_workflow`
  - Synthetic lattice dataset generation, GNN training, inference scripts, and model results.
- `active_projects/lattice_crack_research`
  - Abaqus lattice crack studies, crack-tip propagation scripts, experiment spreadsheets, and related presentation material.
- `active_projects/voronoi_lattice_pipeline`
  - Abaqus Voronoi lattice generation, the Colab GNN stiffness prototype, generated dataset folders, and the original source export used to build them.
- `apps/lattice_predictor`
  - Public browser application for uploading a lattice graph and running the promoted five-member GCN-3 ensemble.
- `docs/reference_materials`
  - Papers, presentation decks, and other supporting reference material.
- `Legacy (unused) Code`
  - Archived pre-organization material kept separate from the active work.

## Entry Points

- Mock GNN workflow:
  - `python active_projects/mock_gnn_workflow/scripts/run_minimal_example.py`
  - `python active_projects/mock_gnn_workflow/scripts/run_full_workflow.py`
- Mock GNN training only:
  - `python active_projects/mock_gnn_workflow/src/train.py`
- Lattice GCN predictor:
  - `cd apps/lattice_predictor && npm install && npm run dev`

Each active project has its own `README.md` with project-specific details.
