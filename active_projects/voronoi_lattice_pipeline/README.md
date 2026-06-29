# Voronoi Lattice Pipeline

This active project contains the organized Voronoi lattice generation and prototype GNN material.

## Layout

- `notes`
  - Interpretation notes about how the Abaqus generator, datasets, and prototype training script fit together.
- `abaqus_generation`
  - `voronoi_lattice_randomness_sweep.py`
  - Abaqus script that generates per-randomness simulation folders and exports CSV graph-style data.
- `gnn_prototype`
  - `colab_gnn_stiffness_prototype.py`
  - `notebooks/`
  - `outputs/`
  - `GCN_Optimization/`
  - Colab-oriented prototype for stiffness prediction, with notebooks and saved artifacts grouped by experiment.
- `datasets`
  - Generated lattice folders used for training and prediction inputs.
- `source_archives`
  - The original lattice data export that the organized dataset copy came from.

## Notes

This project still contains large generated dataset trees. They were kept under `datasets/` so the prototype data layout stays recognizable.
