# Organized Legacy Research Workspace Pt 2

This folder is a clean working copy of the files from `old codebase pt 2` plus the related generated data referenced in its notes.

Folder guide:

- `00_Project_Interpretation_And_Notes`
  - Your notes about what the legacy scripts appear to do and how they connect to the lattice data folder.
- `01_Voronoi_Lattice_Generation_Code`
  - Abaqus-based Voronoi lattice generation script used to create per-simulation output folders and exported CSV data.
- `02_GNN_Training_Prototype_Code_AI_Generated_Colab`
  - AI-generated Colab training prototype that appears to use `Randomness_Sweep` as training data and `Lattice_Guess` as prediction input data.
- `03_Lattice_Generation_Outputs_And_Datasets`
  - Copied generated data folders tied to the Voronoi script:
  - `Randomness_Sweep_Training_Data` for the swept randomness simulations.
  - `Lattice_Guess_Prediction_Input_Data` for the smaller prediction/input set.

Source:

- Original code folder: `old codebase pt 2`
- Related data folder: `lattice_data-20260604T213453Z-3-001\lattice_data`
