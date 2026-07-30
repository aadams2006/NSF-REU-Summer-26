MOOSE CONVERSION: VORONOI LATTICE RANDOMNESS SWEEP
==================================================

Files
-----
voronoi_lattice_randomness_sweep_moose.py
    Standard-Python driver that generates each lattice, writes a Gmsh beam mesh,
    writes the MOOSE input file, optionally launches MOOSE, and converts MOOSE
    CSV output into the original graph-learning CSV format.

Typical use
-----------
1. Generate one test case without running MOOSE:

   python voronoi_lattice_randomness_sweep_moose.py --generate-only --count 1 --seed 42

2. Run one test case with a MOOSE Solid Mechanics executable:

   python voronoi_lattice_randomness_sweep_moose.py \
       --moose-exe /path/to/solid_mechanics-opt --count 1 --seed 42

3. Run the full configured sweep:

   python voronoi_lattice_randomness_sweep_moose.py \
       --moose-exe /path/to/solid_mechanics-opt

You can alternatively set the executable before running:

   export MOOSE_EXE=/path/to/solid_mechanics-opt

Windows/WSL example:

   python3 voronoi_lattice_randomness_sweep_moose.py \
       --moose-exe ~/projects/moose/modules/solid_mechanics/solid_mechanics-opt \
       --count 1 --seed 42

Per-simulation outputs
----------------------
node_features.csv
adjacency_area.csv
lattice_stiffness.csv        (after successful MOOSE solve)
stress_matrix.csv             (after successful MOOSE solve)
lattice.msh                   (Gmsh 2.2 beam mesh)
lattice.i                     (MOOSE input)
moose_results.e               (MOOSE Exodus output)
moose_results.csv             (reaction/stiffness source)
element_to_graph_edge.csv
moose.log

Important model mapping
-----------------------
- The original Abaqus rectangular beam profile is mapped to a MOOSE
  C0 Timoshenko beam with area and rectangular second moments of area.
- The lattice is in the global x-y plane; global z is used as the beam
  y-orientation vector.
- The original bottom-left, bottom-line, and top-displacement constraints are
  retained, with extra out-of-plane constraints to keep the MOOSE beam model
  planar.
- The stress matrix contains axial normal stress (local beam force component 0
  divided by cross-sectional area), corresponding to the original use of S11.
- Stiffness is the summed top-boundary internal y reaction divided by the
  prescribed displacement.

Validation note
---------------
The Python driver was syntax-checked and tested in generate-only mode. The MOOSE
input could not be executed in the conversion environment because no MOOSE
binary was installed. Run a single case first and inspect moose.log. MOOSE input
object names can occasionally differ between releases or custom applications.
