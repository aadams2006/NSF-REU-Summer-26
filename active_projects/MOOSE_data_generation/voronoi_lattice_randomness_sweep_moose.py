#!/usr/bin/env python3
"""2D Voronoi-like lattice randomness sweep for MOOSE beam mechanics.

This is a close MOOSE-oriented conversion of the original Abaqus script.
For every randomness value it:
  1. generates the same seed points and random beam connectivity,
  2. removes crossing beams, thins side walls, and repairs corner nodes,
  3. writes graph CSV files,
  4. writes a Gmsh 2.2 line-element mesh and a MOOSE input file,
  5. optionally runs a MOOSE executable,
  6. converts MOOSE CSV output into lattice_stiffness.csv and stress_matrix.csv.

The script itself uses only the Python standard library. MOOSE is invoked as an
external executable, so run this script with normal Python rather than moose's
Python environment.
"""

from __future__ import print_function

import argparse
import csv
import glob
import math
import os
import random
import shutil
import subprocess
import sys

# -----------------------------
# USER PARAMETERS
# -----------------------------
cell_size = 20.0
cell_height = math.sqrt(3.0) * cell_size
n = 11

randomness_start = 0.0
randomness_end = 0.5
randomness_count = 3000

# Keep units consistent. With the original values these are commonly interpreted
# as N, mm, and MPa (= N/mm^2).
youngs_modulus = 2000.0
poissons_ratio = 0.30

rect_a = 0.01  # out-of-plane dimension
rect_b = 0.10  # in-plane dimension
seed_size = 0.30

displacement_magnitude = 5.0
time_period = 1.0

# Reproducibility. Set to None for non-deterministic generation.
base_random_seed = None

# MOOSE executable. Override with --moose-exe or MOOSE_EXE.
moose_executable = os.environ.get("MOOSE_EXE", "solid_mechanics-opt")


# -----------------------------
# HELPERS
# -----------------------------
def r6(p):
    return (round(p[0], 6), round(p[1], 6))


def seg_key(p, q):
    p2, q2 = r6(p), r6(q)
    return (p2, q2) if p2 <= q2 else (q2, p2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, q1, p2, q2):
    if (r6(p1) == r6(p2) or r6(p1) == r6(q2) or
            r6(q1) == r6(p2) or r6(q1) == r6(q2)):
        return False
    return ((ccw(p1, p2, q2) != ccw(q1, p2, q2)) and
            (ccw(p1, q1, p2) != ccw(p1, q1, q2)))


def is_boundary_point(pt, y_bottom_fixed, y_top_fixed):
    return (abs(pt[1] - y_bottom_fixed) < 1e-6 or
            abs(pt[1] - y_top_fixed) < 1e-6)


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    projx = ax + t * dx
    projy = ay + t * dy
    return math.sqrt((px - projx) ** 2 + (py - projy) ** 2)


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def generate_segments(randomness, rng):
    """Preserves the lattice-generation logic from the Abaqus script."""
    temp_points = []
    for i in range(n):
        n_y = n
        for j in range(n_y):
            x0 = i * 1.5 * cell_size
            y0 = j * cell_height + (i % 2) * (cell_height / 2.0)

            if randomness > 0:
                max_offset = randomness * cell_size
                x_offset = rng.uniform(-max_offset, max_offset)
                y_offset = rng.uniform(-max_offset, max_offset)

                if i == 0 or i == n - 1:
                    x_offset = 0.0
                if j == 0 or j == n_y - 1:
                    y_offset = 0.0

                x0 += x_offset
                y0 += y_offset

            temp_points.append((x0, y0, i, j))

    y_coords = [p[1] for p in temp_points]
    y_min, y_max = min(y_coords), max(y_coords)
    y_tolerance = cell_height * 0.3

    bottom_points, top_points, interior_points = [], [], []
    for x, y, i, j in temp_points:
        if abs(y - y_min) < y_tolerance:
            bottom_points.append((x, y, i, j))
        elif abs(y - y_max) < y_tolerance:
            top_points.append((x, y, i, j))
        else:
            interior_points.append((x, y, i, j))

    bottom_points.sort(key=lambda p: p[0])
    top_points.sort(key=lambda p: p[0])
    y_bottom_fixed = y_min
    y_top_fixed = y_max

    seed_points = []
    seed_points.extend((x, y_bottom_fixed) for x, _y, _i, _j in bottom_points)
    seed_points.extend((x, y) for x, y, _i, _j in interior_points)
    seed_points.extend((x, y_top_fixed) for x, _y, _i, _j in top_points)

    xs = [p[0] for p in seed_points]
    left_bound, right_bound = min(xs), max(xs)

    connection_threshold = cell_size * 2.5
    raw_seg_keys = []

    for i, p1 in enumerate(seed_points):
        neighbors = []
        for j, p2 in enumerate(seed_points):
            if i == j:
                continue
            d = distance(p1, p2)
            if d < connection_threshold:
                neighbors.append((d, j, p2))

        if not neighbors:
            continue

        neighbors.sort(key=lambda item: item[0])
        candidates = neighbors[:min(12, len(neighbors))]
        rng.shuffle(candidates)
        desired = rng.randint(3, 6)

        for _d, _j, p2 in candidates[:min(desired, len(candidates))]:
            raw_seg_keys.append(seg_key(p1, p2))

    segments_dict = {}
    for key in raw_seg_keys:
        segments_dict.setdefault(key, key)

    segment_list = [(pq[0], pq[1]) for pq in segments_dict.values()]

    changed = True
    while changed:
        changed = False
        num_segs = len(segment_list)
        for i in range(num_segs):
            if changed:
                break
            p1, q1 = segment_list[i]
            for j in range(i + 1, num_segs):
                p2, q2 = segment_list[j]
                if not segments_intersect(p1, q1, p2, q2):
                    continue

                s1_boundary = (is_boundary_point(p1, y_bottom_fixed, y_top_fixed) and
                               is_boundary_point(q1, y_bottom_fixed, y_top_fixed))
                s2_boundary = (is_boundary_point(p2, y_bottom_fixed, y_top_fixed) and
                               is_boundary_point(q2, y_bottom_fixed, y_top_fixed))

                if s1_boundary and not s2_boundary:
                    remove_index = j
                elif s2_boundary and not s1_boundary:
                    remove_index = i
                else:
                    remove_index = i if distance(p1, q1) > distance(p2, q2) else j

                del segment_list[remove_index]
                changed = True
                break

    side_tol = cell_size * 1.0
    side_keep_prob = 0.45
    filtered = []
    for p, q in segment_list:
        left_side = p[0] < left_bound + side_tol and q[0] < left_bound + side_tol
        right_side = p[0] > right_bound - side_tol and q[0] > right_bound - side_tol
        if not (left_side or right_side) or rng.random() < side_keep_prob:
            filtered.append((p, q))
    segment_list = filtered

    # Corner-node repair (same logic as original).
    from collections import defaultdict
    corner_joint_coords = set()
    for p, q in segment_list:
        corner_joint_coords.add(r6(p))
        corner_joint_coords.add(r6(q))

    if corner_joint_coords:
        bottom_y = min(pt[1] for pt in corner_joint_coords)
        y_tol_corner = cell_size * 0.2
        bottom_nodes = [pt for pt in corner_joint_coords
                        if abs(pt[1] - bottom_y) < y_tol_corner]
        if bottom_nodes:
            corner_pts = [min(bottom_nodes, key=lambda pt: pt[0]),
                          max(bottom_nodes, key=lambda pt: pt[0])]
            degree = defaultdict(int)
            for p, q in segment_list:
                degree[r6(p)] += 1
                degree[r6(q)] += 1

            for corner_pt in corner_pts:
                if degree[r6(corner_pt)] > 1:
                    continue
                x_c, y_c = corner_pt
                best_pt, best_dy = None, None
                x_tol = cell_size * 0.7
                for pt in corner_joint_coords:
                    if pt[1] <= y_c + 1e-6 or abs(pt[0] - x_c) > x_tol:
                        continue
                    dy = pt[1] - y_c
                    if best_dy is None or dy < best_dy:
                        best_dy, best_pt = dy, pt
                if best_pt is not None and seg_key(corner_pt, best_pt) not in {
                        seg_key(p, q) for p, q in segment_list}:
                    segment_list.append((corner_pt, best_pt))

    segments = {}
    for p, q in segment_list:
        segments[seg_key(p, q)] = (r6(p), r6(q))
    return seed_points, segments


def create_refined_mesh(segments):
    """Split each logical beam into EDGE2 elements close to seed_size."""
    coord_to_id = {}
    nodes = []
    elements = []
    element_to_graph_edge = []

    def add_node(xy):
        key = r6(xy)
        if key not in coord_to_id:
            coord_to_id[key] = len(nodes) + 1  # Gmsh uses 1-based IDs
            nodes.append((key[0], key[1], 0.0))
        return coord_to_id[key]

    logical_edges = list(segments.values())
    for graph_edge_id, (p, q) in enumerate(logical_edges):
        length = distance(p, q)
        divisions = max(1, int(math.ceil(length / seed_size)))
        chain = []
        for k in range(divisions + 1):
            t = k / float(divisions)
            xy = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
            chain.append(add_node(xy))
        for k in range(divisions):
            elements.append((chain[k], chain[k + 1]))
            element_to_graph_edge.append(graph_edge_id)

    return nodes, elements, logical_edges, element_to_graph_edge, coord_to_id


def write_gmsh_mesh(path, nodes, elements):
    """Write Gmsh 2.2 ASCII with a beam block and point physical groups."""
    y_values = [p[1] for p in nodes]
    y_min, y_max = min(y_values), max(y_values)
    tol = max(seed_size * 0.1, 1e-8)

    bottom = [i + 1 for i, p in enumerate(nodes) if abs(p[1] - y_min) < tol]
    top = [i + 1 for i, p in enumerate(nodes) if abs(p[1] - y_max) < tol]
    bottom_left = min(bottom, key=lambda node_id: nodes[node_id - 1][0])
    bottom_other = [node_id for node_id in bottom if node_id != bottom_left]

    # Physical IDs: 1 beams, 2 top, 3 bottom-left, 4 bottom-other.
    point_records = []
    for node_id in top:
        point_records.append((2, node_id))
    point_records.append((3, bottom_left))
    for node_id in bottom_other:
        point_records.append((4, node_id))

    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n4\n")
        f.write('1 1 "beam"\n')
        f.write('0 2 "top"\n')
        f.write('0 3 "bottom_left"\n')
        f.write('0 4 "bottom_other"\n')
        f.write("$EndPhysicalNames\n")
        f.write("$Nodes\n%d\n" % len(nodes))
        for node_id, (x, y, z) in enumerate(nodes, 1):
            f.write("%d %.16g %.16g %.16g\n" % (node_id, x, y, z))
        f.write("$EndNodes\n")

        total_elements = len(elements) + len(point_records)
        f.write("$Elements\n%d\n" % total_elements)
        eid = 1
        for physical_id, node_id in point_records:
            # Gmsh type 15 = point; tags: physical, geometrical.
            f.write("%d 15 2 %d %d %d\n" % (eid, physical_id, physical_id, node_id))
            eid += 1
        for n1, n2 in elements:
            # Gmsh type 1 = 2-node line.
            f.write("%d 1 2 1 1 %d %d\n" % (eid, n1, n2))
            eid += 1
        f.write("$EndElements\n")

    return {"top": top, "bottom_left": [bottom_left], "bottom_other": bottom_other}


def moose_input_text(mesh_filename, output_base):
    area = rect_a * rect_b
    # Local x follows each beam. y_orientation = global z, so local y is
    # out-of-plane and local z is the in-plane transverse direction.
    iy = rect_a * rect_b ** 3 / 12.0
    iz = rect_b * rect_a ** 3 / 12.0

    return """# Auto-generated by voronoi_lattice_randomness_sweep_moose.py
[Mesh]
  [file]
    type = FileMeshGenerator
    file = '{mesh}'
  []
  construct_node_list_from_side_list = true
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
  [rot_x]
  []
  [rot_y]
  []
  [rot_z]
  []
[]

[AuxVariables]
  [reaction_y]
  []
  [force_x]
    order = CONSTANT
    family = MONOMIAL
  []
  [axial_stress]
    order = CONSTANT
    family = MONOMIAL
  []
[]

[Kernels]
  [disp_x]
    type = StressDivergenceBeam
    block = beam
    variable = disp_x
    component = 0
    displacements = 'disp_x disp_y disp_z'
    rotations = 'rot_x rot_y rot_z'
  []
  [disp_y]
    type = StressDivergenceBeam
    block = beam
    variable = disp_y
    component = 1
    displacements = 'disp_x disp_y disp_z'
    rotations = 'rot_x rot_y rot_z'
    save_in = reaction_y
  []
  [disp_z]
    type = StressDivergenceBeam
    block = beam
    variable = disp_z
    component = 2
    displacements = 'disp_x disp_y disp_z'
    rotations = 'rot_x rot_y rot_z'
  []
  [rot_x]
    type = StressDivergenceBeam
    block = beam
    variable = rot_x
    component = 3
    displacements = 'disp_x disp_y disp_z'
    rotations = 'rot_x rot_y rot_z'
  []
  [rot_y]
    type = StressDivergenceBeam
    block = beam
    variable = rot_y
    component = 4
    displacements = 'disp_x disp_y disp_z'
    rotations = 'rot_x rot_y rot_z'
  []
  [rot_z]
    type = StressDivergenceBeam
    block = beam
    variable = rot_z
    component = 5
    displacements = 'disp_x disp_y disp_z'
    rotations = 'rot_x rot_y rot_z'
  []
[]

[Materials]
  [elasticity]
    type = ComputeElasticityBeam
    block = beam
    youngs_modulus = {E:.16g}
    poissons_ratio = {nu:.16g}
    shear_coefficient = 1.0
  []
  [strain]
    type = ComputeIncrementalBeamStrain
    block = beam
    displacements = 'disp_x disp_y disp_z'
    rotations = 'rot_x rot_y rot_z'
    area = {area:.16g}
    Ay = 0.0
    Az = 0.0
    Iy = {iy:.16g}
    Iz = {iz:.16g}
    y_orientation = '0 0 1'
  []
  [resultants]
    type = ComputeBeamResultants
    block = beam
  []
[]

[AuxKernels]
  [force_x]
    type = MaterialRealVectorValueAux
    block = beam
    variable = force_x
    property = forces
    component = 0
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [axial_stress]
    type = QuotientAux
    block = beam
    variable = axial_stress
    numerator = force_x
    denominator = {area:.16g}
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Functions]
  [top_motion]
    type = ParsedFunction
    expression = '{disp:.16g} * t / {period:.16g}'
  []
[]

[BCs]
  # Original BottomLeftFix
  [bottom_left_x]
    type = DirichletBC
    variable = disp_x
    boundary = bottom_left
    value = 0
  []
  [bottom_left_y]
    type = DirichletBC
    variable = disp_y
    boundary = bottom_left
    value = 0
  []
  [bottom_left_rotz]
    type = DirichletBC
    variable = rot_z
    boundary = bottom_left
    value = 0
  []

  # Original BottomLineU2Fix
  [bottom_other_y]
    type = DirichletBC
    variable = disp_y
    boundary = bottom_other
    value = 0
  []

  # Keep the model planar.
  [all_z]
    type = DirichletBC
    variable = disp_z
    boundary = 'top bottom_left bottom_other'
    value = 0
  []
  [all_rotx]
    type = DirichletBC
    variable = rot_x
    boundary = 'top bottom_left bottom_other'
    value = 0
  []
  [all_roty]
    type = DirichletBC
    variable = rot_y
    boundary = 'top bottom_left bottom_other'
    value = 0
  []

  # Original TopDisplacement
  [top_displacement]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = top
    function = top_motion
  []
[]

[Postprocessors]
  [total_reaction_y]
    type = NodalSum
    variable = reaction_y
    boundary = top
    execute_on = TIMESTEP_END
  []
[]

[VectorPostprocessors]
  [element_values]
    type = ElementValueSampler
    block = beam
    variable = 'axial_stress'
    sort_by = id
    execute_on = TIMESTEP_END
  []
[]

[Executioner]
  type = Transient
  start_time = 0
  end_time = {period:.16g}
  dt = {period:.16g}
  solve_type = NEWTON
  nl_rel_tol = 1e-10
  nl_abs_tol = 1e-12
  l_tol = 1e-12
[]

[Outputs]
  file_base = '{out}'
  exodus = true
  csv = true
  execute_on = 'INITIAL TIMESTEP_END'
[]
""".format(mesh=mesh_filename, out=output_base, E=youngs_modulus,
           nu=poissons_ratio, area=area, iy=iy, iz=iz,
           disp=displacement_magnitude, period=time_period)


def export_graph_files(folder, segments):
    joint_coords = set()
    for p, q in segments.values():
        joint_coords.add(r6(p))
        joint_coords.add(r6(q))
    node_list = sorted(joint_coords)
    node_index = {xy: idx for idx, xy in enumerate(node_list)}
    size = len(node_list)
    adjacency = [[0 for _ in range(size)] for __ in range(size)]
    for p, q in segments.values():
        i, j = node_index[r6(p)], node_index[r6(q)]
        if i != j:
            adjacency[i][j] = adjacency[j][i] = 1

    write_csv(os.path.join(folder, "node_features.csv"),
              [["node_id", "x", "y"]] +
              [[i, xy[0], xy[1]] for i, xy in enumerate(node_list)])
    write_csv(os.path.join(folder, "adjacency_area.csv"),
              [[""] + list(range(size))] +
              [[i] + adjacency[i] for i in range(size)])
    return node_list, node_index


def find_csv_with_columns(folder, required_columns):
    candidates = sorted(glob.glob(os.path.join(folder, "*.csv")))
    for path in candidates:
        try:
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
            if set(required_columns).issubset(fields):
                return path
        except (OSError, csv.Error, UnicodeDecodeError):
            pass
    return None


def postprocess_moose(folder, logical_edges, element_to_graph_edge, node_list, node_index):
    # Scalar CSV containing the reaction postprocessor.
    reaction_csv = find_csv_with_columns(folder, ["total_reaction_y"])
    if reaction_csv is None:
        raise RuntimeError("Could not locate MOOSE CSV column 'total_reaction_y'.")
    with open(reaction_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    total_reaction = float(rows[-1]["total_reaction_y"])
    k_lattice = total_reaction / float(displacement_magnitude)
    write_csv(os.path.join(folder, "lattice_stiffness.csv"),
              [["k_lattice_N_per_mm"], [k_lattice]])

    element_csv = find_csv_with_columns(folder, ["id", "axial_stress"])
    if element_csv is None:
        raise RuntimeError("Could not locate ElementValueSampler CSV output.")

    element_stresses = {}
    with open(element_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            element_stresses[int(float(row["id"]))] = float(row["axial_stress"])

    # ElementValueSampler/libMesh IDs are usually zero-based, while our list order
    # is deterministic. Handle both zero- and one-based IDs.
    graph_edge_stress = {}
    for elem_id, stress in element_stresses.items():
        idx = elem_id
        if idx < 0 or idx >= len(element_to_graph_edge):
            idx = elem_id - 1
        if idx < 0 or idx >= len(element_to_graph_edge):
            continue
        graph_edge_id = element_to_graph_edge[idx]
        previous = graph_edge_stress.get(graph_edge_id)
        if previous is None or abs(stress) > abs(previous):
            graph_edge_stress[graph_edge_id] = stress

    size = len(node_list)
    stress_matrix = [[0.0 for _ in range(size)] for __ in range(size)]
    for graph_edge_id, (p, q) in enumerate(logical_edges):
        if graph_edge_id not in graph_edge_stress:
            continue
        i, j = node_index[r6(p)], node_index[r6(q)]
        stress_matrix[i][j] = stress_matrix[j][i] = graph_edge_stress[graph_edge_id]

    write_csv(os.path.join(folder, "stress_matrix.csv"),
              [[""] + list(range(size))] +
              [[i] + stress_matrix[i] for i in range(size)])
    return k_lattice


def run_command(command, cwd, log_path):
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log,
                                   stderr=subprocess.STDOUT)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("MOOSE failed with exit code %d. See %s" %
                           (return_code, log_path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moose-exe", default=moose_executable,
                        help="MOOSE executable (default: %(default)s)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Write meshes/input files but do not run MOOSE")
    parser.add_argument("--count", type=int, default=randomness_count,
                        help="Override number of sweep simulations")
    parser.add_argument("--start", type=float, default=randomness_start)
    parser.add_argument("--end", type=float, default=randomness_end)
    parser.add_argument("--output-dir", default=os.getcwd())
    parser.add_argument("--seed", type=int, default=base_random_seed)
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue sweep after a failed MOOSE solve")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    if not args.generate_only and shutil.which(args.moose_exe) is None and not os.path.isfile(args.moose_exe):
        raise RuntimeError("MOOSE executable not found: %s. Use --moose-exe or --generate-only." % args.moose_exe)

    failures = []
    for sim_idx in range(args.count):
        if args.count > 1:
            randomness = args.start + (args.end - args.start) * sim_idx / float(args.count - 1)
        else:
            randomness = args.start

        rng_seed = None if args.seed is None else args.seed + sim_idx
        rng = random.Random(rng_seed)

        print("\n" + "=" * 70)
        print("SIMULATION %d/%d: randomness = %.4f" %
              (sim_idx + 1, args.count, randomness))
        print("=" * 70)

        folder = os.path.join(output_root, "randomness_%.4f" % randomness)
        os.makedirs(folder, exist_ok=True)

        try:
            seed_points, segments = generate_segments(randomness, rng)
            print("Generated %d seed points" % len(seed_points))
            print("Final logical segments: %d" % len(segments))

            node_list, node_index = export_graph_files(folder, segments)
            nodes, elements, logical_edges, element_to_graph_edge, _coord_to_id = create_refined_mesh(segments)
            mesh_name = "lattice.msh"
            input_name = "lattice.i"
            write_gmsh_mesh(os.path.join(folder, mesh_name), nodes, elements)
            with open(os.path.join(folder, input_name), "w") as handle:
                handle.write(moose_input_text(mesh_name, "moose_results"))

            # Map each mesh element to its original graph edge for deterministic postprocessing.
            write_csv(os.path.join(folder, "element_to_graph_edge.csv"),
                      [["mesh_element_index", "graph_edge_index"]] +
                      [[i, edge] for i, edge in enumerate(element_to_graph_edge)])

            if args.generate_only:
                print("Generated MOOSE files in %s" % folder)
                continue

            print("Running MOOSE...")
            run_command([args.moose_exe, "-i", input_name], folder,
                        os.path.join(folder, "moose.log"))
            k_lattice = postprocess_moose(folder, logical_edges,
                                          element_to_graph_edge,
                                          node_list, node_index)
            print("Lattice stiffness: %.6g" % k_lattice)
            print("All files written to: %s" % folder)

        except Exception as exc:
            failures.append((sim_idx, randomness, str(exc)))
            print("ERROR: %s" % exc, file=sys.stderr)
            if not args.keep_going:
                raise

    print("\n" + "=" * 70)
    print("ALL REQUESTED SIMULATIONS COMPLETE")
    if failures:
        print("Failures: %d" % len(failures))
        for idx, randomness, message in failures:
            print("  simulation %d, randomness %.4f: %s" %
                  (idx, randomness, message))
    print("=" * 70)


if __name__ == "__main__":
    main()
