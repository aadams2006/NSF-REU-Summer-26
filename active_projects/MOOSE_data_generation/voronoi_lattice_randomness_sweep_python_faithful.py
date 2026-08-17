#!/usr/bin/env python3
"""
Standalone native-Python version of the original Abaqus Voronoi/random-lattice
sweep, written to preserve the documented generation / BC / export logic as
closely as practical without Abaqus or MOOSE.

Dependencies:
    pip install numpy scipy

Per-sample outputs (same four core files as the Abaqus script):
    node_features.csv
    adjacency_area.csv
    lattice_stiffness.csv
    stress_matrix.csv

Additional diagnostics:
    sample_metadata.csv

Faithfulness notes
------------------
GEOMETRY / DATASET LOGIC:
    The seed generation, randomness sweep, neighbor selection, random degree,
    overlap removal, side-wall thinning, corner repair, graph-node ordering,
    adjacency export, folder naming, and stiffness definition are kept directly
    aligned with the original script.

FINITE-ELEMENT REPLACEMENT:
    Abaqus is replaced by a linear 2-D Timoshenko frame solver with DOFs
    [ux, uy, rz].  The rectangular section uses
        A  = rect_a * rect_b
        Iz = rect_a * rect_b^3 / 12
    and G = E / (2*(1+nu)).  A rectangular shear factor kappa = 5/6 is used.

    The original Abaqus script meshes every beam using seed_size and then finds
    top/bottom BC nodes from the FE mesh with y_tol = seed_size*0.1.  To retain
    that behavior efficiently, this script explicitly sub-meshes every member
    that touches the top or bottom boundary (where intermediate mesh nodes can
    receive BCs).  Interior straight uniform members are represented by the
    analytically equivalent two-node Timoshenko member stiffness because they
    have no intermediate loads or constraints.  Use --full-submesh to subdivide
    every member at seed_size for a stricter (but much slower) mesh-level check.

STRESS LABEL:
    For every analysis element, local axial force and end bending moments are
    recovered.  S11 is represented as extreme-fiber normal stress
        sigma = N/A +/- M*c/Iz
    at both element ends.  For each original graph beam, the signed S11 value
    with the largest absolute magnitude across all of its analysis elements is
    written into the symmetric stress matrix.  This mirrors the documented
    Abaqus post-processing intent (max-|S11| per original beam), without the
    ODB midpoint-to-segment remapping because the parent beam is known exactly.

LIMITATION:
    This reproduces the documented mechanics assumptions, not Abaqus's internal
    implementation bit-for-bit.  
"""

from __future__ import print_function

import argparse
import csv
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve


# -----------------------------------------------------------------------------
# USER PARAMETERS -- same defaults as the original Abaqus script
# -----------------------------------------------------------------------------
DEFAULT_CELL_SIZE = 20.0
DEFAULT_N = 11

DEFAULT_RANDOMNESS_START = 0.0
DEFAULT_RANDOMNESS_END = 0.5
DEFAULT_RANDOMNESS_COUNT = 3000

DEFAULT_YOUNGS_MODULUS = 2000.0
DEFAULT_POISSONS_RATIO = 0.30

DEFAULT_RECT_A = 0.01  # out of plane
DEFAULT_RECT_B = 0.10  # in plane
DEFAULT_SEED_SIZE = 0.30

DEFAULT_DISPLACEMENT = 5.0
DEFAULT_TIME_PERIOD = 1.0


# -----------------------------------------------------------------------------
# HELPERS -- kept equivalent to the original script
# -----------------------------------------------------------------------------
def r6(p):
    return (round(float(p[0]), 6), round(float(p[1]), 6))


def seg_key(p, q):
    p2, q2 = r6(p), r6(q)
    return (p2, q2) if p2 <= q2 else (q2, p2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, q1, p2, q2):
    if (
        r6(p1) == r6(p2)
        or r6(p1) == r6(q2)
        or r6(q1) == r6(p2)
        or r6(q1) == r6(q2)
    ):
        return False
    return (ccw(p1, p2, q2) != ccw(q1, p2, q2)) and (
        ccw(p1, q1, p2) != ccw(p1, q1, q2)
    )


def is_boundary_point(pt, y_bottom_fixed, y_top_fixed):
    return (abs(pt[1] - y_bottom_fixed) < 1e-6) or (
        abs(pt[1] - y_top_fixed) < 1e-6
    )


# -----------------------------------------------------------------------------
# ORIGINAL LATTICE GENERATION LOGIC
# -----------------------------------------------------------------------------
def generate_lattice(cell_size, n, randomness, rng, verbose=False):
    cell_height = math.sqrt(3.0) * cell_size

    # -----------------------------
    # SEED POINTS (WITH RANDOMNESS)
    # -----------------------------
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

    bottom_points = []
    top_points = []
    interior_points = []

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
    for x, y, i, j in bottom_points:
        seed_points.append((x, y_bottom_fixed))
    for x, y, i, j in interior_points:
        seed_points.append((x, y))
    for x, y, i, j in top_points:
        seed_points.append((x, y_top_fixed))

    if verbose:
        print("Generated %d seed points" % len(seed_points))

    xs = [p[0] for p in seed_points]
    left_bound = min(xs)
    right_bound = max(xs)

    # -----------------------------
    # BUILD SEGMENTS
    # -----------------------------
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

        neighbors.sort(key=lambda x: x[0])
        max_candidates = min(12, len(neighbors))
        candidates = neighbors[:max_candidates]

        rng.shuffle(candidates)
        desired = rng.randint(3, 6)
        num_connections = min(desired, max_candidates)

        for k in range(num_connections):
            _, _, p2 = candidates[k]
            raw_seg_keys.append(seg_key(p1, p2))

    segments_dict = {}
    for key in raw_seg_keys:
        if key not in segments_dict:
            segments_dict[key] = key

    if verbose:
        print("Initial segments: %d" % len(segments_dict))

    # -----------------------------
    # REMOVE OVERLAPPING BEAMS
    # -----------------------------
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
                if segments_intersect(p1, q1, p2, q2):
                    s1_boundary = is_boundary_point(
                        p1, y_bottom_fixed, y_top_fixed
                    ) and is_boundary_point(q1, y_bottom_fixed, y_top_fixed)
                    s2_boundary = is_boundary_point(
                        p2, y_bottom_fixed, y_top_fixed
                    ) and is_boundary_point(q2, y_bottom_fixed, y_top_fixed)

                    if s1_boundary and not s2_boundary:
                        remove_index = j
                    elif s2_boundary and not s1_boundary:
                        remove_index = i
                    else:
                        len1 = distance(p1, q1)
                        len2 = distance(p2, q2)
                        remove_index = i if len1 > len2 else j

                    del segment_list[remove_index]
                    changed = True
                    break

    if verbose:
        print("Segments after overlap removal: %d" % len(segment_list))

    # -----------------------------
    # THIN SIDE WALLS
    # -----------------------------
    side_tol = cell_size * 1.0
    side_keep_prob = 0.45

    filtered_segment_list = []
    for p, q in segment_list:
        left_side = p[0] < left_bound + side_tol and q[0] < left_bound + side_tol
        right_side = p[0] > right_bound - side_tol and q[0] > right_bound - side_tol

        if left_side or right_side:
            if rng.random() < side_keep_prob:
                filtered_segment_list.append((p, q))
        else:
            filtered_segment_list.append((p, q))

    segment_list = filtered_segment_list

    if verbose:
        print("Segments after thinning sides: %d" % len(segment_list))

    # -----------------------------
    # FIX CORNER NODES
    # -----------------------------
    corner_joint_coords = set()
    for p, q in segment_list:
        corner_joint_coords.add(r6(p))
        corner_joint_coords.add(r6(q))

    if corner_joint_coords:
        ys_all = [pt[1] for pt in corner_joint_coords]
        bottom_y = min(ys_all)
        y_tol_corner = cell_size * 0.2

        bottom_nodes = [
            pt for pt in corner_joint_coords if abs(pt[1] - bottom_y) < y_tol_corner
        ]

        if bottom_nodes:
            bottom_left_pt = min(bottom_nodes, key=lambda pt: pt[0])
            bottom_right_pt = max(bottom_nodes, key=lambda pt: pt[0])
            corner_pts = [bottom_left_pt, bottom_right_pt]

            deg = defaultdict(int)
            for p, q in segment_list:
                deg[r6(p)] += 1
                deg[r6(q)] += 1

            for corner_pt in corner_pts:
                if deg[r6(corner_pt)] <= 1:
                    x_c, y_c = corner_pt
                    best_pt = None
                    best_dy = None
                    x_tol = cell_size * 0.7

                    for pt in corner_joint_coords:
                        if pt[1] <= y_c + 1e-6:
                            continue
                        if abs(pt[0] - x_c) > x_tol:
                            continue

                        dy = pt[1] - y_c
                        if best_dy is None or dy < best_dy:
                            best_dy = dy
                            best_pt = pt

                    if best_pt is not None:
                        exists = False
                        for p, q in segment_list:
                            if (
                                r6(p) == r6(corner_pt)
                                and r6(q) == r6(best_pt)
                            ) or (
                                r6(p) == r6(best_pt)
                                and r6(q) == r6(corner_pt)
                            ):
                                exists = True
                                break

                        if not exists:
                            segment_list.append((corner_pt, best_pt))

    # -----------------------------
    # FINAL DEDUPLICATION
    # -----------------------------
    segments = {}
    for p, q in segment_list:
        key = seg_key(p, q)
        # Keep the original point values as the Abaqus source does. r6 is used
        # only for canonical identity / exported graph coordinates.
        segments[key] = (p, q)

    if verbose:
        print("Final segments: %d" % len(segments))

    return segments


# -----------------------------------------------------------------------------
# GRAPH DATA -- same exported graph definition as the original script
# -----------------------------------------------------------------------------
def build_graph_data(segments):
    joint_coords = set()
    for p, q in segments.values():
        joint_coords.add(r6(p))
        joint_coords.add(r6(q))

    node_list = sorted(list(joint_coords))
    node_index = {xy: idx for idx, xy in enumerate(node_list)}
    N = len(node_list)

    adj_binary = np.zeros((N, N), dtype=np.int8)
    graph_edges = []

    for p, q in segments.values():
        i = node_index[r6(p)]
        j = node_index[r6(q)]
        if i != j:
            adj_binary[i, j] = 1
            adj_binary[j, i] = 1
            graph_edges.append((i, j, p, q))

    return node_list, node_index, graph_edges, adj_binary


# -----------------------------------------------------------------------------
# 2-D TIMOSHENKO FRAME FINITE ELEMENT
# -----------------------------------------------------------------------------
def timoshenko_element_matrices(x1, y1, x2, y2, E, nu, A, I, kappa=5.0 / 6.0):
    """Return k_global, T, k_local, L for a 2-D Timoshenko frame member."""
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    L = math.hypot(dx, dy)
    if L <= 1e-12:
        raise ValueError("Zero-length member encountered")

    c = dx / L
    s = dy / L

    G = E / (2.0 * (1.0 + nu))
    shear_stiffness = kappa * G * A
    if shear_stiffness <= 0:
        raise ValueError("Invalid shear stiffness")

    phi = 12.0 * E * I / (shear_stiffness * L * L)
    den = 1.0 + phi

    EA_L = E * A / L
    k22 = 12.0 * E * I / (den * L ** 3)
    k23 = 6.0 * E * I / (den * L ** 2)
    k33 = (4.0 + phi) * E * I / (den * L)
    k36 = (2.0 - phi) * E * I / (den * L)

    k_local = np.array(
        [
            [EA_L, 0.0, 0.0, -EA_L, 0.0, 0.0],
            [0.0, k22, k23, 0.0, -k22, k23],
            [0.0, k23, k33, 0.0, -k23, k36],
            [-EA_L, 0.0, 0.0, EA_L, 0.0, 0.0],
            [0.0, -k22, -k23, 0.0, k22, -k23],
            [0.0, k23, k36, 0.0, -k23, k33],
        ],
        dtype=float,
    )

    # u_local = T @ u_global_element
    T = np.array(
        [
            [c, s, 0.0, 0.0, 0.0, 0.0],
            [-s, c, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c, s, 0.0],
            [0.0, 0.0, 0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    k_global = T.T @ k_local @ T
    return k_global, T, k_local, L, phi


def _n_elements_for_seed(L, seed_size):
    """
    Approximate Abaqus global edge seeding: choose an integer element count whose
    element length is close to target seed_size.  Exact Abaqus internal seeding
    choices are not available outside Abaqus.
    """
    if seed_size <= 0:
        raise ValueError("seed_size must be positive")
    return max(1, int(round(float(L) / float(seed_size))))


def build_analysis_mesh(node_list, graph_edges, seed_size, full_submesh=False):
    """
    Build FE nodes/elements while retaining exact parent graph-edge identity.

    Graph joints are inserted first and retain FE-node ids 0..N_graph-1.
    Members touching the global top/bottom boundary are subdivided at seed_size
    because the original Abaqus BCs act on FE mesh nodes along those boundaries.
    Interior members use one exact Timoshenko member unless --full-submesh.
    """
    fe_nodes = [(float(x), float(y)) for x, y in node_list]
    analysis_elements = []

    ys = [y for _, y in node_list]
    graph_y_min = min(ys)
    graph_y_max = max(ys)
    y_tol = seed_size * 0.1

    for parent_id, (gi, gj, p_raw, q_raw) in enumerate(graph_edges):
        p = node_list[gi]
        q = node_list[gj]
        L = distance(p, q)

        touches_boundary = (
            abs(p[1] - graph_y_min) < y_tol
            or abs(q[1] - graph_y_min) < y_tol
            or abs(p[1] - graph_y_max) < y_tol
            or abs(q[1] - graph_y_max) < y_tol
        )

        if full_submesh or touches_boundary:
            n_elem = _n_elements_for_seed(L, seed_size)
        else:
            n_elem = 1

        chain = [gi]
        for k in range(1, n_elem):
            t = float(k) / float(n_elem)
            x = p[0] + t * (q[0] - p[0])
            y = p[1] + t * (q[1] - p[1])
            fe_nodes.append((x, y))
            chain.append(len(fe_nodes) - 1)
        chain.append(gj)

        for a, b in zip(chain[:-1], chain[1:]):
            analysis_elements.append((a, b, parent_id))

    return fe_nodes, analysis_elements


def connected_components(node_count, elements):
    neighbors = [[] for _ in range(node_count)]
    for i, j, _ in elements:
        neighbors[i].append(j)
        neighbors[j].append(i)

    seen = set()
    comps = []
    for start in range(node_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neighbors[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def solve_frame(
    node_list,
    graph_edges,
    E,
    nu,
    rect_a,
    rect_b,
    seed_size,
    displacement_magnitude,
    full_submesh=False,
):
    """Solve one lattice and return stiffness + max-|S11| per graph beam."""
    A = rect_a * rect_b
    I = rect_a * (rect_b ** 3) / 12.0
    c_extreme = rect_b / 2.0
    kappa = 5.0 / 6.0

    fe_nodes, analysis_elements = build_analysis_mesh(
        node_list=node_list,
        graph_edges=graph_edges,
        seed_size=seed_size,
        full_submesh=full_submesh,
    )

    n_fe = len(fe_nodes)
    ndof = 3 * n_fe

    rows = []
    cols = []
    vals = []
    element_cache = []

    for elem_id, (i, j, parent_id) in enumerate(analysis_elements):
        x1, y1 = fe_nodes[i]
        x2, y2 = fe_nodes[j]
        k_g, T, k_l, L, phi = timoshenko_element_matrices(
            x1, y1, x2, y2, E, nu, A, I, kappa=kappa
        )
        dofs = np.array(
            [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2],
            dtype=int,
        )

        for a in range(6):
            for b in range(6):
                rows.append(int(dofs[a]))
                cols.append(int(dofs[b]))
                vals.append(float(k_g[a, b]))

        element_cache.append((i, j, parent_id, dofs, T, k_l, L, phi))

    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # -----------------------------
    # BOUNDARY CONDITIONS -- same mesh-node selection logic as Abaqus source
    # -----------------------------
    ys = np.array([p[1] for p in fe_nodes], dtype=float)
    actual_y_min = float(ys.min())
    actual_y_max = float(ys.max())
    y_tol = seed_size * 0.1

    bottom_nodes = [
        i for i, (_, y) in enumerate(fe_nodes) if abs(y - actual_y_min) < y_tol
    ]
    top_nodes = [
        i for i, (_, y) in enumerate(fe_nodes) if abs(y - actual_y_max) < y_tol
    ]

    if not bottom_nodes or not top_nodes:
        raise RuntimeError("Could not find top/bottom nodes")

    bottom_left = min(bottom_nodes, key=lambda idx: fe_nodes[idx][0])

    prescribed = {}
    # BottomLeftFix: u1=0, u2=0, ur3=0
    prescribed[3 * bottom_left] = 0.0
    prescribed[3 * bottom_left + 1] = 0.0
    prescribed[3 * bottom_left + 2] = 0.0

    # BottomLineU2Fix: all other bottom FE nodes have only u2 fixed
    for idx in bottom_nodes:
        if idx != bottom_left:
            prescribed[3 * idx + 1] = 0.0

    # TopDisplacement: top FE nodes have only u2 prescribed
    for idx in top_nodes:
        prescribed[3 * idx + 1] = float(displacement_magnitude)

    constrained = np.array(sorted(prescribed.keys()), dtype=int)
    all_dofs = np.arange(ndof, dtype=int)
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[constrained] = False
    free = all_dofs[free_mask]

    u = np.zeros(ndof, dtype=float)
    for dof, value in prescribed.items():
        u[dof] = value

    if len(free) == 0:
        raise RuntimeError("No free DOFs remain")

    # Detect disconnected FE components before sparse solve.
    comps = connected_components(n_fe, analysis_elements)
    if len(comps) != 1:
        raise RuntimeError("Disconnected lattice (%d connected components)" % len(comps))

    K_ff = K[free][:, free]
    K_fc = K[free][:, constrained]
    rhs = -(K_fc @ u[constrained])

    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            u_free = spsolve(K_ff, rhs)
        except MatrixRankWarning as exc:
            raise RuntimeError("Singular frame stiffness matrix / mechanism") from exc

    if not np.all(np.isfinite(u_free)):
        raise RuntimeError("Non-finite displacement solution (likely a mechanism)")

    u[free] = u_free

    # With no applied nodal loads, K*u is the reaction vector at prescribed DOFs.
    reactions = K @ u
    total_RF2 = float(sum(reactions[3 * idx + 1] for idx in top_nodes))
    k_lattice = total_RF2 / float(displacement_magnitude)

    if not np.isfinite(k_lattice):
        raise RuntimeError("Non-finite lattice stiffness")

    # -----------------------------
    # STRESS RECOVERY -- max absolute signed S11 per original graph beam
    # -----------------------------
    parent_stress = {}
    parent_element_count = defaultdict(int)
    max_phi = 0.0

    for i, j, parent_id, dofs, T, k_l, L, phi in element_cache:
        max_phi = max(max_phi, abs(float(phi)))
        parent_element_count[parent_id] += 1

        u_elem_global = u[dofs]
        u_local = T @ u_elem_global
        f_local = k_l @ u_local

        # f_local = [N1, V1, M1, N2, V2, M2] in element-end sign convention.
        # Convert the second-end generalized forces to a consistent section sign.
        N1 = float(f_local[0])
        M1 = float(f_local[2])
        N2 = float(-f_local[3])
        M2 = float(-f_local[5])

        candidates = [
            N1 / A + M1 * c_extreme / I,
            N1 / A - M1 * c_extreme / I,
            N2 / A + M2 * c_extreme / I,
            N2 / A - M2 * c_extreme / I,
        ]
        elem_s11 = max(candidates, key=lambda x: abs(x))

        if parent_id not in parent_stress or abs(elem_s11) > abs(parent_stress[parent_id]):
            parent_stress[parent_id] = float(elem_s11)

    edge_stresses = {}
    for parent_id, (gi, gj, _, _) in enumerate(graph_edges):
        key = (min(gi, gj), max(gi, gj))
        if parent_id in parent_stress:
            edge_stresses[key] = parent_stress[parent_id]

    return {
        "k_lattice": float(k_lattice),
        "edge_stresses": edge_stresses,
        "displacements": u,
        "reactions": reactions,
        "top_nodes": top_nodes,
        "bottom_nodes": bottom_nodes,
        "bottom_left": bottom_left,
        "num_graph_nodes": len(node_list),
        "num_graph_members": len(graph_edges),
        "num_fe_nodes": n_fe,
        "num_fe_elements": len(analysis_elements),
        "num_dofs": ndof,
        "area": A,
        "second_moment": I,
        "shear_modulus": E / (2.0 * (1.0 + nu)),
        "shear_kappa": kappa,
        "max_timoshenko_phi": max_phi,
        "full_submesh": bool(full_submesh),
        "parent_element_count": dict(parent_element_count),
    }


# -----------------------------------------------------------------------------
# CSV EXPORTS -- same core file layout as original
# -----------------------------------------------------------------------------
def write_node_features(folder, node_list):
    path = os.path.join(folder, "node_features.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x", "y"])
        for i, (x, y) in enumerate(node_list):
            w.writerow([i, x, y])


def write_adjacency(folder, adj_binary):
    N = adj_binary.shape[0]
    path = os.path.join(folder, "adjacency_area.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(range(N)))
        for i in range(N):
            w.writerow([i] + [int(v) for v in adj_binary[i, :]])


def write_stiffness(folder, k_lattice):
    path = os.path.join(folder, "lattice_stiffness.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k_lattice_N_per_mm"])
        w.writerow([k_lattice])


def write_stress_matrix(folder, N, edge_stresses):
    stress_matrix = np.zeros((N, N), dtype=float)
    for (i, j), s_val in edge_stresses.items():
        stress_matrix[i, j] = s_val
        stress_matrix[j, i] = s_val

    path = os.path.join(folder, "stress_matrix.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(range(N)))
        for i in range(N):
            w.writerow([i] + [float(v) for v in stress_matrix[i, :]])


def write_metadata(folder, rows):
    path = os.path.join(folder, "sample_metadata.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for key, value in rows:
            w.writerow([key, value])


def remove_partial_label_files(folder):
    for name in ("lattice_stiffness.csv", "stress_matrix.csv"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            os.remove(path)


# -----------------------------------------------------------------------------
# ONE SAMPLE / MAIN SWEEP
# -----------------------------------------------------------------------------
def run_one_sample(sim_idx, randomness, args, rng):
    folder_name = "randomness_%.4f" % randomness
    folder_path = os.path.abspath(os.path.join(args.output_dir, folder_name))
    os.makedirs(folder_path, exist_ok=True)

    t0 = time.perf_counter()

    segments = generate_lattice(
        args.cell_size, args.n, randomness, rng, verbose=args.verbose
    )
    if not segments:
        raise RuntimeError("Generated lattice has no members")

    node_list, node_index, graph_edges, adj_binary = build_graph_data(segments)

    # The original writes these graph files regardless of ODB post-processing.
    write_node_features(folder_path, node_list)
    write_adjacency(folder_path, adj_binary)

    solve_t0 = time.perf_counter()
    result = solve_frame(
        node_list=node_list,
        graph_edges=graph_edges,
        E=args.youngs_modulus,
        nu=args.poissons_ratio,
        rect_a=args.rect_a,
        rect_b=args.rect_b,
        seed_size=args.seed_size,
        displacement_magnitude=args.displacement,
        full_submesh=args.full_submesh,
    )
    solve_seconds = time.perf_counter() - solve_t0

    write_stiffness(folder_path, result["k_lattice"])
    write_stress_matrix(folder_path, len(node_list), result["edge_stresses"])

    total_seconds = time.perf_counter() - t0

    write_metadata(
        folder_path,
        [
            ("status", "complete"),
            ("simulation_index", sim_idx),
            ("randomness", randomness),
            ("graph_nodes", result["num_graph_nodes"]),
            ("graph_members", result["num_graph_members"]),
            ("fe_nodes", result["num_fe_nodes"]),
            ("fe_elements", result["num_fe_elements"]),
            ("dofs", result["num_dofs"]),
            ("youngs_modulus", args.youngs_modulus),
            ("poissons_ratio", args.poissons_ratio),
            ("rect_a_out_of_plane", args.rect_a),
            ("rect_b_in_plane", args.rect_b),
            ("area", result["area"]),
            ("I_in_plane_bending", result["second_moment"]),
            ("shear_modulus", result["shear_modulus"]),
            ("shear_kappa", result["shear_kappa"]),
            ("max_timoshenko_phi", result["max_timoshenko_phi"]),
            ("seed_size", args.seed_size),
            ("displacement_magnitude", args.displacement),
            ("time_period_metadata_only", args.time_period),
            ("full_submesh", result["full_submesh"]),
            ("k_lattice_N_per_mm", result["k_lattice"]),
            ("solve_seconds", solve_seconds),
            ("total_seconds", total_seconds),
        ],
    )

    return {
        "folder": folder_path,
        "nodes": result["num_graph_nodes"],
        "members": result["num_graph_members"],
        "fe_nodes": result["num_fe_nodes"],
        "fe_elements": result["num_fe_elements"],
        "k": result["k_lattice"],
        "solve_seconds": solve_seconds,
        "total_seconds": total_seconds,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate the original random-lattice sweep and solve it with a "
            "native-Python Timoshenko frame FEM replacement for Abaqus."
        )
    )
    p.add_argument("--count", type=int, default=DEFAULT_RANDOMNESS_COUNT)
    p.add_argument("--start", type=float, default=DEFAULT_RANDOMNESS_START)
    p.add_argument("--end", type=float, default=DEFAULT_RANDOMNESS_END)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional global RNG seed. If omitted, stochastic behavior matches the "
            "original script's unseeded random module."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=".",
        help="Output root. Default '.' matches the original per-randomness folders.",
    )

    p.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--youngs-modulus", type=float, default=DEFAULT_YOUNGS_MODULUS)
    p.add_argument("--poissons-ratio", type=float, default=DEFAULT_POISSONS_RATIO)
    p.add_argument("--rect-a", type=float, default=DEFAULT_RECT_A)
    p.add_argument("--rect-b", type=float, default=DEFAULT_RECT_B)
    p.add_argument("--seed-size", type=float, default=DEFAULT_SEED_SIZE)
    p.add_argument("--displacement", type=float, default=DEFAULT_DISPLACEMENT)
    p.add_argument("--time-period", type=float, default=DEFAULT_TIME_PERIOD)

    p.add_argument(
        "--full-submesh",
        action="store_true",
        help=(
            "Subdivide every graph beam at seed_size. More closely mirrors the "
            "Abaqus mesh but is much slower and uses much more memory."
        ),
    )
    p.add_argument(
        "--skip-complete",
        action="store_true",
        help=(
            "Skip solving/writing samples with both label CSVs already present. "
            "Geometry is still regenerated so the global RNG stream stays aligned."
        ),
    )
    p.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the sweep on the first failed solve. Original behavior is to continue.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.rect_a <= 0 or args.rect_b <= 0:
        raise ValueError("Section dimensions must be positive")
    if args.youngs_modulus <= 0:
        raise ValueError("Young's modulus must be positive")
    if args.poissons_ratio <= -1.0:
        raise ValueError("Poisson's ratio must be greater than -1")
    if args.seed_size <= 0:
        raise ValueError("seed_size must be positive")
    if args.displacement == 0:
        raise ValueError("Displacement must be nonzero")

    os.makedirs(args.output_dir, exist_ok=True)

    # One continuous RNG stream matches the original script. Seeding is optional.
    rng = random.Random()
    if args.seed is not None:
        rng.seed(args.seed)

    summary_path = os.path.join(args.output_dir, "sweep_summary.csv")
    summary_exists = os.path.exists(summary_path)
    summary_f = open(summary_path, "a", newline="")
    summary_w = csv.writer(summary_f)
    if not summary_exists:
        summary_w.writerow(
            [
                "sim_idx",
                "randomness",
                "status",
                "graph_nodes",
                "graph_members",
                "fe_nodes",
                "fe_elements",
                "k_lattice_N_per_mm",
                "solve_seconds",
                "total_seconds",
                "message",
            ]
        )

    completed = 0
    failed = 0
    skipped = 0
    sweep_t0 = time.perf_counter()

    try:
        for sim_idx in range(args.count):
            if args.count > 1:
                randomness = args.start + (args.end - args.start) * sim_idx / float(args.count - 1)
            else:
                randomness = args.start

            folder_name = "randomness_%.4f" % randomness
            folder_path = os.path.abspath(os.path.join(args.output_dir, folder_name))
            stiff_path = os.path.join(folder_path, "lattice_stiffness.csv")
            stress_path = os.path.join(folder_path, "stress_matrix.csv")

            print("[%d/%d] randomness=%.4f" % (sim_idx + 1, args.count, randomness))

            # For skip-complete, regenerate geometry to consume exactly the random
            # draws this sample would have consumed in the original global RNG stream.
            if args.skip_complete and os.path.exists(stiff_path) and os.path.exists(stress_path):
                _ = generate_lattice(args.cell_size, args.n, randomness, rng, verbose=False)
                skipped += 1
                print("  SKIP complete (RNG stream advanced)")
                continue

            try:
                info = run_one_sample(sim_idx, randomness, args, rng)
                completed += 1
                print(
                    "  graph=%dN/%dE fe=%dN/%dE k=%.6g solve=%.3fs total=%.3fs"
                    % (
                        info["nodes"],
                        info["members"],
                        info["fe_nodes"],
                        info["fe_elements"],
                        info["k"],
                        info["solve_seconds"],
                        info["total_seconds"],
                    )
                )
                summary_w.writerow(
                    [
                        sim_idx,
                        randomness,
                        "complete",
                        info["nodes"],
                        info["members"],
                        info["fe_nodes"],
                        info["fe_elements"],
                        info["k"],
                        info["solve_seconds"],
                        info["total_seconds"],
                        "",
                    ]
                )
                summary_f.flush()

            except Exception as exc:
                failed += 1
                os.makedirs(folder_path, exist_ok=True)
                # Match the original intent: geometry files may exist even if FE
                # post-processing labels fail; do not leave partial label files.
                remove_partial_label_files(folder_path)
                msg = "%s: %s" % (type(exc).__name__, str(exc))
                print("  FAILED:", msg)
                write_metadata(
                    folder_path,
                    [
                        ("status", "failed"),
                        ("simulation_index", sim_idx),
                        ("randomness", randomness),
                        ("message", msg),
                    ],
                )
                summary_w.writerow(
                    [sim_idx, randomness, "failed", "", "", "", "", "", "", "", msg]
                )
                summary_f.flush()
                if args.stop_on_failure:
                    raise

    finally:
        summary_f.close()

    elapsed = time.perf_counter() - sweep_t0
    print("\n" + "=" * 70)
    print("ALL SIMULATIONS COMPLETE")
    print("Completed: %d" % completed)
    print("Failed:    %d" % failed)
    print("Skipped:   %d" % skipped)
    print("Elapsed:   %.2f s" % elapsed)
    print("Output:    %s" % os.path.abspath(args.output_dir))
    print("=" * 70)


if __name__ == "__main__":
    main()
