# 2D VORONOI LATTICE WITH SWEPT RANDOMNESS PARAMETER
# Updated:
# - Sweeps randomness from start to end value
# - Creates separate folder for each simulation
# - Exports simplified files: node_features, adjacency (binary), stiffness (k only), stress_matrix

from abaqus import *
from abaqusConstants import *
import regionToolset, mesh, math
import job
import random
import csv, os

# -----------------------------
# USER PARAMETERS
# -----------------------------
cell_size      = 20.0
cell_height    = math.sqrt(3.0) * cell_size
n              = 11  # Grid density

# SWEPT RANDOMNESS PARAMETERS:
randomness_start = 0.0    # Starting randomness value
randomness_end   = 0.5    # Ending randomness value
randomness_count = 3000     # Number of simulations (will create this many increments)

youngs_modulus = 2000.0
poissons_ratio = 0.30

rect_a         = 0.01 #Out of Plane
rect_b         = 0.10 #In plane
seed_size      = 0.30

# Displacement parameters
displacement_magnitude = 5.0
time_period = 1.0

# -----------------------------
# HELPERS
# -----------------------------
def r6(p):
    return (round(p[0], 6), round(p[1], 6))

def seg_key(p, q):
    p2, q2 = r6(p), r6(q)
    return (p2, q2) if p2 <= q2 else (q2, p2)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def ccw(a, b, c):
    return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

def segments_intersect(p1, q1, p2, q2):
    if (r6(p1) == r6(p2) or r6(p1) == r6(q2) or
        r6(q1) == r6(p2) or r6(q1) == r6(q2)):
        return False
    return (ccw(p1, p2, q2) != ccw(q1, p2, q2)) and \
           (ccw(p1, q1, p2) != ccw(p1, q1, q2))

def is_boundary_point(pt, y_bottom_fixed, y_top_fixed):
    return (abs(pt[1] - y_bottom_fixed) < 1e-6) or (abs(pt[1] - y_top_fixed) < 1e-6)

# -----------------------------
# MAIN LOOP OVER RANDOMNESS VALUES
# -----------------------------
for sim_idx in range(randomness_count):
    # Calculate current randomness value
    if randomness_count > 1:
        randomness = randomness_start + (randomness_end - randomness_start) * sim_idx / float(randomness_count - 1)
    else:
        randomness = randomness_start
    
    print("\n" + "="*70)
    print("SIMULATION %d/%d: randomness = %.4f" % (sim_idx + 1, randomness_count, randomness))
    print("="*70)
    
    # Create output folder for this simulation
    folder_name = "randomness_%.4f" % randomness
    folder_path = os.path.abspath(folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Store original directory
    original_dir = os.getcwd()
    print("Output folder: %s" % folder_path)
    
    # -----------------------------
    # SEED POINTS (WITH RANDOMNESS)
    # -----------------------------
    seed_points = []
    
    temp_points = []
    for i in range(n):
        n_y = n
        for j in range(n_y):
            x0 = i * 1.5 * cell_size
            y0 = j * cell_height + (i % 2) * (cell_height/2.0)

            if randomness > 0:
                max_offset = randomness * cell_size
                x_offset = random.uniform(-max_offset, max_offset)
                y_offset = random.uniform(-max_offset, max_offset)

                if i == 0 or i == n-1:
                    x_offset = 0.0
                if j == 0 or j == n_y-1:
                    y_offset = 0.0

                x0 += x_offset
                y0 += y_offset

            temp_points.append((x0, y0, i, j))

    # Identify y bounds to tag top and bottom rows
    y_coords = [p[1] for p in temp_points]
    y_min, y_max = min(y_coords), max(y_coords)
    y_tolerance = cell_height * 0.3

    bottom_points = []
    top_points = []
    interior_points = []

    for (x, y, i, j) in temp_points:
        if abs(y - y_min) < y_tolerance:
            bottom_points.append((x, y, i, j))
        elif abs(y - y_max) < y_tolerance:
            top_points.append((x, y, i, j))
        else:
            interior_points.append((x, y, i, j))

    bottom_points.sort(key=lambda p: p[0])
    top_points.sort(key=lambda p: p[0])

    y_bottom_fixed = y_min
    y_top_fixed    = y_max

    seed_points = []
    for (x, y, i, j) in bottom_points:
        seed_points.append((x, y_bottom_fixed))
    for (x, y, i, j) in interior_points:
        seed_points.append((x, y))
    for (x, y, i, j) in top_points:
        seed_points.append((x, y_top_fixed))

    print("Generated %d seed points" % len(seed_points))

    xs = [p[0] for p in seed_points]
    ys = [p[1] for p in seed_points]
    left_bound   = min(xs)
    right_bound  = max(xs)
    bottom_bound = min(ys)
    top_bound    = max(ys)

    padding = cell_size * 2.0
    xmin = left_bound  - padding
    xmax = right_bound + padding
    ymin = bottom_bound - padding
    ymax = top_bound    + padding

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

        random.shuffle(candidates)
        desired = random.randint(3, 6)
        num_connections = min(desired, max_candidates)

        for k in range(num_connections):
            d, j, p2 = candidates[k]
            key = seg_key(p1, p2)
            raw_seg_keys.append(key)

    segments_dict = {}
    for key in raw_seg_keys:
        if key not in segments_dict:
            segments_dict[key] = key

    print("Initial segments: %d" % len(segments_dict))

    # -----------------------------
    # REMOVE OVERLAPPING BEAMS
    # -----------------------------
    segment_list = [(pq[0], pq[1]) for pq in segments_dict.values()]

    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        num_segs = len(segment_list)
        for i in range(num_segs):
            if changed:
                break
            p1, q1 = segment_list[i]
            for j in range(i+1, num_segs):
                p2, q2 = segment_list[j]
                if segments_intersect(p1, q1, p2, q2):
                    s1_boundary = is_boundary_point(p1, y_bottom_fixed, y_top_fixed) and is_boundary_point(q1, y_bottom_fixed, y_top_fixed)
                    s2_boundary = is_boundary_point(p2, y_bottom_fixed, y_top_fixed) and is_boundary_point(q2, y_bottom_fixed, y_top_fixed)

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

    print("Segments after overlap removal: %d" % len(segment_list))

    # -----------------------------
    # THIN SIDE WALLS
    # -----------------------------
    side_tol = cell_size * 1.0
    side_keep_prob = 0.45

    filtered_segment_list = []
    for (p, q) in segment_list:
        left_side  = (p[0] < left_bound  + side_tol and q[0] < left_bound  + side_tol)
        right_side = (p[0] > right_bound - side_tol and q[0] > right_bound - side_tol)

        if left_side or right_side:
            if random.random() < side_keep_prob:
                filtered_segment_list.append((p, q))
        else:
            filtered_segment_list.append((p, q))

    segment_list = filtered_segment_list
    print("Segments after thinning sides: %d" % len(segment_list))

    # -----------------------------
    # FIX CORNER NODES
    # -----------------------------
    from collections import defaultdict

    corner_joint_coords = set()
    for (p, q) in segment_list:
        corner_joint_coords.add(r6(p))
        corner_joint_coords.add(r6(q))

    if corner_joint_coords:
        ys_all = [pt[1] for pt in corner_joint_coords]
        bottom_y = min(ys_all)
        y_tol_corner = cell_size * 0.2

        bottom_nodes = [pt for pt in corner_joint_coords
                        if abs(pt[1] - bottom_y) < y_tol_corner]

        if bottom_nodes:
            bottom_left_pt  = min(bottom_nodes, key=lambda pt: pt[0])
            bottom_right_pt = max(bottom_nodes, key=lambda pt: pt[0])
            corner_pts = [bottom_left_pt, bottom_right_pt]

            deg = defaultdict(int)
            for (p, q) in segment_list:
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
                        for (p, q) in segment_list:
                            if (r6(p) == r6(corner_pt) and r6(q) == r6(best_pt)) or \
                               (r6(p) == r6(best_pt) and r6(q) == r6(corner_pt)):
                                exists = True
                                break

                        if not exists:
                            segment_list.append((corner_pt, best_pt))

    segments = {}
    for (p, q) in segment_list:
        key = seg_key(p, q)
        segments[key] = (p, q)

    print("Final segments: %d" % len(segments))

    # -----------------------------
    # PART / MATERIAL / SECTION
    # -----------------------------
    model_name = "VoronoiLattice_%d" % sim_idx
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    sk = model.ConstrainedSketch(name='__profile__',
                                 sheetSize=max(xmax-xmin, ymax-ymin)*1.5)
    for (p, q) in segments.values():
        sk.Line(point1=p, point2=q)

    part = model.Part(name='VoronoiPart',
                      dimensionality=TWO_D_PLANAR,
                      type=DEFORMABLE_BODY)
    part.BaseWire(sketch=sk)

    if "LatticeMaterial" in model.materials.keys():
        del model.materials["LatticeMaterial"]
    model.Material(name="LatticeMaterial")
    model.materials["LatticeMaterial"].Elastic(table=((youngs_modulus, poissons_ratio),))

    if "WireProfile" in model.profiles.keys():
        del model.profiles["WireProfile"]
    model.RectangularProfile(name="WireProfile", a=rect_a, b=rect_b)

    if "LatticeSection" in model.sections.keys():
        del model.sections["LatticeSection"]
    model.BeamSection(name="LatticeSection",
                      integration=DURING_ANALYSIS,
                      profile="WireProfile",
                      material="LatticeMaterial")

    all_edges = part.edges[:]
    reg = part.Set(edges=all_edges, name='AllEdges')
    part.SectionAssignment(region=reg, sectionName="LatticeSection")
    part.assignBeamSectionOrientation(region=reg,
                                      method=N1_COSINES,
                                      n1=(0.0, 0.0, 1.0))

    # -----------------------------
    # MESH
    # -----------------------------
    part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    # -----------------------------
    # ASSEMBLY
    # -----------------------------
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    if "VoronoiInstance" in a.instances.keys():
        del a.instances["VoronoiInstance"]
    inst = a.Instance(name='VoronoiInstance', part=part, dependent=ON)

    # -----------------------------
    # STEP
    # -----------------------------
    model.StaticStep(name='LoadStep', previous='Initial',
                     description='Uniaxial displacement',
                     timePeriod=time_period, nlgeom=OFF,
                     initialInc=0.1, minInc=1e-5, maxInc=0.5,
                     maxNumInc=1000)

    # -----------------------------
    # BOUNDARY CONDITIONS
    # -----------------------------
    all_node_y = [nd.coordinates[1] for nd in inst.nodes]
    actual_y_min = min(all_node_y)
    actual_y_max = max(all_node_y)

    y_tol = seed_size * 0.1

    bottom_nodes = [nd for nd in inst.nodes
                    if abs(nd.coordinates[1] - actual_y_min) < y_tol]
    top_nodes = [nd for nd in inst.nodes
                 if abs(nd.coordinates[1] - actual_y_max) < y_tol]

    if len(bottom_nodes) == 0 or len(top_nodes) == 0:
        raise RuntimeError("Could not find top/bottom nodes.")

    top_region = a.Set(nodes=mesh.MeshNodeArray(top_nodes), name='TopSet')

    bottom_left_node = min(bottom_nodes, key=lambda nd: nd.coordinates[0])
    bottom_left_set = a.Set(nodes=mesh.MeshNodeArray((bottom_left_node,)),
                            name='BottomLeftNode')

    model.DisplacementBC(name='BottomLeftFix', createStepName='Initial',
                         region=bottom_left_set, u1=0.0, u2=0.0, ur3=0.0)

    bottom_other_nodes = [nd for nd in bottom_nodes
                          if nd.label != bottom_left_node.label]

    if bottom_other_nodes:
        bottom_other_set = a.Set(nodes=mesh.MeshNodeArray(bottom_other_nodes),
                                 name='BottomOtherNodes')
        model.DisplacementBC(name='BottomLineU2Fix', createStepName='Initial',
                             region=bottom_other_set, u1=UNSET, u2=0.0, ur3=UNSET)

    model.DisplacementBC(name='TopDisplacement', createStepName='LoadStep',
                         region=top_region, u1=UNSET, u2=displacement_magnitude, ur3=UNSET)

    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'U', 'RF', 'COORD', 'SF')
    )

    # -----------------------------
    # JOB
    # -----------------------------
    job_name = 'VoronoiJob_%d' % sim_idx
    if job_name in mdb.jobs.keys():
        del mdb.jobs[job_name]

    mdb.Job(name=job_name, model=model_name,
            description='Voronoi lattice randomness=%.4f' % randomness)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    print("Job submitted. Waiting for completion...")
    mdb.jobs[job_name].waitForCompletion()
    print("Job completed!")

    # -----------------------------
    # EXPORT FILES TO FOLDER
    # -----------------------------
    joint_coords = set()
    for (p, q) in segments.values():
        joint_coords.add(r6(p))
        joint_coords.add(r6(q))
    node_list = sorted(list(joint_coords))
    node_index = {xy: idx for idx, xy in enumerate(node_list)}

    N = len(node_list)

    # Binary adjacency matrix (1 if connected, 0 otherwise)
    adj_binary = [[0 for _ in range(N)] for __ in range(N)]

    for (p, q) in segments.values():
        i = node_index[r6(p)]
        j = node_index[r6(q)]
        if i != j:
            adj_binary[i][j] = 1
            adj_binary[j][i] = 1

    # Write node_features.csv
    os.chdir(folder_path)
    with open('node_features.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'x', 'y'])
        for i, (x, y) in enumerate(node_list):
            w.writerow([i, x, y])

    # Write adjacency_area.csv (binary version)
    with open('adjacency_area.csv', 'w') as f:
        w = csv.writer(f)
        header = [''] + list(range(N))
        w.writerow(header)
        for i in range(N):
            row = [i] + adj_binary[i]
            w.writerow(row)

    # -----------------------------
    # STIFFNESS & STRESS EXPORT
    # -----------------------------
    try:
        from odbAccess import openOdb

        odb_path = os.path.join(original_dir, job_name + '.odb')
        odb = openOdb(odb_path, readOnly=True)
        step = odb.steps['LoadStep']
        last_frame = step.frames[-1]

        ra = odb.rootAssembly

        # LATTICE STIFFNESS
        topset = ra.nodeSets['TOPSET']
        rf_field = last_frame.fieldOutputs['RF'].getSubset(region=topset)
        total_RF2 = 0.0
        for v in rf_field.values:
            total_RF2 += float(v.data[1])

        k_lattice = total_RF2 / float(displacement_magnitude)

        # Write simplified lattice_stiffness.csv
        with open('lattice_stiffness.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['k_lattice_N_per_mm'])
            w.writerow([k_lattice])

        print("Lattice stiffness: %.6f N/mm" % k_lattice)

        # STRESS EXPORT
        try:
            s_field = last_frame.fieldOutputs['S']
        except KeyError:
            print("No stress field found")
            s_field = None

        if s_field is not None:
            try:
                inst_odb = ra.instances['VORONOIINSTANCE']
            except KeyError:
                first_key = list(ra.instances.keys())[0]
                inst_odb = ra.instances[first_key]

            nodes_by_label = {}
            for nd in inst_odb.nodes:
                nodes_by_label[nd.label] = nd

            elem_mid = {}
            for elem in inst_odb.elements:
                conn = elem.connectivity
                if len(conn) < 2:
                    continue
                n1 = nodes_by_label[conn[0]]
                n2 = nodes_by_label[conn[-1]]
                x1, y1 = n1.coordinates[0], n1.coordinates[1]
                x2, y2 = n2.coordinates[0], n2.coordinates[1]
                xm = 0.5 * (x1 + x2)
                ym = 0.5 * (y1 + y2)
                elem_mid[elem.label] = (xm, ym)

            element_max_s11 = {}
            for v in s_field.values:
                elem_lab = v.elementLabel
                if len(v.data) == 0:
                    continue
                s11 = float(v.data[0])
                if elem_lab not in element_max_s11 or abs(s11) > abs(element_max_s11[elem_lab]):
                    element_max_s11[elem_lab] = s11

            def point_to_segment_distance(px, py, ax, ay, bx, by):
                dx = bx - ax
                dy = by - ay
                L2 = dx*dx + dy*dy
                if L2 <= 0.0:
                    return math.sqrt((px - ax)**2 + (py - ay)**2)
                t = ((px - ax)*dx + (py - ay)*dy) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                projx = ax + t*dx
                projy = ay + t*dy
                return math.sqrt((px - projx)**2 + (py - projy)**2)

            seg_tol = seed_size * 0.8
            beam_stress = {}

            seg_list_for_stress = []
            for (p, q) in segments.values():
                i = node_index[r6(p)]
                j = node_index[r6(q)]
                ax, ay = p[0], p[1]
                bx, by = q[0], q[1]
                seg_list_for_stress.append((i, j, ax, ay, bx, by))

            elem_mid_items = list(elem_mid.items())

            for (i, j, ax, ay, bx, by) in seg_list_for_stress:
                key = (min(i, j), max(i, j))

                xmin_box = min(ax, bx) - seg_tol
                xmax_box = max(ax, bx) + seg_tol
                ymin_box = min(ay, by) - seg_tol
                ymax_box = max(ay, by) + seg_tol

                for elem_lab, (xm, ym) in elem_mid_items:
                    if elem_lab not in element_max_s11:
                        continue

                    if xm < xmin_box or xm > xmax_box or ym < ymin_box or ym > ymax_box:
                        continue

                    d = point_to_segment_distance(xm, ym, ax, ay, bx, by)
                    if d > seg_tol:
                        continue

                    s11 = element_max_s11[elem_lab]
                    if key not in beam_stress or abs(s11) > abs(beam_stress[key]):
                        beam_stress[key] = s11

            stress_matrix = [[0.0 for _ in range(N)] for __ in range(N)]

            for (i, j, _, _, _, _) in seg_list_for_stress:
                key = (min(i, j), max(i, j))
                if key in beam_stress:
                    s_val = beam_stress[key]
                    stress_matrix[i][j] = s_val
                    stress_matrix[j][i] = s_val

            # Write stress_matrix.csv
            with open('stress_matrix.csv', 'w') as f:
                w = csv.writer(f)
                header = [''] + list(range(N))
                w.writerow(header)
                for i in range(N):
                    row = [i] + stress_matrix[i]
                    w.writerow(row)

            print("Wrote stress_matrix.csv")

        odb.close()

    except Exception as e:
        print("Error in post-processing: %s" % str(e))
    
    # Return to original directory for next iteration
    os.chdir(original_dir)
    print("All files written to folder: %s" % folder_path)

print("\n" + "="*70)
print("ALL SIMULATIONS COMPLETE")
print("="*70)