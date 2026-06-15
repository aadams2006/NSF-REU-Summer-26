# === 2D HEX LATTICE with CRACK + Mode-I test
#     -> Exports per-step curve + derived columns (1-5) and a one-row metrics summary
from abaqus import *
from abaqusConstants import *
import regionToolset, mesh, math, os
from odbAccess import openOdb

# ========= USER PARAMETERS (edit these) =========
cell_size      = 20.0                                   # ℓ
cell_height    = math.sqrt(3.0) * cell_size
n              = 15

youngs_modulus = 2000.0
poissons_ratio = 0.30

# Cross-section: rect_b (in-plane thickness), rect_a (out-of-plane specimen thickness B)
rect_a         = 0.01                                   # B (used for nominal stress area B*W)
rect_b         = 0.10                                   # in-plane strut thickness (for 2-D relative density)

seed_size      = 0.50
displacement   = 1.0                                    # top opening U2 (Mode I)
tol            = seed_size/10.0

# Crack shape/guards (left-open, tip near center)
enable_crack               = True
crack_half_height_left     = 0.55*cell_height
left_guard_horiz_keep      = 1.0*cell_size

# -- Fracture/toughness post-processing params --
parent_sigma_f   = 100.0          # <-- set to parent solid tensile strength (units of stress)
YI_shape_factor  = 1.0            # <-- shape factor Y_I; use 1.0 as rough first pass, replace with calibrated value
drop_fraction    = 0.95           # 5% drop criterion (step 1)
linear_window_fr = 0.10           # first 10% points for initial stiffness/E_eff

# ========= helpers =========
def r6(p): return (round(p[0], 6), round(p[1], 6))
def seg_key(p, q):
    p2, q2 = r6(p), r6(q);  return (p2, q2) if p2 <= q2 else (q2, p2)
def lerp(p, q, t): return (p[0] + t*(q[0]-p[0]), p[1] + t*(q[1]-p[1]))
def is_horizontal(p, q, eps=1e-9): return abs(p[1]-q[1]) <= eps
def dist(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])

# ========= build honeycomb segments =========
segments = {}
xs, ys = [], []

for i in range(n):
    n_y = n-1 if (i % 2 == 1) else n
    for j in range(n_y):
        x0 = i * 1.5 * cell_size
        y0 = j * cell_height + (i % 2) * (cell_height/2.0)

        a1 = (x0,                           y0)
        a2 = (x0 - cell_size/2.0,           y0 + math.sqrt(3.0)*cell_size/2.0)
        a3 = (x0,                           y0 + cell_height)
        a4 = (x0 + cell_size,               y0 + cell_height)
        a5 = (x0 + cell_size + cell_size/2.0, y0 + math.sqrt(3.0)*cell_size/2.0)
        a6 = (x0 + cell_size,               y0)

        poly = [a1, a2, a3, a4, a5, a6, a1]
        for k in range(6):
            p, q = poly[k], poly[k+1]
            segments[seg_key(p, q)] = (p, q)

        xs.extend([a1[0], a2[0], a3[0], a4[0], a5[0], a6[0]])
        ys.extend([a1[1], a2[1], a3[1], a4[1], a5[1], a6[1]])

xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
xc, yc     = 0.5*(xmin + xmax), 0.5*(ymin + ymax)

# nearest lattice node to center -> tip node (left-opening crack, tip ~ center)
all_nodes = set()
for (p, q) in segments.values():
    all_nodes.add(r6(p)); all_nodes.add(r6(q))
tip_node = min(all_nodes, key=lambda v: (v[0]-xc)**2 + (v[1]-yc)**2)
tip_x    = tip_node[0]

# preserve beam(s) attached to the crack tip going RIGHT
preserve_tip_right = set()
for key, (p, q) in segments.items():
    P, Q = r6(p), r6(q)
    if (P == tip_node and Q[0] >= tip_x + 1e-9) or (Q == tip_node and P[0] >= tip_x + 1e-9):
        preserve_tip_right.add(key)

# crack interior predicate
def half_height(x):
    if x < xmin - 1e-9 or x > xc + 1e-9: return 0.0
    denom = max(xc - xmin, 1e-9);  t = (xc - x)/denom
    return max(0.0, min(1.0, t)) * crack_half_height_left

def interior_of_crack(pt, eps=1e-9):
    x, y = pt
    if x <= xmin + eps or x >= xc - eps: return False
    return abs(y - yc) < half_height(x) - eps

def segment_interior_hits_crack(p, q):
    if is_horizontal(p, q) and min(p[0], q[0]) <= xmin + left_guard_horiz_keep:  # keep leftmost horizontals
        return False
    for t in (0.2, 0.4, 0.6, 0.8):
        if interior_of_crack(lerp(p, q, t)): return True
    return interior_of_crack(p) and interior_of_crack(q)

if enable_crack:
    kept = {}
    for key, (p, q) in segments.items():
        if (key not in preserve_tip_right) and segment_interior_hits_crack(p, q):  # remove only interior
            continue
        kept[key] = (p, q)
    segments = kept

# total strut length for 2-D relative density  (FIX: avoid generator expression)
L_total = 0.0
for (p, q) in segments.values():
    L_total += dist(p, q)

# ========= sketch & part =========
model_name = "HexCrack"
if model_name in mdb.models: del mdb.models[model_name]
model = mdb.Model(name=model_name)

sk = model.ConstrainedSketch(name='__profile__', sheetSize=max(xmax, ymax)*1.5)
for (p, q) in segments.values():
    sk.Line(point1=p, point2=q)

part = model.Part(name='HexPart', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
part.BaseWire(sketch=sk)

# ========= material / section =========
model.Material(name="LatticeMaterial")
model.materials["LatticeMaterial"].Elastic(table=((youngs_modulus, poissons_ratio),))
model.RectangularProfile(name="WireProfile", a=rect_a, b=rect_b)
model.BeamSection(name="LatticeSection", integration=DURING_ANALYSIS,
                  profile="WireProfile", material="LatticeMaterial")
reg = regionToolset.Region(edges=part.edges)
part.SectionAssignment(region=reg, sectionName="LatticeSection")
part.assignBeamSectionOrientation(region=reg, method=N1_COSINES, n1=(0.0, 0.0, 1.0))

# ========= mesh =========
part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
part.generateMesh()

# ========= assembly / step / BCs =========
a = model.rootAssembly
a.DatumCsysByDefault(CARTESIAN)
inst = a.Instance(name='HexInstance', part=part, dependent=ON)

model.StaticStep(name="Loading", previous="Initial", nlgeom=ON,
                 timePeriod=10.0, initialInc=0.1, minInc=1e-5, maxInc=1.0)

# sets: bottom fixed in U2, top pulled +U2
y_coords = [nd.coordinates[1] for nd in inst.nodes]
y_min, y_max = min(y_coords), max(y_coords)
bottom_nodes = inst.nodes.getByBoundingBox(yMin=y_min - tol, yMax=y_min + tol)
top_nodes    = inst.nodes.getByBoundingBox(yMin=y_max - tol, yMax=y_max + tol)
a.Set(name="BottomNodes", nodes=bottom_nodes)
a.Set(name="TopNodes",    nodes=top_nodes)

model.DisplacementBC(name="BC_FixedBottom", createStepName="Initial",
                     region=a.sets["BottomNodes"], u1=UNSET, u2=0.0, ur3=UNSET)
model.DisplacementBC(name="BC_TensionTop", createStepName="Loading",
                     region=a.sets["TopNodes"], u1=UNSET, u2=displacement, ur3=UNSET)

# history outputs for P-u
model.HistoryOutputRequest(name="H-TopSurface", createStepName="Loading",
                           region=a.sets["TopNodes"], variables=("U2", "RF2"))

# ========= job =========
job_name = "HexModeI"
if job_name in mdb.jobs: del mdb.jobs[job_name]
job = mdb.Job(name=job_name, model=model_name, numCpus=4, numDomains=4,
              memory=90, memoryUnits=PERCENTAGE,
              description="2D hex lattice Mode-I with derived metrics export")
job.submit(consistencyChecking=OFF)
job.waitForCompletion()

# ========= postprocess: curve + metrics (steps 1-5) =========
def find_instance_name(ra):
    keys = sorted(ra.instances.keys())
    return keys[0] if keys else None

def find_set_case_insensitive(ra, target):
    t = target.upper()
    for nm in ra.nodeSets.keys():
        if nm.upper() == t:
            return nm
    raise RuntimeError("Node set '%s' not found in ODB." % target)

odb = openOdb(path=job_name + ".odb", readOnly=True)
ra  = odb.rootAssembly
inst_name = find_instance_name(ra)
if inst_name is None:
    odb.close();  raise RuntimeError("No instances found in ODB.")

# collect TopNodes labels
set_name = find_set_case_insensitive(ra, "TopNodes")
ns = ra.nodeSets[set_name]
labels = []
for nodeArray in ns.nodes:
    for nd in nodeArray: labels.append(nd.label)
labels = sorted(set(labels))

step = odb.steps["Loading"]
times, sum_RF2, sum_U2, count = None, None, None, 0

# sum RF2 and average U2 over TopNodes
for lab in labels:
    key = "Node %s.%d" % (inst_name, lab)
    if key not in step.historyRegions.keys(): continue
    hreg = step.historyRegions[key]
    if ("RF2" not in hreg.historyOutputs.keys()) or ("U2" not in hreg.historyOutputs.keys()): continue
    rf = hreg.historyOutputs["RF2"];  uu = hreg.historyOutputs["U2"]
    rf_data, uu_data = rf.data, uu.data
    if times is None:
        times   = [t for (t, _) in rf_data]
        sum_RF2 = [0.0]*len(times)
        sum_U2  = [0.0]*len(times)
    for i in range(len(times)):
        sum_RF2[i] += rf_data[i][1]
        sum_U2[i]  += uu_data[i][1]
    count += 1

if count == 0:
    odb.close();  raise RuntimeError("No RF2/U2 histories found for TopNodes.")

# global series
u = [v/float(count) for v in sum_U2]      # avg U2 on top edge
P = [-v for v in sum_RF2]                 # tension positive
H = (ymax - ymin);  Wspec = (xmax - xmin);  B = rect_a

eps = [ui / H for ui in u]                # nominal strain
sig = [Pi / (B * Wspec) for Pi in P]      # nominal stress

# ---- Step 1: tangent stiffness & failure by 5% drop ----
N = len(u)
def central_diff(y, x):
    d = [0.0]*N
    if N >= 2:
        d[0] = (y[1]-y[0])/(x[1]-x[0] if x[1]!=x[0] else 1.0)
        d[-1]= (y[-1]-y[-2])/(x[-1]-x[-2] if x[-1]!=x[-2] else 1.0)
    for i in range(1, N-1):
        dx = (x[i+1]-x[i-1])
        d[i] = (y[i+1]-y[i-1])/(dx if dx!=0.0 else 1.0)
    return d

k_tangent = central_diff(P, u)  # dP/du
M = max(3, min(N-2, int(linear_window_fr*N)))
k0 = sum(k_tangent[:M])/float(M)

# failure index = first drop below 0.95*k0; fallback to P peak
threshold = drop_fraction * k0
fail_idx  = None
for i in range(1, N):
    if k_tangent[i] <= threshold:
        fail_idx = i; break
if fail_idx is None:
    fail_idx = max(range(N), key=lambda i: P[i])

u_f, P_f = u[fail_idx], P[fail_idx]
eps_f, sig_f = eps[fail_idx], sig[fail_idx]

# work/energy (trapezoid in P-u)
W_cum = [0.0]*N
for i in range(1, N):
    du = (u[i]-u[i-1]);  W_cum[i] = W_cum[i-1] + 0.5*(P[i]+P[i-1])*du
W_f = W_cum[fail_idx]

# ---- Step 2: effective modulus from small strain ----
dEdE = central_diff(sig, eps)      # dσ/dε
E_eff_init = sum(dEdE[:M])/float(M)

# ---- Step 3: estimate a, then K_Ic = YI * σ_f* sqrt(pi*a) using σ at onset ----
# For this left-opening crack, take a as tip-to-left distance:
a = abs(tip_x - xmin)
K_Ic = YI_shape_factor * sig_f * math.sqrt(math.pi * a)

# ---- Step 4: normalized toughness ----
barK_Ic = K_Ic / (parent_sigma_f * math.sqrt(cell_size))

# ---- Step 5: 2-D relative density (area fraction) ----
bar_rho = (L_total * rect_b) / (Wspec * H)

# ---- Write per-step CSV with derived columns ----
du_list  = [0.0] + [u[i]-u[i-1] for i in range(1, N)]
dP_list  = [0.0] + [P[i]-P[i-1] for i in range(1, N)]
is_fail  = [0]*N;  is_fail[fail_idx] = 1

with open("PU_curve_enhanced.csv", "w") as f:
    # metadata header (Excel will still open fine)
    f.write("# W=%.6e, H=%.6e, B=%.6e, a=%.6e, ell=%.6e, rect_b=%.6e\n" % (Wspec, H, B, a, cell_size, rect_b))
    f.write("# YI=%.6e, parent_sigma_f=%.6e, bar_rho=%.6e, k0=%.6e, E_eff_init=%.6e\n" %
            (YI_shape_factor, parent_sigma_f, bar_rho, k0, E_eff_init))
    f.write("# u_f=%.6e, P_f=%.6e, eps_f=%.6e, sig_f=%.6e, W_f=%.6e, K_Ic=%.6e, barK_Ic=%.6e\n" %
            (u_f, P_f, eps_f, sig_f, W_f, K_Ic, barK_Ic))
    # columns
    f.write("time,u,P,eps,sig,du,dP,k_tangent,W_cum,is_failure\n")
    for t, ui, Pi, ei, si, dui, dPi, ki, Wi, fl in zip(
            times, u, P, eps, sig, du_list, dP_list, k_tangent, W_cum, is_fail):
        f.write("{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{}\n"
                .format(t, ui, Pi, ei, si, dui, dPi, ki, Wi, fl))

# ---- One-row metrics CSV (easier to aggregate across runs) ----
with open("PU_metrics.csv", "w") as f:
    f.write("W,H,B,a,ell,rect_b,bar_rho,k0,E_eff_init,u_f,P_f,eps_f,sig_f,W_f,YI,parent_sigma_f,K_Ic,barK_Ic\n")
    f.write("{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n".format(
        Wspec, H, B, a, cell_size, rect_b, bar_rho, k0, E_eff_init, u_f, P_f, eps_f, sig_f, W_f,
        YI_shape_factor, parent_sigma_f, K_Ic, barK_Ic))

odb.close()

print("Done. Wrote PU_curve_enhanced.csv and PU_metrics.csv")
