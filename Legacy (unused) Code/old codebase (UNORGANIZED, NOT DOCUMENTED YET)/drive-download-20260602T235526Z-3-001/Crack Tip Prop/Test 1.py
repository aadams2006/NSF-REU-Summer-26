# === 2D HEX LATTICE with CRACK + Mode-I opening test (P–u export) ===
from abaqus import *
from abaqusConstants import *
import regionToolset, mesh, math, os
from odbAccess import openOdb

# ---------- USER PARAMETERS ----------
cell_size      = 20.0
cell_height    = math.sqrt(3.0) * cell_size
n              = 15

youngs_modulus = 2000.0
poissons_ratio = 0.30


#   rect_b = in-plane beam thicknes
#   rect_a = out-of-plane
rect_a         = 0.01
rect_b         = 0.10

seed_size      = 0.50
displacement   = 1.0   # top opening displacement (Mode I)
tol            = seed_size/10.0


enable_crack               = True
crack_half_height_left     = 0.55*cell_height
left_guard_horiz_keep      = 1.0*cell_size


def r6(p): return (round(p[0], 6), round(p[1], 6))
def seg_key(p, q):
    p2, q2 = r6(p), r6(q)
    return (p2, q2) if p2 <= q2 else (q2, p2)
def lerp(p, q, t): return (p[0] + t*(q[0]-p[0]), p[1] + t*(q[1]-p[1]))
def is_horizontal(p, q, eps=1e-9): return abs(p[1]-q[1]) <= eps
#geometery
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

#looking for crack tip
all_nodes = set()
for (p, q) in segments.values():
    all_nodes.add(r6(p)); all_nodes.add(r6(q))
tip_node = min(all_nodes, key=lambda v: (v[0]-xc)**2 + (v[1]-yc)**2)
tip_x = tip_node[0]


preserve_tip_right = set()
for key, (p, q) in segments.items():
    P, Q = r6(p), r6(q)
    if P == tip_node and Q[0] >= tip_x + 1e-9:
        preserve_tip_right.add(key)
    elif Q == tip_node and P[0] >= tip_x + 1e-9:
        preserve_tip_right.add(key)


def half_height(x):
    if x < xmin - 1e-9 or x > xc + 1e-9: return 0.0
    denom = max(xc - xmin, 1e-9)
    t = (xc - x)/denom
    return max(0.0, min(1.0, t)) * crack_half_height_left

def interior_of_crack(pt, eps=1e-9):
    x, y = pt
    if x <= xmin + eps or x >= xc - eps:
        return False
    return abs(y - yc) < half_height(x) - eps

def segment_interior_hits_crack(p, q):
    if is_horizontal(p, q) and min(p[0], q[0]) <= xmin + left_guard_horiz_keep:
        return False
    for t in (0.2, 0.4, 0.6, 0.8):
        m = lerp(p, q, t)
        if interior_of_crack(m):
            return True
    return interior_of_crack(p) and interior_of_crack(q)

if enable_crack:
    kept = {}
    for key, (p, q) in segments.items():
        if (key not in preserve_tip_right) and segment_interior_hits_crack(p, q):
            continue
        kept[key] = (p, q)
    segments = kept


model_name = "HexCrack"
if model_name in mdb.models: del mdb.models[model_name]
model = mdb.Model(name=model_name)

sk = model.ConstrainedSketch(name='__profile__', sheetSize=max(xmax, ymax)*1.5)
for (p, q) in segments.values():
    sk.Line(point1=p, point2=q)

part = model.Part(name='HexPart', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
part.BaseWire(sketch=sk)

model.Material(name="LatticeMaterial")
model.materials["LatticeMaterial"].Elastic(table=((youngs_modulus, poissons_ratio),))
model.RectangularProfile(name="WireProfile", a=rect_a, b=rect_b)
model.BeamSection(name="LatticeSection", integration=DURING_ANALYSIS,
                  profile="WireProfile", material="LatticeMaterial")
reg = regionToolset.Region(edges=part.edges)
part.SectionAssignment(region=reg, sectionName="LatticeSection")
part.assignBeamSectionOrientation(region=reg, method=N1_COSINES, n1=(0.0, 0.0, 1.0))


part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
part.generateMesh()


a = model.rootAssembly
a.DatumCsysByDefault(CARTESIAN)
inst = a.Instance(name='HexInstance', part=part, dependent=ON)

model.StaticStep(name="Loading", previous="Initial", nlgeom=ON,
                 timePeriod=10.0, initialInc=0.1, minInc=1e-5, maxInc=1.0)


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


model.HistoryOutputRequest(name="H-TopSurface", createStepName="Loading",
                           region=a.sets["TopNodes"], variables=("U2", "RF2"))


job_name = "HexModeI"
if job_name in mdb.jobs: del mdb.jobs[job_name]
job = mdb.Job(name=job_name, model=model_name, numCpus=4, numDomains=4,
              memory=90, memoryUnits=PERCENTAGE,
              description="2D hex lattice Mode-I opening with P–u export")
job.submit(consistencyChecking=OFF)
job.waitForCompletion()


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
    odb.close()
    raise RuntimeError("No instances found in ODB rootAssembly.")


set_name = find_set_case_insensitive(ra, "TopNodes")
ns = ra.nodeSets[set_name]

labels = []
try:

    for nodeArray in ns.nodes:
        for nd in nodeArray:
            labels.append(nd.label)
except:

    for nd in ns.nodes:
        labels.append(nd.label)

labels = sorted(set(labels))  # unique

step = odb.steps["Loading"]


times   = None
sum_RF2 = None
sum_U2  = None
count   = 0


for lab in labels:
    key = "Node %s.%d" % (inst_name, lab)
    if key not in step.historyRegions.keys():

        continue
    hreg = step.historyRegions[key]

    if ("RF2" not in hreg.historyOutputs.keys()) or ("U2" not in hreg.historyOutputs.keys()):
        continue

    rf = hreg.historyOutputs["RF2"]
    uu = hreg.historyOutputs["U2"]

    rf_data = rf.data  # list[(t, val)]
    uu_data = uu.data

    if times is None:
        times   = [t for (t, _) in rf_data]
        sum_RF2 = [0.0]*len(times)
        sum_U2  = [0.0]*len(times)


    for i in range(len(times)):
        sum_RF2[i] += rf_data[i][1]
        sum_U2[i]  += uu_data[i][1]
    count += 1

if count == 0:
    odb.close()
    raise RuntimeError("No RF2/U2 histories found for TopNodes in step 'Loading'.")

u_avg = [v / float(count) for v in sum_U2]

P = [-v for v in sum_RF2]

H = (ymax - ymin)
W = (xmax - xmin)
B = rect_a  # out-of-plane thickness
eps = [ui / H for ui in u_avg]
sig = [Pi / (B * W) for Pi in P]

csv_path = os.path.abspath("PU_curve.csv")
with open(csv_path, "w") as f:
    f.write("time,u_top_avg,P_total,eps_nom,sig_nom\n")
    for t, u, p, e, s in zip(times, u_avg, P, eps, sig):
        f.write("{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n".format(t, u, p, e, s))

odb.close()

print("Bounds: xmin=%.3f xmax=%.3f, ymin=%.3f ymax=%.3f" % (xmin, xmax, ymin, ymax))
print("Crack tip (nearest node) = (%.3f, %.3f)" % tip_node)
print("TopNodes used: %d nodes" % count)
print("Wrote P–u data to: %s" % csv_path)
