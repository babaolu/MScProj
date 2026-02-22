import gmsh
import numpy as np

# --- Configuration ---
gmsh.initialize()
gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.05)
gmsh.model.add("forearm_electrode")

# Dimensions (mm)
r = 42.0
R_bone = r * np.sqrt(0.107)              # ~14.7 mm
R_muscle = r * np.sqrt(0.611 + 0.107)    # ~38.0 mm
R_fat = r * np.sqrt(0.222 + 0.612 + 0.107) # ~43.0 mm (Wait, check math)
# Let's trust your original ratios, but R_fat must be < R_skin (45)
# Sqrt(0.942) * 45 = 43.6 mm. Correct.
R_skin = r

# Band Dimensions
t = 2.0
Ed = 8.0
Et = 0.4
count = 16
L0 = 2 * np.pi * 28
IED = (L0 - (Ed * count))/count
theta_e = Ed / R_skin
theta_gap = IED / R_skin

# --- 1. Create Base Shapes ---
# Create Solid Disks (we will slice them later)
bone_disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_bone, R_bone)
muscle_disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_muscle, R_muscle)
fat_disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_fat, R_fat)
skin_disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_skin, R_skin)

# Create Band (Hollow)
band_outer = gmsh.model.occ.addDisk(0, 0, 0, R_skin + t, R_skin + t)
band_inner = gmsh.model.occ.addDisk(0, 0, 0, R_skin, R_skin)
band = gmsh.model.occ.cut([(2, band_outer)], [(2, band_inner)])

# Create Electrodes
electrodes = []
e_surfaces = []
theta = 0.0
for i in range(count):
    e_outer = gmsh.model.occ.addCircle(0, 0, 0, R_skin + Et, angle1=theta, angle2=theta + theta_e)
    e_surfaces.append(e_outer)
    e_inner = gmsh.model.occ.addCircle(0, 0, 0, R_skin, angle1=theta, angle2=theta + theta_e)
    l1 = gmsh.model.occ.addLine(gmsh.model.occ.addPoint(R_skin*np.cos(theta), R_skin*np.sin(theta), 0),
                                gmsh.model.occ.addPoint((R_skin+Et)*np.cos(theta), (R_skin+Et)*np.sin(theta), 0))
    l2 = gmsh.model.occ.addLine(gmsh.model.occ.addPoint(R_skin*np.cos(theta+theta_e), R_skin*np.sin(theta+theta_e), 0),
                                gmsh.model.occ.addPoint((R_skin+Et)*np.cos(theta+theta_e), (R_skin+Et)*np.sin(theta+theta_e), 0))
    loop = gmsh.model.occ.addCurveLoop([e_outer, l2, -e_inner, -l1])
    electrodes.append(gmsh.model.occ.addPlaneSurface([loop]))
    theta += theta_e + theta_gap

# Create Ground
theta_total = L0 / R_skin
theta += (((2 * np.pi) - theta_total) - theta_e - theta_gap) / 2
g_inner = gmsh.model.occ.addCircle(0, 0, 0, R_skin + t - Et, angle1=theta, angle2=theta + theta_e)
g_outer = gmsh.model.occ.addCircle(0, 0, 0, R_skin + t, angle1=theta, angle2=theta + theta_e)
gl1 = gmsh.model.occ.addLine(gmsh.model.occ.addPoint((R_skin + t - Et)*np.cos(theta), (R_skin + t - Et)*np.sin(theta), 0),
                             gmsh.model.occ.addPoint((R_skin + t)*np.cos(theta), (R_skin + t)*np.sin(theta), 0))
gl2 = gmsh.model.occ.addLine(gmsh.model.occ.addPoint((R_skin + t - Et)*np.cos(theta+theta_e), (R_skin + t - Et)*np.sin(theta+theta_e), 0),
                             gmsh.model.occ.addPoint((R_skin + t)*np.cos(theta+theta_e), (R_skin + t)*np.sin(theta+theta_e), 0))
gsurf = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([g_outer, gl2, -g_inner, -gl1])])

wrapped_substrate = gmsh.model.occ.cut(band[0],
    [(2, e) for e in electrodes] + [(2, gsurf)],
    removeObject=True, removeTool=False)[0][0][1]

# --- 2. GLOBAL FRAGMENT (The Fix) ---
all_devices = [(2, wrapped_substrate)] + [(2, e) for e in electrodes] + [(2, gsurf)]

# 1. Create shapes and CUT them so they don't overlap.
# We do this so we can name them variables like 'skin_ring' and 'fat_ring'
skin_ring = gmsh.model.occ.cut([(2, skin_disk)], [(2, fat_disk)], removeTool=False)[0][0][1]
fat_ring = gmsh.model.occ.cut([(2, fat_disk)], [(2, muscle_disk)], removeTool=False)[0][0][1]
muscle_ring = gmsh.model.occ.cut([(2, muscle_disk)], [(2, bone_disk)], removeTool=False)[0][0][1]
# Bone doesn't need cutting
bone = bone_disk

# 2. Prepare the list for the "Grand Fragment"
# The ORDER here matters for tagging later!
# Index 0: Skin
# Index 1: Fat
# Index 2: Muscle
# Index 3: Bone
tissue_inputs = [(2, skin_ring), (2, fat_ring), (2, muscle_ring), (2, bone)]

# 3. Fragment (The Glue)
# This merges the boundaries so they share nodes.
ov, ovv = gmsh.model.occ.fragment(tissue_inputs + all_devices, [])
gmsh.model.occ.synchronize()


# Identify which input corresponds to the electrodes (indices 5 onwards)
# Input order: Skin(0), Fat(1), Muscle(2), Bone(3), Band(4), Elec...(5+)
electrode_parents_indices = range(len(tissue_inputs) + 1, len(tissue_inputs) + len(all_devices))
electrode_children_tags = []
for idx in electrode_parents_indices:
    for child in ovv[idx]:
        electrode_children_tags.append(child[1])


# 4. Tagging by Lineage (The Safe Way)
# ovv[i] contains the new tag for the input at index i.

# Since we defined the order in step 2, we know:
final_skin_tag = [tag[1] for tag in ovv[0]]
final_fat_tag = [tag[1] for tag in ovv[1]]
final_muscle_tag = [tag[1] for tag in ovv[2]]
final_bone_tag = [tag[1] for tag in ovv[3]]
substrate_tag = [tag[1] for tag in ovv[4]]

# 5. Create Physical Groups
gmsh.model.addPhysicalGroup(2, final_skin_tag, name="Skin")
gmsh.model.addPhysicalGroup(2, final_fat_tag, name="Fat")
gmsh.model.addPhysicalGroup(2, final_muscle_tag, name="Muscle")
gmsh.model.addPhysicalGroup(2, final_bone_tag, name="Bone")
gmsh.model.addPhysicalGroup(2, substrate_tag, name="Substrate")
gmsh.model.addPhysicalGroup(2, electrode_children_tags, name="Conductor_Array")

gmsh.model.addPhysicalGroup(1, [g_outer], name="Ground")
for i in range(len(e_surfaces)):
    gmsh.model.addPhysicalGroup(1, [e_surfaces[i]], name=f"E{i}")

gmsh.model.mesh.generate(2)
gmsh.write("2d_wrapped.msh")
gmsh.fltk.run()
gmsh.finalize()
