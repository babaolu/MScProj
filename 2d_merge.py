import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("forearm_electrode")

#Define forearm geometry (mm)
r = 45			# Radius of forearm
R_bone = r * np.sqrt(0.107)
R_muscle = r * np.sqrt(0.612 + 0.107)
R_fat = r * np.sqrt(0.223 + 0.612 + 0.107)
R_skin = r


# Band dimensions (mm)
L0 = 2 * np.pi * 30
t = 1		# Band thickness
Ed = 7
Et = 0.02
count = 16
IED = (L0 - (Ed * count))/count

bone = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_bone, R_bone)
muscle = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_muscle, R_muscle)
fat = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_fat, R_fat)
skin = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_skin, R_skin)

gmsh.model.occ.synchronize()

# Subtracting to get annuli
skin_ring = gmsh.model.occ.cut([(2, skin)], [(2, fat)], removeObject=True,
            removeTool=False)
fat_ring = gmsh.model.occ.cut([(2, fat)], [(2, muscle)], removeObject=True,
            removeTool=False)
muscle_ring = gmsh.model.occ.cut([(2, muscle)], [(2, bone)], removeObject=True,
            removeTool=False)

gmsh.model.occ.synchronize()
print("Skin ring:", skin_ring)

# Collect resulting surface tags
skin_surfaces = [ent[1] for ent in skin_ring[0]]
fat_surfaces = [ent[1] for ent in fat_ring[0]]
muscle_surfaces = [ent[1] for ent in muscle_ring[0]]
bone_surfaces = [bone]
print("Skin surfaces:", skin_surfaces)

# Creating physical groups for surfaces
skin_tag = gmsh.model.addPhysicalGroup(2, skin_surfaces, name="Skin")
fat_tag = gmsh.model.addPhysicalGroup(2, fat_surfaces, name="Fat")
muscle_tag = gmsh.model.addPhysicalGroup(2, muscle_surfaces, name="Muscle")
bone_tag = gmsh.model.addPhysicalGroup(2, bone_surfaces, name="Bone")

# Creating physical groups for boundaries
bone_boundaries = gmsh.model.getBoundary([(2, tag) for tag in bone_surfaces], oriented=False)
muscle_boundaries = gmsh.model.getBoundary([(2, tag) for tag in muscle_surfaces], oriented=False)
fat_boundaries = gmsh.model.getBoundary([(2, tag) for tag in fat_surfaces], oriented=False)
skin_boundaries = gmsh.model.getBoundary([(2, tag) for tag in skin_surfaces], oriented=False)

gmsh.model.occ.synchronize()
print("Bone boundary curves:", bone_boundaries)
print("Muscle boundary curves:", muscle_boundaries)
print("Fat boundary curves:", fat_boundaries)

bone_muscle_interface = gmsh.model.addPhysicalGroup(1,
        [tag[1] for tag in bone_boundaries],
        name="Bone_Muscle")

muscle_fat = list(set(muscle_boundaries) - set(bone_boundaries))
muscle_fat_interface = gmsh.model.addPhysicalGroup(1,
        [tag[1] for tag in muscle_fat],
        name="Muscle_Fat")

fat_skin = list(set(fat_boundaries) - set(muscle_boundaries))
fat_skin_interface = gmsh.model.addPhysicalGroup(1,
        [tag[1] for tag in fat_skin],
        name="Fat_Skin")

skin_electrode = list(set(skin_boundaries) - set(fat_boundaries))
skin_electrode_interface = gmsh.model.addPhysicalGroup(1,
        [tag[1] for tag in skin_electrode],
        name="Skin_Electrode")
gmsh.model.occ.synchronize()

print("Muscle_Fat:", muscle_fat)
print("Fat_Skin:", fat_skin)
print("Skin_Electrode:", skin_electrode)

rect = gmsh.model.occ.addRectangle(-L0 / 2, -(r + t), 0, L0, t)
conductor_array = [gmsh.model.occ.addRectangle(IED / 2 + (Ed + IED) * i - (L0 / 2),
                    t - Et - (r + t), 0, Ed, Et) for i in range(count)]

gmsh.model.occ.synchronize()
print("Conductor array:", conductor_array)

# Subtracting to get embedded conductors
encaps_matrix = gmsh.model.occ.cut([(2, rect)],
        [(2, tag) for tag in conductor_array],
        removeObject=True, removeTool=False)

gmsh.model.occ.synchronize()

# Creating physical groups for surfaces
substrate_surfaces = [ent[1] for ent in encaps_matrix[0]]

substrate_tag = gmsh.model.addPhysicalGroup(2, substrate_surfaces,
        name="Substrate")
conductor_tag = gmsh.model.addPhysicalGroup(2, conductor_array,
        name="Conductor_Array")

gmsh.model.occ.synchronize()

boundaries = gmsh.model.getBoundary([(2, tag) for tag in substrate_surfaces],
        oriented=False)
conductor_boundaries = gmsh.model.getBoundary([(2, tag) for tag in conductor_array],
        oriented=False)

left_curves = []
right_curves = []
top_curves = []

tol = 1e-6    # Tolerance

for dim, tag in boundaries:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim,tag)
    if abs(xmin + (L0 / 2)) < tol and abs(xmax + (L0 / 2)) < tol:
        left_curves.append(tag)
    if abs(xmin - (L0 / 2)) < tol and abs(xmax - (L0 / 2)) < tol:
        right_curves.append(tag)
    if abs(ymin + r) < tol and abs(ymax + r) < tol:
        top_curves.append(tag)

for dim, tag in conductor_boundaries:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim,tag)
    if abs(ymin + r) < tol and abs(ymax + r) < tol:
        top_curves.append(tag)


if left_curves:
    gmsh.model.addPhysicalGroup(1, left_curves, name="Left")
if right_curves:
    gmsh.model.addPhysicalGroup(1, right_curves, name="Right")
if top_curves:
    gmsh.model.addPhysicalGroup(1, top_curves, name="Top")

print("Left curves:", left_curves)
print("Right curves:", right_curves)
print("Top curves:", top_curves)

gmsh.model.occ.synchronize()

# Group all band entities
#band_entities = encaps_matrix[0] + [(2, tag) for tag in conductor_array]
#gmsh.model.occ.translate(band_entities, -L0 / 2, -(r + t), 0)

gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write("2d_forearm_electrode.msh")
gmsh.fltk.run()
gmsh.finalize()
