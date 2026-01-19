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
L0 = 2 * np.pi * 40
t = 1		# Band thickness
Ed = 8
Et = 0.1
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

"""rect = gmsh.model.occ.addRectangle(-L0 / 2, -(r + t), 0, L0, t)
conductor_array = [gmsh.model.occ.addRectangle(IED / 2 + (Ed + IED) * i - (L0 / 2),
                    t - Et - (r + t), 0, Ed, Et) for i in range(count)]"""

theta_total = L0 / R_skin
theta_e = Ed / R_skin
theta_gap = IED / R_skin

band_outer = gmsh.model.occ.addDisk(0, 0, 0, R_skin + t, R_skin + t)
band_inner = gmsh.model.occ.addDisk(0, 0, 0, R_skin, R_skin)

band = gmsh.model.occ.cut([(2, band_outer)], [(2, band_inner)],
                          removeObject=True, removeTool=True)
print("Band:", band)

electrodes = []
theta = 0.0

for i in range(count):
    p1 = gmsh.model.occ.addPoint(0, 0, 0)

    e_outer = gmsh.model.occ.addCircle(0, 0, 0, R_skin + Et,
                                       angle1=theta,
                                       angle2=theta + theta_e)

    e_inner = gmsh.model.occ.addCircle(0, 0, 0, R_skin,
                                       angle1=theta,
                                       angle2=theta + theta_e)

    l1 = gmsh.model.occ.addLine(
        gmsh.model.occ.addPoint(R_skin*np.cos(theta),
                                R_skin*np.sin(theta), 0),
        gmsh.model.occ.addPoint((R_skin+Et)*np.cos(theta),
                                (R_skin+Et)*np.sin(theta), 0))

    l2 = gmsh.model.occ.addLine(
        gmsh.model.occ.addPoint(R_skin*np.cos(theta+theta_e),
                                R_skin*np.sin(theta+theta_e), 0),
        gmsh.model.occ.addPoint((R_skin+Et)*np.cos(theta+theta_e),
                                (R_skin+Et)*np.sin(theta+theta_e), 0))

    loop = gmsh.model.occ.addCurveLoop([e_outer, l2, -e_inner, -l1])
    surf = gmsh.model.occ.addPlaneSurface([loop])
    electrodes.append(surf)

    theta += theta_e + theta_gap


gmsh.model.occ.synchronize()
print("Conductor array:", electrodes)

# Subtracting to get ground conductors
theta += (((2 * np.pi) - theta_total) - theta_e - theta_gap) / 2
g_inner = gmsh.model.occ.addCircle(0, 0, 0, R_skin + t - Et,
                                   angle1=theta, angle2=theta + theta_e)
g_outer = gmsh.model.occ.addCircle(0, 0, 0, R_skin + t,
                                   angle1=theta, angle2=theta + theta_e)

gl1 = gmsh.model.occ.addLine(
    gmsh.model.occ.addPoint((R_skin + t - Et)*np.cos(theta),
                            (R_skin + t - Et)*np.sin(theta), 0),
    gmsh.model.occ.addPoint((R_skin + t)*np.cos(theta),
                            (R_skin + t)*np.sin(theta), 0))

gl2 = gmsh.model.occ.addLine(
    gmsh.model.occ.addPoint((R_skin + t - Et)*np.cos(theta+theta_e),
                            (R_skin + t - Et)*np.sin(theta+theta_e), 0),
    gmsh.model.occ.addPoint((R_skin + t)*np.cos(theta+theta_e),
                            (R_skin + t)*np.sin(theta+theta_e), 0))

gloop = gmsh.model.occ.addCurveLoop([g_outer, gl2, -g_inner, -gl1])
gsurf = gmsh.model.occ.addPlaneSurface([gloop])


wrapped_substrate = gmsh.model.occ.cut(band[0],
    [(2, e) for e in electrodes] + [(2, gsurf)],
    removeObject=True, removeTool=False)

gmsh.model.occ.synchronize()

# Creating physical groups for surfaces
"""substrate_surfaces = [ent[1] for ent in encaps_matrix[0]]

substrate_tag = gmsh.model.addPhysicalGroup(2, substrate_surfaces,
        name="Substrate")"""

gmsh.model.addPhysicalGroup(2,
    [s[1] for s in wrapped_substrate[0]],
    name="Substrate")

gmsh.model.addPhysicalGroup(2, electrodes, name="Conductor_Array")
gmsh.model.addPhysicalGroup(2, [gsurf], name="Ground")

#conductor_tag = gmsh.model.addPhysicalGroup(2, conductor_array,
 #       name="Conductor_Array")

gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write("2d_wrapped.msh")
gmsh.fltk.run()
gmsh.finalize()
