import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType        # type: ignore
else:
    print("This demo requires petsc4py")
    exit(0)

from mpi4py import MPI

import numpy as np

import gmsh
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

gmsh.initialize()
gmsh.model.add("concentric_layers")

#Define geometry (mm)
r = 45			# Radius of forearm
R_bone = r * np.sqrt(0.107)
R_muscle = r * np.sqrt(0.612 + 0.107)
R_fat = r * np.sqrt(0.223 + 0.612 + 0.107)
R_skin = r

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

gmsh.model.mesh.generate(2)
gmsh.write("forearm.msh")
gmsh.write("forearm.geo")
gmsh.fltk.run()
gmsh.finalize()
