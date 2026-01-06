import gmsh
import numpy as np
from mpi4py import MPI

gmsh.initialize()
gmsh.model.add("electrode_band")

# Band dimensions (mm)
L0 = 2 * np.pi * 30
t = 1		# Band thickness
Ed = 7
Et = 0.02
count = 16
IED = (L0 - (Ed * count))/count

rect = gmsh.model.occ.addRectangle(0, 0, 0, L0, t)
conductor_array = [gmsh.model.occ.addRectangle(IED / 2 + (Ed + IED) * i, t - Et, 0, Ed, Et) for i in range(count)]

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
left_curves = []
right_curves = []
mid_curves = []

tol = 1e-6    # Tolerance

for dim, tag in boundaries:
    xmin, ymin, zmin, xmax, ymax, ymax = gmsh.model.getBoundingBox(dim,tag)
    if abs(xmin - 0) < tol and abs(xmax - 0) < tol:
        left_curves.append(tag)
    elif abs(xmin - L0) < tol and abs(xmax - L0) < tol:
        right_curves.append(tag)
    elif abs(xmin - (L0 / 2)) < tol and abs(xmax - (L0 / 2)) < tol:
        mid_curves.append(tag)

if left_curves:
    gmsh.model.addPhysicalGroup(1, left_curves, name="Left")
if right_curves:
    gmsh.model.addPhysicalGroup(1, right_curves, name="Right")

gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write("electrode_band.msh")
gmsh.fltk.run()
gmsh.finalize()
