from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    dirichletbc,
    Function,
    functionspace,
    assemble_scalar,
    form,
    locate_dofs_geometrical,
    locate_dofs_topological
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmsh as gmshio
from dolfinx.plot import vtk_mesh
from dolfinx.mesh import compute_midpoints

from ufl import SpatialCoordinate, TestFunction, TrialFunction, Measure, dx, grad, inner

from mpi4py import MPI

import numpy as np
import pyvista
from pathlib import Path

msh_file = "2d_wrapped.msh"

comm = MPI.COMM_WORLD

# Conductivities in S/mm
σ_bone = 2.0e-2 / 1000
σ_muscle = 0.3 / 1000
σ_fat = 4.07e-2 / 1000
σ_skin = 4.88e-4 / 1000
σ_cnt = 4.21 # 49.49 / 1000
σ_eco = 1e-8 / 1000

mesh_data = gmshio.read_from_msh(msh_file, comm, 0, gdim=2)

mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
groups = mesh_data.physical_groups

print("Groups:", groups)

tdim = mesh.topology.dim
fdim = tdim - 1
gdim = mesh.topology.dim

Q = functionspace(mesh, ("DG", 0))

σ = Function(Q)
Is = Function(Q)

bone_cells = cell_tags.find(groups["Bone"].tag)
muscle_cells = cell_tags.find(groups["Muscle"].tag)
fat_cells = cell_tags.find(groups["Fat"].tag)
skin_cells = cell_tags.find(groups["Skin"].tag)
cnt_cells = cell_tags.find(groups["Conductor_Array"].tag)
eco_cells = cell_tags.find(groups["Substrate"].tag)

σ.x.array[bone_cells] = σ_bone
σ.x.array[muscle_cells] = σ_muscle
σ.x.array[fat_cells] = σ_fat
σ.x.array[skin_cells] = σ_skin
σ.x.array[cnt_cells] = σ_cnt
σ.x.array[eco_cells] = σ_eco

Is.x.array[:] = 0.0

cell_centers = compute_midpoints(mesh, tdim, muscle_cells)[:, :2]

x_plus = np.array([20.0, 0.0])
x_minus = np.array([24.0, 0.0])

x1_plus = np.array([0.0, 22.0])
x1_minus = np.array([0.0, 28.0])

sigma_s = 0.05
I0 = 7.0e-6     # A/mm^3

r_plus  = np.sum((cell_centers - x_plus)**2, axis=1)
r_minus = np.sum((cell_centers - x_minus)**2, axis=1)

r1_plus  = np.sum((cell_centers - x1_plus)**2, axis=1)
r1_minus = np.sum((cell_centers - x1_minus)**2, axis=1)

Is_vals = I0 * (
#    np.exp(-r_plus / (2*sigma_s**2)) #-
#    np.exp(-r_minus / (2*sigma_s**2))
    + np.exp(-r1_plus / (2*sigma_s**2)) #-
#    np.exp(-r1_minus / (2*sigma_s**2))
)

print("Max |Is_vals|:", np.max(np.abs(Is_vals)) if len(Is_vals) > 0 else 0.0)
print("Min distance to x_plus:", np.min(np.sqrt(r_plus)) if len(r_plus) > 0 else "None")
print("Min distance to x_minus:", np.min(np.sqrt(r_minus)) if len(r_minus) > 0 else "None")

Is.x.array[muscle_cells] = Is_vals

ground_facets = facet_tags.find(groups["Ground"].tag)

V = functionspace(mesh, ("Lagrange", 1))
"""
dof0 = np.array([0], dtype=np.int32)
bc_gauge = dirichletbc(
    default_scalar_type(0.0),
    dof0,
    V
)
"""
ground_dofs = locate_dofs_topological(V, fdim, ground_facets)
bc_gauge = dirichletbc(0.0, ground_dofs, V)

u, v = TrialFunction(V), TestFunction(V)
a = inner(σ * grad(u), grad(v)) * dx
L = Is * v * dx

print("Number of muscle cells:", len(muscle_cells))
print("Muscle cell centers min:", np.min(cell_centers, axis=0) if len(cell_centers) > 0 else "None")
print("Muscle cell centers max:", np.max(cell_centers, axis=0) if len(cell_centers) > 0 else "None")

problem = LinearProblem(
    a,
    L,
    bcs=[bc_gauge],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)
uh = problem.solve()

results_folder = Path("/home/itunz/Work/MScProj")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "2d_parview"
with XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)

u_vals = uh.x.array.real

print("V min:", np.min(u_vals))
print("V max:", np.max(u_vals))
print("Any NaN:", np.isnan(u_vals).any())

ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)

electrode_voltages = {}

for i in range(16):
    name = f"E{i}"
    tag = groups[name].tag

    # Integral of potential over electrode
    integral_u = assemble_scalar(form(uh * ds(tag)))
    integral_u = mesh.comm.allreduce(integral_u, op=MPI.SUM)

    # Electrode length (measure of boundary)
    electrode_area = assemble_scalar(form(1.0 * ds(tag)))
    electrode_area = mesh.comm.allreduce(electrode_area, op=MPI.SUM)

    if electrode_area > 0:
        V_avg = integral_u / electrode_area
    else:
        V_avg = 0.0

    electrode_voltages[name] = V_avg

text_path = "fat.txt"
selected = []
for i in ["E0", "E5", "E10", "E15"]:
    selected.append(electrode_voltages[i])

with open(text_path, 'a') as file:
    file.write(str(selected).strip("[]"))
    file.write("\n")

if mesh.comm.rank == 0:
    print("\nElectrode Voltages (Volts):")
    for k, v in electrode_voltages.items():
        print(f"{k}: {v:.6e}")
"""
mesh.topology.create_connectivity(tdim, tdim)
u_topology, u_cell_types, u_geometry = vtk_mesh(V)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["V"] = uh.x.array.real
u_grid.set_active_scalars("V")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()"""
