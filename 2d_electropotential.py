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

from ufl import SpatialCoordinate, TestFunction, TrialFunction, dx, grad, inner

from mpi4py import MPI

import numpy as np
import pyvista

msh_file = "2d_wrapped.msh"

comm = MPI.COMM_WORLD

σ_bone = 2.0e-2
σ_tranverse_muscle = 0.107
σ_fat = 4.07e-2
σ_skin = 4.88e-4
σ_cnt = 0.9135
σ_eco = 1e-5

mesh_data = gmshio.read_from_msh(msh_file, comm, 0, gdim=2)

mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
groups = mesh_data.physical_groups

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
cnt_cells = np.hstack((
    cell_tags.find(groups["Conductor_Array"].tag),
    cell_tags.find(groups["Ground"].tag)
))
eco_cells = cell_tags.find(groups["Substrate"].tag)

σ.x.array[bone_cells] = σ_bone
σ.x.array[muscle_cells] = σ_tranverse_muscle
σ.x.array[fat_cells] = σ_fat
σ.x.array[skin_cells] = σ_skin
σ.x.array[cnt_cells] = σ_cnt
σ.x.array[eco_cells] = σ_eco

Is.x.array[:] = 0.0
Is.x.array[muscle_cells] = 1.0    # A/m^3

ground_facets = facet_tags.find(groups["Ground"].tag)

mesh.topology.create_connectivity(fdim, tdim)
facet_to_cell = mesh.topology.connectivity(fdim, tdim)

ground_boundary_facets = [f for f in ground_facets if len(facet_to_cell.links(f)) == 1]
ground_facets = np.array(ground_boundary_facets, dtype=np.int32)

V = functionspace(mesh, ("Lagrange", 1))

ground_dofs = locate_dofs_topological(V, fdim, ground_facets)
ground_bc = dirichletbc(0.0, ground_dofs, V)

u, v = TrialFunction(V), TestFunction(V)
a = inner(σ * grad(u), grad(v)) * dx
L = Is * v * dx

problem = LinearProblem(
    a,
    L,
    bcs=[ground_bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)
uh = problem.solve()


mesh.topology.create_connectivity(tdim, tdim)
u_topology, u_cell_types, u_geometry = vtk_mesh(V)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["V"] = uh.x.array.real
u_grid.set_active_scalars("V")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
