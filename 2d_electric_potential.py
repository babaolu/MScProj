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

import ufl
from dolfinx import fem, plot, default_scalar_type
from dolfinx.fem import form
from dolfinx.io import gmshio, XDMFFile
from dolfinx.fem.petsc import LinearProblem
import pyvista

msh_file = "2d_forearm_electrode.msh"

comm = MPI.COMM_WORLD

# (mm, Pa,)
L0 = 2 * np.pi * 30	# Band length 
t = 1		# Band thickness
r = 45			# Radius of forearm
ν_eco = 0.49
E_eco = 68947.6		# Modulus
μ_eco = E_eco / (2.0 * (1.0 + ν_eco))
λ_eco = E_eco * ν_eco / ((1.0 + ν_eco) * (1.0 - 2.0 * ν_eco))
ν_cnt = 0.34
E_cnt = 1.1E+12
μ_cnt = E_cnt / (2.0 * (1.0 + ν_cnt))
λ_cnt = E_cnt * ν_cnt / ((1.0 + ν_cnt) * (1.0 - 2.0 * ν_cnt))
ν_rigid = 0.04
E_rigid = 1.1E+20
μ_rigid = E_rigid / (2.0 * (1.0 + ν_rigid))
λ_rigid = E_rigid * ν_rigid / ((1.0 + ν_rigid) * (1.0 - 2.0 * ν_rigid))

msh, cell_tags, facet_tags = gmshio.read_from_msh(msh_file, comm, 0, gdim=2)

tdim = msh.topology.dim
fdim = tdim - 1

# Create arrays matching number of cells
λ = np.zeros(msh.topology.index_map(tdim).size_local,
        dtype=default_scalar_type)
μ = np.zeros_like(λ)

for cell_id, tag in enumerate(cell_tags.values):
    if tag == 1:
        λ[cell_id] = λ_eco
        μ[cell_id] = μ_eco
    elif tag == 2:
        λ[cell_id] = λ_cnt
        μ[cell_id] = μ_cnt
    else:  # All other forearm tissues rigid
        λ[cell_id] = λ_rigid
        μ[cell_id] = μ_rigid

V0 = fem.functionspace(msh, ("DG", 0))
λ_func = fem.Function(V0)
μ_func = fem.Function(V0)
λ_func.x.array[:] = λ
μ_func.x.array[:] = μ

print("Values: ", facet_tags.values)
print("Facet dim:", facet_tags.dim)
print(f"Whole face tag of type {facet_tags.name}=>", dir(facet_tags))

V = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim, )))

left_facets = facet_tags.find(11)
right_facets = facet_tags.find(12)

left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)

#u_left = np.array([L0 / 2, 2 * r], dtype=default_scalar_type)
u_left = np.array([L0/2, 0], dtype=default_scalar_type)
#u_right = np.array([-L0 / 2, 2 * r], dtype=default_scalar_type)
u_right = np.array([-L0/2, 0], dtype=default_scalar_type)

bc_left = fem.dirichletbc(u_left, left_dofs, V)
bc_right = fem.dirichletbc(u_right, right_dofs, V)

bcs = [bc_left, bc_right]

ds = ufl.Measure("ds", domain=msh)

def σ(u):
    """Return an expression for the stress σ given a displacement field"""
    return (2.0 * μ_func * ufl.sym(ufl.grad(u))
        + λ_func * ufl.nabla_div(u) * ufl.Identity(len(u)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(σ(u), ufl.grad(v)) * ufl.dx
T = fem.Constant(msh, default_scalar_type((0, 0)))
f = fem.Constant(msh, default_scalar_type((0, 0)))
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
)
uh = problem.solve()

sigma_dev = σ(uh) - (1 / 3) * ufl.tr(σ(uh)) * ufl.Identity(len(uh))
sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))

W = fem.functionspace(msh, ("DG", 0))
sigma_vm_expr = fem.Expression(sigma_vm, W.element.interpolation_points())
sigma_vm_h = fem.Function(W)
sigma_vm_h.interpolate(sigma_vm_expr)
sigma_vm_h.name = "σ"

#with XDMFFile(msh.comm, "out_elasticity/displacements.xdmf", "w") as file:
#    file.write_mesh(msh)
#    file.write_function(uh)

# Save solution to XDMF format
with XDMFFile(msh.comm, "out_elasticity/von_mises_stress.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(sigma_vm_h)



"""
# --- create VTK mesh for plotting (same as you had) ---
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# --- interpolate uh to a P1 vector function so dofs map to mesh vertices ---
# create a vector-valued P1 space for visualization

try:
    V_vis = fem.vector_functionspace(msh, ("Lagrange", 2))
except Exception:
    # alternate API
    V_vis = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim,)))

uh_vis = fem.Function(V_vis)
# Interpolate (copy) uh values into uh_vis (works when shapes are compatible)
uh_vis.interpolate(uh)

# Define tensor function space for stress
Vsig = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, msh.geometry.dim)))
σh = fem.Function(Vsig)

# Define expression for stress in terms of uh
λ_const = fem.Constant(msh, default_scalar_type(np.mean(λ)))
μ_const = fem.Constant(msh, default_scalar_type(np.mean(μ)))

# --- define stress expression ---
σ_expr = fem.Expression(
    2.0 * μ_const * ufl.sym(ufl.grad(uh)) +
    λ_const * ufl.nabla_div(uh) * ufl.Identity(msh.geometry.dim),
    Vsig.element.interpolation_points()
)
σh.interpolate(σ_expr)

# --- compute von Mises stress robustly ---
σ_array = σh.x.array.reshape((-1, msh.geometry.dim, msh.geometry.dim))
sxx = σ_array[:, 0, 0]
syy = σ_array[:, 1, 1]
szz = σ_array[:, 2, 2] if msh.geometry.dim == 3 else np.zeros_like(sxx)
sxy = σ_array[:, 0, 1]
syz = σ_array[:, 1, 2] if msh.geometry.dim == 3 else np.zeros_like(sxx)
szx = σ_array[:, 2, 0] if msh.geometry.dim == 3 else np.zeros_like(sxx)

σ_vm = np.sqrt(
    0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) +
    3.0 * (sxy**2 + syz**2 + szx**2)
)

grid.cell_data["von Mises"] = σ_vm  # use cell_data since σh is DG(0)

# Now build an array of shape (num_points, dim)
dim = msh.geometry.dim
num_points = geometry.shape[0]  # number of points in the VTK geometry

# uh_vis.x.array holds the DOF vector; reshape into (num_dofs, dim)
uh_vals = uh_vis.x.array.reshape((-1, dim))  # should match num_points x dim

# Safety checks
if uh_vals.shape[0] != num_points:
    raise RuntimeError(f"Point count mismatch: uh_vis has {uh_vals.shape[0]} rows, "
                       f"VTK geometry has {num_points} points. Interpolation step failed.")

# If 2D, pad to 3 components (z = 0) because VTK expects 3-component vectors
if dim == 2:
    zeros = np.zeros((uh_vals.shape[0], 1), dtype=uh_vals.dtype)
    uh_3 = np.hstack([uh_vals, zeros])
else:
    uh_3 = uh_vals

# Ensure dtype is float64 for PyVista
uh_3 = uh_3.astype(np.float64)

# Attach as point data and warp
grid["Displacement"] = uh_3  # must be (n_points, 3)
warped = grid.warp_by_vector("Displacement", factor=1e2)  # adjust factor as needed

# Plot
p = pyvista.Plotter()
p.add_mesh(grid, style="wireframe", color="black", line_width=0.5, label="Original")
p.add_mesh(warped, scalars=np.linalg.norm(uh_3[:, :3], axis=1), cmap="viridis",
           show_edges=True, label="Deformed")
p.add_mesh(
    warped,
    scalars="von Mises",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={"title": "Von Mises Stress [Pa]"},
)
p.add_scalar_bar(title="|u|")
p.show_axes()
p.view_xy()
p.add_axes()
p.show_bounds(grid='front', location='outer', all_edges=True)
if not pyvista.OFF_SCREEN:
    p.show()
else:
    p.screenshot("deformation.png")
"""
