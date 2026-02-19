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
from dolfinx.io import gmsh as gmshio, XDMFFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import meshtags
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

# Penalty/Augmented Lagrangian parameters
rho = 1E+6		# Penalty parameter (tune E/h scale)
tol = 1E-6		# Convergence tolerance on displacement updates
maxiter = 40

mesh_data = gmshio.read_from_msh(msh_file, comm, 0, gdim=2)

print("Mesh Data:", mesh_data)

msh = mesh_data.mesh 
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
physical_groups = mesh_data.physical_groups

print("Cell tags:", cell_tags)
print("Facet tags:", facet_tags)
print("Physical groups:", physical_groups)

tdim = msh.topology.dim
fdim = tdim - 1
gdim = msh.geometry.dim

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

print(f"Geometry dimension is {msh.geometry.dim}")

V = fem.functionspace(msh, ("Lagrange", 2, (gdim, )))

left_facets = facet_tags.find(11)
right_facets = facet_tags.find(12)

left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)

u_left = np.array([L0 / 2, 2 * r], dtype=default_scalar_type)
#u_left = np.array([0, 0], dtype=default_scalar_type)
u_right = np.array([-L0 / 2, 2 * r], dtype=default_scalar_type)
#u_right = np.array([-L0/2, 0], dtype=default_scalar_type)

bc_left = fem.dirichletbc(u_left, left_dofs, V)
bc_right = fem.dirichletbc(u_right, right_dofs, V)

bcs = [bc_left, bc_right]

ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

contact_tag = physical_groups["Skin_Electrode"].tag
print("Contact_tag:", contact_tag)

contact_facets = facet_tags.find(contact_tag)
R_skin = r
center = np.array([0.0, 0.0])

#MeshTags for contact *domain* (value 1 indicates contact candidate facets)
contact_values = np.ones(contact_facets.shape, dtype=np.int32)
contact_mt = meshtags(msh, fdim, contact_facets, contact_values)

#helper: UFL measure restricted to the contact tag (all contact cadidate facets)
ds_contact_all = ufl.Measure("ds", domain=msh, subdomain_data=contact_mt)

# Normal on facets (UFL)
n = ufl.FacetNormal(msh)
x = ufl.SpatialCoordinate(msh)

def σ(u):
    """Return an expression for the stress σ given a displacement field"""
    return (2.0 * μ_func * ufl.sym(ufl.grad(u))
        + λ_func * ufl.nabla_div(u) * ufl.Identity(len(u)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(σ(u), ufl.grad(v)) * ufl.dx
f = fem.Constant(msh, np.array([0.0] * gdim, dtype=default_scalar_type))
L = ufl.dot(f, v) * ufl.dx

# Initial guess (zero)
u_k = fem.Function(V)
u_k.x.array[:] = 0.0

# Initialze lambda per candidate facet (as and array indexed by contact_facets order)
#lambda_vals = np.zeros(contact_facets.shape[0], dtype=np.float64)

# Precompute facet->vertex connectivity for centroid sampling (useful for speed)
# Ensure connectivity exists
if not msh.topology.index_map(fdim).size_local:
    msh.topology.create_connectivity(fdim, 0)
facet_to_vertices = msh.topology.connectivity(fdim, 0)

# Outward normal from the skin circle 
r_vec = x - ufl.as_vector(center)
n_skin = r_vec / ufl.sqrt(ufl.dot(r_vec, r_vec))
gap = ufl.dot(u_k + r_vec, n_skin) - R_skin
g_pos = ufl.min_value(gap, 0.0)

# To sample u at vertices, creat a CG1 vector function space (vertex-based)
# interpolate/projection
V_vert = fem.functionspace(msh, ("Lagrange", 1, (gdim, )))


# Begin ALM iterations
for it in range(maxiter):
    # --- 1) Produce vertex-valued displacement u_vert from current iterate u_k ---
    u_vert = fem.Function(V_vert)
    try:
        u_vert.interpolate(u_k)
    except Exception:
        # fallback L2 projection
        trial = ufl.TrialFunction(V_vert)
        test = ufl.TestFunction(V_vert)
        a_p = ufl.inner(trial, test) * ufl.dx
        L_p = ufl.inner(u_k, test) * ufl.dx
        proj = LinearProblem(a_p, L_p, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u_vert = proj.solve()
    u_vert_vals = u_vert.x.array.reshape((-1, gdim))

    #lambda_avg = np.mean(lambda_vals)
    #g_avg = fem.assemble_scalar(form(gap * ds(contact_tag)))


    #L_contact = (lambda_avg + rho * g_avg) * ufl.dot(v, n_skin) * ds(contact_tag)
    L_contact = rho * g_pos * ufl.dot(v, n_skin) * ds(contact_tag)
    # --- 4) assemble total problem and solve linear system ---
    L_total = L + L_contact
    
    problem = LinearProblem(
        a,
        L_total,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="linear_elasticity",
    )
    u_new = problem.solve()

    # --- 5) Update lambda per contact facet: lambda <- max(0, lambda + rho*(g - u_new·n)) ---
    u_vert_new = fem.Function(V_vert)
    try:
        u_vert_new.interpolate(u_new)
    except Exception:
        trial = ufl.TrialFunction(V_vert)
        test = ufl.TestFunction(V_vert)
        a_p = ufl.inner(trial, test) * ufl.dx
        L_p = ufl.inner(u_new, test) * ufl.dx
        proj = LinearProblem(a_p, L_p,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u_vert_new = proj.solve()

    u_vert_new_vals = u_vert_new.x.array.reshape((-1, gdim))
    
    g_val = fem.assemble_scalar(form(gap * ds(contact_tag)))

    #lambda_vals[:] = np.maximum(0.0, lambda_vals + rho * g_val)

    # compute convergence measure
    du = np.linalg.norm(u_new.x.array)
    print(f"ALM iter {it}: ||u||={du:.3e}")
    u_k.x.array[:] = u_new.x.array[:]

    if du < tol:
        print("ALM converged.")
        break

# final solution is u_k
u_vis = fem.Function(V_vert)
u_vis.interpolate(u_k) 
uh = u_k

# --- Prepare displacement output (interpolate to CG1 vector space) ---
print(f"geometry dimesion is {gdim}, with shape of {msh.geometry.x.shape}")
num_points = msh.geometry.x.shape[0]
u_vals = u_vis.x.array.reshape((-1, gdim))

if u_vals.shape[0] != num_points:
    raise RuntimeError(
        f"DOF/vertex mismatch: u_vals has {u_vals.shape[0]} rows, mesh has" 
        + f"{num_points} points"
        )

# Extend 2D displacement to 3D (add zero z-component)
if u_vals.shape[1] == 2 and msh.geometry.x.shape[1] == 3:
    pad = np.zeros((u_vals.shape[0], 1))
    u_vals = np.hstack([u_vals, pad])
orig_coords = msh.geometry.x.copy()

#msh.geometry.x[:] = msh.geometry.x + u_vals

sigma_dev = σ(uh) - (1.0 / 3) * ufl.tr(σ(uh)) * ufl.Identity(len(uh))
sigma_vm = ufl.sqrt((3.0 / 2) * ufl.inner(sigma_dev, sigma_dev))

W = fem.functionspace(msh, ("DG", 0))
sigma_vm_expr = fem.Expression(sigma_vm, W.element.interpolation_points)
sigma_vm_h = fem.Function(W)
sigma_vm_h.interpolate(sigma_vm_expr)
sigma_vm_h.name = "σ"

with XDMFFile(msh.comm, "out_elasticity/displacements.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u_vis)

# Save solution to XDMF format
with XDMFFile(msh.comm, "out_elasticity/von_mises_stress.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(sigma_vm_h)

# Create the VTK mesh from your function space
topology, cell_types, geometry = plot.vtk_mesh(V_vert)

# Create an unstructured grid (2D domain)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach displacement data (reshape correctly for 2D)
uh_array = u_vis.x.array.reshape((geometry.shape[0], gdim))

# If 2D, pad to 3 components (z = 0) because VTK expects 3-component vectors
if gdim == 2:
    zeros = np.zeros((uh_array.shape[0], 1), dtype=uh_array.dtype)
    uh_3 = np.hstack([uh_array, zeros])
else:
    uh_3 = uh_array

# Ensure dtype is float64 for PyVista
uh_3 = uh_3.astype(np.float64)

grid["u"] = uh_3
"""uh_x = np.zeros(uh_3.shape)
uh_y = np.zeros(uh_3.shape)
uh_x[:, 0] = uh_3[:, 0]
uh_y[:, 1] = uh_3[:, 1]
grid["u_xy"] = uh_x + uh_y"""

grid.set_active_vectors("u")
# Warp geometry by displacement (for visualization)
# Factor scales how much deformation is shown (visual only)
warped = grid.warp_by_vector("u", factor=1.0)

# Initialize the plotter
if msh.comm.rank == 0:
    p = pyvista.Plotter()
    #p.add_mesh(grid, style="wireframe", color="black", line_width=0.5, label="Original")
    ##p.add_mesh(warped, scalars=np.linalg.norm(uh_array, axis=1),
    ##           cmap="viridis", show_edges=True, label="Deformed")
    #p.add_mesh(grid, scalars="u_xy", cmap="coolwarm", show_edges=True)
    p.add_mesh(grid, style="wireframe", color="k")
    p.add_mesh(warped, show_edges=True)

    p.add_scalar_bar(title="u_xy (signed displacement)")
    p.show_axes()
    p.view_xy()

    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("deformation.png")
	
