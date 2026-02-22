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
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.mesh import meshtags
import pyvista

msh_file = "2d_forearm_electrode.msh"

comm = MPI.COMM_WORLD

# (mm, Pa,)
L0 = 2 * np.pi * 28	# Band length 
t = 1		# Band thickness
r = 42			# Radius of forearm
ν_eco = 0.49
E_eco = 68947.6		# Modulus
μ_eco = E_eco / (2.0 * (1.0 + ν_eco))
λ_eco = E_eco * ν_eco / ((1.0 + ν_eco) * (1.0 - 2.0 * ν_eco))
ν_cnt = 0.34
E_cnt = 4.7E+9
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

V0 = fem.functionspace(msh, ("DG", 0))
λ = fem.Function(V0)
μ = fem.Function(V0)
C10 = fem.Function(V0)
C01 = fem.Function(V0)

for cell_id, tag in enumerate(cell_tags.values):
    if tag == physical_groups["Substrate"].tag:
        λ.x.array[cell_id] = λ_eco
        C10.x.array[cell_id] = μ_eco / 4.0   # example split; C10 + C01 = mu_eco/2
        C01.x.array[cell_id] = μ_eco / 4.0
    elif tag ==  physical_groups["Conductor_Array"].tag:
        λ.x.array[cell_id] = λ_cnt
        C10.x.array[cell_id] = μ_cnt / 4.0   # example split; C10 + C01 = mu_eco/2
        C01.x.array[cell_id] = μ_cnt / 4.0

    else:  # All other forearm tissues rigid
        λ.x.array[cell_id] = λ_rigid
        C10.x.array[cell_id] = μ_rigid / 4.0   # example split; C10 + C01 = mu_eco/2
        C01.x.array[cell_id] = μ_rigid / 4.0

print("Values: ", facet_tags.values)
print("Facet dim:", facet_tags.dim)
print(f"Whole face tag of type {facet_tags.name}=>", dir(facet_tags))

print(f"Geometry dimension is {msh.geometry.dim} -> {msh.geometry.x.shape[1]}")

V = fem.functionspace(msh, ("Lagrange", 2, (gdim, )))
u = fem.Function(V)
v = ufl.TestFunction(V)

left_facets = facet_tags.find(11)
right_facets = facet_tags.find(12)

left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)

u_left = np.array([L0 / 2, 2 * r], dtype=default_scalar_type)
#u_left = np.array([0, 2 * r], dtype=default_scalar_type)
u_right = np.array([-L0 / 2, 2 * r], dtype=default_scalar_type)
#u_right = np.array([0, 2 * r], dtype=default_scalar_type)

def σ(u):
    """Return an expression for the stress σ given a displacement field"""
    return (2.0 * μ * ufl.sym(ufl.grad(u))
        + λ * ufl.nabla_div(u) * ufl.Identity(gdim))

dx = ufl.dx
ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

# --- 1. Define Kinematics (plane-strain consistent) ---
I = ufl.variable(ufl.Identity(gdim))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
Ic = ufl.variable(ufl.tr(C))
J_det = ufl.variable(ufl.det(F))

I1 = ufl.variable(Ic + 1.0)                                   # 3D I1 with F33=1
I2 = ufl.variable((Ic**2 - ufl.tr(C * C))/2.0 + Ic)          # 3D I2 with F33=1

# --- 2. Mooney-Rivlin energy (exactly recovers your Neo-Hookean when C01=0) ---
psi = (C10 * (I1 - 3.0) +
       C01 * (I2 - 3.0) -
       (2.0 * C10 + 4.0 * C01) * ufl.ln(J_det) +
       (λ / 2.0) * (ufl.ln(J_det))**2)
# --- 3. Derive Stress (First Piola-Kirchhoff) ---
# We take the derivative of energy w.r.t F to get stress
P = ufl.diff(psi, F)

# --- 4. Update the Weak Form ---
# Replace your old 'a_bulk' with this:
a_bulk = ufl.inner(P, ufl.grad(v)) * dx
contact_tag = physical_groups["Top"].tag
print("Contact_tag:", contact_tag)

ds_c = ds(contact_tag)

# Cells adjacent to contact facets
# Ensure connectivity exists
msh.topology.create_connectivity(fdim, tdim)
facet_to_cell = msh.topology.connectivity(fdim, tdim)

contact_facets = facet_tags.find(contact_tag)

# Collect all cells connected to contact facets
contact_cells = np.unique(
    np.hstack([facet_to_cell.links(f) for f in contact_facets])
)

x = ufl.SpatialCoordinate(msh)
R_skin = r


center = ufl.as_vector([0.0, 0.0])

# Current configuration
x_current = x + u

dist = ufl.sqrt(ufl.dot(x_current, x_current))
n_contact = x_current / dist

# Gap (negative = penetration)
gap = R_skin - dist

Vλ = fem.functionspace(msh, ("DG", 0))
lambda_n = fem.Function(Vλ)
lambda_n.name = "lambda_contact"
gamma = fem.Constant(msh, default_scalar_type(1e8))

t_n = lambda_n + gamma * gap

v_n = ufl.dot(v, n_contact)

R_contact = -ufl.conditional(
    ufl.gt(gap, 0.0),
    t_n * v_n,
    0.0
) * ds_c

R = a_bulk + R_contact
J = ufl.derivative(R, u)

# Create a loop to apply load in 10 steps
al_maxiter = 15
al_tol = 1e-6
steps = 10
for step in range(1, steps + 1):
    fraction = step / steps
    print(f"Solving Step {step}/{steps} (Load: {fraction*100:.0f}%)")
    
    # Update BC values
    current_left = u_left * fraction
    current_right = u_right * fraction
    
    # You must update the values inside the BC objects. 
    # Since DirichletBC in dolfinx is immutable, we usually update a Function 
    # that defines the BC, or recreate the BCs inside the loop.
    # Easiest way here: Re-create BCs every step
    bc_left = fem.dirichletbc(current_left, left_dofs, V)
    bc_right = fem.dirichletbc(current_right, right_dofs, V)
    bcs = [bc_left, bc_right]
    for al_iter in range(al_maxiter):
        print(f"  AL iteration {al_iter+1}")

        problem = NonlinearProblem(
            R, u, bcs=bcs, J=J,
            petsc_options_prefix="contact_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-8,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

        problem.solve()

        # --- Update Lagrange multiplier ---
        gap_expr = fem.Expression(gap, Vλ.element.interpolation_points)
        gap_h = fem.Function(Vλ)
        gap_h.interpolate(gap_expr)

        lambda_n.x.array[contact_cells] = np.maximum(
            lambda_n.x.array[contact_cells] + gamma.value * gap_h.x.array[contact_cells],
            0.0
        )
        
        penetration = np.max(gap_h.x.array[contact_cells])

        print(f"    min gap = {penetration:.3e}")

        if abs(penetration) < al_tol:
            print("    AL converged")
            break
        
    # Update u for the next step (Newton solver does this automatically on 'u')
    # final solution is u
    V_vis = fem.functionspace(msh, ("Lagrange", 1, (gdim, )))
    u_vis = fem.Function(V_vis)
    u_vis.interpolate(u) 

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

    """sigma_dev = σ(u) - (1.0 / 3) * ufl.tr(σ(u)) * ufl.Identity(len(u))
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
        file.write_function(sigma_vm_h)"""

    # Create the VTK mesh from your function space
    topology, cell_types, geometry = plot.vtk_mesh(V_vis)

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

    grid.set_active_vectors("u")
    # Warp geometry by displacement (for visualization)
    # Factor scales how much deformation is shown (visual only)
    warped = grid.warp_by_vector("u", factor=1.0)

    # Initialize the plotter
    if msh.comm.rank == 0:
        p = pyvista.Plotter()
        p.add_mesh(grid, style="wireframe", color="k")
        p.add_mesh(warped, show_edges=True)

        p.add_scalar_bar(title="u_xy (signed displacement)")
        p.show_axes()
        p.view_xy()

        if not pyvista.OFF_SCREEN:
            p.show()
        else:
            figure_as_array = p.screenshot("deformation.png")
	
