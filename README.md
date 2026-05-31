# Forearm HD-EMG Multiphysics Modeling & Simulation Pipeline

This repository contains a 2D multi-physics finite element simulation framework designed to model, analyze, and optimize flexible, wearable high-density electromyography (HD-EMG) electrode array bands wrapped around a human forearm. 

The framework is implemented using **FEniCSx (DOLFINx)** and **GMSH**, utilizing high-performance parallel linear algebra solvers via **PETSc** and advanced data visualization in **R**.

---

## 1. Scientific Motivation & Project Purpose

High-Density Electromyography (HD-EMG) is a non-invasive technique that uses grid arrays of closely spaced skin electrodes to record electrical muscle activity. While traditional bipolar EMG only monitors whether a muscle is active, HD-EMG provides high-resolution spatial mapping, enabling the identification of individual motor units, non-invasive motor unit tracking, and precise gesture recognition.

Designing wearable HD-EMG armbands requires balancing two conflicting physical requirements:
1.  **Electromechanical Contact Quality:** The flexible elastomer band must be wrapped tightly enough to ensure uniform skin contact and minimize electrode-skin contact impedance, without causing pain or tissue ischemia (blood flow constriction).
2.  **Bioelectric Signal Resolution:** Design parameters such as electrode diameter (Ed), inter-electrode distance (IED), fat thickness, and limb geometry act as spatial filters, altering the volume conduction of current from deep muscle active fibers to the skin surface.

This project provides an **in-silico design-space exploration tool** to systematically model, sweep, and analyze these design parameters, allowing researchers to optimize sensor layout configurations and evaluate signal degradation caused by anatomical variations (e.g., patient fat layers) before manufacturing physical prototypes.

---

## 2. System Architecture & Module Status

The repository contains two distinct, independent modules:

```
                                  +---------------------------------------+
                                  |            GMSH OCC Engine            |
                                  |      (Fragment & Lineage Tagging)     |
                                  +---------------------------------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |   2D Conforming Mesh      |
                                        |     (2d_wrapped.msh)      |
                                        +---------------------------+
                                                      |
                                    +-----------------+-----------------+
                                    |                                   |
                                    v                                   v
                      +---------------------------+       +---------------------------+
                      |         Module A          |       |         Module B          |
                      |   Bioelectric Sweeps      |       |  Nonlinear Solid Mechanics|
                      |     (Fully Working)       |       |  (Decoupled Prototypes)   |
                      +---------------------------+       +---------------------------+
                      | - Steady-State Poisson    |       | - Mooney-Rivlin Elasticity|
                      | - Monopole Source Sweeps  |       | - ALM Contact Solver      |
                      | - Async Process Sweep     |       | - Nitsche Contact Solver  |
                      | - Voltage Logger (.txt)   |       | - Unstable / Standalone   |
                      | - R Data Visualization    |       |                           |
                      +---------------------------+       +---------------------------+
```

### Module A: Parametric Bioelectric Sweeps (Fully Working)
This is the completed, data-generating research pipeline. It sequentially runs parametric geometric builders in GMSH, solves the bioelectric volume conduction problem in DOLFINx, and saves the recording profile to flat `.txt` files. These files are subsequently loaded and analyzed in R.

### Module B: Nonlinear Solid Mechanics (Decoupled prototypes)
This is an advanced, standalone computational mechanics development testbed. It implements hyperelastic material laws, Augmented Lagrangian contact, and Nitsche-based contact formulations to simulate the physical wrapping of the band around the arm. **These scripts are completely disconnected from the active sweep pipeline.** The bioelectric sweeps are executed on statically constructed concentric polar geometries rather than deformed mechanical configurations.

---

## 3. Module A: Bioelectric Sweep Pipeline (In-Depth)

The bioelectric pipeline models the steady-state volume conduction of action potentials from muscle fibers to recording electrodes.

### 3.1 Anatomical Model & Properties
The forearm is modeled as a concentric, multi-layered 2D cylinder with the following physiological area fractions and isotropic conductivities (extracted directly from the solver code):

*   **Bone Core:** Radius $R_{\text{bone}} = r \times \sqrt{0.107}$ (Area Fraction: $10.7\%$)
    *   *Conductivity:* $\sigma_{\text{bone}} = 2.0 \times 10^{-5}$ S/mm ($0.02$ S/m)
*   **Muscle Layer:** Radius $R_{\text{muscle}} = r \times \sqrt{f_{\text{mus}} + 0.107}$ (Typical Area Fraction: $61.1\%$)
    *   *Conductivity:* $\sigma_{\text{muscle}} = 3.0 \times 10^{-4}$ S/mm ($0.3$ S/m)
*   **Fat Ring:** Radius $R_{\text{fat}} = r \times \sqrt{0.222 + 0.611 + 0.107}$ (Typical Area Fraction: $22.2\%$)
    *   *Conductivity:* $\sigma_{\text{fat}} = 4.07 \times 10^{-5}$ S/mm ($0.0407$ S/m)
*   **Skin Boundary:** Outer radius $R_{\text{skin}} = r$ (Typical Area Fraction: $6.0\%$)
    *   *Conductivity:* $\sigma_{\text{skin}} = 4.88 \times 10^{-7}$ S/mm ($0.000488$ S/m)
*   **Sensor Substrate (Ecoflex elastomer):** Thickness $t = 2.0$ mm
    *   *Conductivity:* $\sigma_{\text{eco}} = 1.0 \times 10^{-11}$ S/mm ($10^{-8}$ S/m)
*   **Conductor Array (CNT electrodes):** Array of $16$ contact pads, thickness $Et = 0.4$ mm
    *   *Conductivity:* $\sigma_{\text{cnt}} = 4.21$ S/mm ($4210$ S/m)

### 3.2 Conforming Mesh Generation (OpenCASCADE Fragment)
To prevent numerical current bottlenecks, the mesh must be fully conforming (nodes must be shared along the contact boundaries between tissues and electrodes). This is achieved using the **GMSH OpenCASCADE (OCC) kernel** via the following steps:
1.  Separate surfaces for the tissue disks, elastomer band, and conductor pads are declared.
2.  The `gmsh.model.occ.fragment` boolean operator intersects all overlapping entities, cuts out nested boundaries, and glues matching nodes:
    ```python
    ov, ovv = gmsh.model.occ.fragment(tissue_inputs + all_devices, [], removeObject=False)
    ```
3.  The parent lineage tracking array `ovv` maps original entities to their newly fragmented, boundary-shared sub-surfaces, allowing the script to safely tag physical cell and facet groups:
    ```python
    final_skin_tag = [tag[1] for tag in ovv[0]]
    substrate_tag = [tag[1] for tag in ovv[4]]
    ```

### 3.3 Poisson Finite Element Formulation
Current propagation through tissues is modeled as steady-state bioelectric volume conduction.

#### Variational Weak Form
The electric potential $V \in H^1(\Omega)$ satisfies Poisson's equation with a spatial volume current source $I_s$:
$$-\nabla \cdot (\sigma \nabla V) = I_s \quad \text{in } \Omega$$

Multiplying by a test function $v \in H^1_0(\Omega)$ and integrating by parts yields the bilinear form $a(u, v)$ and linear form $L(v)$ solved in DOLFINx:
$$a(u, v) = \int_{\Omega} \sigma \nabla u \cdot \nabla v \, dx$$
$$L(v) = \int_{\Omega} I_s v \, dx$$

#### Boundary Conditions
*   **Ground Electrode:** Dirichlet boundary condition $V = 0.0$ applied to the 1D physical group `Ground` (the back facet of the ground pad).
*   **Outer Boundaries:** Insulating Neumann boundary condition $\sigma \nabla V \cdot \mathbf{n} = 0$ on all other external surfaces.

#### Electrode Voltage Extraction
The average potential recorded by each of the 16 electrodes $E_i$ is computed by taking the line integral of $V$ over the electrode-skin contact boundary segment $\Gamma_{E_i}$ (labeled `E0` through `E15`) and dividing by the segment's arc length:
$$V_{\text{avg}} = \frac{\int_{\Gamma_{E_i}} V \, ds}{\int_{\Gamma_{E_i}} 1 \, ds}$$

This is implemented directly via DOLFINx boundary integrations:
```python
integral_u = assemble_scalar(form(uh * ds(tag)))
electrode_area = assemble_scalar(form(1.0 * ds(tag)))
V_avg = integral_u / electrode_area
```

---

### 3.4 Detailed Parameter Sweeps
The pipeline sweeps six design parameters sequentially. The orchestrator scripts and their parameters, ranges, and simulation counts are summarized below:

| Sweep Script | GMSH Generator | Poisson Solver | Parameter Swept | Actual Sweep Range | Steps (Simulations) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ed.py`** | `2d_wrap_ed.py` | `2d_elect_ed.py` | Electrode Diameter ($E_d$) | $1.0\text{ mm}$ to $11.0\text{ mm}$ | 300 |
| **`fat.py`** | `2d_wrap_fat.py` | `2d_elect_fat.py` | Muscle Fraction $f_{\text{mus}}$ (varies Fat thickness) | $0.90$ to $0.30$ ($Fat/Muscle$ ratios: $0.10$ to $0.70$) | 300 |
| **`depth.py`** | *Static Mesh* | `2d_elect_depth.py` | Source Depth (Radial position within muscle) | $14.0\text{ mm}$ to $35.59\text{ mm}$ (from bone to fat) | 500 |
| **`ied.py`** | `2d_wrap_ied.py` | `2d_elect_ied.py` | Band Radius $b_{\text{radius}}$ (changes derived IED) | $22.0\text{ mm}$ to $43.0\text{ mm}$ | 300 |
| **`spread.py`**| *Static Mesh* | `2d_elect_spread.py`| Gaussian Source Spatial Spread ($\sigma_s$) | $0.05$ to $15.05$ | 300 |
| **`size.py`** | `2d_wrap_size.py`| `2d_elect_size.py`| Core Forearm Skin Radius ($r$) | $30.0\text{ mm}$ to $45.0\text{ mm}$ | 300 |

#### Orchestrator Code Execution Model
The python orchestrators invoke the subprocesses synchronously. Inside the parameter loop, `asyncio.run()` executes the tasks in a blocking sequence:
```python
async def run_sim(i):
    proc = await asyncio.create_subprocess_exec("python3", "2d_wrap_ed.py", f"{i:.2f}")
    await proc.wait()
    proc = await asyncio.create_subprocess_exec("python3", "2d_elect_ed.py")
    await proc.wait()

for i in np.linspace(1, 11, 300):
    asyncio.run(run_sim(i))
```
This isolates the execution of each DOLFINx and PETSc solver instantiation, preventing the memory leaks and PETSc solver state accumulation that typically occur during massive sequential loops within a single Python process.

### 3.5 Logging & Analysis in R
For each parameter step, the average voltage at four representative electrodes (**E0, E5, E10, E15**) is logged to flat `.txt` files (e.g., `ed.txt`, `fat.txt`). 

These files are read in R using `readr`, reshaped into tidy long-format dataframes using `tidyr::pivot_longer()`, and plotted using `ggplot2` to analyze design sensitivities:
```R
library(ggplot2)
library(tidyr)
df <- read_csv("ed.txt", col_names = FALSE)
colnames(df) <- c("E0", "E5", "E10", "E15")
df$ed <- seq(from = 1, to = 11, length.out = 300)
f_long <- pivot_longer(df, cols = -ed, names_to = "Line", values_to = "Value")
ggplot(f_long, aes(x = ed, y = Value, color = Line)) + geom_line() + labs(x = "Electrode diameter (mm)", y = "Volts (V)")
```

---

## 4. Module B: Nonlinear Contact Mechanics (Prototype Details)

Module B consists of standalone files exploring the mechanics of wrapping the elastic band around the arm.

### 4.1 Constitutive & Mechanics Model
Because the elastomer band undergoes massive rotations and strains during wrapping, linear elasticity is insufficient.
*   **Kinematics:** Implements finite strain kinematics. Given the displacement field $\mathbf{u}$, the deformation gradient is $\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}$, the right Cauchy-Green tensor is $\mathbf{C} = \mathbf{F}^T \mathbf{F}$, and the Jacobian volume ratio is $J = \det(\mathbf{F})$.
*   **Constitutive Law (Mooney-Rivlin):** Implemented in `2d_mr_deformation.py` using first and second invariants $I_1 = \operatorname{tr}(\mathbf{C})$ and $I_2 = \frac{1}{2}(\operatorname{tr}(\mathbf{C})^2 - \operatorname{tr}(\mathbf{C}^2))$:
    $$\psi = C_{10}(I_1 - 3) + C_{01}(I_2 - 3) - (2C_{10} + 4C_{01})\ln(J) + \frac{\lambda}{2}(\ln(J))^2$$
*   **Stress Derivation:** The First Piola-Kirchhoff stress tensor $\mathbf{P}$ is derived automatically via symbolical UFL differentiation with respect to the deformation gradient:
    ```python
    P = ufl.diff(psi, F)
    ```

### 4.2 Mechanical Contact Formulations
To prevent the band from clipping into the arm, the solver implements and evaluates two different boundary contact techniques:

#### 1. Augmented Lagrangian Method (ALM)
In `2d_mr_deformation.py`, contact constraints are enforced using a dual method. Non-penetration is modeled using a penalty parameter $\gamma$ and a discrete Lagrange multiplier field $\lambda_n$ defined on the contact boundary cells. The multiplier is updated in an outer iteration loop:
```python
t_n = lambda_n + gamma * gap
lambda_n.x.array[contact_cells] = np.maximum(lambda_n.x.array[contact_cells] + gamma.value * gap_h.x.array[contact_cells], 0.0)
```

#### 2. Nitsche's Method
In `2d_electric_potential4.py` (which is a mechanics solver despite its misleading name), contact is enforced weakly without adding multipliers. The variational contact contribution $R_{\text{contact}}$ incorporates a boundary traction term, a symmetric test traction term, and a penalty term:
$$R_{\text{contact}} = -\int_{\Gamma_C} (\boldsymbol{\sigma}(\mathbf{u})\mathbf{n} \cdot \mathbf{n})(\mathbf{v} \cdot \mathbf{n}) \, ds - \int_{\Gamma_C} (\boldsymbol{\sigma}(\mathbf{v})\mathbf{n} \cdot \mathbf{n})(\mathbf{g} \cdot \mathbf{n}) \, ds + \int_{\Gamma_C} \frac{\gamma}{h} (\mathbf{g} \cdot \mathbf{n})(\mathbf{v} \cdot \mathbf{n}) \, ds$$
```python
R_contact = (
    - ufl.inner(σ(u) * n, v_n * n) * ds_c
    - ufl.inner(σ(v) * n, gap_neg * n) * ds_c
    + gamma / h * gap_neg * v_n * ds_c
)
```

### 4.3 Technical Failure Modes (Why It Was Not Integrated)
As standalone scripts, the mechanics models do not achieve stable band-wrapping and collapse into numerical divergence partway through loading. Code analysis reveals the following critical technical flaws:

1.  **Singular Dirichlet Pulling Boundary Conditions:** 
    The left and right ends of the flat band are pulled directly to the exact same spatial coordinates at the top of the circular arm:
    ```python
    u_left = np.array([L0 / 2, 2 * r])
    u_right = np.array([-L0 / 2, 2 * r])
    ```
    This forces a topological collision of the left and right mesh boundaries at `(0, r)`. This singular boundary configuration causes massive mesh shear, leading to elements folding over themselves and producing a non-physical negative Jacobian determinant ($J \le 0$), causing the Newton solver to fail instantly.
2.  **Severe Material Stiffness Mismatches:**
    The Young's modulus of the soft Ecoflex substrate is $E_{\text{eco}} = 6.89 \times 10^4$ Pa, while the CNT conductor pads have $E_{\text{cnt}} = 4.7 \times 10^9$ Pa (a $68,000$-fold discrepancy). This massive stiffness gradient creates severe numerical ill-conditioning and stress singularities at the interfaces during bending, disrupting solver convergence.
3.  **Radial-Only Local Contact Gap Limitation:**
    The contact gap is evaluated only in the radial coordinate direction relative to the center of the arm:
    ```python
    gap = R_skin - dist
    ```
    As the band wraps around the limb, its elements rotate. Radial distance comparison fails to prevent self-penetration and lateral boundaries from clipping through the skin.
4.  **ALM Penalty Sensitivity & conditioning:**
    The forearm tissues are set to an artificial Young's modulus of $E_{\text{rigid}} = 1.1 \times 10^{20}$ Pa. Enforcing contact against an infinitely rigid boundary using high penalty parameters ($\gamma = 10 \times E_{\text{cnt}}$) severely ill-conditions the tangential Jacobian matrix, causing the PETSc SNES solver to stall and diverge.

---

## 5. File Directory & Descriptions

```
.
├── 2d_elastic_deformation.py     # Linear elastic contact solver using Augmented Lagrangian Method (ALM).
├── 2d_elect_depth.py             # Poisson solver for the muscle source depth parametric sweep.
├── 2d_elect_ed.py                # Poisson solver for the electrode diameter parametric sweep.
├── 2d_elect_fat.py               # Poisson solver for the fat layer thickness parametric sweep.
├── 2d_elect_ied.py               # Poisson solver for the inter-electrode distance parametric sweep.
├── 2d_elect_size.py              # Poisson solver for the forearm size parametric sweep.
├── 2d_elect_spread.py            # Poisson solver for the current source spatial spread parametric sweep.
├── 2d_electric_potential.py      # Standalone prototype of Mooney-Rivlin mechanics with penalty contact.
├── 2d_electric_potential1.py     # Mechanics prototype exploring linear elastic contact boundaries.
├── 2d_electric_potential2.py     # Experimental mechanics script combining ALM contact loops on a circular arm.
├── 2d_electric_potential3.py     # Standalone contact mechanics testing ground using linear elasticity and ALM.
├── 2d_electric_potential4.py     # Standalone contact mechanics script implementing Nitsche's weak contact formulation.
├── 2d_electrode.py               # Flat electrode band geometry mesh generator script in GMSH.
├── 2d_electropotential.py        # Reference Poisson bioelectric solver script with visual plotting in PyVista.
├── 2d_forearm.py                 # Multi-layered concentric forearm geometry mesh generator in GMSH.
├── 2d_forearm_electrode.msh      # Pre-generated flat forearm-electrode assembly mesh file.
├── 2d_merge.py                   # Combines flat electrode and forearm geometry in GMSH without fragmenting.
├── 2d_mr_deformation.py          # Nonlinear Mooney-Rivlin hyperelastic solver with ALM contact.
├── 2d_parview.h5                 # HDF5 grid file containing results of reference Poisson solves.
├── 2d_parview.xdmf               # Paraview XML wrapper file pointing to HDF5 volume conduction data.
├── 2d_wrap_ed.py                 # GMSH generator wrapping an electrode band around a forearm (varying Ed).
├── 2d_wrap_fat.py                # GMSH generator wrapping an electrode band around a forearm (varying fat fraction).
├── 2d_wrap_fragment.py           # GMSH meshing script utilizing OCC fragment for conformal interface tagging.
├── 2d_wrap_fragment2.py          # Test version of conforming GMSH wrapped mesh script.
├── 2d_wrap_fragment3.py          # Alternative test variant of conforming GMSH wrapped mesh script.
├── 2d_wrap_ied.py                # GMSH generator wrapping an electrode band around a forearm (varying band radius).
├── 2d_wrap_merge.py              # Analytical wrapped mesh builder in GMSH using polar coordinates.
├── 2d_wrap_merge2.py             # Polar-wrapped forearm and electrode mesh script variant.
├── 2d_wrap_merge3.py             # Polar-wrapped forearm and electrode mesh script variant 3.
├── 2d_wrap_size.py               # GMSH generator wrapping an electrode band around a forearm (varying forearm size).
├── 2d_wrapped.msh                # Pre-generated conforming analytical polar-wrapped mesh file.
├── README.md                     # This documentation file.
├── demo_elastic.py               # Reference linear elasticity demo script from official FEniCSx tutorial.
├── demo_poisson.py               # Reference Poisson equation solver demo script from official FEniCSx tutorial.
├── depth.py                      # Orchestrator script running the muscle source depth sweep sequentially.
├── depth.txt                     # Voltage output logs for the depth sweep.
├── depth1.txt                    # Backup/comparison log file for the depth sweep.
├── ed.py                         # Orchestrator script running the electrode diameter sweep sequentially.
├── ed.txt                        # Voltage output logs for the electrode diameter sweep.
├── electrode_band.msh            # Pre-generated isolated electrode band mesh file.
├── fat.py                        # Orchestrator script running the fat layer thickness sweep sequentially.
├── fat.txt                       # Voltage output logs for the fat layer thickness sweep (contains data integrity bugs).
├── fat1.txt                      # Backup/comparison log file for the fat layer thickness sweep.
├── forearm.msh                   # Pre-generated concentric forearm tissue mesh file.
├── ied.py                        # Orchestrator script running the band radius (derived IED) sweep sequentially.
├── ied.txt                       # Voltage output logs for the IED sweep.
├── ied1.txt                      # Backup/comparison log file for the IED sweep.
├── size.py                       # Orchestrator script running the forearm circumference size sweep sequentially.
├── size.txt                      # Voltage output logs for the forearm size sweep.
├── size1.txt                     # Backup/comparison log file for the forearm size sweep.
├── spread.py                     # Orchestrator script running the source spatial spread sweep sequentially.
├── spread.txt                    # Voltage output logs for the spatial spread sweep.
└── spread1.txt                   # Backup/comparison log file for the spatial spread sweep.
```

---

## 6. Dependencies

To execute the simulation pipeline, your system must have the following software installed:

*   **FEniCSx (DOLFINx v0.7+)** (Finite Element computational library)
*   **GMSH** (with OpenCASCADE geometry support and Python API)
*   **PETSc / petsc4py** (Linear algebra solver backend, compiled with MUMPS)
*   **MPI / mpi4py** (Parallel message passing interface)
*   **PyVista** (VTK-based 3D mesh and scalar plotting)
*   **numpy** (Numerical operations and matrix manipulation)
*   **R Environment** (with the following packages for statistical plotting: `ggplot2`, `tidyr`, `readr`)

---

## 7. Known Limitations & Numerical Issues

1.  **Fully Decoupled Multiphysics:** 
    The parametric sweeps are executed on an analytical concentric polar mesh (`2d_wrapped.msh`) where all interfaces are perfectly circular. It does not account for tissue deformation, compression, or thickness changes caused by mechanical tightening of the band.
2.  **Monopole Bioelectric Source:**
    While biological action potentials act as electric dipoles, the current sweeps are executed with a **spatial Gaussian monopole**. In 2D, monopole current sources decay much more slowly ($\sim 1/r$) than dipoles ($\sim 1/r^2$), which overestimates recorded amplitudes on the skin and distorts the spatial resolution profiles.
3.  **Data Invalidation in `fat.txt`:**
    For small fat layer values, GMSH's OCC kernel fails to resolve conforming concentric boundaries, causing GMSH to crash silently without writing a new mesh. The subsequent Poisson solver runs on the pre-existing `2d_wrapped.msh` file from the previous step. This creates large blocks of **completely identical rows** in the output data file, which must be filtered or corrected before publication.
4.  **Source Position and Spread Inconsistencies:**
    The sweeps use inconsistent spatial source parameters. The spatial width is set to $\sigma_s = 0.4$ in some prototypes, $\sigma_s = 0.05$ in the `ed` sweep, and is swept dynamically in the `spread` sweep. Additionally, the spread sweep places the source on the horizontal x-axis (`x_plus`), whereas all other sweeps locate it on the vertical y-axis (`x1_plus`). This makes direct comparison across different sweeps non-trivial.
5.  **Isotropic Tissue assumption:**
    All tissue conductivities are treated as isotropic scalars. Real muscle tissue is highly anisotropic, with conductivity parallel to muscle fibers being up to 10 times higher than in the transverse direction.
6.  **2D Plane-Strain Simplification:**
    The 2D plane-strain mechanics model ignores longitudinal band shear, and the 2D Poisson solver ignores axial current spread along the length of muscle fibers, which limits physiological fidelity.
