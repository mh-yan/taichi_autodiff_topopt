# MPM-TopOpt: MPM-Based Topology Optimization Framework (2D/3D)

A topology optimization framework based on the **Material Point Method (MPM)** concept, supporting **2D and 3D** problems with **irregular point-cloud domains**. Uses Taichi for GPU-accelerated matrix-free PCG solving and automatic differentiation.

## Key Features

- **Arbitrary-shape design domains** via point cloud input (.npy / .csv / .stl)
- **2D (Q4) and 3D (H8)** unified codebase with a single `--dim` flag
- **Taichi AD** for filter/projection gradients + **adjoint method** for FEA sensitivity
- **Matrix-free PCG** solver (CPU / CUDA / Vulkan backends)
- Heaviside projection with beta/penal continuation
- P-norm **stress constraint** with augmented Lagrangian
- Built-in **reference SIMP** (scipy direct solver) for validation
- Interactive 3D visualization (Plotly HTML) with boundary condition annotations
- **Additive manufacturing pipeline**: STL → point cloud → optimize → export STL

## Quick Start

```bash
pip install numpy scipy taichi matplotlib plotly

# 2D cantilever beam (matches classic SIMP exactly)
python main.py --dim 2 --nelx 60 --nely 20 --volfrac 0.5 --iters 200

# 3D cantilever beam
python main.py --dim 3 --nelx 30 --nely 10 --nelz 6 --volfrac 0.3 --iters 200

# MBB beam / L-bracket / 3D bridge
python main.py --dim 2 --problem mbb --iters 200
python main.py --dim 2 --problem lbracket --nelx 40 --nely 40 --volfrac 0.4 --iters 200
python main.py --dim 3 --problem bridge --nelx 30 --nely 10 --nelz 6 --iters 200

# Irregular point cloud
python main.py --dim 3 --point-cloud my_cloud.npy --nelx 30 --nely 10 --nelz 6 --iters 200

# Additive manufacturing demo (generates sample geometry if no STL provided)
python am_demo.py --generate-sample --dim 3 --nelx 24 --nely 8 --nelz 6 --iters 200

# Run all benchmarks with comparison tables
python benchmark.py
```

## Benchmark Results (200 iterations)

All regular-grid benchmarks match classical SIMP to machine precision:

| Problem | Grid | V_f | C_mpm | C_ref | Correlation | Gray Mnd |
|---------|------|-----|-------|-------|-------------|----------|
| Cantilever 2D | 60x20 | 0.50 | 175.39 | 175.39 | 1.000000 | 0.021 |
| MBB 2D | 60x20 | 0.50 | 195.21 | 195.21 | 1.000000 | 0.035 |
| L-bracket 2D | 40x40 | 0.40 | 90.55 | 90.55 | 1.000000 | 0.028 |
| Cantilever 3D | 30x10x6 | 0.30 | 46.05 | 46.05 | 1.000000 | 0.055 |
| Bridge 3D | 30x10x6 | 0.30 | 6.21 | 6.21 | 1.000000 | 0.044 |

## Pre-computed Examples

The `examples/` directory contains results from all benchmark problems, ready for inspection:

```
examples/
├── cantilever_2d/         2D cantilever beam (classic benchmark)
├── mbb_2d/                2D MBB beam (half-symmetry)
├── lbracket_2d/           2D L-bracket
├── cantilever_3d/         3D cantilever beam
├── bridge_3d/             3D bridge (two supports)
├── irregular_2d/          2D with irregular point cloud domain
├── irregular_3d/          3D with irregular point cloud domain
├── am_demo_2d/            Additive manufacturing demo (2D, with STL export)
├── am_demo_3d/            Additive manufacturing demo (3D, with STL export)
└── benchmark_comparison/  Convergence curves + comparison tables for all problems
```

Each example directory contains:
- `final_mpm.png` / `final_ref.png` — final topology (MPM vs reference SIMP)
- `comparison.png` — side-by-side comparison with difference map (2D)
- `history.png` — convergence curves (compliance, volume, tip displacement, PCG residual)
- `interactive_mpm.html` — interactive 3D visualization (open in browser)
- `summary.txt` — numerical metrics (compliance, volume, correlation, IoU, Dice)
- CSV history files for post-processing

## Project Structure

```
mpm_to/
├── main.py            Entry point: optimization loop + SIMP comparison + visualization
├── config.py          Configuration dataclass + CLI argument parsing
├── engine.py          Taichi AD engine: PCG solver + auto-diff sensitivities
├── mesh.py            Background grid, DOF connectivity, boundary conditions
│                      (cantilever / MBB / L-bracket / bridge)
├── particles.py       Point cloud generation/loading, rasterization, P2G/G2P
├── stiffness.py       2D Q4 / 3D H8 element stiffness matrices
├── filter_utils.py    Density filter, Heaviside projection, OC update
├── solver.py          Reference SIMP solver (scipy spsolve)
├── stress.py          Von Mises stress, p-norm constraint, augmented Lagrangian
├── viz.py             Matplotlib static plots + Plotly interactive HTML
├── benchmark.py       Automated benchmark suite with comparison tables
├── am_demo.py         Additive manufacturing demo: STL → optimize → STL
├── my_cloud.npy       Sample 2D irregular point cloud
├── my_cloud_3d.npy    Sample 3D irregular point cloud
├── requirements.txt   Python dependencies
└── examples/          Pre-computed results for all benchmark problems
```

## Why MPM for Topology Optimization?

| Feature | Classic SIMP (top88/top3d) | MPM-TopOpt |
|---------|---------------------------|------------|
| Design domain | Rectangular only | **Any shape** (point cloud) |
| Mesh generation | Must match domain | **Mesh-free**: auto background grid |
| Resolution | Globally uniform | Locally adaptive point density |
| Large deformation | Requires remeshing | Natural extension (particle tracking) |
| Input format | Structured grid | **Point cloud** (.npy/.csv/.stl) |
| Gradient computation | Manual chain rule | **Taichi AD** + adjoint method |

**Mathematical equivalence**: When one particle per element, MPM-TopOpt degenerates to classic SIMP exactly (correlation > 0.999999 on all benchmarks).

## Automatic Differentiation

The filter and Heaviside projection gradients are computed by **Taichi's reverse-mode AD** (`kernel.grad()`), while the FEA sensitivity uses the **adjoint method** (for compliance: adjoint = displacement, so `dC/dx_phys = -p * x^(p-1) * (E0-Emin) * u^T K_e u`). This hybrid approach means:

- Changing the filter or projection formula automatically updates gradients
- No manual gradient derivation needed for the density processing chain
- The PCG solver is not differentiated (adjoint handles it analytically)

## Command-Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dim` | 2 | Dimension (2 or 3) |
| `--problem` | cantilever | Problem type: cantilever / mbb / lbracket / bridge |
| `--nelx/nely/nelz` | 60/20/4 | Background grid resolution |
| `--volfrac` | 0.5 | Volume fraction constraint |
| `--penal-start/max/every` | 3.0/4.0/25 | SIMP penalization continuation |
| `--beta-start/max/every` | 1.0/8.0/20 | Heaviside projection continuation |
| `--rmin` | 1.5 | Filter radius (element widths) |
| `--iters` | 200 | Optimization iterations |
| `--point-cloud` | (none) | Path to external point cloud (.npy/.csv) |
| `--arch` | cpu | Taichi backend: cpu / cuda / vulkan |
| `--stress-constraint` | off | Enable p-norm stress constraint |
| `--sigma-allow` | 1.0 | Allowable stress for constraint |
| `--no-show-bc` | off | Hide boundary condition markers |

---

## Technical Documentation

For detailed algorithmic descriptions, mathematical formulations, and implementation notes, see the sections below.

### Algorithm Flow

```
Initialization:
  Load/generate point cloud → Rasterize to active elements → Build background grid → Build filter

Optimization loop (per iteration):
  P2G: particle density → element density
  Forward (Taichi AD tracked):
    Density filter: x → x_tilde = H·x / Hs
    Heaviside projection: x_tilde → x_phys
  SIMP interpolation: E_e = Emin + x_phys^p · (E0 - Emin)
  Matrix-free PCG: K(x_phys) · u = F
  Adjoint sensitivity: dC/dx_phys = -p · x^(p-1) · (E0-Emin) · u_e^T K_e u_e
  Taichi AD backward: x_phys.grad → x_tilde.grad → x.grad
  G2P: element sensitivity → particle sensitivity
  OC update: bisection on Lagrange multiplier

Post-processing:
  Reference SIMP comparison → similarity metrics → visualization
```

### Finite Element Details

**2D Q4 (plane stress):** 4-node quadrilateral, 8 DOF/element. Analytical stiffness matrix from top88.

**3D H8 (hexahedral):** 8-node brick, 24 DOF/element. Numerical integration via 2x2x2 Gauss quadrature. Verified: symmetric, 6 zero eigenvalues (rigid body modes), 18 positive eigenvalues.

### Irregular Point Cloud Processing

1. KD-Tree distance query from element centers to nearest particle
2. Threshold activation (distance <= 1.8 element widths)
3. Morphological closing with edge-padded boundary protection
4. Largest connected component extraction
5. Adaptive boundary conditions (leftmost active nodes = fixed, rightmost = load)

### Stress Constraint

Von Mises stress with qp-relaxation (`sigma_tilde = x^(p-1) * sigma_vm`) to avoid stress singularity in void regions. P-norm aggregation (`sigma_pn = (sum (sigma/sigma_allow)^p)^(1/p)`) as smooth max. Augmented Lagrangian method for constraint handling with adaptive penalty parameter.
