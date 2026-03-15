#!/usr/bin/env python3
"""Additive manufacturing demo: STL → point cloud → topology optimization → STL.

Usage:
    python am_demo.py input.stl --dim 3 --nelx 30 --nely 10 --nelz 6 --iters 100
    python am_demo.py --generate-sample --dim 2 --nelx 60 --nely 20 --iters 100
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

from config import TopOptConfig
from stiffness import get_ke
from mesh import build_mesh
from particles import assign_particles, rasterize_active, p2g, g2p
from filter_utils import attach_filter, schedule, oc_update


# ---------------------------------------------------------------------------
# STL I/O
# ---------------------------------------------------------------------------

def stl_to_points(stl_path: str, n_samples: int = 10000) -> np.ndarray:
    """Load an STL file and return interior + surface points.

    Tries trimesh first; falls back to a minimal binary STL parser with
    rejection sampling for interior fill.
    """
    path = Path(stl_path)
    if not path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    try:
        import trimesh
        mesh = trimesh.load(str(path), force="mesh")
        surface_pts, _ = trimesh.sample.sample_surface(mesh, n_samples)

        try:
            vox = mesh.voxelized(pitch=mesh.extents.max() / 50)
            interior_pts = vox.points
        except Exception:
            interior_pts = _rejection_sample(mesh.bounds[0], mesh.bounds[1],
                                             n_samples, mesh.contains)

        pts = np.vstack([surface_pts, interior_pts])
        return np.unique(pts.round(decimals=6), axis=0)

    except ImportError:
        pass

    vertices, faces = _parse_stl_binary(path)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    pts = _surface_sample_from_faces(vertices, faces, n_samples)

    rng = np.random.default_rng(42)
    grid_pts = []
    span = hi - lo
    pitch = span.max() / 40
    for _ in range(3):
        candidates = lo + rng.random((n_samples * 2, 3)) * span
        grid_pts.append(candidates)
    grid_pts = np.vstack([pts] + grid_pts)
    return np.unique(grid_pts.round(decimals=6), axis=0)


def _rejection_sample(lo: np.ndarray, hi: np.ndarray, n: int,
                      contains_fn) -> np.ndarray:
    """Fill interior of a watertight mesh via rejection sampling."""
    rng = np.random.default_rng(42)
    collected = []
    span = hi - lo
    remaining = n
    for _ in range(20):
        candidates = lo + rng.random((remaining * 3, 3)) * span
        inside = contains_fn(candidates)
        collected.append(candidates[inside])
        remaining = n - sum(c.shape[0] for c in collected)
        if remaining <= 0:
            break
    return np.vstack(collected)[:n] if collected else np.empty((0, 3))


def _parse_stl_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal binary STL parser.  Returns (vertices, faces)."""
    data = path.read_bytes()
    if data[:5] == b"solid" and b"\n" in data[:80]:
        raise ValueError("ASCII STL not supported by minimal parser; install trimesh")
    n_tri = struct.unpack_from("<I", data, 80)[0]
    vertices = []
    faces = []
    offset = 84
    for i in range(n_tri):
        _nx, _ny, _nz = struct.unpack_from("<3f", data, offset)
        offset += 12
        tri_verts = []
        for _ in range(3):
            vx, vy, vz = struct.unpack_from("<3f", data, offset)
            offset += 12
            tri_verts.append([vx, vy, vz])
        offset += 2  # attribute byte count
        base = len(vertices)
        vertices.extend(tri_verts)
        faces.append([base, base + 1, base + 2])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def _surface_sample_from_faces(vertices: np.ndarray, faces: np.ndarray,
                               n: int) -> np.ndarray:
    """Uniform random sampling on triangle surfaces."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5
    probs = areas / areas.sum()

    rng = np.random.default_rng(42)
    chosen = rng.choice(len(faces), size=n, p=probs)
    r1, r2 = rng.random(n), rng.random(n)
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2
    pts = (u[:, None] * v0[chosen]
           + v[:, None] * v1[chosen]
           + w[:, None] * v2[chosen])
    return pts


def write_stl_binary(path: str, vertices: np.ndarray, faces: np.ndarray,
                     normals: np.ndarray | None = None) -> None:
    """Write a binary STL file.

    Parameters
    ----------
    path : output file path
    vertices : (N, 3) float array
    faces : (M, 3) int array – indices into vertices
    normals : optional (M, 3) face normals; computed if None
    """
    if normals is None:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normals /= norms

    n_tri = faces.shape[0]
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n_tri))
        for i in range(n_tri):
            f.write(struct.pack("<3f", *normals[i].astype(np.float32)))
            for vi in faces[i]:
                f.write(struct.pack("<3f", *vertices[vi].astype(np.float32)))
            f.write(struct.pack("<H", 0))


# ---------------------------------------------------------------------------
# Sample geometry generators
# ---------------------------------------------------------------------------

def generate_sample_2d(nelx: int, nely: int) -> np.ndarray:
    """L-bracket point cloud for 2D demo (no STL needed)."""
    rng = np.random.default_rng(0)
    pts = []
    nx, ny = nelx * 4, nely * 4
    for ix in range(nx):
        for iy in range(ny):
            x = (ix + 0.5) / nx * nelx
            y = (iy + 0.5) / ny * nely
            in_bottom = y < nely * 0.5
            in_left = x < nelx * 0.4
            if in_bottom or in_left:
                pts.append([x + rng.uniform(-0.05, 0.05),
                            y + rng.uniform(-0.05, 0.05)])
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] = np.clip(pts[:, 0], 0.1, nelx - 0.1)
    pts[:, 1] = np.clip(pts[:, 1], 0.1, nely - 0.1)
    return pts


def generate_sample_3d(nelx: int, nely: int, nelz: int) -> np.ndarray:
    """Rectangular bracket with a cylindrical through-hole for 3D demo."""
    rng = np.random.default_rng(0)
    pts = []
    cx, cy = nelx * 0.65, nely * 0.5
    radius = min(nely, nelz) * 0.25

    nx, ny, nz = max(nelx * 2, 20), max(nely * 2, 10), max(nelz * 2, 6)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = (ix + 0.5) / nx * nelx
                y = (iy + 0.5) / ny * nely
                z = (iz + 0.5) / nz * nelz
                dist_xy = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist_xy > radius:
                    pts.append([x + rng.uniform(-0.05, 0.05),
                                y + rng.uniform(-0.05, 0.05),
                                z + rng.uniform(-0.05, 0.05)])
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] = np.clip(pts[:, 0], 0.1, nelx - 0.1)
    pts[:, 1] = np.clip(pts[:, 1], 0.1, nely - 0.1)
    pts[:, 2] = np.clip(pts[:, 2], 0.1, nelz - 0.1)
    return pts


# ---------------------------------------------------------------------------
# Density field → STL export
# ---------------------------------------------------------------------------

def _voxel_mesh(field_3d: np.ndarray, threshold: float):
    """Convert a 3D boolean/density field into a triangle mesh (voxel cubes).

    Each solid voxel becomes a set of exposed faces (no internal faces).
    """
    nz, nx, ny = field_3d.shape
    solid = field_3d > threshold
    vertices = []
    faces = []
    vert_map = {}

    def _get_vert(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vert_map:
            vert_map[key] = len(vertices)
            vertices.append([x, y, z])
        return vert_map[key]

    cube_faces_offsets = [
        ([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], (0, 0, -1)),  # -z
        ([0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], (0, 0, 1)),   # +z
        ([0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], (-1, 0, 0)),  # -x
        ([1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], (1, 0, 0)),   # +x
        ([0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], (0, -1, 0)),  # -y
        ([0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], (0, 1, 0)),   # +y
    ]
    neighbor_dirs = [(0, 0, -1), (0, 0, 1), (-1, 0, 0),
                     (1, 0, 0), (0, -1, 0), (0, 1, 0)]

    for iz in range(nz):
        for ix in range(nx):
            for iy in range(ny):
                if not solid[iz, ix, iy]:
                    continue
                for fi, (offsets, _norm) in enumerate(cube_faces_offsets):
                    dz, dx, dy = neighbor_dirs[fi]
                    niz, nix, niy = iz + dz, ix + dx, iy + dy
                    if (0 <= niz < nz and 0 <= nix < nx
                            and 0 <= niy < ny and solid[niz, nix, niy]):
                        continue
                    v = [_get_vert(ix + o[0], iy + o[1], iz + o[2])
                         for o in offsets]
                    faces.append([v[0], v[1], v[2]])
                    faces.append([v[0], v[2], v[3]])

    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def _marching_cubes(field_3d: np.ndarray, threshold: float):
    """Extract isosurface via marching cubes (scikit-image) or voxel fallback."""
    try:
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(field_3d, level=threshold)
        return verts, faces.astype(np.int32)
    except ImportError:
        pass
    return _voxel_mesh(field_3d, threshold)


def _extrude_2d(field_2d: np.ndarray, threshold: float,
                depth: float = 1.0):
    """Create extruded mesh from 2D density field (nelx × nely → 3D slab)."""
    nx, ny = field_2d.shape
    solid = field_2d > threshold
    vertices = []
    faces = []
    vert_map = {}

    def _vert(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vert_map:
            vert_map[key] = len(vertices)
            vertices.append([x, y, z])
        return vert_map[key]

    for ix in range(nx):
        for iy in range(ny):
            if not solid[ix, iy]:
                continue
            x0, y0 = float(ix), float(iy)
            x1, y1 = x0 + 1, y0 + 1
            z0, z1 = 0.0, depth

            corners = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
            ]
            vi = [_vert(*c) for c in corners]

            # bottom (-z)
            faces.append([vi[0], vi[2], vi[1]])
            faces.append([vi[0], vi[3], vi[2]])
            # top (+z)
            faces.append([vi[4], vi[5], vi[6]])
            faces.append([vi[4], vi[6], vi[7]])

            if ix == 0 or not solid[ix - 1, iy]:
                faces.append([vi[0], vi[4], vi[7]])
                faces.append([vi[0], vi[7], vi[3]])
            if ix == nx - 1 or not solid[ix + 1, iy]:
                faces.append([vi[1], vi[2], vi[6]])
                faces.append([vi[1], vi[6], vi[5]])
            if iy == 0 or not solid[ix, iy - 1]:
                faces.append([vi[0], vi[1], vi[5]])
                faces.append([vi[0], vi[5], vi[4]])
            if iy == ny - 1 or not solid[ix, iy + 1]:
                faces.append([vi[3], vi[7], vi[6]])
                faces.append([vi[3], vi[6], vi[2]])

    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def points_to_stl(field: np.ndarray, nelx: int, nely: int, nelz: int,
                  threshold: float, out_path: str, dim: int) -> str:
    """Export optimized density field as STL.

    Returns the path written.
    """
    try:
        import trimesh as _tm
        has_trimesh = True
    except ImportError:
        has_trimesh = False

    if dim == 2:
        field_2d = field.reshape(nelx, nely)
        verts, faces = _extrude_2d(field_2d, threshold)
    else:
        field_3d = field.reshape(nelz, nelx, nely)
        verts, faces = _marching_cubes(field_3d, threshold)

    if faces.shape[0] == 0:
        print("WARNING: no solid voxels above threshold; STL is empty")
        write_stl_binary(out_path, np.zeros((1, 3)), np.zeros((1, 3), dtype=np.int32))
        return out_path

    if has_trimesh:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()
        mesh.export(out_path)
    else:
        write_stl_binary(out_path, verts, faces)

    return out_path


# ---------------------------------------------------------------------------
# Optimization loop (adapted from main.run_mpm_topopt)
# ---------------------------------------------------------------------------

def run_am_topopt(cfg: TopOptConfig, points: np.ndarray, out_dir: Path):
    """Run topology optimization on the given point cloud and export results."""
    import taichi as ti
    ti.init(arch=getattr(ti, cfg.arch), default_fp=ti.f64, unrolling_limit=0)
    from engine import make_engine_class
    TopOptEngine = make_engine_class()

    p2e, counts = assign_particles(points, cfg.nelx, cfg.nely, cfg.nelz, cfg.dim)
    active_mask = rasterize_active(points, cfg.nelx, cfg.nely, cfg.nelz,
                                   cfg.dim, close_iters=2)

    mesh = build_mesh(cfg, active_mask)
    mesh = attach_filter(mesh, cfg.rmin)

    KE = get_ke(cfg.dim, cfg.E0, cfg.nu)
    engine = TopOptEngine(mesh, KE, cfg.E0, cfg.Emin,
                          cfg.penal_start, cfg.eta, cfg.beta_start)

    x_p = np.full(points.shape[0], cfg.volfrac)
    x = p2g(x_p, p2e, counts, mesh.n_elem)
    x[~mesh.active_mask] = 0.0

    t0 = time.perf_counter()
    for it in range(1, cfg.n_iter + 1):
        beta = schedule(it, cfg.beta_start, cfg.beta_max, cfg.beta_every)
        penal = schedule(it, cfg.penal_start, cfg.penal_max, cfg.penal_every)
        engine.set_params(penal, beta)
        engine.set_x(x)
        engine.forward()

        U, pcg_iters, pcg_res = engine.solve_pcg(cfg.cg_maxiter, cfg.cg_tol)
        dc, compliance = engine.compute_compliance_grad()
        dv, volume = engine.compute_volume_grad()

        x = oc_update(x, dc, dv, cfg.volfrac, mesh,
                      cfg.move, cfg.xmin, beta, cfg.eta)
        x_p = x[p2e]

        if it == 1 or it % cfg.save_every == 0 or it == cfg.n_iter:
            print(f"  it={it:3d}  C={compliance:11.5e}  V={volume:.4f}  "
                  f"pcg={pcg_iters:4d}  res={pcg_res:9.2e}  "
                  f"beta={beta:.1f}  penal={penal:.1f}")

    elapsed = time.perf_counter() - t0

    engine.set_params(
        schedule(cfg.n_iter, cfg.penal_start, cfg.penal_max, cfg.penal_every),
        schedule(cfg.n_iter, cfg.beta_start, cfg.beta_max, cfg.beta_every),
    )
    engine.set_x(x)
    engine.forward()
    U_final, _, _ = engine.solve_pcg(cfg.cg_maxiter, cfg.cg_tol)
    _, comp_final = engine.compute_compliance_grad()
    _, vol_final = engine.compute_volume_grad()
    x_phys = engine.get_x_phys()

    return x_phys, comp_final, vol_final, elapsed, mesh


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_am_args():
    p = argparse.ArgumentParser(
        description="Additive manufacturing demo: STL → topology optimization → STL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("stl_input", nargs="?", default=None,
                   help="Input STL file (omit if using --generate-sample)")
    p.add_argument("--generate-sample", action="store_true",
                   help="Use a built-in sample geometry instead of an STL file")
    p.add_argument("--dim", type=int, default=3, choices=[2, 3])
    p.add_argument("--nelx", type=int, default=30)
    p.add_argument("--nely", type=int, default=10)
    p.add_argument("--nelz", type=int, default=6)
    p.add_argument("--volfrac", type=float, default=0.3)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rmin", type=float, default=1.5)
    p.add_argument("--penal-start", type=float, default=3.0)
    p.add_argument("--penal-max", type=float, default=4.0)
    p.add_argument("--penal-every", type=int, default=25)
    p.add_argument("--beta-start", type=float, default=1.0)
    p.add_argument("--beta-max", type=float, default=8.0)
    p.add_argument("--beta-every", type=int, default=20)
    p.add_argument("--arch", type=str, default="cpu",
                   choices=["cpu", "cuda", "vulkan"])
    p.add_argument("--out-dir", type=str, default="output_am")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Density threshold for STL extraction")
    p.add_argument("--n-samples", type=int, default=10000,
                   help="Number of surface samples for STL → point cloud")
    return p.parse_args()


def main():
    args = parse_am_args()

    if args.stl_input is None and not args.generate_sample:
        print("ERROR: provide an STL file or use --generate-sample")
        sys.exit(1)

    print("=" * 60)
    print("  Additive Manufacturing Topology Optimization Demo")
    print("=" * 60)

    # --- Step 1: obtain point cloud ---
    if args.generate_sample:
        print(f"\n[1/4] Generating sample {args.dim}D geometry …")
        if args.dim == 2:
            points = generate_sample_2d(args.nelx, args.nely)
        else:
            points = generate_sample_3d(args.nelx, args.nely, args.nelz)
        print(f"  Generated {points.shape[0]} points")
    else:
        print(f"\n[1/4] Loading STL: {args.stl_input}")
        raw_pts = stl_to_points(args.stl_input, n_samples=args.n_samples)
        print(f"  Sampled {raw_pts.shape[0]} raw points")

        lo, hi = raw_pts.min(axis=0), raw_pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-12)
        points = (raw_pts - lo) / span
        if args.dim == 2:
            points = points[:, :2]
            points[:, 0] *= args.nelx
            points[:, 1] *= args.nely
        else:
            points[:, 0] *= args.nelx
            points[:, 1] *= args.nely
            points[:, 2] *= args.nelz
        print(f"  Rescaled to grid [{args.nelx} x {args.nely}"
              + (f" x {args.nelz}]" if args.dim == 3 else "]"))

    # --- Step 2: configure and run optimization ---
    print(f"\n[2/4] Running topology optimization ({args.iters} iterations) …")
    cfg = TopOptConfig(
        dim=args.dim, nelx=args.nelx, nely=args.nely, nelz=args.nelz,
        volfrac=args.volfrac, n_iter=args.iters, rmin=args.rmin,
        penal_start=args.penal_start, penal_max=args.penal_max,
        penal_every=args.penal_every, beta_start=args.beta_start,
        beta_max=args.beta_max, beta_every=args.beta_every,
        arch=args.arch, out_dir=args.out_dir,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "input_points.npy", points)

    x_phys, comp, vol, elapsed, mesh = run_am_topopt(cfg, points, out)

    # --- Step 3: export STL ---
    stl_out = str(out / "optimized.stl")
    print(f"\n[3/4] Exporting STL (threshold={args.threshold}) …")
    points_to_stl(x_phys, args.nelx, args.nely, args.nelz,
                  args.threshold, stl_out, args.dim)

    n_solid = int((x_phys[mesh.active_ids] > args.threshold).sum())
    print(f"  Wrote {stl_out}")
    print(f"  Solid elements: {n_solid} / {mesh.n_active}")

    # --- Step 4: summary ---
    print(f"\n[4/4] Summary")
    print("-" * 40)
    print(f"  Dimension:         {args.dim}D")
    print(f"  Grid:              {args.nelx} x {args.nely}"
          + (f" x {args.nelz}" if args.dim == 3 else ""))
    print(f"  Input points:      {points.shape[0]}")
    print(f"  Active elements:   {mesh.n_active}")
    print(f"  Final compliance:  {comp:.6e}")
    print(f"  Final volume frac: {vol:.4f}")
    print(f"  Wall time:         {elapsed:.1f} s")
    print(f"  Output directory:  {out}")
    print(f"  STL file:          {stl_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
