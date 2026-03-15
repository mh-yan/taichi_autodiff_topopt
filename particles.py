"""Material-point (particle) management: generation, loading, P2G / G2P."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Point cloud generation
# ---------------------------------------------------------------------------

def generate_particles_2d(nelx: int, nely: int, spacing: float = 0.0) -> np.ndarray:
    """Regular grid of particles for 2D cantilever benchmark.

    If *spacing* <= 0, places exactly one particle at each element center.
    Otherwise fills the domain [0, nelx] x [0, nely] with given spacing.
    """
    if spacing <= 0:
        ex = np.arange(nelx)
        ey = np.arange(nely)
        gx, gy = np.meshgrid(ex + 0.5, ey + 0.5, indexing="xy")
        return np.column_stack([gx.ravel(), gy.ravel()])
    xs = np.arange(0.5 * spacing, nelx, spacing)
    ys = np.arange(0.5 * spacing, nely, spacing)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel()])


def generate_particles_3d(nelx: int, nely: int, nelz: int,
                          spacing: float = 0.0) -> np.ndarray:
    """Regular grid of particles for 3D cantilever benchmark."""
    if spacing <= 0:
        pts = []
        for ez in range(nelz):
            for ex in range(nelx):
                for ey in range(nely):
                    pts.append([ex + 0.5, ey + 0.5, ez + 0.5])
        return np.array(pts)
    xs = np.arange(0.5 * spacing, nelx, spacing)
    ys = np.arange(0.5 * spacing, nely, spacing)
    zs = np.arange(0.5 * spacing, nelz, spacing)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def load_point_cloud(path: str, dim: int,
                     nelx: int, nely: int, nelz: int = 1) -> np.ndarray:
    """Load point cloud from file and rescale to grid coordinates."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    if p.suffix.lower() == ".npy":
        pts = np.load(p)
    else:
        try:
            pts = np.loadtxt(p, delimiter=",")
        except Exception:
            pts = np.loadtxt(p)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < dim:
        raise ValueError(f"Point cloud must be Nx{dim} (got shape {pts.shape})")
    pts = pts[:, :dim]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    pts = (pts - lo) / span
    if dim == 2:
        pts[:, 0] *= nelx
        pts[:, 1] *= nely
    else:
        pts[:, 0] *= nelx
        pts[:, 1] *= nely
        pts[:, 2] *= nelz
    return pts


# ---------------------------------------------------------------------------
# Rasterisation – map particles to background-grid elements
# ---------------------------------------------------------------------------

def assign_particles(points: np.ndarray, nelx: int, nely: int,
                     nelz: int = 1, dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each particle to its containing element.

    Returns
    -------
    p2e : (n_particles,) element index for each particle
    counts : (n_elem,) number of particles in each element
    """
    if dim == 2:
        ex = np.clip(points[:, 0].astype(int), 0, nelx - 1)
        ey = np.clip(points[:, 1].astype(int), 0, nely - 1)
        p2e = ex * nely + ey
        n_elem = nelx * nely
    else:
        ex = np.clip(points[:, 0].astype(int), 0, nelx - 1)
        ey = np.clip(points[:, 1].astype(int), 0, nely - 1)
        ez = np.clip(points[:, 2].astype(int), 0, nelz - 1)
        p2e = ez * nelx * nely + ex * nely + ey
        n_elem = nelx * nely * nelz
    counts = np.bincount(p2e, minlength=n_elem)
    return p2e.astype(np.int32), counts.astype(np.int32)


def rasterize_active(points: np.ndarray, nelx: int, nely: int,
                     nelz: int = 1, dim: int = 2,
                     close_iters: int = 2,
                     distance_threshold: float = 1.8) -> np.ndarray:
    """Determine which elements are active from a point cloud.

    For regular benchmark grids every element is active.  For irregular
    clouds, elements within *distance_threshold* (in element widths) of
    any particle are activated, then gaps are closed with morphological
    operations.
    """
    if dim == 2:
        n_elem = nelx * nely
        centers = np.zeros((n_elem, 2))
        for ix in range(nelx):
            for iy in range(nely):
                centers[ix * nely + iy] = [ix + 0.5, iy + 0.5]
        mask_nd_shape = (nelx, nely)
    else:
        n_elem = nelx * nely * nelz
        centers = np.zeros((n_elem, 3))
        for iz in range(nelz):
            for ix in range(nelx):
                for iy in range(nely):
                    centers[iz * nelx * nely + ix * nely + iy] = [
                        ix + 0.5, iy + 0.5, iz + 0.5]
        mask_nd_shape = (nelz, nelx, nely)

    tree = cKDTree(points)
    distances, _ = tree.query(centers)
    mask_flat = distances <= distance_threshold
    mask_nd = mask_flat.reshape(mask_nd_shape)

    fill_ratio = mask_nd.sum() / mask_nd.size
    if close_iters > 0 and fill_ratio < 0.85:
        # Pad boundaries to prevent erosion from eating the domain edges
        pad_w = close_iters
        padded = np.pad(mask_nd, pad_w, mode="edge")
        struct = ndimage.generate_binary_structure(dim, 2)
        padded = ndimage.binary_closing(padded, structure=struct,
                                        iterations=close_iters)
        padded = ndimage.binary_fill_holes(padded)
        sl = tuple(slice(pad_w, -pad_w) for _ in range(dim))
        mask_nd = padded[sl]

    labels, nlab = ndimage.label(mask_nd)
    if nlab > 1:
        areas = ndimage.sum(mask_nd, labels, index=np.arange(1, nlab + 1))
        mask_nd = labels == (1 + int(np.argmax(areas)))

    return mask_nd.ravel().astype(bool)


# ---------------------------------------------------------------------------
# P2G / G2P transfers
# ---------------------------------------------------------------------------

def p2g(x_p: np.ndarray, p2e: np.ndarray, counts: np.ndarray,
        n_elem: int) -> np.ndarray:
    """Particle-to-grid: average particle densities per element."""
    x_e = np.zeros(n_elem, dtype=np.float64)
    np.add.at(x_e, p2e, x_p)
    active = counts > 0
    x_e[active] /= counts[active]
    return x_e


def g2p(field_e: np.ndarray, p2e: np.ndarray, counts: np.ndarray,
        n_particles: int) -> np.ndarray:
    """Grid-to-particle: distribute element field to particles."""
    c = np.maximum(counts[p2e], 1).astype(np.float64)
    return field_e[p2e] / c
