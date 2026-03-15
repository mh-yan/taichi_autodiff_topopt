"""Density filter, Heaviside projection, and OC update for SIMP."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from mesh import MeshData


# ---------------------------------------------------------------------------
# Density filter
# ---------------------------------------------------------------------------

def build_filter(mesh: MeshData, rmin: float) -> Tuple[csr_matrix, np.ndarray]:
    """Build filter matrix H (sparse) and row-sum vector Hs.

    Works for both 2D and 3D via element center coordinates.
    """
    nelx, nely, nelz, dim = mesh.nelx, mesh.nely, mesh.nelz, mesh.dim
    active_set = set(int(i) for i in mesh.active_ids)
    radius = int(np.ceil(rmin))

    rows, cols, vals = [], [], []

    if dim == 2:
        for eid in mesh.active_ids:
            ex = int(eid) // nely
            ey = int(eid) % nely
            for nx in range(max(0, ex - radius), min(nelx, ex + radius + 1)):
                for ny in range(max(0, ey - radius), min(nely, ey + radius + 1)):
                    nid = nx * nely + ny
                    if nid not in active_set:
                        continue
                    dist = np.sqrt(float((ex - nx)**2 + (ey - ny)**2))
                    w = rmin - dist
                    if w > 0:
                        rows.append(int(eid))
                        cols.append(int(nid))
                        vals.append(float(w))
    else:
        for eid in mesh.active_ids:
            rem = int(eid)
            ez = rem // (nelx * nely); rem -= ez * nelx * nely
            ex = rem // nely; ey = rem % nely
            for nz in range(max(0, ez - radius), min(nelz, ez + radius + 1)):
                for nx in range(max(0, ex - radius), min(nelx, ex + radius + 1)):
                    for ny in range(max(0, ey - radius), min(nely, ey + radius + 1)):
                        nid = nz * nelx * nely + nx * nely + ny
                        if nid not in active_set:
                            continue
                        dist = np.sqrt(float((ex-nx)**2 + (ey-ny)**2 + (ez-nz)**2))
                        w = rmin - dist
                        if w > 0:
                            rows.append(int(eid))
                            cols.append(int(nid))
                            vals.append(float(w))

    H = coo_matrix((vals, (rows, cols)), shape=(mesh.n_elem, mesh.n_elem)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    Hs[Hs == 0.0] = 1.0
    return H, Hs


def attach_filter(mesh: MeshData, rmin: float) -> MeshData:
    """Build filter and store it inside the MeshData object."""
    H, Hs = build_filter(mesh, rmin)
    mesh.H = H
    mesh.Hs = Hs
    mesh.row_ptr = H.indptr.astype(np.int32)
    mesh.col_idx = H.indices.astype(np.int32)
    mesh.weights = H.data.astype(np.float64)
    return mesh


# ---------------------------------------------------------------------------
# Heaviside projection
# ---------------------------------------------------------------------------

def projection(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def projection_derivative(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    t = np.tanh(beta * (x - eta))
    return beta * (1.0 - t * t) / den


# ---------------------------------------------------------------------------
# Continuation schedule
# ---------------------------------------------------------------------------

def schedule(it: int, start: float, maxv: float, every: int) -> float:
    if every <= 0:
        return start
    steps = max(0, (it - 1) // every)
    return min(maxv, start * (2 ** steps))


# ---------------------------------------------------------------------------
# OC update (projected volume constraint)
# ---------------------------------------------------------------------------

def oc_update(x: np.ndarray, dc: np.ndarray, dv: np.ndarray,
              volfrac: float, mesh: MeshData,
              move: float, xmin: float,
              beta: float, eta: float) -> np.ndarray:
    """OC update with bisection on Lagrange multiplier.

    The volume constraint is evaluated on filtered+projected densities.
    """
    active = mesh.active_ids
    H, Hs = mesh.H, mesh.Hs

    def _candidate(lmid: float) -> np.ndarray:
        ratio = np.sqrt(np.maximum(1e-30, -dc[active] /
                                   np.maximum(1e-30, dv[active] * lmid)))
        cand = np.maximum(xmin,
                          np.maximum(x[active] - move,
                                     np.minimum(1.0,
                                                np.minimum(x[active] + move,
                                                           x[active] * ratio))))
        out = x.copy()
        out[active] = cand
        out[~mesh.active_mask] = 0.0
        return out

    def _proj_vol(xc: np.ndarray) -> float:
        xt = np.zeros_like(xc)
        xt[active] = (H @ xc)[active] / Hs[active]
        xp = projection(xt, beta, eta)
        return float(xp[active].mean())

    l1, l2 = 0.0, 1.0
    cand = _candidate(l2)
    vol = _proj_vol(cand)
    guard = 0
    while vol > volfrac and guard < 80:
        l2 *= 2.0
        cand = _candidate(l2)
        vol = _proj_vol(cand)
        guard += 1

    for _ in range(80):
        lmid = 0.5 * (l1 + l2)
        cand = _candidate(lmid)
        vol = _proj_vol(cand)
        if vol > volfrac:
            l1 = lmid
        else:
            l2 = lmid
        if (l2 - l1) / (l1 + l2 + 1e-12) <= 1e-4:
            break
    return cand
