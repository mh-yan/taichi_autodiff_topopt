from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

from config import TopOptConfig


@dataclass
class MeshData:
    dim: int
    nelx: int
    nely: int
    nelz: int
    n_elem: int
    n_nodes: int
    n_dof: int
    dpe: int                    # DOFs per element (8 for 2D, 24 for 3D)
    edof: np.ndarray            # (n_elem, dpe) DOF connectivity
    active_mask: np.ndarray     # (n_elem,) bool
    active_ids: np.ndarray      # indices of active elements
    n_active: int
    fixed_dofs: np.ndarray
    free_dofs: np.ndarray
    free_mask: np.ndarray       # (n_dof,) bool
    F: np.ndarray               # load vector (n_dof,)
    load_nodes: np.ndarray
    elem_centers: np.ndarray    # (n_elem, dim)
    node_coords: np.ndarray     # (n_nodes, dim)
    # filter data (populated by filter_utils)
    H: csr_matrix | None = None
    Hs: np.ndarray | None = None
    row_ptr: np.ndarray | None = None
    col_idx: np.ndarray | None = None
    weights: np.ndarray | None = None

    @property
    def edof_active(self) -> np.ndarray:
        return self.edof[self.active_ids]


# ---------------------------------------------------------------------------
# 2D mesh helpers
# ---------------------------------------------------------------------------

def _build_edof_2d(nelx: int, nely: int) -> np.ndarray:
    """Build DOF connectivity for 2D Q4 mesh (top88 convention)."""
    n_elem = nelx * nely
    edof = np.zeros((n_elem, 8), dtype=np.int32)
    for ex in range(nelx):
        for ey in range(nely):
            el = ex * nely + ey
            n1 = ex * (nely + 1) + ey
            n2 = (ex + 1) * (nely + 1) + ey
            edof[el] = [
                2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                2*n2,   2*n2+1, 2*n1,   2*n1+1,
            ]
    return edof


def _elem_centers_2d(nelx: int, nely: int) -> np.ndarray:
    ex = np.arange(nelx)
    ey = np.arange(nely)
    cx, cy = np.meshgrid(ex + 0.5, ey + 0.5, indexing="xy")
    centers = np.zeros((nelx * nely, 2))
    for ix in range(nelx):
        for iy in range(nely):
            centers[ix * nely + iy] = [ix + 0.5, iy + 0.5]
    return centers


def _node_coords_2d(nelx: int, nely: int) -> np.ndarray:
    n_nodes = (nelx + 1) * (nely + 1)
    coords = np.zeros((n_nodes, 2))
    for ix in range(nelx + 1):
        for iy in range(nely + 1):
            coords[ix * (nely + 1) + iy] = [ix, iy]
    return coords


def _cantilever_bc_2d(nelx: int, nely: int, n_dof: int, edof: np.ndarray,
                      active_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left edge fixed, right-edge mid-height downward unit load.

    For irregular domains the fixed / load nodes are chosen from nodes
    actually connected to active elements.
    """
    active_nodes_mask = np.zeros((nelx + 1) * (nely + 1), dtype=bool)
    for dofs in edof[active_ids]:
        active_nodes_mask[np.unique(dofs // 2)] = True
    active_nodes = np.flatnonzero(active_nodes_mask)
    node_x = active_nodes // (nely + 1)
    node_y = active_nodes % (nely + 1)

    # fixed: leftmost active column
    xmin = int(node_x.min())
    fixed_nodes = active_nodes[node_x <= xmin]
    fixed_dofs = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))

    # load: rightmost active nodes, closest to mid-height
    xmax = int(node_x.max())
    right_nodes = active_nodes[node_x >= xmax]
    right_y = right_nodes % (nely + 1)
    mid = nely / 2.0
    order = np.argsort(np.abs(right_y - mid))
    load_nodes = right_nodes[order[:max(1, min(4, len(order)))]]

    F = np.zeros(n_dof)
    F[2 * load_nodes + 1] = -1.0 / len(load_nodes)

    active_dofs = np.unique(edof[active_ids].ravel())
    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[active_dofs] = True
    free_mask[fixed_dofs] = False
    free_dofs = np.flatnonzero(free_mask)
    return fixed_dofs, free_dofs, free_mask, F, load_nodes.astype(np.int32)


def _mbb_bc_2d(nelx: int, nely: int, n_dof: int, edof: np.ndarray,
               active_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MBB beam (half-domain with left-edge x-symmetry).

    Top-left corner downward load, bottom-right y-roller, left edge ux=0.
    """
    top_left_node = 0                                    # ix=0, iy=0
    bottom_right_node = nelx * (nely + 1) + nely         # ix=nelx, iy=nely

    left_nodes = np.arange(nely + 1, dtype=np.int32)     # ix=0, iy=0..nely

    fixed_dofs = np.sort(np.unique(np.concatenate([
        2 * left_nodes,                                   # ux=0 symmetry
        [2 * bottom_right_node + 1],                      # uy=0 roller
    ])))

    F = np.zeros(n_dof)
    F[2 * top_left_node + 1] = -1.0

    load_nodes = np.array([top_left_node], dtype=np.int32)

    active_dofs = np.unique(edof[active_ids].ravel())
    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[active_dofs] = True
    free_mask[fixed_dofs] = False
    free_dofs = np.flatnonzero(free_mask)
    return fixed_dofs, free_dofs, free_mask, F, load_nodes


def _lbracket_bc_2d(nelx: int, nely: int, n_dof: int, edof: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """L-bracket (2D): full rectangle minus top-right quadrant.

    Returns (active_mask, fixed_dofs, free_dofs, free_mask, F, load_nodes).
    """
    n_elem = nelx * nely
    n_nodes = (nelx + 1) * (nely + 1)

    el = np.arange(n_elem)
    active_mask = (el // nely < nelx // 2) | (el % nely < nely // 2)
    active_ids = np.flatnonzero(active_mask)

    # Fixed: top-left edge — all nodes at iy=nely, ix <= nelx//2
    fixed_nodes = np.array(
        [ix * (nely + 1) + nely for ix in range(nelx // 2 + 1)], dtype=np.int32,
    )
    fixed_dofs = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))

    # Load: rightmost mid-height active node, pointing downward
    active_nodes_mask = np.zeros(n_nodes, dtype=bool)
    for dofs in edof[active_ids]:
        active_nodes_mask[np.unique(dofs // 2)] = True
    active_nodes = np.flatnonzero(active_nodes_mask)
    node_x = active_nodes // (nely + 1)

    xmax = int(node_x.max())
    right_nodes = active_nodes[node_x == xmax]
    right_y = right_nodes % (nely + 1)
    mid = (right_y.min() + right_y.max()) / 2.0
    load_node = right_nodes[np.argmin(np.abs(right_y - mid))]
    load_nodes = np.array([load_node], dtype=np.int32)

    F = np.zeros(n_dof)
    F[2 * load_node + 1] = -1.0

    active_dofs = np.unique(edof[active_ids].ravel())
    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[active_dofs] = True
    free_mask[fixed_dofs] = False
    free_dofs = np.flatnonzero(free_mask)
    return active_mask, fixed_dofs, free_dofs, free_mask, F, load_nodes


# ---------------------------------------------------------------------------
# 3D mesh helpers
# ---------------------------------------------------------------------------

def _build_edof_3d(nelx: int, nely: int, nelz: int) -> np.ndarray:
    """Build DOF connectivity for 3D H8 mesh."""
    nxy = (nelx + 1) * (nely + 1)
    n_elem = nelx * nely * nelz
    edof = np.zeros((n_elem, 24), dtype=np.int32)
    for ez in range(nelz):
        for ex in range(nelx):
            for ey in range(nely):
                el = ez * nelx * nely + ex * nely + ey
                n = np.array([
                    ez * nxy + ex * (nely+1) + ey,
                    ez * nxy + (ex+1) * (nely+1) + ey,
                    ez * nxy + (ex+1) * (nely+1) + (ey+1),
                    ez * nxy + ex * (nely+1) + (ey+1),
                    (ez+1) * nxy + ex * (nely+1) + ey,
                    (ez+1) * nxy + (ex+1) * (nely+1) + ey,
                    (ez+1) * nxy + (ex+1) * (nely+1) + (ey+1),
                    (ez+1) * nxy + ex * (nely+1) + (ey+1),
                ], dtype=np.int32)
                dofs = np.zeros(24, dtype=np.int32)
                for i in range(8):
                    dofs[3*i:3*i+3] = [3*n[i], 3*n[i]+1, 3*n[i]+2]
                edof[el] = dofs
    return edof


def _elem_centers_3d(nelx: int, nely: int, nelz: int) -> np.ndarray:
    n_elem = nelx * nely * nelz
    centers = np.zeros((n_elem, 3))
    for ez in range(nelz):
        for ex in range(nelx):
            for ey in range(nely):
                el = ez * nelx * nely + ex * nely + ey
                centers[el] = [ex + 0.5, ey + 0.5, ez + 0.5]
    return centers


def _node_coords_3d(nelx: int, nely: int, nelz: int) -> np.ndarray:
    nxy = (nelx + 1) * (nely + 1)
    n_nodes = nxy * (nelz + 1)
    coords = np.zeros((n_nodes, 3))
    for iz in range(nelz + 1):
        for ix in range(nelx + 1):
            for iy in range(nely + 1):
                nid = iz * nxy + ix * (nely + 1) + iy
                coords[nid] = [ix, iy, iz]
    return coords


def _cantilever_bc_3d(nelx: int, nely: int, nelz: int, n_dof: int,
                      edof: np.ndarray, active_ids: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left face fixed, right-face center downward unit load.

    Adapts to irregular domains by selecting nodes actually connected
    to active elements.
    """
    nxy = (nelx + 1) * (nely + 1)
    n_nodes = nxy * (nelz + 1)
    active_nodes_mask = np.zeros(n_nodes, dtype=bool)
    for dofs in edof[active_ids]:
        active_nodes_mask[np.unique(dofs // 3)] = True
    active_nodes = np.flatnonzero(active_nodes_mask)
    node_ix = (active_nodes % nxy) // (nely + 1)

    # fixed: leftmost active x-plane
    xmin = int(node_ix.min())
    fixed_nodes = active_nodes[node_ix <= xmin]
    fixed_dofs = np.sort(np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2]))

    # load: rightmost active nodes, closest to domain centre
    xmax = int(node_ix.max())
    right_nodes = active_nodes[node_ix >= xmax]
    right_iy = right_nodes % (nely + 1)
    right_iz = right_nodes // nxy
    cy, cz = nely / 2.0, nelz / 2.0
    dist = np.sqrt((right_iy - cy)**2 + (right_iz - cz)**2)
    order = np.argsort(dist)
    load_nodes = right_nodes[order[:max(1, min(4, len(order)))]]

    F = np.zeros(n_dof)
    F[3 * load_nodes + 1] = -1.0 / len(load_nodes)

    active_dofs = np.unique(edof[active_ids].ravel())
    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[active_dofs] = True
    free_mask[fixed_dofs] = False
    free_dofs = np.flatnonzero(free_mask)
    return fixed_dofs, free_dofs, free_mask, F, load_nodes.astype(np.int32)


def _bridge_bc_3d(nelx: int, nely: int, nelz: int, n_dof: int,
                  edof: np.ndarray, active_ids: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3D bridge: two bottom pin supports, top-centre downward load."""
    nxy = (nelx + 1) * (nely + 1)

    sup1 = (nelz // 2) * nxy + 0                          # (x=0, y=0, z=nelz//2)
    sup2 = (nelz // 2) * nxy + nelx * (nely + 1)          # (x=nelx, y=0, z=nelz//2)
    support_nodes = np.array([sup1, sup2], dtype=np.int32)
    fixed_dofs = np.sort(np.concatenate([
        3 * support_nodes, 3 * support_nodes + 1, 3 * support_nodes + 2,
    ]))

    load_node = (nelz // 2) * nxy + (nelx // 2) * (nely + 1) + nely
    load_nodes = np.array([load_node], dtype=np.int32)

    F = np.zeros(n_dof)
    F[3 * load_node + 1] = -1.0

    active_dofs = np.unique(edof[active_ids].ravel())
    free_mask = np.zeros(n_dof, dtype=bool)
    free_mask[active_dofs] = True
    free_mask[fixed_dofs] = False
    free_dofs = np.flatnonzero(free_mask)
    return fixed_dofs, free_dofs, free_mask, F, load_nodes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_mesh(cfg: TopOptConfig, active_mask: np.ndarray | None = None) -> MeshData:
    """Build background grid and boundary conditions.

    Parameters
    ----------
    cfg : TopOptConfig
    active_mask : optional bool array (n_elem,). If None all elements active.
    """
    nelx, nely, nelz = cfg.nelx, cfg.nely, cfg.nelz
    dim = cfg.dim
    problem = cfg.problem

    if dim == 2:
        n_elem = nelx * nely
        n_nodes = (nelx + 1) * (nely + 1)
        n_dof = 2 * n_nodes
        dpe = 8
        edof = _build_edof_2d(nelx, nely)
        centers = _elem_centers_2d(nelx, nely)
        node_coords = _node_coords_2d(nelx, nely)
    else:
        n_elem = nelx * nely * nelz
        n_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)
        n_dof = 3 * n_nodes
        dpe = 24
        edof = _build_edof_3d(nelx, nely, nelz)
        centers = _elem_centers_3d(nelx, nely, nelz)
        node_coords = _node_coords_3d(nelx, nely, nelz)

    if active_mask is None:
        active_mask = np.ones(n_elem, dtype=bool)

    if problem == "lbracket":
        if dim != 2:
            raise ValueError("L-bracket problem is only supported in 2D")
        lbracket_mask, fixed_dofs, free_dofs, free_mask, F, load_nodes = \
            _lbracket_bc_2d(nelx, nely, n_dof, edof)
        active_mask = lbracket_mask
        active_ids = np.flatnonzero(active_mask)
    else:
        active_ids = np.flatnonzero(active_mask)
        if problem == "cantilever":
            if dim == 2:
                fixed_dofs, free_dofs, free_mask, F, load_nodes = \
                    _cantilever_bc_2d(nelx, nely, n_dof, edof, active_ids)
            else:
                fixed_dofs, free_dofs, free_mask, F, load_nodes = \
                    _cantilever_bc_3d(nelx, nely, nelz, n_dof, edof, active_ids)
        elif problem == "mbb":
            if dim != 2:
                raise ValueError("MBB beam problem is only supported in 2D")
            fixed_dofs, free_dofs, free_mask, F, load_nodes = \
                _mbb_bc_2d(nelx, nely, n_dof, edof, active_ids)
        elif problem == "bridge":
            if dim != 3:
                raise ValueError("Bridge problem is only supported in 3D")
            fixed_dofs, free_dofs, free_mask, F, load_nodes = \
                _bridge_bc_3d(nelx, nely, nelz, n_dof, edof, active_ids)
        else:
            raise ValueError(f"Unknown problem type: {problem}")

    return MeshData(
        dim=dim, nelx=nelx, nely=nely, nelz=nelz,
        n_elem=n_elem, n_nodes=n_nodes, n_dof=n_dof, dpe=dpe,
        edof=edof, active_mask=active_mask, active_ids=active_ids,
        n_active=int(active_ids.size),
        fixed_dofs=fixed_dofs, free_dofs=free_dofs, free_mask=free_mask,
        F=F, load_nodes=load_nodes,
        elem_centers=centers, node_coords=node_coords,
    )
