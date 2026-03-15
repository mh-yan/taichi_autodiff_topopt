"""Reference FEA solver (scipy spsolve) and full reference SIMP loop."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from config import TopOptConfig
from mesh import MeshData
from filter_utils import projection, projection_derivative, schedule, oc_update


def solve_fem(mesh: MeshData, x_phys: np.ndarray, KE: np.ndarray,
              penal: float, E0: float, Emin: float) -> np.ndarray:
    """Assemble global K and solve Ku=F with scipy direct solver."""
    active = mesh.active_ids
    edof_a = mesh.edof[active]
    Ee = Emin + np.maximum(x_phys[active], 1e-9) ** penal * (E0 - Emin)

    dpe = mesh.dpe
    n_act = edof_a.shape[0]
    iK = np.repeat(edof_a, dpe, axis=1).ravel()
    jK = np.tile(edof_a, (1, dpe)).ravel()
    sK = (Ee[:, None, None] * KE[None, :, :]).reshape(dpe * dpe * n_act)

    K = coo_matrix((sK, (iK, jK)), shape=(mesh.n_dof, mesh.n_dof)).tocsr()
    U = np.zeros(mesh.n_dof, dtype=np.float64)
    free = mesh.free_dofs
    U[free] = spsolve(K[free][:, free], mesh.F[free])
    return U


def compute_compliance_sensitivity(mesh: MeshData, U: np.ndarray,
                                   x_phys: np.ndarray, KE: np.ndarray,
                                   penal: float, E0: float, Emin: float):
    """Compute compliance and its sensitivity w.r.t. x_phys."""
    active = mesh.active_ids
    edof_a = mesh.edof[active]
    Ue = U[edof_a]
    ce = np.einsum("ij,ij->i", Ue @ KE, Ue)
    Ee = Emin + np.maximum(x_phys[active], 1e-9) ** penal * (E0 - Emin)
    compliance = float(np.dot(Ee, ce))

    dc_phys = np.zeros(mesh.n_elem, dtype=np.float64)
    dc_phys[active] = -penal * np.maximum(x_phys[active], 1e-9) ** (penal - 1) \
                      * (E0 - Emin) * ce
    return compliance, dc_phys


def run_reference_simp(mesh: MeshData, KE: np.ndarray,
                       cfg: TopOptConfig) -> tuple:
    """Full reference SIMP optimisation using scipy direct solver.

    Returns (x_phys_final, U_final, history, final_compliance).
    """
    x = np.zeros(mesh.n_elem, dtype=np.float64)
    x[mesh.active_ids] = cfg.volfrac
    hist = []

    for it in range(1, cfg.n_iter + 1):
        beta = schedule(it, cfg.beta_start, cfg.beta_max, cfg.beta_every)
        penal = schedule(it, cfg.penal_start, cfg.penal_max, cfg.penal_every)

        x_tilde = np.zeros_like(x)
        x_tilde[mesh.active_ids] = (mesh.H @ x)[mesh.active_ids] / \
                                    mesh.Hs[mesh.active_ids]
        x_phys = np.zeros_like(x)
        x_phys[mesh.active_ids] = projection(x_tilde[mesh.active_ids], beta, cfg.eta)

        U = solve_fem(mesh, x_phys, KE, penal, cfg.E0, cfg.Emin)
        comp, dc_phys = compute_compliance_sensitivity(
            mesh, U, x_phys, KE, penal, cfg.E0, cfg.Emin)

        dproj = np.zeros_like(x)
        dproj[mesh.active_ids] = projection_derivative(
            x_tilde[mesh.active_ids], beta, cfg.eta)

        dc = np.asarray(mesh.H.T @ ((dc_phys * dproj) / mesh.Hs)).ravel()
        dc[~mesh.active_mask] = 0.0
        dv = np.asarray(mesh.H.T @ (dproj / mesh.Hs)).ravel()
        dv[~mesh.active_mask] = 0.0

        x = oc_update(x, dc, dv, cfg.volfrac, mesh,
                      cfg.move, cfg.xmin, beta, cfg.eta)

        dpn = mesh.dim
        tip_dy = float(U[dpn * mesh.load_nodes + 1].mean()) if mesh.load_nodes.size else 0.0
        hist.append((it, comp, float(x_phys[mesh.active_ids].mean()), tip_dy))

    # final evaluation
    beta = schedule(cfg.n_iter, cfg.beta_start, cfg.beta_max, cfg.beta_every)
    penal = schedule(cfg.n_iter, cfg.penal_start, cfg.penal_max, cfg.penal_every)
    x_tilde = np.zeros_like(x)
    x_tilde[mesh.active_ids] = (mesh.H @ x)[mesh.active_ids] / mesh.Hs[mesh.active_ids]
    x_phys = np.zeros_like(x)
    x_phys[mesh.active_ids] = projection(x_tilde[mesh.active_ids], beta, cfg.eta)
    U = solve_fem(mesh, x_phys, KE, penal, cfg.E0, cfg.Emin)
    comp, _ = compute_compliance_sensitivity(mesh, U, x_phys, KE, penal, cfg.E0, cfg.Emin)
    return x_phys, U, np.asarray(hist, dtype=np.float64), comp
