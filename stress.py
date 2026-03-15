"""P-norm stress constraint for SIMP-based topology optimization.

Von Mises stress computation, p-norm aggregation, sensitivities,
and augmented Lagrangian updates for 2D Q4 and 3D H8 elements.
All pure numpy — runs once per iteration on CPU.
"""

import numpy as np


# ---------------------------------------------------------------------------
# B matrices at element centre (ξ=η=0 for 2D, ξ=η=ζ=0 for 3D)
# ---------------------------------------------------------------------------

def _B_center_2d():
    """Strain-displacement matrix (3x8) at centre of Q4 unit element.

    Node order matches edof convention in mesh.py (_build_edof_2d):
        0: (-1,+1)  top-left        phys (ex, ey+1)
        1: (+1,+1)  top-right       phys (ex+1, ey+1)
        2: (+1,-1)  bottom-right    phys (ex+1, ey)
        3: (-1,-1)  bottom-left     phys (ex, ey)

    Jacobian for [-1,1]^2 -> [0,1]^2 is J = I/2, so dN/dx = 2 * dN/dxi.
    At centre (xi=eta=0): dN_i/dxi = xi_i/4, so dN_i/dx = xi_i/2.
    """
    xi = np.array([-1.0, 1.0, 1.0, -1.0])
    eta = np.array([1.0, 1.0, -1.0, -1.0])
    dNdx = xi / 2.0
    dNdy = eta / 2.0

    B = np.zeros((3, 8), dtype=np.float64)
    for i in range(4):
        B[0, 2 * i] = dNdx[i]
        B[1, 2 * i + 1] = dNdy[i]
        B[2, 2 * i] = dNdy[i]
        B[2, 2 * i + 1] = dNdx[i]
    return B


def _B_center_3d():
    """Strain-displacement matrix (6x24) at centre of H8 unit element.

    Node order matches stiffness.py / mesh.py (_build_edof_3d):
        0: (-1,-1,-1)  1: (+1,-1,-1)  2: (+1,+1,-1)  3: (-1,+1,-1)
        4: (-1,-1,+1)  5: (+1,-1,+1)  6: (+1,+1,+1)  7: (-1,+1,+1)

    At centre: dN_i/dx_j = nat[i,j] / 4  (Jacobian I/2 for unit element).
    """
    nat = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    dNdx = nat / 4.0  # (8, 3)

    B = np.zeros((6, 24), dtype=np.float64)
    for i in range(8):
        c = 3 * i
        B[0, c] = dNdx[i, 0]
        B[1, c + 1] = dNdx[i, 1]
        B[2, c + 2] = dNdx[i, 2]
        B[3, c] = dNdx[i, 1]
        B[3, c + 1] = dNdx[i, 0]
        B[4, c + 1] = dNdx[i, 2]
        B[4, c + 2] = dNdx[i, 1]
        B[5, c] = dNdx[i, 2]
        B[5, c + 2] = dNdx[i, 0]
    return B


# ---------------------------------------------------------------------------
# Constitutive matrices
# ---------------------------------------------------------------------------

def _D_plane_stress(E, nu):
    """Plane-stress constitutive matrix (3x3)."""
    c = E / (1.0 - nu ** 2)
    return c * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ], dtype=np.float64)


def _D_3d(E, nu):
    """3-D isotropic constitutive matrix (6x6)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return np.array([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Von Mises stress (vectorised)
# ---------------------------------------------------------------------------

def compute_von_mises_2d(U, edof_active, elem_ids, x_phys, KE, E0, Emin, penal, nu):
    """Von Mises stress with qp-relaxation for active 2D Q4 elements.

    Computes solid-material stress  sigma = D(E0,nu) . B_centre . u_e,
    then applies qp-relaxation:  sigma_tilde = x_phys^(p-1) * sigma_vm
    to avoid the stress singularity in void regions.

    Parameters
    ----------
    U          : (n_dof,)    global displacement vector.
    edof_active: (n_active, 8) DOF connectivity for active elements.
    elem_ids   : (n_active,) global element indices of active elements.
    x_phys     : (n_elem,)   physical densities (after filter + projection).
    KE         : (8, 8)      unit element stiffness (unused, kept for API consistency).
    E0, Emin   : float       Young's modulus bounds.
    penal      : float       SIMP penalisation exponent.
    nu         : float       Poisson's ratio.

    Returns
    -------
    sigma_vm : (n_active,) qp-relaxed von Mises stress per active element.
    """
    DB = _D_plane_stress(E0, nu) @ _B_center_2d()     # (3, 8)
    u_e = U[edof_active]                               # (n_active, 8)
    stress = u_e @ DB.T                                # (n_active, 3): [sx, sy, txy]

    sx, sy, txy = stress[:, 0], stress[:, 1], stress[:, 2]
    vm = np.sqrt(np.maximum(sx ** 2 - sx * sy + sy ** 2 + 3.0 * txy ** 2, 0.0))

    xp = np.maximum(x_phys[elem_ids], 1e-9)
    return xp ** (penal - 1.0) * vm


def compute_von_mises_3d(U, edof_active, elem_ids, x_phys, KE, E0, Emin, penal, nu):
    """Von Mises stress with qp-relaxation for active 3D H8 elements.

    Same approach as the 2D version but with the full 3-D constitutive law.
    sigma_vm = sqrt(0.5 * ((sx-sy)^2 + (sy-sz)^2 + (sz-sx)^2
                           + 6*(txy^2 + tyz^2 + txz^2)))

    Parameters
    ----------
    U          : (n_dof,)     global displacement vector.
    edof_active: (n_active, 24) DOF connectivity for active elements.
    elem_ids   : (n_active,)  global element indices of active elements.
    x_phys     : (n_elem,)    physical densities.
    KE         : (24, 24)     unit element stiffness (unused, API consistency).
    E0, Emin   : float
    penal      : float
    nu         : float

    Returns
    -------
    sigma_vm : (n_active,) qp-relaxed von Mises stress per active element.
    """
    DB = _D_3d(E0, nu) @ _B_center_3d()               # (6, 24)
    u_e = U[edof_active]                               # (n_active, 24)
    stress = u_e @ DB.T                                # (n_active, 6)

    sx, sy, sz = stress[:, 0], stress[:, 1], stress[:, 2]
    txy, tyz, txz = stress[:, 3], stress[:, 4], stress[:, 5]
    vm = np.sqrt(np.maximum(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2
               + 6.0 * (txy ** 2 + tyz ** 2 + txz ** 2)),
        0.0))

    xp = np.maximum(x_phys[elem_ids], 1e-9)
    return xp ** (penal - 1.0) * vm


# ---------------------------------------------------------------------------
# P-norm aggregation
# ---------------------------------------------------------------------------

def pnorm_stress(sigma_vm, sigma_allow, pn=8):
    """P-norm stress aggregation.

    sigma_pn = (sum (sigma_vm / sigma_allow)^pn)^(1/pn)

    Smooth, differentiable approximation of max(sigma_vm).
    """
    ratio = sigma_vm / sigma_allow
    return (np.sum(ratio ** pn)) ** (1.0 / pn)


def pnorm_stress_sensitivity(sigma_vm, sigma_allow, pn, dc_dsigma_vm=None):
    """Derivative of p-norm w.r.t. each element's relaxed sigma_vm.

    d(sigma_pn)/d(sigma_vm[e]) =
        sigma_pn^(1-pn) * (sigma_vm[e] / sigma_allow)^(pn-1) / sigma_allow

    Parameters
    ----------
    sigma_vm    : (n_active,) relaxed von Mises stresses.
    sigma_allow : float       allowable stress.
    pn          : int         p-norm exponent.
    dc_dsigma_vm: (n_active,) optional upstream gradient to chain-multiply.

    Returns
    -------
    d : (n_active,) sensitivities (or chained product if dc_dsigma_vm given).
    """
    sigma_pn = pnorm_stress(sigma_vm, sigma_allow, pn)
    safe_pn = max(sigma_pn, 1e-30)

    d = (safe_pn ** (1.0 - pn)
         * (sigma_vm / sigma_allow) ** (pn - 1.0)
         / sigma_allow)

    if dc_dsigma_vm is not None:
        return dc_dsigma_vm * d
    return d


# ---------------------------------------------------------------------------
# Stress sensitivity w.r.t. x_phys (explicit / local part)
# ---------------------------------------------------------------------------

def stress_sensitivity_2d(U, edof_active, elem_ids, x_phys,
                          KE, E0, Emin, penal, nu, sigma_vm):
    """Explicit sensitivity of qp-relaxed sigma_vm w.r.t. x_phys.

    From the relaxation  sigma_tilde = x^(p-1) * sigma_vm_solid:
        d(sigma_tilde)/d(x_phys[e]) = (p-1) * sigma_tilde / x_phys[e]

    This is the *explicit* (local) part only.  The implicit part
    (through the displacement field) requires a separate adjoint solve.

    The formula is dimension-independent; the full function signature
    is kept for API consistency with compute_von_mises_2d.

    Parameters
    ----------
    sigma_vm : (n_active,) relaxed von Mises stresses from compute_von_mises_*.

    Returns
    -------
    dsigma_dx : (n_elem,) sensitivity mapped to full element array.
    """
    dsigma_dx = np.zeros_like(x_phys)
    xp = np.maximum(x_phys[elem_ids], 1e-9)
    dsigma_dx[elem_ids] = (penal - 1.0) * sigma_vm / xp
    return dsigma_dx


# ---------------------------------------------------------------------------
# Augmented Lagrangian
# ---------------------------------------------------------------------------

def augmented_lagrangian_update(sigma_pn, sigma_pn_prev, lam, mu,
                                sigma_allow=1.0):
    """Update Lagrange multiplier and penalty for the stress constraint.

    lambda_new = max(0, lambda + mu * (sigma_pn / sigma_allow - 1))
    mu_new     = 2 * mu   if |sigma_pn - sigma_pn_prev| > 0.25 * |sigma_pn_prev|

    Parameters
    ----------
    sigma_pn      : float  current p-norm stress value.
    sigma_pn_prev : float  previous iteration's p-norm stress.
    lam           : float  current Lagrange multiplier.
    mu            : float  current penalty parameter.
    sigma_allow   : float  allowable stress limit.

    Returns
    -------
    lam_new, mu_new : updated multiplier and penalty.
    """
    lam_new = max(0.0, lam + mu * (sigma_pn / sigma_allow - 1.0))

    if abs(sigma_pn_prev) > 1e-30:
        if abs(sigma_pn - sigma_pn_prev) > 0.25 * abs(sigma_pn_prev):
            mu *= 2.0

    return lam_new, mu
