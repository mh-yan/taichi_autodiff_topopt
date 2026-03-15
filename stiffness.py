import numpy as np


def ke_2d(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """8x8 element stiffness matrix for 2D Q4 plane-stress (unit-size element)."""
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0],
                    [-6, 3, 12, -3], [-3, 0, -3, 12]], dtype=np.float64)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6],
                    [0, -3, -6, 3], [3, -6, 3, -6]], dtype=np.float64)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4],
                    [-2, -9, -4, -3], [9, 4, -3, -4]], dtype=np.float64)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2],
                    [4, 9, 2, 3], [-9, -2, 3, 2]], dtype=np.float64)
    KE = E / (1 - nu**2) / 24.0 * (
        np.block([[A11, A12], [A12.T, A11]])
        + nu * np.block([[B11, B12], [B12.T, B11]])
    )
    return KE


def ke_3d(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """24x24 element stiffness matrix for 3D H8 (unit-size hexahedral element).

    Computed via 2x2x2 Gauss quadrature on reference element [-1,1]^3
    mapped to physical element [0,1]^3.
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.array([
        [lam + 2*mu, lam,       lam,       0,  0,  0],
        [lam,       lam + 2*mu, lam,       0,  0,  0],
        [lam,       lam,       lam + 2*mu, 0,  0,  0],
        [0,         0,         0,         mu,  0,  0],
        [0,         0,         0,          0, mu,  0],
        [0,         0,         0,          0,  0, mu],
    ], dtype=np.float64)

    nat = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)

    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = np.array([
        [-gp, -gp, -gp], [gp, -gp, -gp], [-gp, gp, -gp], [gp, gp, -gp],
        [-gp, -gp,  gp], [gp, -gp,  gp], [-gp, gp,  gp], [gp, gp,  gp],
    ])

    KE = np.zeros((24, 24), dtype=np.float64)
    for xi, eta, zeta in gauss_pts:
        dN = np.zeros((3, 8))
        for i in range(8):
            si, ei, zi = nat[i]
            dN[0, i] = si * (1 + ei * eta) * (1 + zi * zeta) / 8
            dN[1, i] = (1 + si * xi) * ei * (1 + zi * zeta) / 8
            dN[2, i] = (1 + si * xi) * (1 + ei * eta) * zi / 8

        # Physical coords of nodes = (nat + 1) / 2  =>  J = I/2, det(J) = 1/8
        dNdx = 2.0 * dN
        det_J = 1.0 / 8.0

        B = np.zeros((6, 24))
        for i in range(8):
            c = 3 * i
            B[0, c]     = dNdx[0, i]
            B[1, c + 1] = dNdx[1, i]
            B[2, c + 2] = dNdx[2, i]
            B[3, c]     = dNdx[1, i];  B[3, c + 1] = dNdx[0, i]
            B[4, c + 1] = dNdx[2, i];  B[4, c + 2] = dNdx[1, i]
            B[5, c]     = dNdx[2, i];  B[5, c + 2] = dNdx[0, i]

        KE += B.T @ D @ B * det_J

    return KE


def get_ke(dim: int, E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    if dim == 2:
        return ke_2d(E, nu)
    return ke_3d(E, nu)
