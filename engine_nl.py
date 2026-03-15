"""Differentiable nonlinear MLS-MPM engine for topology optimisation.

Explicit time integration, neo-Hookean constitutive law, SIMP density
interpolation. Gradients computed via stochastic parameter-shift
(avoids Taichi 1.7 AD tape bugs on macOS ARM).

Call ``make_nl_engine(cfg)`` after ``ti.init()``.
"""
import numpy as np


def make_nl_engine(cfg):
    import taichi as ti

    dim = 2
    n_grid = cfg.nl_n_grid
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    dt = cfg.nl_dt
    p_vol = (dx * 0.5) ** dim
    gravity = cfg.nl_gravity
    steps = cfg.nl_steps
    penal = 3.0

    E_val = cfg.nl_E
    nu_val = cfg.nu
    mu_0 = E_val / (2.0 * (1.0 + nu_val))
    la_0 = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    Emin_lame = 1e-4
    bound = 3

    ppd = cfg.nl_ppd
    bx0, by0 = cfg.nl_beam_x0, cfg.nl_beam_y0
    bw, bh = cfg.nl_beam_w, cfg.nl_beam_h
    nx = max(1, int(bw / dx * ppd))
    ny = max(1, int(bh / dx * ppd))
    n_particles = nx * ny

    init_x_np = np.zeros((n_particles, dim), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            init_x_np[i * ny + j] = [bx0 + (i + 0.5) * bw / nx,
                                      by0 + (j + 0.5) * bh / ny]

    tip_threshold = bx0 + bw * 0.92
    tip_ids = np.where(init_x_np[:, 0] > tip_threshold)[0]
    if len(tip_ids) == 0:
        tip_ids = np.array([n_particles - 1], dtype=np.int32)
    n_tip = len(tip_ids)

    # ---- Taichi fields (no needs_grad, no AD tape) --------------------
    x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
    rho_f = ti.field(dtype=ti.f32, shape=n_particles)
    init_x_f = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    tip_id_f = ti.field(dtype=ti.i32, shape=n_tip)

    init_x_f.from_numpy(init_x_np)
    tip_id_f.from_numpy(tip_ids.astype(np.int32))

    # ---- Kernels -------------------------------------------------------
    @ti.kernel
    def reset_sim():
        for p in range(n_particles):
            x[p] = init_x_f[p]
            v[p] = ti.Vector([0.0, 0.0])
            C[p] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])

    @ti.kernel
    def clear_grid():
        for i, j in grid_m:
            grid_v[i, j] = ti.Vector([0.0, 0.0])
            grid_m[i, j] = 0.0

    @ti.kernel
    def p2g():
        for p in range(n_particles):
            base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
            fx = x[p] * inv_dx - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
                 0.5 * (fx - 0.5)**2]
            new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[p]) @ F[p]
            F[p] = new_F
            J = new_F.determinant()
            r, s = ti.polar_decompose(new_F)
            rp = ti.max(rho_f[p], 1e-4)
            scale = ti.pow(rp, penal)
            mu_eff = Emin_lame + scale * (mu_0 - Emin_lame)
            la_eff = Emin_lame + scale * (la_0 - Emin_lame)
            cauchy = 2.0 * mu_eff * (new_F - r) @ new_F.transpose() \
                     + ti.Matrix.diag(dim=2, val=1) * la_eff * (J - 1.0) * J
            stress = -(dt * p_vol * 4.0 * inv_dx * inv_dx) * cauchy
            affine = stress + 1.0 * C[p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (ti.cast(offset, ti.f32) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    grid_v[base + offset] += weight * (v[p] + affine @ dpos)
                    grid_m[base + offset] += weight * 1.0

    @ti.kernel
    def g2p():
        for p in range(n_particles):
            base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
            fx = x[p] * inv_dx - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j]), ti.f32) - fx
                    g_m = grid_m[base[0] + i, base[1] + j]
                    g_v = grid_v[base[0] + i, base[1] + j]
                    v_out = g_v / ti.max(g_m, 1e-10)
                    v_out[1] -= dt * gravity
                    if base[0] + i < bound:
                        v_out = ti.Vector([0.0, 0.0])
                    if base[0] + i > n_grid - bound:
                        v_out[0] = ti.min(v_out[0], 0.0)
                    if base[1] + j < bound:
                        v_out[1] = ti.max(v_out[1], 0.0)
                    if base[1] + j > n_grid - bound:
                        v_out[1] = ti.min(v_out[1], 0.0)
                    weight = w[i][0] * w[j][1]
                    new_v += weight * v_out
                    new_C += 4.0 * weight * v_out.outer_product(dpos) * inv_dx
            v[p] = new_v
            x[p] = x[p] + dt * new_v
            C[p] = new_C

    def run_forward(rho_np):
        """Simulate and return average tip y-displacement."""
        rho_f.from_numpy(rho_np.astype(np.float32))
        reset_sim()
        for _ in range(steps):
            clear_grid()
            p2g()
            g2p()
        xf = x.to_numpy()
        return float(xf[tip_ids, 1].mean())

    # ---- Engine class --------------------------------------------------
    class NonlinearMPMEngine:
        def __init__(self):
            self.n_particles = n_particles
            self.n_tip = n_tip
            self.tip_ids = tip_ids
            self.init_x = init_x_np
            self.steps = steps
            self.n_grid = n_grid
            self.dx = dx

        def optimise(self, n_design_iters, volfrac, move=0.2, xmin=1e-3,
                     save_callback=None, n_spsa=3, fd_eps=0.1):
            """Topology optimization via SPSA gradient + projected gradient descent.

            Uses multiple SPSA samples per iteration for variance reduction.
            """
            rho_np = np.full(n_particles, volfrac, dtype=np.float32)
            hist = []
            rng = np.random.default_rng(42)
            lr = 0.5

            for it in range(1, n_design_iters + 1):
                tip_y_0 = run_forward(rho_np)
                loss_0 = -tip_y_0

                # Multi-sample SPSA gradient
                dc_accum = np.zeros(n_particles, dtype=np.float64)
                for _ in range(n_spsa):
                    delta = rng.choice([-1.0, 1.0], size=n_particles).astype(np.float32)
                    rho_plus = np.clip(rho_np + fd_eps * delta, xmin, 1.0)
                    rho_minus = np.clip(rho_np - fd_eps * delta, xmin, 1.0)
                    tip_plus = run_forward(rho_plus)
                    tip_minus = run_forward(rho_minus)
                    dc_accum += ((-tip_plus) - (-tip_minus)) / (2.0 * fd_eps) * delta
                dc = (dc_accum / n_spsa).astype(np.float32)

                # Projected gradient descent with volume constraint
                rho_trial = rho_np - lr * dc
                rho_trial = np.clip(rho_trial, xmin, 1.0)
                # Project onto volume constraint via bisection on threshold
                if rho_trial.mean() > volfrac:
                    lo, hi = 0.0, dc.max() * lr + 1.0
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        proj = np.clip(rho_np - lr * dc + mid * (volfrac - 1.0), xmin, 1.0)
                        if proj.mean() > volfrac:
                            hi = mid
                        else:
                            lo = mid
                    rho_trial = np.clip(rho_np - lr * dc + hi * (volfrac - 1.0), xmin, 1.0)
                rho_trial = np.clip(rho_trial, xmin, 1.0)
                # Move limit
                rho_np = np.clip(rho_trial, rho_np - move, rho_np + move).astype(np.float32)
                rho_np = np.clip(rho_np, xmin, 1.0)

                hist.append((it, loss_0, float(rho_np.mean()), tip_y_0))

                if save_callback and (it == 1 or it % 5 == 0 or it == n_design_iters):
                    rho_f.from_numpy(rho_np); reset_sim()
                    for _ in range(steps):
                        clear_grid(); p2g(); g2p()
                    save_callback(it, loss_0, rho_np,
                                  x.to_numpy(), self.init_x)

                if it <= 5 or it % 5 == 0:
                    print(f"  design={it:3d}  loss={loss_0:11.5e}  "
                          f"vol={rho_np.mean():.4f}  tip_y={tip_y_0:.6f}")

            return rho_np, np.array(hist)

    return NonlinearMPMEngine()
