"""Differentiable nonlinear MLS-MPM in JAX.

End-to-end differentiable via ``jax.grad``.  Neo-Hookean (fixed corotated)
constitutive law with SIMP density interpolation.

Usage:
    from engine_nl_jax import NLConfig, NLEngine
    eng = NLEngine(NLConfig(n_steps=512, gravity=30, E=100))
    rho_final, hist = eng.optimise(n_iters=60, volfrac=0.5)
"""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class NLConfig:
    n_grid: int = 64
    dt: float = 1e-4
    n_steps: int = 512
    gravity: float = 30.0
    E: float = 100.0
    nu: float = 0.3
    penal: float = 3.0
    Emin_lame: float = 1e-4
    beam_x0: float = 0.05
    beam_y0: float = 0.40
    beam_w: float = 0.50
    beam_h: float = 0.12
    ppd: int = 2
    bound: int = 3
    volfrac: float = 0.5
    move: float = 0.2
    xmin: float = 1e-3
    n_design_iters: int = 60


def _polar_2d(F):
    C = F.T @ F
    det_C = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    s = jnp.sqrt(jnp.maximum(det_C, 1e-20))
    denom = jnp.sqrt(jnp.maximum(jnp.trace(C) + 2.0 * s, 1e-20))
    S = (C + s * jnp.eye(2)) / denom
    S_inv = jnp.linalg.inv(S + 1e-12 * jnp.eye(2))
    R = F @ S_inv
    return R


def _bspline_w(fx):
    """Quadratic B-spline weights: returns (3,2) array."""
    return jnp.array([
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
    ])


class NLEngine:
    def __init__(self, cfg: NLConfig):
        self.cfg = cfg
        dx = 1.0 / cfg.n_grid
        inv_dx = float(cfg.n_grid)
        ppd = cfg.ppd
        nx = max(1, int(cfg.beam_w / dx * ppd))
        ny = max(1, int(cfg.beam_h / dx * ppd))
        self.n_particles = nx * ny

        xs = cfg.beam_x0 + (jnp.arange(nx) + 0.5) * cfg.beam_w / nx
        ys = cfg.beam_y0 + (jnp.arange(ny) + 0.5) * cfg.beam_h / ny
        gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
        self.init_x = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)

        tip_thresh = cfg.beam_x0 + cfg.beam_w * 0.9
        self.tip_mask = self.init_x[:, 0] > tip_thresh
        self.n_tip = max(1, int(self.tip_mask.sum()))

        mu_0 = cfg.E / (2.0 * (1.0 + cfg.nu))
        la_0 = cfg.E * cfg.nu / ((1.0 + cfg.nu) * (1.0 - 2.0 * cfg.nu))

        n_grid = cfg.n_grid
        dt = cfg.dt
        p_vol = (dx * 0.5) ** 2
        gravity = cfg.gravity
        bound = cfg.bound
        penal = cfg.penal
        Emin = cfg.Emin_lame
        n_p = self.n_particles
        init_x_arr = self.init_x
        tip_mask = self.tip_mask
        n_tip = self.n_tip
        n_steps = cfg.n_steps

        offsets = jnp.array([[i, j] for i in range(3) for j in range(3)])

        def _p2g_one(xp, vp, Cp, Fp, rp):
            """Compute grid contributions for one particle.
            Returns (indices(9,2), vel_contribs(9,2), mass_contribs(9,), new_F).
            """
            base = jnp.floor(xp * inv_dx - 0.5).astype(jnp.int32)
            fx = xp * inv_dx - base
            ws = _bspline_w(fx)

            new_F = (jnp.eye(2) + dt * Cp) @ Fp
            J = jnp.linalg.det(new_F)
            R = _polar_2d(new_F)

            scale = jnp.power(jnp.clip(rp, 1e-4, 1.0), penal)
            mu_eff = Emin + scale * (mu_0 - Emin)
            la_eff = Emin + scale * (la_0 - Emin)

            cauchy = (2.0 * mu_eff * (new_F - R) @ new_F.T
                      + la_eff * (J - 1.0) * J * jnp.eye(2))
            mass = jnp.clip(rp, 1e-4, 1.0)
            stress = -(dt * p_vol * 4.0 * inv_dx * inv_dx) * cauchy
            affine = stress + mass * Cp

            def _contrib(off):
                dpos = (off - fx) * dx
                w = ws[off[0], 0] * ws[off[1], 1]
                v_c = w * (mass * vp + affine @ dpos)
                m_c = w * mass
                idx = base + off
                return idx, v_c, m_c

            idxs, v_cs, m_cs = jax.vmap(_contrib)(offsets)
            return idxs, v_cs, m_cs, new_F

        vmap_p2g = jax.vmap(_p2g_one)

        def _g2p_one(xp, grid_v_out):
            base = jnp.floor(xp * inv_dx - 0.5).astype(jnp.int32)
            fx = xp * inv_dx - base
            ws = _bspline_w(fx)

            def _gather(off):
                dpos = off - fx
                idx = base + off
                ix = jnp.clip(idx[0], 0, n_grid - 1)
                iy = jnp.clip(idx[1], 0, n_grid - 1)
                gv = grid_v_out[ix, iy]
                w = ws[off[0], 0] * ws[off[1], 1]
                return w * gv, 4.0 * w * jnp.outer(gv, dpos) * inv_dx

            v_parts, C_parts = jax.vmap(_gather)(offsets)
            new_v = v_parts.sum(axis=0)
            new_C = C_parts.sum(axis=0)
            new_x = xp + dt * new_v
            return new_x, new_v, new_C

        vmap_g2p = jax.vmap(_g2p_one, in_axes=(0, None))

        def substep(state, rho):
            x, v, C, F = state

            # P2G (vectorized over particles)
            all_idx, all_vc, all_mc, new_F = vmap_p2g(x, v, C, F, rho)
            # all_idx: (n_p, 9, 2), all_vc: (n_p, 9, 2), all_mc: (n_p, 9)

            flat_idx = all_idx.reshape(-1, 2)
            flat_ix = jnp.clip(flat_idx[:, 0], 0, n_grid - 1)
            flat_iy = jnp.clip(flat_idx[:, 1], 0, n_grid - 1)
            flat_lin = flat_ix * n_grid + flat_iy

            grid_v = jnp.zeros((n_grid * n_grid, 2))
            grid_m = jnp.zeros(n_grid * n_grid)
            grid_v = grid_v.at[flat_lin].add(all_vc.reshape(-1, 2))
            grid_m = grid_m.at[flat_lin].add(all_mc.reshape(-1))
            grid_v = grid_v.reshape(n_grid, n_grid, 2)
            grid_m = grid_m.reshape(n_grid, n_grid)

            # Grid operation
            inv_m = 1.0 / jnp.maximum(grid_m, 1e-10)
            gv_out = grid_v * inv_m[..., None]
            gv_out = gv_out.at[:, :, 1].add(-dt * gravity)

            # BCs
            ix_arr = jnp.arange(n_grid)[:, None]
            iy_arr = jnp.arange(n_grid)[None, :]
            left = ix_arr < bound
            gv_out = jnp.where(left[..., None], 0.0, gv_out)
            right_wall = (ix_arr > n_grid - bound) & (gv_out[:, :, 0:1] > 0)
            gv_out = gv_out.at[:, :, 0].set(
                jnp.where((ix_arr > n_grid - bound) & (gv_out[:, :, 0] > 0),
                          0.0, gv_out[:, :, 0]))
            gv_out = gv_out.at[:, :, 1].set(
                jnp.where((iy_arr < bound) & (gv_out[:, :, 1] < 0),
                          0.0, gv_out[:, :, 1]))
            gv_out = gv_out.at[:, :, 1].set(
                jnp.where((iy_arr > n_grid - bound) & (gv_out[:, :, 1] > 0),
                          0.0, gv_out[:, :, 1]))

            # G2P
            new_x, new_v, new_C = vmap_g2p(x, gv_out)

            return (new_x, new_v, new_C, new_F)

        def simulate(rho):
            """Run full simulation, return (loss, final_x)."""
            x0 = init_x_arr
            v0 = jnp.zeros((n_p, 2))
            C0 = jnp.zeros((n_p, 2, 2))
            F0 = jnp.tile(jnp.eye(2), (n_p, 1, 1))

            def scan_fn(state, _):
                new_state = substep(state, rho)
                return new_state, None

            final, _ = jax.lax.scan(scan_fn, (x0, v0, C0, F0), None, length=n_steps)
            xf = final[0]
            tip_y = jnp.where(tip_mask, xf[:, 1], 0.0).sum() / n_tip
            return -tip_y, xf

        def loss_fn(rho):
            l, _ = simulate(rho)
            return l

        self._simulate_jit = jax.jit(simulate)
        self._loss_jit = jax.jit(loss_fn)
        self._grad_jit = jax.jit(jax.grad(loss_fn))

    def optimise(self, n_iters=None, volfrac=None, move=None, xmin=None,
                 save_callback=None):
        cfg = self.cfg
        n_iters = n_iters or cfg.n_design_iters
        volfrac = volfrac or cfg.volfrac
        move = move or cfg.move
        xmin = xmin or cfg.xmin

        rho = jnp.full(self.n_particles, volfrac)
        hist = []

        for it in range(1, n_iters + 1):
            loss_val, xf = self._simulate_jit(rho)
            loss_val = float(loss_val)
            dc = np.array(self._grad_jit(rho))

            # OC update
            dc_neg = np.maximum(-dc, 1e-30)
            rho_np = np.array(rho)
            l1, l2 = 1e-9, 1e9
            for _ in range(60):
                lmid = 0.5 * (l1 + l2)
                ratio = np.sqrt(dc_neg / np.maximum(lmid, 1e-30))
                rho_new = np.maximum(xmin,
                          np.maximum(rho_np - move,
                          np.minimum(1.0,
                          np.minimum(rho_np + move,
                                     rho_np * ratio))))
                if rho_new.mean() > volfrac:
                    l1 = lmid
                else:
                    l2 = lmid
                if (l2 - l1) / (l1 + l2 + 1e-12) < 1e-4:
                    break

            rho = jnp.array(rho_new)
            tip_y = -loss_val
            hist.append((it, loss_val, float(rho.mean()), tip_y))

            if save_callback and (it == 1 or it % 5 == 0 or it == n_iters):
                save_callback(it, loss_val, np.array(rho),
                              np.array(xf), np.array(self.init_x))

            if it <= 5 or it % 5 == 0:
                print(f"  design={it:3d}  loss={loss_val:11.5e}  "
                      f"vol={float(rho.mean()):.4f}  tip_y={tip_y:.6f}")

        return np.array(rho), np.array(hist)
