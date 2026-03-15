#!/usr/bin/env python3
"""Visualise the P2G → filter → projection → FEA → sensitivity → G2P → OC
pipeline on a small 2D problem, showing every intermediate field at each
iteration so the reader can see exactly how particles and grid interact.

Usage:
    python viz_p2g_g2p.py                       # default irregular cloud
    python viz_p2g_g2p.py --regular             # one-particle-per-element
    python viz_p2g_g2p.py --iters 8 --nelx 16 --nely 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np

from config import TopOptConfig
from stiffness import ke_2d
from mesh import build_mesh
from particles import (generate_particles_2d, assign_particles,
                       rasterize_active, p2g, g2p)
from filter_utils import (attach_filter, projection, projection_derivative,
                          schedule, oc_update)
from solver import solve_fem, compute_compliance_sensitivity


# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def _draw_grid(ax, nelx, nely, active_mask, field, cmap, vmin, vmax, title,
               show_grid_lines=True):
    """Draw element grid coloured by *field* (nelx*nely flat array)."""
    img = field.reshape(nelx, nely).T
    mask_2d = active_mask.reshape(nelx, nely).T.astype(float)
    img_masked = np.where(mask_2d > 0, img, np.nan)
    ax.imshow(img_masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
              extent=[0, nelx, 0, nely], aspect="equal", interpolation="nearest")
    if show_grid_lines:
        for ix in range(nelx + 1):
            ax.axvline(ix, color="gray", lw=0.3, alpha=0.5)
        for iy in range(nely + 1):
            ax.axhline(iy, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlim(0, nelx); ax.set_ylim(0, nely)
    ax.set_title(title, fontsize=9)


def _draw_particles(ax, points, values, cmap, vmin, vmax, title,
                    nelx, nely, active_mask, s=12):
    """Draw particles as coloured dots on top of the grid outline."""
    mask_2d = active_mask.reshape(nelx, nely).T.astype(float)
    bg = np.full_like(mask_2d, np.nan)
    bg[mask_2d > 0] = 0.95
    ax.imshow(bg, origin="lower", cmap="Greys", vmin=0, vmax=1,
              extent=[0, nelx, 0, nely], aspect="equal", interpolation="nearest")
    for ix in range(nelx + 1):
        ax.axvline(ix, color="gray", lw=0.3, alpha=0.4)
    for iy in range(nely + 1):
        ax.axhline(iy, color="gray", lw=0.3, alpha=0.4)
    sc = ax.scatter(points[:, 0], points[:, 1], c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax, s=s, edgecolors="k", linewidths=0.2,
                    zorder=5)
    ax.set_xlim(0, nelx); ax.set_ylim(0, nely)
    ax.set_title(title, fontsize=9)
    return sc


def _draw_arrows(ax, points, values, scale, cmap, vmin, vmax, title,
                 nelx, nely, active_mask):
    """Draw particles with arrow glyphs showing sensitivity magnitude."""
    _draw_particles(ax, points, np.abs(values), cmap, vmin, vmax, title,
                    nelx, nely, active_mask, s=8)
    nv = np.abs(values)
    nv_norm = nv / max(nv.max(), 1e-30)
    for i in range(len(points)):
        if nv_norm[i] > 0.05:
            ax.annotate("", xy=(points[i, 0], points[i, 1] - nv_norm[i] * scale),
                        xytext=(points[i, 0], points[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="red",
                                        lw=0.6 + 1.2 * nv_norm[i]))


# -----------------------------------------------------------------------
# Main pipeline visualisation
# -----------------------------------------------------------------------

def run_viz(nelx=12, nely=6, n_iters=5, volfrac=0.5, regular=False,
            out_dir="examples/p2g_g2p_pipeline", snapshot_every=10):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- setup ---
    cfg = TopOptConfig(dim=2, nelx=nelx, nely=nely, volfrac=volfrac,
                       rmin=1.5, penal_start=3.0, penal_max=3.0, penal_every=0,
                       beta_start=1.0, beta_max=1.0, beta_every=0, Emin=1e-9)

    if regular:
        points = generate_particles_2d(nelx, nely, spacing=0)
    else:
        rng = np.random.default_rng(42)
        n_pts = int(nelx * nely * 6)
        points = np.column_stack([
            rng.uniform(0, nelx, n_pts),
            rng.uniform(0, nely, n_pts),
        ])
        hole_mask = ~((points[:, 0] > nelx * 0.35) & (points[:, 0] < nelx * 0.65)
                      & (points[:, 1] > nely * 0.3) & (points[:, 1] < nely * 0.7))
        points = points[hole_mask]

    p2e, counts = assign_particles(points, nelx, nely, 1, 2)
    active_mask = rasterize_active(points, nelx, nely, 1, 2, close_iters=1)
    mesh = build_mesh(cfg, active_mask)
    mesh = attach_filter(mesh, cfg.rmin)
    KE = ke_2d(cfg.E0, cfg.nu)

    x_p = np.full(points.shape[0], volfrac)
    x = p2g(x_p, p2e, counts, mesh.n_elem)
    x[~mesh.active_mask] = 0.0

    cmap_dens = "YlOrRd"
    cmap_sens = "coolwarm"
    cmap_disp = "RdBu_r"

    # ---------------------------------------------------------------
    # Figure 1: Single-iteration pipeline breakdown (7 panels)
    # ---------------------------------------------------------------
    fig_pipe, axes_pipe = plt.subplots(2, 4, figsize=(20, 8), dpi=150)
    for ax in axes_pipe.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    step_labels = [
        "(a) Particles xₚ",
        "(b) P2G → x_elem",
        "(c) Filter → x̃",
        "(d) Projection → x_phys",
        "(e) FEA → displacement u",
        "(f) Sensitivity ∂C/∂x on grid",
        "(g) G2P → ∂C/∂xₚ on particles",
        "(h) OC update → new xₚ",
    ]

    # (a) initial particles
    _draw_particles(axes_pipe[0, 0], points, x_p, cmap_dens, 0, 1,
                    step_labels[0], nelx, nely, active_mask)

    # (b) P2G
    _draw_grid(axes_pipe[0, 1], nelx, nely, active_mask, x, cmap_dens, 0, 1,
               step_labels[1])

    # (c) filter
    x_tilde = np.zeros_like(x)
    x_tilde[mesh.active_ids] = (mesh.H @ x)[mesh.active_ids] / mesh.Hs[mesh.active_ids]
    _draw_grid(axes_pipe[0, 2], nelx, nely, active_mask, x_tilde, cmap_dens, 0, 1,
               step_labels[2])

    # (d) projection
    beta = cfg.beta_start
    x_phys = np.zeros_like(x)
    x_phys[mesh.active_ids] = projection(x_tilde[mesh.active_ids], beta, cfg.eta)
    _draw_grid(axes_pipe[0, 3], nelx, nely, active_mask, x_phys, cmap_dens, 0, 1,
               step_labels[3])

    # (e) FEA
    U = solve_fem(mesh, x_phys, KE, cfg.penal_start, cfg.E0, cfg.Emin)
    disp_mag = np.zeros(mesh.n_elem)
    edof_a = mesh.edof[mesh.active_ids]
    for idx, eid in enumerate(mesh.active_ids):
        ue = U[edof_a[idx]]
        disp_mag[eid] = np.sqrt(np.mean(ue[0::2]**2 + ue[1::2]**2))
    dmax = max(disp_mag.max(), 1e-30)
    _draw_grid(axes_pipe[1, 0], nelx, nely, active_mask, disp_mag / dmax,
               cmap_disp, 0, 1, step_labels[4])

    # (f) sensitivity on grid
    comp, dc_phys = compute_compliance_sensitivity(
        mesh, U, x_phys, KE, cfg.penal_start, cfg.E0, cfg.Emin)
    dproj = np.zeros_like(x)
    dproj[mesh.active_ids] = projection_derivative(x_tilde[mesh.active_ids], beta, cfg.eta)
    dc = np.asarray(mesh.H.T @ ((dc_phys * dproj) / mesh.Hs)).ravel()
    dc[~mesh.active_mask] = 0.0
    dc_abs = np.abs(dc)
    dc_max = max(dc_abs[mesh.active_ids].max(), 1e-30)
    _draw_grid(axes_pipe[1, 1], nelx, nely, active_mask, dc_abs / dc_max,
               "magma", 0, 1, step_labels[5])

    # (g) G2P sensitivity to particles
    dc_p = g2p(dc, p2e, counts, points.shape[0])
    dc_p_abs = np.abs(dc_p)
    dc_p_max = max(dc_p_abs.max(), 1e-30)
    _draw_particles(axes_pipe[1, 2], points, dc_p_abs / dc_p_max, "magma", 0, 1,
                    step_labels[6], nelx, nely, active_mask, s=10)

    # (h) OC update → new particle densities
    dv = np.asarray(mesh.H.T @ (dproj / mesh.Hs)).ravel()
    dv[~mesh.active_mask] = 0.0
    x_new = oc_update(x, dc, dv, volfrac, mesh, cfg.move, cfg.xmin, beta, cfg.eta)
    x_p_new = x_new[p2e]
    _draw_particles(axes_pipe[1, 3], points, x_p_new, cmap_dens, 0, 1,
                    step_labels[7], nelx, nely, active_mask)

    # arrows between panels
    for row in range(2):
        for col in range(3):
            ax_from = axes_pipe[row, col]
            ax_to = axes_pipe[row, col + 1]
            fig_pipe.add_artist(FancyArrowPatch(
                posA=(ax_from.get_position().x1 + 0.005, ax_from.get_position().y0 +
                      ax_from.get_position().height / 2),
                posB=(ax_to.get_position().x0 - 0.005, ax_to.get_position().y0 +
                      ax_to.get_position().height / 2),
                transform=fig_pipe.transFigure, arrowstyle="->,head_width=6,head_length=4",
                color="#2563EB", lw=2, zorder=10))
    # row transition arrow
    ax_end = axes_pipe[0, 3]
    ax_start2 = axes_pipe[1, 0]
    fig_pipe.add_artist(FancyArrowPatch(
        posA=(ax_end.get_position().x1 - ax_end.get_position().width / 2,
              ax_end.get_position().y0 - 0.01),
        posB=(ax_start2.get_position().x0 + ax_start2.get_position().width / 2,
              ax_start2.get_position().y1 + 0.01),
        transform=fig_pipe.transFigure, arrowstyle="->,head_width=6,head_length=4",
        connectionstyle="arc3,rad=0.3", color="#2563EB", lw=2, zorder=10))

    fig_pipe.suptitle("MPM Topology Optimization: Single-Iteration Pipeline",
                      fontsize=14, fontweight="bold", y=0.98)
    fig_pipe.tight_layout(rect=[0, 0, 1, 0.95])
    fig_pipe.savefig(out / "pipeline_single_iteration.png")
    plt.close(fig_pipe)
    print(f"  Saved {out / 'pipeline_single_iteration.png'}")

    # ---------------------------------------------------------------
    # Figure 2: Multi-iteration evolution (particles + grid + topology)
    # ---------------------------------------------------------------
    x_p = np.full(points.shape[0], volfrac)
    x = p2g(x_p, p2e, counts, mesh.n_elem)
    x[~mesh.active_mask] = 0.0

    iters_to_show = [0] + list(range(snapshot_every, n_iters + 1, snapshot_every))
    if iters_to_show[-1] != n_iters:
        iters_to_show.append(n_iters)
    n_show = len(iters_to_show)
    fig_evo, axes_evo = plt.subplots(3, n_show, figsize=(3.2 * n_show, 9), dpi=150)
    for ax in axes_evo.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    row_labels = ["Particles xₚ", "Grid x_phys (after filter+proj)", "Topology (threshold 0.5)"]
    for r, label in enumerate(row_labels):
        axes_evo[r, 0].set_ylabel(label, fontsize=9, fontweight="bold")

    pt_size = max(2, min(10, 800.0 / len(points)))
    def _snapshot(col, it_label, x_p_cur, x_phys_cur):
        _draw_particles(axes_evo[0, col], points, x_p_cur, cmap_dens, 0, 1,
                        f"iter {it_label}", nelx, nely, active_mask, s=pt_size)
        _draw_grid(axes_evo[1, col], nelx, nely, active_mask, x_phys_cur,
                   "gray_r", 0, 1, f"iter {it_label}", show_grid_lines=False)
        topo = np.where(x_phys_cur > 0.5, 1.0, 0.0)
        topo[~active_mask] = np.nan
        _draw_grid(axes_evo[2, col], nelx, nely, active_mask, x_phys_cur,
                   "gray_r", 0, 1, f"iter {it_label}", show_grid_lines=False)

    col = 0
    x_phys_init = np.zeros_like(x)
    x_phys_init[mesh.active_ids] = projection(
        (mesh.H @ x)[mesh.active_ids] / mesh.Hs[mesh.active_ids], beta, cfg.eta)
    _snapshot(col, 0, x_p, x_phys_init)
    col += 1

    for it in range(1, n_iters + 1):
        penal = cfg.penal_start
        x_tilde_it = np.zeros_like(x)
        x_tilde_it[mesh.active_ids] = (mesh.H @ x)[mesh.active_ids] / mesh.Hs[mesh.active_ids]
        x_phys_it = np.zeros_like(x)
        x_phys_it[mesh.active_ids] = projection(x_tilde_it[mesh.active_ids], beta, cfg.eta)
        U_it = solve_fem(mesh, x_phys_it, KE, penal, cfg.E0, cfg.Emin)
        _, dc_phys_it = compute_compliance_sensitivity(
            mesh, U_it, x_phys_it, KE, penal, cfg.E0, cfg.Emin)
        dproj_it = np.zeros_like(x)
        dproj_it[mesh.active_ids] = projection_derivative(x_tilde_it[mesh.active_ids], beta, cfg.eta)
        dc_it = np.asarray(mesh.H.T @ ((dc_phys_it * dproj_it) / mesh.Hs)).ravel()
        dc_it[~mesh.active_mask] = 0.0
        dv_it = np.asarray(mesh.H.T @ (dproj_it / mesh.Hs)).ravel()
        dv_it[~mesh.active_mask] = 0.0
        x = oc_update(x, dc_it, dv_it, volfrac, mesh, cfg.move, cfg.xmin, beta, cfg.eta)
        x_p = x[p2e]

        if it in iters_to_show:
            x_tilde_s = np.zeros_like(x)
            x_tilde_s[mesh.active_ids] = (mesh.H @ x)[mesh.active_ids] / mesh.Hs[mesh.active_ids]
            x_phys_s = np.zeros_like(x)
            x_phys_s[mesh.active_ids] = projection(x_tilde_s[mesh.active_ids], beta, cfg.eta)
            _snapshot(col, it, x_p, x_phys_s)
            col += 1

    fig_evo.suptitle("MPM Topology Optimization: Iteration-by-Iteration Evolution",
                     fontsize=14, fontweight="bold", y=0.99)
    fig_evo.tight_layout(rect=[0, 0, 1, 0.96])
    fig_evo.savefig(out / "evolution_iterations.png")
    plt.close(fig_evo)
    print(f"  Saved {out / 'evolution_iterations.png'}")

    # ---------------------------------------------------------------
    # Figure 3: P2G / G2P detail — zoom into a few cells
    # ---------------------------------------------------------------
    fig_detail, axes_d = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    zoom_x = (nelx // 3, nelx // 3 + 4)
    zoom_y = (nely // 3, nely // 3 + 3)

    for ax in axes_d:
        ax.set_xlim(zoom_x[0], zoom_x[1])
        ax.set_ylim(zoom_y[0], zoom_y[1])
        ax.set_aspect("equal")
        for ix in range(zoom_x[0], zoom_x[1] + 1):
            ax.axvline(ix, color="gray", lw=0.8, alpha=0.6)
        for iy in range(zoom_y[0], zoom_y[1] + 1):
            ax.axhline(iy, color="gray", lw=0.8, alpha=0.6)

    in_zoom = ((points[:, 0] >= zoom_x[0]) & (points[:, 0] <= zoom_x[1]) &
               (points[:, 1] >= zoom_y[0]) & (points[:, 1] <= zoom_y[1]))
    zp = points[in_zoom]
    zv = x_p[in_zoom]

    # Panel 1: particles with density values annotated
    axes_d[0].scatter(zp[:, 0], zp[:, 1], c=zv, cmap=cmap_dens, vmin=0, vmax=1,
                      s=60, edgecolors="k", linewidths=0.5, zorder=5)
    for i in range(len(zp)):
        axes_d[0].annotate(f"{zv[i]:.2f}", (zp[i, 0], zp[i, 1]),
                           fontsize=6, ha="center", va="bottom",
                           xytext=(0, 4), textcoords="offset points")
    axes_d[0].set_title("(a) Particles with density xₚ", fontsize=10)

    # Panel 2: P2G — grid cells coloured by averaged density
    x_elem = p2g(x_p, p2e, counts, mesh.n_elem)
    for ex in range(zoom_x[0], zoom_x[1]):
        for ey in range(zoom_y[0], zoom_y[1]):
            eid = ex * nely + ey
            if not active_mask[eid]:
                continue
            val = x_elem[eid]
            c = plt.cm.YlOrRd(val)
            rect = Rectangle((ex, ey), 1, 1, facecolor=c, edgecolor="gray", lw=0.8)
            axes_d[1].add_patch(rect)
            axes_d[1].text(ex + 0.5, ey + 0.5, f"{val:.3f}",
                           ha="center", va="center", fontsize=7, fontweight="bold")
            cnt = counts[eid]
            axes_d[1].text(ex + 0.5, ey + 0.15, f"n={cnt}",
                           ha="center", va="center", fontsize=6, color="blue")
    axes_d[1].scatter(zp[:, 0], zp[:, 1], c="white", s=15, edgecolors="k",
                      linewidths=0.3, zorder=5, alpha=0.5)
    axes_d[1].set_title("(b) P2G: x_elem = mean(xₚ in cell)", fontsize=10)

    # Panel 3: G2P — sensitivity arrows on particles
    dc_elem_last = dc.copy()
    dc_p_last = g2p(dc_elem_last, p2e, counts, points.shape[0])
    zdc = dc_p_last[in_zoom]
    zdc_abs = np.abs(zdc)
    zdc_max = max(zdc_abs.max(), 1e-30)
    for ex in range(zoom_x[0], zoom_x[1]):
        for ey in range(zoom_y[0], zoom_y[1]):
            eid = ex * nely + ey
            if not active_mask[eid]:
                continue
            val = np.abs(dc_elem_last[eid])
            c = plt.cm.magma(val / max(dc_abs[mesh.active_ids].max(), 1e-30))
            rect = Rectangle((ex, ey), 1, 1, facecolor=c, edgecolor="gray",
                              lw=0.8, alpha=0.3)
            axes_d[2].add_patch(rect)
    sc3 = axes_d[2].scatter(zp[:, 0], zp[:, 1], c=zdc_abs / zdc_max,
                            cmap="magma", vmin=0, vmax=1, s=50,
                            edgecolors="k", linewidths=0.3, zorder=5)
    for i in range(len(zp)):
        axes_d[2].annotate(f"{zdc[i]:.1f}", (zp[i, 0], zp[i, 1]),
                           fontsize=5, ha="center", va="bottom",
                           xytext=(0, 4), textcoords="offset points", color="red")
    axes_d[2].set_title("(c) G2P: ∂C/∂xₚ = ∂C/∂x_elem / count", fontsize=10)

    fig_detail.suptitle("P2G / G2P Transfer Detail (zoomed view)",
                        fontsize=13, fontweight="bold")
    fig_detail.tight_layout(rect=[0, 0, 1, 0.93])
    fig_detail.savefig(out / "p2g_g2p_detail.png")
    plt.close(fig_detail)
    print(f"  Saved {out / 'p2g_g2p_detail.png'}")

    print(f"\nAll figures saved to {out}/")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nelx", type=int, default=12)
    p.add_argument("--nely", type=int, default=6)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--volfrac", type=float, default=0.5)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--regular", action="store_true")
    p.add_argument("--out-dir", default="examples/p2g_g2p_pipeline")
    args = p.parse_args()
    run_viz(args.nelx, args.nely, args.iters, args.volfrac, args.regular,
            args.out_dir, args.snapshot_every)
