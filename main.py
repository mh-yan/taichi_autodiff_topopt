#!/usr/bin/env python3
"""MPM topology optimisation – 2D / 3D cantilever beam benchmark.

Material points define the (possibly irregular) design domain.
A regular background grid provides the FEA mesh.
The Taichi PCG engine solves KU=F; a reference scipy-SIMP loop
is run afterwards for comparison.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import taichi as ti

from config import TopOptConfig, parse_args
from stiffness import get_ke
from mesh import build_mesh
from particles import (generate_particles_2d, generate_particles_3d,
                       load_point_cloud, assign_particles, rasterize_active,
                       p2g, g2p)
from filter_utils import attach_filter, schedule, oc_update
from solver import run_reference_simp
from viz import (plot_density_2d, plot_density_3d, plot_comparison_2d,
                 plot_history, render_frame, try_make_gif, compute_similarity,
                 interactive_2d, interactive_3d, interactive_comparison_3d)


def _save_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _save_summary(path: Path, d: dict):
    path.write_text("\n".join(f"{k}: {v}" for k, v in d.items()), encoding="utf-8")


# ------------------------------------------------------------------
# Main optimisation loop (MPM + Taichi PCG)
# ------------------------------------------------------------------

def run_mpm_topopt(cfg: TopOptConfig):
    ti.init(arch=getattr(ti, cfg.arch), default_fp=ti.f64, unrolling_limit=0)
    from engine import make_engine_class
    TopOptEngine = make_engine_class()

    # ---- 1. particles ------------------------------------------------
    if cfg.point_cloud:
        points = load_point_cloud(cfg.point_cloud, cfg.dim,
                                  cfg.nelx, cfg.nely, cfg.nelz)
    elif cfg.dim == 2:
        points = generate_particles_2d(cfg.nelx, cfg.nely, cfg.point_spacing)
    else:
        points = generate_particles_3d(cfg.nelx, cfg.nely, cfg.nelz,
                                       cfg.point_spacing)

    # ---- 2. background mesh ------------------------------------------
    p2e, counts = assign_particles(points, cfg.nelx, cfg.nely, cfg.nelz, cfg.dim)
    if cfg.point_cloud:
        active_mask = rasterize_active(points, cfg.nelx, cfg.nely, cfg.nelz,
                                       cfg.dim, close_iters=2)
    else:
        active_mask = None  # full domain for benchmark

    mesh = build_mesh(cfg, active_mask)
    mesh = attach_filter(mesh, cfg.rmin)

    # ---- 3. BC dict for visualisation ----------------------------------
    dpn = cfg.dim
    fixed_nodes = np.unique(mesh.fixed_dofs // dpn)
    bc = dict(fixed_nodes=fixed_nodes, load_nodes=mesh.load_nodes, F=mesh.F)

    # ---- 4. stiffness matrix & engine ---------------------------------
    KE = get_ke(cfg.dim, cfg.E0, cfg.nu)
    engine = TopOptEngine(mesh, KE, cfg.E0, cfg.Emin,
                          cfg.penal_start, cfg.eta, cfg.beta_start)

    # ---- 4. output directory ------------------------------------------
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    if cfg.save_frames or cfg.make_gif:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- 5. initialise particle & element densities -------------------
    x_p = np.full(points.shape[0], cfg.volfrac)          # particle densities
    x = p2g(x_p, p2e, counts, mesh.n_elem)               # element densities
    x[~mesh.active_mask] = 0.0

    # ---- 6. optimisation loop -----------------------------------------
    hist = []
    for it in range(1, cfg.n_iter + 1):
        beta = schedule(it, cfg.beta_start, cfg.beta_max, cfg.beta_every)
        penal = schedule(it, cfg.penal_start, cfg.penal_max, cfg.penal_every)
        engine.set_params(penal, beta)
        engine.set_x(x)
        engine.forward()

        U, pcg_iters, pcg_res = engine.solve_pcg(cfg.cg_maxiter, cfg.cg_tol)
        dc, compliance = engine.compute_compliance_grad()
        dv, volume = engine.compute_volume_grad()
        x_phys = engine.get_x_phys()

        # G2P sensitivities → OC on particles → P2G back to elements
        dc_p = g2p(dc, p2e, counts, points.shape[0])
        dv_p = g2p(dv, p2e, counts, points.shape[0])

        # OC operates on element-level (equivalent due to ratio invariance)
        x = oc_update(x, dc, dv, cfg.volfrac, mesh,
                      cfg.move, cfg.xmin, beta, cfg.eta)

        # keep particles consistent with elements
        x_p = x[p2e]

        dpn = cfg.dim
        tip_dy = float(U[dpn * mesh.load_nodes + 1].mean()) \
            if mesh.load_nodes.size else 0.0
        hist.append((it, compliance, volume, tip_dy,
                      pcg_res, beta, penal, pcg_iters))

        if it == 1 or it % cfg.save_every == 0 or it == cfg.n_iter:
            tag = (f"MPM density  it={it}  beta={beta:.1f}  "
                   f"penal={penal:.1f}  C={compliance:.3e}")
            if cfg.dim == 2:
                plot_density_2d(x_phys, cfg.nelx, cfg.nely, tag,
                                out / f"density_{it:04d}.png",
                                bc=bc, show_bc=cfg.show_bc)
                if cfg.save_frames or cfg.make_gif:
                    render_frame(x_phys, cfg.nelx, cfg.nely, it,
                                 compliance, frames_dir / f"frame_{it:04d}.png")
            else:
                plot_density_3d(x_phys, cfg.nelx, cfg.nely, cfg.nelz, tag,
                                out / f"density_{it:04d}.png",
                                bc=bc, show_bc=cfg.show_bc)
            print(f"it={it:3d}  C={compliance:11.5e}  V={volume:.4f}  "
                  f"tip_dy={tip_dy:11.5e}  pcg={pcg_iters:4d}  "
                  f"res={pcg_res:9.2e}  beta={beta:.1f}  penal={penal:.1f}")

    # ---- 7. final evaluation -----------------------------------------
    beta_f = schedule(cfg.n_iter, cfg.beta_start, cfg.beta_max, cfg.beta_every)
    penal_f = schedule(cfg.n_iter, cfg.penal_start, cfg.penal_max, cfg.penal_every)
    engine.set_params(penal_f, beta_f)
    engine.set_x(x)
    engine.forward()
    U_final, _, _ = engine.solve_pcg(cfg.cg_maxiter, cfg.cg_tol)
    _, comp_final = engine.compute_compliance_grad()
    _, vol_final = engine.compute_volume_grad()
    x_phys_final = engine.get_x_phys()

    # ---- 8. reference SIMP --------------------------------------------
    print("\nRunning reference SIMP (scipy direct solver) …")
    x_phys_ref, U_ref, hist_ref, comp_ref = run_reference_simp(mesh, KE, cfg)
    similarity = compute_similarity(x_phys_final, x_phys_ref, mesh.active_ids)

    # ---- 9. outputs ---------------------------------------------------
    hist_np = np.asarray(hist, dtype=np.float64)

    if cfg.dim == 2:
        plot_density_2d(x_phys_final, cfg.nelx, cfg.nely,
                        "Final MPM topology", out / "final_mpm.png",
                        bc=bc, show_bc=cfg.show_bc)
        plot_density_2d(x_phys_ref, cfg.nelx, cfg.nely,
                        "Final reference SIMP", out / "final_ref.png",
                        bc=bc, show_bc=cfg.show_bc)
        plot_comparison_2d(x_phys_final, x_phys_ref, cfg.nelx, cfg.nely,
                           "MPM vs Reference SIMP", out / "comparison.png",
                           bc=bc, show_bc=cfg.show_bc)
    else:
        plot_density_3d(x_phys_final, cfg.nelx, cfg.nely, cfg.nelz,
                        "Final MPM topology", out / "final_mpm.png",
                        bc=bc, show_bc=cfg.show_bc)
        plot_density_3d(x_phys_ref, cfg.nelx, cfg.nely, cfg.nelz,
                        "Final reference SIMP", out / "final_ref.png",
                        bc=bc, show_bc=cfg.show_bc)

    plot_history(hist_np[:, :5], out / "history.png", ref_hist=hist_ref)

    _save_csv(out / "history_mpm.csv",
              ["iter", "compliance", "volume", "tip_dy",
               "pcg_res", "beta", "penal", "pcg_iters"], hist)
    _save_csv(out / "history_ref.csv",
              ["iter", "compliance", "volume", "tip_dy"], hist_ref.tolist())

    gif_status = "not requested"
    if cfg.make_gif:
        ok, msg = try_make_gif(frames_dir, out / "animation.gif", cfg.gif_fps)
        gif_status = msg if ok else f"failed: {msg}"

    summary = {
        "dim": cfg.dim,
        "grid": f"{cfg.nelx}x{cfg.nely}" + (f"x{cfg.nelz}" if cfg.dim == 3 else ""),
        "particles": int(points.shape[0]),
        "active_elements": int(mesh.n_active),
        "mpm_final_compliance": float(comp_final),
        "mpm_final_volume": float(vol_final),
        "ref_final_compliance": float(comp_ref),
        "ref_final_volume": float(x_phys_ref[mesh.active_ids].mean()),
        **similarity,
        "gif": gif_status,
    }
    _save_summary(out / "summary.txt", summary)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to {out}/")

    # ---- 10. interactive visualisation (HTML) -------------------------
    print("\nGenerating interactive HTML visualisations …")
    try:
        if cfg.dim == 2:
            interactive_2d(x_phys_final, cfg.nelx, cfg.nely,
                           "MPM Final Topology (2D)",
                           path=out / "interactive_mpm.html",
                           active_mask=mesh.active_mask,
                           bc=bc, show_bc=cfg.show_bc)
        else:
            interactive_3d(x_phys_final, cfg.nelx, cfg.nely, cfg.nelz,
                           "MPM Final Topology (3D)",
                           path=out / "interactive_mpm.html",
                           active_mask=mesh.active_mask,
                           bc=bc, show_bc=cfg.show_bc)
            interactive_comparison_3d(
                x_phys_final, x_phys_ref,
                cfg.nelx, cfg.nely, cfg.nelz,
                "MPM vs Reference SIMP (3D)",
                path=out / "interactive_comparison.html",
                active_mask=mesh.active_mask)
        print(f"  -> Open {out / 'interactive_mpm.html'} in your browser")
    except Exception as e:
        print(f"  Interactive viz skipped: {e}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    run_mpm_topopt(parse_args())
