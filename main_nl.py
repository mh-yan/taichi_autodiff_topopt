#!/usr/bin/env python3
"""Nonlinear MPM topology optimisation — cantilever beam benchmark.

Runs a differentiable MLS-MPM simulation with neo-Hookean constitutive
law and SIMP density interpolation.  Gravity loads a cantilever (left
wall fixed) and the optimiser maximises downward tip displacement
(= minimises compliance under self-weight).

Usage:
    python main_nl.py                              # default cantilever
    python main_nl.py --nl-gravity 20 --nl-steps 512 --nl-design-iters 80
    python main_nl.py --arch cuda                  # GPU acceleration
"""
import argparse
from pathlib import Path

import numpy as np
import taichi as ti

from config import TopOptConfig
from viz import (plot_particles_nl, plot_nl_deformation,
                 plot_nl_evolution, plot_nl_history)


def parse_nl_args() -> TopOptConfig:
    p = argparse.ArgumentParser(description="Nonlinear MPM topology optimization")
    p.add_argument("--arch", default="cpu", choices=["cpu", "cuda", "vulkan"])
    p.add_argument("--volfrac", type=float, default=0.5)
    p.add_argument("--nl-dt", type=float, default=1e-4)
    p.add_argument("--nl-steps", type=int, default=256)
    p.add_argument("--nl-n-grid", type=int, default=64)
    p.add_argument("--nl-gravity", type=float, default=9.8)
    p.add_argument("--nl-E", type=float, default=500.0)
    p.add_argument("--nl-ppd", type=int, default=2,
                   help="particles per cell per dimension")
    p.add_argument("--nl-beam-x0", type=float, default=0.05)
    p.add_argument("--nl-beam-y0", type=float, default=0.4)
    p.add_argument("--nl-beam-w", type=float, default=0.55)
    p.add_argument("--nl-beam-h", type=float, default=0.15)
    p.add_argument("--nl-design-iters", type=int, default=60)
    p.add_argument("--move", type=float, default=0.2)
    p.add_argument("--xmin", type=float, default=1e-3)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--out-dir", type=str, default="output_nl")
    args = p.parse_args()

    return TopOptConfig(
        arch=args.arch, volfrac=args.volfrac, nu=args.nu,
        move=args.move, xmin=args.xmin, out_dir=args.out_dir,
        nl_dt=args.nl_dt, nl_steps=args.nl_steps,
        nl_n_grid=args.nl_n_grid, nl_gravity=args.nl_gravity,
        nl_E=args.nl_E, nl_ppd=args.nl_ppd,
        nl_beam_x0=args.nl_beam_x0, nl_beam_y0=args.nl_beam_y0,
        nl_beam_w=args.nl_beam_w, nl_beam_h=args.nl_beam_h,
        nl_design_iters=args.nl_design_iters,
    )


def main():
    cfg = parse_nl_args()
    ti.init(arch=getattr(ti, cfg.arch), default_fp=ti.f32)

    from engine_nl import make_nl_engine
    engine = make_nl_engine(cfg)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    xlim = (0.0, cfg.nl_beam_x0 + cfg.nl_beam_w + 0.1)
    ylim = (0.0, cfg.nl_beam_y0 + cfg.nl_beam_h + 0.15)

    print("=" * 60)
    print("  Nonlinear MPM Topology Optimization")
    print("=" * 60)
    print(f"  Particles:    {engine.n_particles}")
    print(f"  Grid:         {engine.n_grid}x{engine.n_grid}")
    print(f"  Time steps:   {engine.steps}")
    print(f"  dt:           {cfg.nl_dt}")
    print(f"  E:            {cfg.nl_E},  nu: {cfg.nu}")
    print(f"  Gravity:      {cfg.nl_gravity}")
    print(f"  Design iters: {cfg.nl_design_iters}")
    print(f"  Volume frac:  {cfg.volfrac}")
    print("=" * 60)

    snapshots = []

    def save_cb(it, loss_val, rho_np, def_p, init_p):
        snapshots.append((it, rho_np.copy(), init_p.copy(), def_p.copy()))
        plot_particles_nl(init_p, rho_np,
                          f"Design iter {it}  loss={loss_val:.4f}",
                          out / f"density_{it:04d}.png",
                          xlim=xlim, ylim=ylim)
        plot_nl_deformation(init_p, def_p, rho_np,
                            f"Deformation iter {it}  loss={loss_val:.4f}",
                            out / f"deformed_{it:04d}.png",
                            xlim=xlim, ylim=ylim)

    rho_final, hist = engine.optimise(
        cfg.nl_design_iters, cfg.volfrac,
        move=cfg.move, xmin=cfg.xmin,
        save_callback=save_cb,
    )

    # Final outputs
    plot_nl_history(hist, out / "history.png")
    if snapshots:
        sel = snapshots[::max(1, len(snapshots) // 6)]
        if sel[-1][0] != snapshots[-1][0]:
            sel.append(snapshots[-1])
        plot_nl_evolution(sel, out / "evolution.png", xlim=xlim, ylim=ylim)

    np.save(out / "rho_final.npy", rho_final)
    np.save(out / "init_positions.npy", engine.init_x)

    print(f"\n{'=' * 60}")
    print(f"  Results saved to {out}/")
    print(f"  Final loss:   {hist[-1, 1]:.6f}")
    print(f"  Final volume: {hist[-1, 2]:.4f}")
    print(f"  Final tip_y:  {hist[-1, 3]:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
