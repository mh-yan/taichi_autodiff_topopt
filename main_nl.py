#!/usr/bin/env python3
"""Nonlinear MPM topology optimisation with JAX auto-differentiation.

Runs a differentiable MLS-MPM simulation with neo-Hookean constitutive
law and SIMP density interpolation, using JAX for end-to-end gradients.

Usage:
    python main_nl.py
    python main_nl.py --gravity 50 --E 80 --n-steps 1024 --n-iters 80
"""
import argparse
from pathlib import Path

import numpy as np

from engine_nl_jax import NLConfig, NLEngine, generate_irregular_cloud
from viz import (plot_particles_nl, plot_nl_deformation,
                 plot_nl_evolution, plot_nl_history)


def main():
    p = argparse.ArgumentParser(description="Nonlinear MPM topology optimization (JAX)")
    p.add_argument("--n-grid", type=int, default=64)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--gravity", type=float, default=30.0)
    p.add_argument("--E", type=float, default=100.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--ppd", type=int, default=2)
    p.add_argument("--beam-x0", type=float, default=0.05)
    p.add_argument("--beam-y0", type=float, default=0.40)
    p.add_argument("--beam-w", type=float, default=0.50)
    p.add_argument("--beam-h", type=float, default=0.12)
    p.add_argument("--volfrac", type=float, default=0.5)
    p.add_argument("--n-iters", type=int, default=60)
    p.add_argument("--move", type=float, default=0.2)
    p.add_argument("--out-dir", type=str, default="output_nl")
    args = p.parse_args()

    cfg = NLConfig(
        n_grid=args.n_grid, dt=args.dt, n_steps=args.n_steps,
        gravity=args.gravity, E=args.E, nu=args.nu, ppd=args.ppd,
        beam_x0=args.beam_x0, beam_y0=args.beam_y0,
        beam_w=args.beam_w, beam_h=args.beam_h,
        volfrac=args.volfrac, move=args.move,
        n_design_iters=args.n_iters,
    )

    eng = NLEngine(cfg)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    xlim = (0.0, cfg.beam_x0 + cfg.beam_w + 0.1)
    ylim = (0.0, cfg.beam_y0 + cfg.beam_h + 0.2)

    print("=" * 60)
    print("  Nonlinear MPM Topology Optimization (JAX AD)")
    print("=" * 60)
    print(f"  Particles:    {eng.n_particles}")
    print(f"  Grid:         {cfg.n_grid}x{cfg.n_grid}")
    print(f"  Time steps:   {cfg.n_steps}")
    print(f"  E={cfg.E}  nu={cfg.nu}  gravity={cfg.gravity}")
    print(f"  Design iters: {cfg.n_design_iters}")
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

    rho_final, hist = eng.optimise(save_callback=save_cb)

    plot_nl_history(hist, out / "history.png")
    if snapshots:
        sel = snapshots[::max(1, len(snapshots) // 6)]
        if sel[-1][0] != snapshots[-1][0]:
            sel.append(snapshots[-1])
        plot_nl_evolution(sel, out / "evolution.png", xlim=xlim, ylim=ylim)

    np.save(out / "rho_final.npy", rho_final)
    print(f"\nResults saved to {out}/")
    print(f"  Final loss:   {hist[-1, 1]:.6f}")
    print(f"  Final volume: {hist[-1, 2]:.4f}")
    print(f"  Final tip_y:  {hist[-1, 3]:.6f}")


if __name__ == "__main__":
    main()
