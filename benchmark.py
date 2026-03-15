#!/usr/bin/env python3
"""Run all benchmark problems and generate comparison tables/plots.

Usage:
    python benchmark.py                 # run all benchmarks
    python benchmark.py --quick         # fewer iterations for quick test
    python benchmark.py --problems cantilever mbb   # subset
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import taichi as ti

from config import TopOptConfig
from stiffness import get_ke
from mesh import build_mesh
from particles import generate_particles_2d, generate_particles_3d, assign_particles, p2g, g2p
from filter_utils import attach_filter, schedule, oc_update
from solver import run_reference_simp


BENCHMARKS = {
    "cantilever_2d": dict(dim=2, nelx=60, nely=20, volfrac=0.5, problem="cantilever"),
    "mbb_2d":        dict(dim=2, nelx=60, nely=20, volfrac=0.5, problem="mbb"),
    "lbracket_2d":   dict(dim=2, nelx=40, nely=40, volfrac=0.4, problem="lbracket"),
    "cantilever_3d": dict(dim=3, nelx=30, nely=10, nelz=6, volfrac=0.3, problem="cantilever"),
    "bridge_3d":     dict(dim=3, nelx=30, nely=10, nelz=6, volfrac=0.3, problem="bridge"),
}


def gray_indicator(x_phys, active_ids):
    """Measure of non-discreteness: Mnd = 4/N * sum x(1-x). Ideal = 0."""
    xa = x_phys[active_ids]
    return float(4.0 / max(1, len(xa)) * np.sum(xa * (1.0 - xa)))


def run_single(name, overrides, n_iter, out_root):
    from engine import make_engine_class
    TopOptEngine = make_engine_class()

    cfg = TopOptConfig(**{k: v for k, v in overrides.items()})
    cfg.n_iter = n_iter

    if cfg.dim == 2:
        points = generate_particles_2d(cfg.nelx, cfg.nely)
    else:
        points = generate_particles_3d(cfg.nelx, cfg.nely, cfg.nelz)

    mesh = build_mesh(cfg)
    mesh = attach_filter(mesh, cfg.rmin)
    KE = get_ke(cfg.dim, cfg.E0, cfg.nu)
    p2e, counts = assign_particles(points, cfg.nelx, cfg.nely, cfg.nelz, cfg.dim)

    engine = TopOptEngine(mesh, KE, cfg.E0, cfg.Emin, cfg.penal_start, cfg.eta, cfg.beta_start)

    x = np.zeros(mesh.n_elem, dtype=np.float64)
    x[mesh.active_ids] = cfg.volfrac
    hist = []

    t0 = time.perf_counter()
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
        x = oc_update(x, dc, dv, cfg.volfrac, mesh, cfg.move, cfg.xmin, beta, cfg.eta)
        gind = gray_indicator(x_phys, mesh.active_ids)
        hist.append([it, compliance, volume, gind, pcg_iters])
    mpm_time = time.perf_counter() - t0

    engine.set_x(x); engine.forward()
    engine.solve_pcg(cfg.cg_maxiter, cfg.cg_tol)
    _, mpm_comp = engine.compute_compliance_grad()
    _, mpm_vol = engine.compute_volume_grad()
    mpm_xp = engine.get_x_phys()
    mpm_gind = gray_indicator(mpm_xp, mesh.active_ids)

    t0 = time.perf_counter()
    ref_xp, _, ref_hist, ref_comp = run_reference_simp(mesh, KE, cfg)
    ref_time = time.perf_counter() - t0
    ref_gind = gray_indicator(ref_xp, mesh.active_ids)

    aa = mpm_xp[mesh.active_ids]; bb = ref_xp[mesh.active_ids]
    corr = float(np.corrcoef(aa, bb)[0, 1]) if len(aa) > 1 else 1.0

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "history.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "compliance", "volume", "gray_indicator", "pcg_iters"])
        w.writerows(hist)

    result = dict(
        name=name, dim=cfg.dim, problem=cfg.problem,
        grid=f"{cfg.nelx}x{cfg.nely}" + (f"x{cfg.nelz}" if cfg.dim == 3 else ""),
        n_active=mesh.n_active, volfrac=cfg.volfrac,
        mpm_compliance=mpm_comp, ref_compliance=ref_comp,
        mpm_volume=mpm_vol, ref_volume=float(ref_xp[mesh.active_ids].mean()),
        mpm_gray=mpm_gind, ref_gray=ref_gind,
        correlation=corr, mpm_time_s=mpm_time, ref_time_s=ref_time,
    )
    return result, np.array(hist), ref_hist


def plot_all(results, histories, ref_histories, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Convergence comparison
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n), dpi=140, squeeze=False)
    for i, (res, hist, ref_h) in enumerate(zip(results, histories, ref_histories)):
        axes[i, 0].plot(hist[:, 0], hist[:, 1], "b-", lw=1.5, label="MPM (Taichi AD)")
        axes[i, 0].plot(ref_h[:, 0], ref_h[:, 1], "r--", lw=1.2, label="Ref SIMP")
        axes[i, 0].set_ylabel("compliance")
        axes[i, 0].set_title(res["name"])
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(hist[:, 0], hist[:, 3], "b-", lw=1.5, label="MPM")
        axes[i, 1].set_ylabel("gray indicator Mnd")
        axes[i, 1].set_title(f"{res['name']} (ideal=0)")
        axes[i, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("iteration")
    axes[-1, 1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(out_root / "convergence_comparison.png")
    plt.close(fig)

    # Summary table
    fig, ax = plt.subplots(figsize=(14, 1.0 + 0.5 * n), dpi=140)
    ax.axis("off")
    cols = ["Problem", "Grid", "V_f", "C_mpm", "C_ref", "Corr", "Mnd_mpm",
            "Mnd_ref", "t_mpm(s)", "t_ref(s)"]
    rows = []
    for r in results:
        rows.append([
            r["name"], r["grid"], f"{r['volfrac']:.2f}",
            f"{r['mpm_compliance']:.2f}", f"{r['ref_compliance']:.2f}",
            f"{r['correlation']:.6f}",
            f"{r['mpm_gray']:.4f}", f"{r['ref_gray']:.4f}",
            f"{r['mpm_time_s']:.1f}", f"{r['ref_time_s']:.1f}",
        ])
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.5)
    fig.tight_layout()
    fig.savefig(out_root / "summary_table.png")
    plt.close(fig)

    # CSV
    with (out_root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="50 iterations instead of 200")
    p.add_argument("--problems", nargs="*", default=None)
    p.add_argument("--out-dir", default="output_benchmark")
    p.add_argument("--arch", default="cpu")
    args = p.parse_args()

    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f64, unrolling_limit=0)

    n_iter = 50 if args.quick else 200
    problems = args.problems or list(BENCHMARKS.keys())
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results, histories, ref_histories = [], [], []
    for name in problems:
        if name not in BENCHMARKS:
            print(f"Unknown problem: {name}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"  Running: {name}  (iters={n_iter})")
        print(f"{'='*60}")
        res, hist, ref_h = run_single(name, BENCHMARKS[name], n_iter, out_root)
        results.append(res)
        histories.append(hist)
        ref_histories.append(ref_h)
        print(f"  MPM  C={res['mpm_compliance']:.4f}  Mnd={res['mpm_gray']:.4f}  t={res['mpm_time_s']:.1f}s")
        print(f"  Ref  C={res['ref_compliance']:.4f}  Mnd={res['ref_gray']:.4f}  t={res['ref_time_s']:.1f}s")
        print(f"  Corr={res['correlation']:.6f}")

    plot_all(results, histories, ref_histories, out_root)
    print(f"\n{'='*60}")
    print(f"All results saved to {out_root}/")
    print(f"  convergence_comparison.png  — convergence curves")
    print(f"  summary_table.png          — comparison table")
    print(f"  summary.csv                — raw data")


if __name__ == "__main__":
    main()
