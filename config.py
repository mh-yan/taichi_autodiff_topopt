from dataclasses import dataclass, field
import argparse


@dataclass
class TopOptConfig:
    dim: int = 2
    nelx: int = 60
    nely: int = 20
    nelz: int = 4
    volfrac: float = 0.5
    penal_start: float = 3.0
    penal_max: float = 4.0
    penal_every: int = 25
    rmin: float = 1.5
    beta_start: float = 1.0
    beta_max: float = 8.0
    beta_every: int = 20
    eta: float = 0.5
    E0: float = 1.0
    Emin: float = 1e-9
    nu: float = 0.3
    n_iter: int = 200
    cg_maxiter: int = 2000
    cg_tol: float = 1e-8
    move: float = 0.2
    xmin: float = 1e-3
    arch: str = "cpu"
    out_dir: str = "output"
    point_cloud: str = ""
    point_spacing: float = 0.0
    save_every: int = 10
    save_frames: bool = False
    make_gif: bool = False
    gif_fps: int = 8
    show_bc: bool = True
    problem: str = "cantilever"
    stress_constraint: bool = False
    sigma_allow: float = 1.0
    stress_pnorm: int = 8
    # nonlinear MPM parameters
    nl_dt: float = 1e-4
    nl_steps: int = 256
    nl_n_grid: int = 64
    nl_gravity: float = 9.8
    nl_E: float = 500.0
    nl_ppd: int = 2
    nl_beam_x0: float = 0.05
    nl_beam_y0: float = 0.4
    nl_beam_w: float = 0.55
    nl_beam_h: float = 0.15
    nl_design_iters: int = 60

    @property
    def nel(self):
        if self.dim == 2:
            return (self.nelx, self.nely)
        return (self.nelx, self.nely, self.nelz)

    @property
    def n_elem(self):
        if self.dim == 2:
            return self.nelx * self.nely
        return self.nelx * self.nely * self.nelz

    @property
    def dpn(self):
        """DOFs per node."""
        return self.dim

    @property
    def dpe(self):
        """DOFs per element."""
        return 2**self.dim * self.dim


def parse_args() -> TopOptConfig:
    p = argparse.ArgumentParser(description="MPM topology optimization (2D/3D)")
    p.add_argument("--dim", type=int, default=2, choices=[2, 3])
    p.add_argument("--arch", type=str, default="cpu", choices=["cpu", "cuda", "vulkan"])
    p.add_argument("--nelx", type=int, default=60)
    p.add_argument("--nely", type=int, default=20)
    p.add_argument("--nelz", type=int, default=4)
    p.add_argument("--volfrac", type=float, default=0.5)
    p.add_argument("--penal-start", type=float, default=3.0)
    p.add_argument("--penal-max", type=float, default=4.0)
    p.add_argument("--penal-every", type=int, default=25)
    p.add_argument("--rmin", type=float, default=1.5)
    p.add_argument("--beta-start", type=float, default=1.0)
    p.add_argument("--beta-max", type=float, default=8.0)
    p.add_argument("--beta-every", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--E0", type=float, default=1.0)
    p.add_argument("--Emin", type=float, default=1e-9)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--cg-maxiter", type=int, default=2000)
    p.add_argument("--cg-tol", type=float, default=1e-8)
    p.add_argument("--move", type=float, default=0.2)
    p.add_argument("--xmin", type=float, default=1e-3)
    p.add_argument("--out-dir", type=str, default="output")
    p.add_argument("--point-cloud", type=str, default="")
    p.add_argument("--point-spacing", type=float, default=0.0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--make-gif", action="store_true")
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--no-show-bc", action="store_true",
                   help="Hide boundary condition markers on plots")
    p.add_argument("--problem", type=str, default="cantilever",
                   choices=["cantilever", "mbb", "lbracket", "bridge"])
    p.add_argument("--stress-constraint", action="store_true",
                   help="Enable p-norm stress constraint via augmented Lagrangian")
    p.add_argument("--sigma-allow", type=float, default=1.0,
                   help="Allowable stress limit for stress constraint")
    p.add_argument("--stress-pnorm", type=int, default=8,
                   help="Exponent for p-norm stress aggregation")
    args = p.parse_args()

    cfg = TopOptConfig(
        dim=args.dim, nelx=args.nelx, nely=args.nely, nelz=args.nelz,
        volfrac=args.volfrac, penal_start=args.penal_start,
        penal_max=args.penal_max, penal_every=args.penal_every,
        rmin=args.rmin, beta_start=args.beta_start, beta_max=args.beta_max,
        beta_every=args.beta_every, eta=args.eta, E0=args.E0, Emin=args.Emin,
        nu=args.nu, n_iter=args.iters, cg_maxiter=args.cg_maxiter,
        cg_tol=args.cg_tol, move=args.move, xmin=args.xmin, arch=args.arch,
        out_dir=args.out_dir, point_cloud=args.point_cloud,
        point_spacing=args.point_spacing, save_every=args.save_every,
        save_frames=args.save_frames, make_gif=args.make_gif,
        gif_fps=args.gif_fps,
        show_bc=not args.no_show_bc,
        problem=args.problem,
        stress_constraint=args.stress_constraint,
        sigma_allow=args.sigma_allow,
        stress_pnorm=args.stress_pnorm,
    )
    if cfg.dim == 3 and cfg.volfrac == 0.5:
        cfg.volfrac = 0.3
    return cfg
