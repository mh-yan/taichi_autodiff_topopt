"""Visualisation helpers for 2D / 3D topology optimisation results."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------

def _draw_bc_2d(ax, fixed_nodes, load_nodes, F, nely, dpn=2):
    """Draw boundary condition markers on a 2D axes."""
    # Fixed supports: blue triangles on the left
    fy = np.array([n % (nely + 1) for n in fixed_nodes])
    fx = np.array([n // (nely + 1) for n in fixed_nodes])
    seen = set()
    for x, y in zip(fx, fy):
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        ax.plot(x, y, marker="<", color="#2563EB", markersize=6,
                markeredgecolor="white", markeredgewidth=0.4, zorder=5)

    # Load arrows: red arrows at load nodes
    for ln in load_nodes:
        lx = ln // (nely + 1)
        ly = ln % (nely + 1)
        fy_val = F[dpn * ln + 1] if dpn * ln + 1 < len(F) else 0
        fx_val = F[dpn * ln] if dpn * ln < len(F) else 0
        scale = 2.5
        if abs(fy_val) > 1e-12 or abs(fx_val) > 1e-12:
            ax.annotate("", xy=(lx, ly),
                        xytext=(lx - fx_val * scale, ly - fy_val * scale),
                        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.3",
                                        color="#DC2626", lw=1.8),
                        zorder=5)


def plot_density_2d(field: np.ndarray, nelx: int, nely: int,
                    title: str, path: Path,
                    active_mask: np.ndarray | None = None,
                    bc: dict | None = None, show_bc: bool = True):
    plt = _mpl()
    img = field.reshape(nelx, nely).T
    fig, ax = plt.subplots(figsize=(9, 3), dpi=160)
    im = ax.imshow(img, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0,
                   extent=[0, nelx, 0, nely], aspect="equal")
    if show_bc and bc is not None:
        _draw_bc_2d(ax, bc["fixed_nodes"], bc["load_nodes"], bc["F"], nely)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_comparison_2d(field_a: np.ndarray, field_b: np.ndarray,
                       nelx: int, nely: int, title: str, path: Path,
                       bc: dict | None = None, show_bc: bool = True):
    plt = _mpl()
    a = field_a.reshape(nelx, nely).T
    b = field_b.reshape(nelx, nely).T
    d = np.abs(a - b)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), dpi=160)
    axes[0].imshow(a, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                   extent=[0, nelx, 0, nely], aspect="equal")
    axes[0].set_title("MPM (Taichi PCG)")
    axes[1].imshow(b, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                   extent=[0, nelx, 0, nely], aspect="equal")
    axes[1].set_title("Reference SIMP")
    im = axes[2].imshow(d, origin="lower", cmap="magma",
                        vmin=0, vmax=max(1e-6, float(d.max())),
                        extent=[0, nelx, 0, nely], aspect="equal")
    axes[2].set_title("|difference|")
    if show_bc and bc is not None:
        for ax in axes[:2]:
            _draw_bc_2d(ax, bc["fixed_nodes"], bc["load_nodes"], bc["F"], nely)
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------
# 3D
# ------------------------------------------------------------------


def _draw_bc_3d(ax, fixed_nodes, load_nodes, F, nely, nelz, dpn=3):
    """Draw boundary condition markers on a 3D axes."""
    nely1 = nely + 1
    n_nodes = len(F) // dpn
    nelz1 = nelz + 1
    nxy = n_nodes // nelz1  # (nelx+1)*(nely+1)

    def _decode(nid):
        iz = nid // nxy
        rem = nid % nxy
        ix = rem // nely1
        iy = rem % nely1
        return ix, iy, iz

    # Fixed supports: only draw a sparse subset to avoid clutter
    fixed_coords = set()
    for n in fixed_nodes:
        fixed_coords.add(_decode(int(n)))
    if len(fixed_coords) > 0:
        fc = np.array(list(fixed_coords))
        # Sub-sample if too many
        if len(fc) > 60:
            step = max(1, len(fc) // 40)
            fc = fc[::step]
        ax.scatter(fc[:, 0], fc[:, 1], fc[:, 2],
                   marker="s", c="#2563EB", s=25, alpha=0.7,
                   edgecolors="white", linewidths=0.3,
                   zorder=5, label="Fixed BC")

    # Load arrows
    for ln in load_nodes:
        ix, iy, iz = _decode(int(ln))
        fx_v = float(F[dpn * ln]) if dpn * ln < len(F) else 0
        fy_v = float(F[dpn * ln + 1]) if dpn * ln + 1 < len(F) else 0
        fz_v = float(F[dpn * ln + 2]) if dpn * ln + 2 < len(F) else 0
        s = 5.0
        ax.quiver(ix, iy, iz, fx_v * s, fy_v * s, fz_v * s,
                  color="#DC2626", arrow_length_ratio=0.25,
                  linewidth=2.5, zorder=6)
    if len(load_nodes) > 0:
        lc = np.array([_decode(int(ln)) for ln in load_nodes])
        ax.scatter(lc[:, 0], lc[:, 1], lc[:, 2],
                   marker="o", c="#DC2626", s=40,
                   edgecolors="white", linewidths=0.5,
                   zorder=6, label="Load BC")


def plot_density_3d(field: np.ndarray, nelx: int, nely: int, nelz: int,
                    title: str, path: Path, threshold: float = 0.5,
                    bc: dict | None = None, show_bc: bool = True):
    plt = _mpl()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    voxels = field.reshape(nelz, nelx, nely).transpose(1, 2, 0) > threshold
    colors = np.empty(voxels.shape, dtype=object)
    flat = field.reshape(nelz, nelx, nely).transpose(1, 2, 0)
    for ix in range(nelx):
        for iy in range(nely):
            for iz in range(nelz):
                if voxels[ix, iy, iz]:
                    v = float(np.clip(flat[ix, iy, iz], 0.0, 1.0))
                    gray = 1.0 - v * 0.85
                    colors[ix, iy, iz] = (gray, gray, gray, 0.9)
    fig = plt.figure(figsize=(10, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(voxels, facecolors=colors,
              edgecolor=(0.3, 0.3, 0.3, 0.15))
    if show_bc and bc is not None:
        _draw_bc_3d(ax, bc["fixed_nodes"], bc["load_nodes"], bc["F"],
                    nely, nelz, dpn=3)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------

def plot_history(hist: np.ndarray, path: Path,
                 ref_hist: np.ndarray | None = None):
    """hist columns: iter, compliance, volume, tip_dy, pcg_res, beta, penal, pcg_iters."""
    plt = _mpl()
    n_cols = min(4, hist.shape[1])
    fig, axes = plt.subplots(n_cols, 1, figsize=(8, 2.5 * n_cols), dpi=160, sharex=True)
    if n_cols == 1:
        axes = [axes]
    labels = ["compliance", "volume", "tip_dy", "pcg_res"]
    for k in range(n_cols):
        axes[k].plot(hist[:, 0], hist[:, k + 1], linewidth=1.8, label="MPM")
        if ref_hist is not None and ref_hist.shape[1] > k + 1:
            axes[k].plot(ref_hist[:, 0], ref_hist[:, k + 1],
                         linewidth=1.4, linestyle="--", label="Ref SIMP")
            axes[k].legend()
        axes[k].set_ylabel(labels[k])
        axes[k].grid(True, alpha=0.3)
    axes[-1].set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def render_frame(field: np.ndarray, nelx: int, nely: int,
                 iteration: int, comp: float, path: Path):
    plt = _mpl()
    img = field.reshape(nelx, nely).T
    fig, ax = plt.subplots(figsize=(9, 3), dpi=140)
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1,
              extent=[0, nelx, 0, nely], aspect="equal")
    ax.set_title(f"iter={iteration}  C={comp:.3e}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def try_make_gif(frames_dir: Path, out_path: Path, fps: int = 8):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        return False, f"imageio import failed: {exc}"
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return False, "No PNG frames found"
    imageio.mimsave(out_path,
                    [imageio.imread(fp) for fp in frames],
                    duration=1.0 / max(1, fps), loop=0)
    return True, f"GIF created from {len(frames)} frames"


# ------------------------------------------------------------------
# Interactive visualisation (plotly)
# ------------------------------------------------------------------

def _plotly_bc_2d(fig, bc, nely):
    """Add BC annotations to a plotly 2D figure."""
    import plotly.graph_objects as go
    fixed_nodes = bc["fixed_nodes"]
    load_nodes = bc["load_nodes"]
    F = bc["F"]
    nely1 = nely + 1

    seen = set()
    fx_list, fy_list = [], []
    for n in fixed_nodes:
        ix = n // nely1; iy = n % nely1
        if (ix, iy) not in seen:
            seen.add((ix, iy))
            fx_list.append(ix); fy_list.append(iy)
    fig.add_trace(go.Scatter(
        x=fx_list, y=fy_list, mode="markers", name="Fixed (Dirichlet)",
        marker=dict(symbol="triangle-left", size=10, color="#2563EB",
                    line=dict(width=1, color="white")),
        hoverinfo="text",
        text=[f"Fixed node ({x},{y})" for x, y in zip(fx_list, fy_list)],
    ))

    for ln in load_nodes:
        ix = ln // nely1; iy = ln % nely1
        fy_val = F[2 * ln + 1] if 2 * ln + 1 < len(F) else 0
        fx_val = F[2 * ln] if 2 * ln < len(F) else 0
        s = 3.5
        fig.add_annotation(
            x=ix, y=iy,
            ax=ix - fx_val * s, ay=iy - fy_val * s,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=2.5,
            arrowcolor="#DC2626",
        )
    fig.add_trace(go.Scatter(
        x=[ln // nely1 for ln in load_nodes],
        y=[ln % nely1 for ln in load_nodes],
        mode="markers", name="Load (Neumann)",
        marker=dict(symbol="circle", size=8, color="#DC2626",
                    line=dict(width=1, color="white")),
        hoverinfo="text",
        text=[f"Load F=({F[2*ln]:.3f},{F[2*ln+1]:.3f})" for ln in load_nodes],
    ))


def interactive_2d(field: np.ndarray, nelx: int, nely: int,
                   title: str = "Topology", path: Path | None = None,
                   active_mask: np.ndarray | None = None,
                   bc: dict | None = None, show_bc: bool = True):
    """Open an interactive 2D density heatmap in the browser."""
    import plotly.graph_objects as go
    img = field.reshape(nelx, nely).T
    if active_mask is not None:
        mask_2d = active_mask.reshape(nelx, nely).T.astype(float)
        img = np.where(mask_2d > 0, img, np.nan)
    fig = go.Figure(data=go.Heatmap(
        z=img, colorscale="Greys", reversescale=True,
        zmin=0, zmax=1, colorbar=dict(title="density"),
    ))
    if show_bc and bc is not None:
        _plotly_bc_2d(fig, bc, nely)
    fig.update_layout(
        title=title, xaxis_title="x", yaxis_title="y",
        yaxis=dict(scaleanchor="x"), width=900, height=500,
    )
    if path:
        fig.write_html(str(path))
    fig.show()
    return fig


def _field_to_voxels(field, nelx, nely, nelz, active_mask, threshold):
    """Helper: build plotly Isosurface data from a flat density field."""
    import plotly.graph_objects as go

    vol = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=np.float64)
    cnt = np.zeros_like(vol)
    data = field.copy()
    if active_mask is not None:
        data[~active_mask] = 0.0
    for eid in range(nelx * nely * nelz):
        if data[eid] < 1e-6:
            continue
        ez = eid // (nelx * nely)
        rem = eid - ez * nelx * nely
        ex = rem // nely
        ey = rem % nely
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    vol[ex + di, ey + dj, ez + dk] += data[eid]
                    cnt[ex + di, ey + dj, ez + dk] += 1.0
    cnt[cnt == 0] = 1.0
    vol /= cnt

    gx, gy, gz = np.mgrid[0:nelx + 1, 0:nely + 1, 0:nelz + 1]
    return go.Isosurface(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(),
        value=vol.ravel(),
        isomin=threshold, isomax=1.0,
        surface_count=3,
        colorscale="Greys", reversescale=True,
        caps=dict(x_show=True, y_show=True, z_show=True),
        showscale=True, colorbar=dict(title="density"),
        opacity=0.6,
    )


def _plotly_bc_3d(fig, bc, nely, nelz):
    """Add BC annotations to a plotly 3D figure."""
    import plotly.graph_objects as go
    fixed_nodes = bc["fixed_nodes"]
    load_nodes = bc["load_nodes"]
    F = bc["F"]
    nely1 = nely + 1
    nxy_plane = len(F) // 3 // (nelz + 1)  # approximate

    seen = set()
    fx, fy, fz = [], [], []
    for n in fixed_nodes:
        iz = n // nxy_plane
        rem = n % nxy_plane
        ix = rem // nely1
        iy = rem % nely1
        key = (ix, iy, iz)
        if key not in seen:
            seen.add(key)
            fx.append(ix); fy.append(iy); fz.append(iz)
    fig.add_trace(go.Scatter3d(
        x=fx, y=fy, z=fz, mode="markers", name="Fixed (Dirichlet)",
        marker=dict(symbol="diamond", size=3, color="#2563EB",
                    line=dict(width=0.5, color="white")),
        hoverinfo="text",
        text=[f"Fixed ({x},{y},{z})" for x, y, z in zip(fx, fy, fz)],
    ))

    lx, ly, lz = [], [], []
    for ln in load_nodes:
        iz = ln // nxy_plane
        rem = ln % nxy_plane
        ix = rem // nely1
        iy = rem % nely1
        lx.append(ix); ly.append(iy); lz.append(iz)
    fig.add_trace(go.Scatter3d(
        x=lx, y=ly, z=lz, mode="markers", name="Load (Neumann)",
        marker=dict(symbol="circle", size=6, color="#DC2626",
                    line=dict(width=1, color="white")),
        hoverinfo="text",
        text=[f"Load F=({F[3*ln]:.3f},{F[3*ln+1]:.3f},{F[3*ln+2]:.3f})"
              for ln in load_nodes],
    ))

    # Draw load arrows as Cone traces
    u_arr, v_arr, w_arr = [], [], []
    for ln in load_nodes:
        u_arr.append(float(F[3*ln]))
        v_arr.append(float(F[3*ln+1]))
        w_arr.append(float(F[3*ln+2]))
    fig.add_trace(go.Cone(
        x=lx, y=ly, z=lz, u=u_arr, v=v_arr, w=w_arr,
        sizemode="absolute", sizeref=2.0, anchor="tail",
        colorscale=[[0, "#DC2626"], [1, "#DC2626"]], showscale=False,
        name="Load direction",
    ))


def interactive_3d(field: np.ndarray, nelx: int, nely: int, nelz: int,
                   title: str = "3D Topology", path: Path | None = None,
                   threshold: float = 0.3,
                   active_mask: np.ndarray | None = None,
                   bc: dict | None = None, show_bc: bool = True):
    """Interactive 3D isosurface view in the browser."""
    import plotly.graph_objects as go

    iso = _field_to_voxels(field, nelx, nely, nelz, active_mask, threshold)
    fig = go.Figure(data=iso)
    if show_bc and bc is not None:
        _plotly_bc_3d(fig, bc, nely, nelz)
    fig.update_layout(
        title=title, width=1000, height=750,
        scene=dict(
            xaxis_title="x (length)", yaxis_title="y (height)",
            zaxis_title="z (depth)", aspectmode="data",
        ),
    )
    if path:
        fig.write_html(str(path))
    fig.show()
    return fig


def interactive_comparison_3d(field_a: np.ndarray, field_b: np.ndarray,
                              nelx: int, nely: int, nelz: int,
                              title: str = "MPM vs Ref SIMP",
                              path: Path | None = None,
                              threshold: float = 0.3,
                              active_mask: np.ndarray | None = None):
    """Side-by-side 3D comparison using dual isosurfaces."""
    import plotly.graph_objects as go

    data_a = field_a.copy()
    data_b = field_b.copy()
    if active_mask is not None:
        data_a[~active_mask] = 0.0
        data_b[~active_mask] = 0.0

    def _make_scatter(data, name, offset_x=0.0):
        xs, ys, zs, vals = [], [], [], []
        for eid in range(nelx * nely * nelz):
            if data[eid] < threshold:
                continue
            ez = eid // (nelx * nely)
            rem = eid - ez * nelx * nely
            ex = rem // nely
            ey = rem % nely
            xs.append(ex + 0.5 + offset_x)
            ys.append(ey + 0.5)
            zs.append(ez + 0.5)
            vals.append(float(data[eid]))
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers", name=name,
            marker=dict(size=4, color=vals, colorscale="Greys",
                        reversescale=True, cmin=threshold, cmax=1.0,
                        opacity=0.85, line=dict(width=0)),
            text=[f"{name} ρ={v:.3f}" for v in vals],
            hoverinfo="text+x+y+z",
        )

    fig = go.Figure(data=[
        _make_scatter(data_a, "MPM", offset_x=0),
        _make_scatter(data_b, "Ref SIMP", offset_x=nelx + 5),
    ])
    fig.update_layout(
        title=title, width=1200, height=700,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                   aspectmode="data"),
    )
    if path:
        fig.write_html(str(path))
    fig.show()
    return fig



# ------------------------------------------------------------------
# Similarity metrics
# ------------------------------------------------------------------

def compute_similarity(a: np.ndarray, b: np.ndarray,
                       active_ids: np.ndarray, threshold: float = 0.5) -> dict:
    aa = np.asarray(a[active_ids], dtype=np.float64)
    bb = np.asarray(b[active_ids], dtype=np.float64)
    corr = float(np.corrcoef(aa, bb)[0, 1]) if aa.size > 1 else 1.0
    rmse = float(np.sqrt(np.mean((aa - bb) ** 2))) if aa.size else 0.0
    ma = aa > threshold; mb = bb > threshold
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    sa = int(ma.sum()); sb = int(mb.sum())
    iou = inter / max(1, union)
    dice = 2.0 * inter / max(1, sa + sb)
    return {
        "corr": corr, "rmse": rmse,
        "iou@0.5": float(iou), "dice@0.5": float(dice),
        "active_binary_mpm": sa, "active_binary_ref": sb,
        "matches_simp": bool(corr >= 0.90 and dice >= 0.80),
    }
