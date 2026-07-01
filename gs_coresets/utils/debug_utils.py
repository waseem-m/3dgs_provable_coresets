# gs_coresets.utils/debug_utils.py
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union, Iterable
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from gs_coresets.utils.scene_utils import index_to_world
from gs_coresets.utils.types_devices_utils import torch_ctx

__all__ = [
    # debug plumbing
    "IS_DEBUG_MODE",
    "print_debug",
    "DebugContext",
    "maybe_enable_debugging",
    # histograms
    "AxisHist",
    "compute_gauss_histograms",
    "plot_axis_hist",
    "plot_gauss_histograms",
    "plot_gauss_histograms_stacked",
    "debug_gauss_histograms",
    # plot scene projections
    "compute_room_aabbs_from_masks_torch",
    "plot_three_projections_torch",
    "debug_scene_plots",
]

# toggle once
IS_DEBUG_MODE: bool = False
IS_DEBUG_PRINT: bool = True
IS_DEBUG_INTERACTIVE: bool = False

# ----------------------------------------------------------
# Basic debug plumbing
# ----------------------------------------------------------

def print_debug(msg: str | None = None) -> None:
    if not IS_DEBUG_MODE or not IS_DEBUG_PRINT:
        # Both IS_DEBUG_MODE and IS_PRINT_DEBUG need to be True.
        return
    if msg is None:
        print()
    else:
        print(f"[DEBUG] {msg}")


class DebugContext:
    """Context object that resets debug mode on exit."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def close(self) -> None:
        if self.enabled:
            disable_debugging()
            self.enabled = False

    def __enter__(self) -> "DebugContext":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.close()


def maybe_enable_debugging(debug: bool) -> Optional[DebugContext]:
    """
    Toggle global debug mode when requested and return a context to restore it.
    """
    if not debug:
        disable_debugging()
        return None
    enable_debugging()
    return DebugContext(True)


def enable_debugging() -> None:
    global IS_DEBUG_MODE
    IS_DEBUG_MODE = True


def disable_debugging() -> None:
    global IS_DEBUG_MODE
    IS_DEBUG_MODE = False

# ----------------------------------------------------------
# Gaussian histograms
# ----------------------------------------------------------

ArrayLike = Union["torch.Tensor", np.ndarray]

@dataclass
class AxisHist:
    name: str
    # Full-range histogram (outliers visible)
    full_counts: np.ndarray
    full_centers: np.ndarray
    full_edges: np.ndarray
    full_min: float
    full_max: float
    # Mass-window histogram (bulk visible)
    mass_counts: np.ndarray
    mass_centers: np.ndarray
    mass_edges: np.ndarray
    mass_lo: float
    mass_hi: float
    # Center = (mass_lo + mass_hi) / 2 in world units
    center: float

def _to_numpy(a: ArrayLike) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def _percentiles_torch(x: "torch.Tensor", qs: np.ndarray, sample: Optional[int]) -> np.ndarray:
    if sample is not None and x.numel() > sample:
        idx = torch.randint(0, x.numel(), (sample,), device=x.device)
        x = x[idx]
    q = torch.tensor(qs, device=x.device, dtype=x.dtype)
    vals = torch.quantile(x, q)
    return vals.detach().cpu().numpy()

def _percentiles_numpy(x: np.ndarray, qs: np.ndarray, sample: Optional[int]) -> np.ndarray:
    if sample is not None and x.size > sample:
        idx = np.random.randint(0, x.size, size=sample)
        x = x[idx]
    return np.quantile(x, qs)

def _hist_torch(x: "torch.Tensor", bins: int, mn: float, mx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import math as _math
    x = torch.nan_to_num(x.reshape(-1))
    if not _math.isfinite(mn) or not _math.isfinite(mx):
        mn, mx = float(x.min().item()), float(x.max().item())
    if mn == mx:
        mx = mn + 1e-12
    counts = torch.histc(x, bins=bins, min=mn, max=mx)
    edges = torch.linspace(mn, mx, steps=bins + 1, device=x.device, dtype=x.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts.cpu().numpy(), centers.cpu().numpy(), edges.cpu().numpy()

def _hist_numpy(x: np.ndarray, bins: int, mn: float, mx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(mn):
        mn = float(np.nanmin(x))
    if not np.isfinite(mx):
        mx = float(np.nanmax(x))
    if mn == mx:
        mx = mn + 1e-12
    counts, edges = np.histogram(x, bins=bins, range=(mn, mx))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, centers, edges

def _compute_one_axis(a: ArrayLike,
                      name: str,
                      bins_full: int,
                      bins_mass: int,
                      mass_q: tuple[float, float],
                      sample_for_quantile: Optional[int]) -> AxisHist:
    qlo, qhi = mass_q
    if isinstance(a, torch.Tensor):
        x = torch.nan_to_num(a.reshape(-1))
        mn = float(x.min().item()); mx = float(x.max().item())
        lo, hi = _percentiles_torch(x, np.array([qlo, qhi], dtype=np.float64), sample_for_quantile)
        center = float((lo + hi) / 2.0)
        full_counts, full_centers, full_edges = _hist_torch(x, bins_full, mn, mx)
        span = max(1e-12, hi - lo)
        mass_counts, mass_centers, mass_edges = _hist_torch(x, bins_mass, float(lo - 1e-6*span), float(hi + 1e-6*span))
    else:
        x = np.asarray(a).reshape(-1)
        x = np.nan_to_num(x, copy=False)
        mn = float(np.min(x)); mx = float(np.max(x))
        lo, hi = _percentiles_numpy(x, np.array([qlo, qhi], dtype=np.float64), sample_for_quantile)
        center = float((lo + hi) / 2.0)
        full_counts, full_centers, full_edges = _hist_numpy(x, bins_full, mn, mx)
        span = max(1e-12, hi - lo)
        mass_counts, mass_centers, mass_edges = _hist_numpy(x, bins_mass, float(lo - 1e-6*span), float(hi + 1e-6*span))

    return AxisHist(
        name=name,
        full_counts=full_counts, full_centers=full_centers, full_edges=full_edges, full_min=float(mn), full_max=float(mx),
        mass_counts=mass_counts, mass_centers=mass_centers, mass_edges=mass_edges, mass_lo=float(lo), mass_hi=float(hi),
        center=center,
    )

def compute_gauss_histograms(xyz: ArrayLike,
                             bins_full: int = 256,
                             bins_mass: int = 256,
                             mass_q: tuple[float, float] = (0.005, 0.995),
                             sample_for_quantile: Optional[int] = 2_000_000) -> Dict[str, AxisHist]:
    """
    Compute fast per-axis histograms of Gaussian centers.
    Returns dict {'x': AxisHist, 'y': AxisHist, 'z': AxisHist}.
    """
    if isinstance(xyz, torch.Tensor):
        assert xyz.shape[-1] == 3, "xyz must have shape (N,3)"
        hx = _compute_one_axis(xyz[:, 0], "x", bins_full, bins_mass, mass_q, sample_for_quantile)
        hy = _compute_one_axis(xyz[:, 1], "y", bins_full, bins_mass, mass_q, sample_for_quantile)
        hz = _compute_one_axis(xyz[:, 2], "z", bins_full, bins_mass, mass_q, sample_for_quantile)
    else:
        arr = np.asarray(xyz)
        assert arr.shape[-1] == 3, "xyz must have shape (N,3)"
        hx = _compute_one_axis(arr[:, 0], "x", bins_full, bins_mass, mass_q, sample_for_quantile)
        hy = _compute_one_axis(arr[:, 1], "y", bins_full, bins_mass, mass_q, sample_for_quantile)
        hz = _compute_one_axis(arr[:, 2], "z", bins_full, bins_mass, mass_q, sample_for_quantile)
    return {"x": hx, "y": hy, "z": hz}

def _draw_asymptotes(ax,
                     lo: float,
                     hi: float,
                     *,
                     pivot_value: float,
                     extra_asym_scales: Iterable[float],
                     clamp_min: float,
                     clamp_max: float,
                     color_main="tab:red",
                     color_extra="tab:purple") -> None:
    # main window
    ax.axvline(lo, color=color_main, linestyle="--", linewidth=1.0)
    ax.axvline(hi, color=color_main, linestyle="--", linewidth=1.0)
    # extra symmetric expansions around pivot (e.g., ×2 span)
    span = hi - lo
    for s in extra_asym_scales:
        if s is None:
            continue
        try:
            s = float(s)
        except Exception:
            continue
        if s <= 1.0:
            continue
        half = 0.5 * s * span
        a = max(clamp_min, pivot_value - half)
        b = min(clamp_max, pivot_value + half)
        ax.axvline(a, color=color_extra, linestyle=":", linewidth=0.9)
        ax.axvline(b, color=color_extra, linestyle=":", linewidth=0.9)

def plot_axis_hist(axis_hist: AxisHist,
                   out_path: Optional[str] = None,
                   show: bool = True,
                   *,
                   extra_asym_scales: Iterable[float] = (2.0,),
                   pivot: str = "center",      # "center" | "lo" | "hi" | float
                   clamp_to_full: bool = True) -> plt.Figure:
    """
    Fast single-axis histogram plotter (line/step for speed; log Y).
    Draws full-range curve and overlays mass-window curve.
    Also draws vertical asymptotes at mass_lo/hi and optional symmetric expansions around a pivot.
    """
    h = axis_hist
    fig, ax = plt.subplots(figsize=(8, 3))

    # Full-range (thin line)
    ax.plot(h.full_centers, h.full_counts, linewidth=1.0, label=f"{h.name}-full")

    # Mass-window (emphasized step)
    ax.step(h.mass_centers, h.mass_counts, where="mid", linewidth=1.4, label=f"{h.name}-mass")

    ax.set_yscale("log")
    ax.set_xlabel(f"{h.name.upper()} (world units)")
    ax.set_ylabel("count")
    ax.set_title(f"Gaussians along {h.name.upper()} — full vs. mass window")

    # Decide pivot value
    if isinstance(pivot, (float, int)):
        pivot_val = float(pivot)
    else:
        p = str(pivot).lower()
        if p == "center":
            pivot_val = float(h.center)
        elif p == "lo":
            pivot_val = float(h.mass_lo)
        elif p == "hi":
            pivot_val = float(h.mass_hi)
        else:
            warnings.warn(f"Unknown pivot '{pivot}', defaulting to 'center'")
            pivot_val = float(h.center)

    # Asymptotes (mass window + scaled)
    clamp_min = float(h.full_min) if clamp_to_full else float("-inf")
    clamp_max = float(h.full_max) if clamp_to_full else float("inf")
    _draw_asymptotes(
        ax, h.mass_lo, h.mass_hi,
        pivot_value=pivot_val,
        extra_asym_scales=extra_asym_scales,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

    ax.legend(loc="upper right")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=110)
    if show:
        plt.show()
    return fig

def plot_gauss_histograms(hists: Dict[str, AxisHist],
                          out_prefix: Optional[str] = None,
                          show: bool = True,
                          *,
                          extra_asym_scales: Iterable[float] = (2.0,),
                          pivot: str = "center",
                          clamp_to_full: bool = True) -> Dict[str, plt.Figure]:
    """
    Draw three separate figures (X/Y/Z); optionally save as <out_prefix>_{x,y,z}.png.
    """
    figs: Dict[str, plt.Figure] = {}
    for k in ("x", "y", "z"):
        fig = plot_axis_hist(
            hists[k], out_path=(f"{out_prefix}_{k}.png" if out_prefix else None), show=show,
            extra_asym_scales=extra_asym_scales, pivot=pivot, clamp_to_full=clamp_to_full,
        )
        figs[k] = fig
    return figs

def plot_gauss_histograms_stacked(hists: Dict[str, AxisHist],
                                  out_path: Optional[str] = None,
                                  show: bool = True,
                                  *,
                                  extra_asym_scales: Iterable[float] = (2.0,),
                                  pivot: str = "center",
                                  clamp_to_full: bool = True) -> plt.Figure:
    """
    One window with 3 stacked subplots: X on top, Y middle, Z bottom.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    for ax, k in zip(axes, ("x", "y", "z")):
        h = hists[k]
        ax.plot(h.full_centers, h.full_counts, linewidth=1.0, label=f"{k}-full")
        ax.step(h.mass_centers, h.mass_counts, where="mid", linewidth=1.4, label=f"{k}-mass")
        ax.set_yscale("log")
        ax.set_ylabel("count")
        ax.set_title(f"{k.upper()} axis")

        # Choose pivot
        if isinstance(pivot, (float, int)):
            pivot_val = float(pivot)
        else:
            p = str(pivot).lower()
            pivot_val = float(h.center if p == "center" else (h.mass_lo if p == "lo" else (h.mass_hi if p == "hi" else h.center)))

        clamp_min = float(h.full_min) if clamp_to_full else float("-inf")
        clamp_max = float(h.full_max) if clamp_to_full else float("inf")
        _draw_asymptotes(
            ax, h.mass_lo, h.mass_hi,
            pivot_value=pivot_val,
            extra_asym_scales=extra_asym_scales,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("world units")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    return fig

# --- new: convenience wrapper for Gaussian histograms ---
def debug_gauss_histograms(xyz,
                           *, low_pct: float = 0.005,
                           high_pct: float = 0.995,
                           mul_fact: float = 2.0,
                           bins_full: int = 256,
                           bins_mass: int = 256,
                           out_path: str | None = None,
                           show: bool = True):
    """
    One-call Gaussian histogram debug: compute + stacked plot.
    Returns: (hists_dict, fig)
    """
    if not IS_DEBUG_MODE or not IS_DEBUG_INTERACTIVE:
        # Both IS_DEBUG_MODE and IS_DEBUG_INTERACTIVE need to be True.
        return None, None
    hists = compute_gauss_histograms(
        xyz, bins_full=bins_full, bins_mass=bins_mass, mass_q=(low_pct, high_pct)
    )
    fig = plot_gauss_histograms_stacked(
        hists, out_path=out_path, show=show, extra_asym_scales=(mul_fact,)
    )
    return hists, fig



# ----------------------------------------------------------
# Plot 3 Scene Projections
# ----------------------------------------------------------

@torch.no_grad()
def compute_room_aabbs_from_masks_torch(
    grid_torch: Dict[str, object],
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    pad_vox: float = 0.0,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Given boolean masks over a (nx,ny,nz) grid, compute:
      • room_air_aabb  = AABB of the 'interior' (enclosed air) mask, padded by `pad_vox` voxels in world units
      • room_wall_aabb = AABB of the 'occ' (occupied/walls) mask, padded by `pad_vox` voxels in world units

    Returns:
        (room_air_aabb, room_wall_aabb) where each is (min_xyz, max_xyz) as 3D torch tensors.
    """
    occ: torch.Tensor       = grid_torch["occ"]        # bool (nx,ny,nz)
    interior: torch.Tensor  = grid_torch["interior"]   # bool (nx,ny,nz)
    nx, ny, nz = grid_torch["dims"]                    # tuple[int,int,int]

    dev = occ.device
    dt  = torch_ctx.dtype

    # ---- index-space mins/maxs for interior (air) ----
    if interior.any():
        air_idx = interior.nonzero(as_tuple=False)          # (M,3)
        air_mins = air_idx.min(dim=0).values                # (3,)
        air_maxs = air_idx.max(dim=0).values                # (3,)
    else:
        air_mins = torch.tensor([0, 0, 0], device=dev)
        air_maxs = torch.tensor([nx - 1, ny - 1, nz - 1], device=dev)

    # ---- index-space mins/maxs for occupied (walls) ----
    if occ.any():
        wall_idx = occ.nonzero(as_tuple=False)              # (K,3)
        wall_mins = wall_idx.min(dim=0).values
        wall_maxs = wall_idx.max(dim=0).values
    else:
        wall_mins = air_mins
        wall_maxs = air_maxs

    # ---- convert to world using voxel-edge convention (same as CameraGenerator) ----
    air_min_w  = index_to_world(air_mins,  aabb_min=aabb_min,  aabb_max=aabb_max, dims=(nx, ny, nz), as_numpy=False)
    air_max_w  = index_to_world(air_maxs,  aabb_min=aabb_min,  aabb_max=aabb_max, dims=(nx, ny, nz), as_numpy=False)
    wall_min_w = index_to_world(wall_mins, aabb_min=aabb_min,  aabb_max=aabb_max, dims=(nx, ny, nz), as_numpy=False)
    wall_max_w = index_to_world(wall_maxs, aabb_min=aabb_min,  aabb_max=aabb_max, dims=(nx, ny, nz), as_numpy=False)

    # ---- pad by a fraction of a voxel in *world* units (keeps CG behavior) ----
    size_w  = (aabb_max - aabb_min).to(device=dev, dtype=dt)
    # den = torch.tensor([nx, ny, nz], device=dev, dtype=dt).clamp_min(1)  # center-grid convention
    den = torch.tensor([nx - 1, ny - 1, nz - 1], device=dev, dtype=dt).clamp_min(1)  # edge-based convention
    vsize_w = size_w / den
    pad_w   = float(pad_vox) * vsize_w

    room_air_aabb  = (air_min_w  - pad_w, air_max_w  + pad_w)
    room_wall_aabb = (wall_min_w - pad_w, wall_max_w + pad_w)
    return room_air_aabb, room_wall_aabb


def plot_three_projections_torch(
    grid_torch,            # dict with "walls"/"interior"/"outside"/"dims" (torch.bool tensors)
    aabb_min, aabb_max,    # 3D torch tensors
    room_air_aabb=None,    # tuple(min,max) of 3D torch tensors or None
    room_wall_aabb=None,   # tuple(min,max) of 3D torch tensors or None
    *, draw_boxes=True, figsize=(8, 12), show=True
):
    """
    Show 3 stacked projections for a voxel grid:
      Top    : X–Y (collapsed Z)
      Middle : X–Z (collapsed Y)
      Bottom : Y–Z (collapsed X)

    Colors:
      outside = light gray, walls = black, interior = white.
    Room AABBs (if provided):
      red = interior air AABB, green = wall-shell AABB.

    Returns:
      (fig, (ax_xy, ax_xz, ax_yz), imgs_cpu) where imgs_cpu is a dict of numpy HxWx3 uint8 arrays.
    """

    g = grid_torch
    # Prefer "walls" if present; fall back to "occ" for backward-compatibility.
    occ = g["walls"] if "walls" in g else g["occ"]   # (nx,ny,nz) bool, device tensor
    interior = g["interior"]
    outside = g.get("outside", None)  # optional
    nx, ny, nz = g["dims"]
    dev = occ.device

    aabb_min_np = aabb_min.detach().cpu().numpy()
    aabb_max_np = aabb_max.detach().cpu().numpy()
    x_min, y_min, z_min = map(float, aabb_min_np)
    x_max, y_max, z_max = map(float, aabb_max_np)

    # If boxes not provided, compute them here
    if draw_boxes and (room_air_aabb is None):
        room_air_aabb, room_wall_aabb = compute_room_aabbs_from_masks_torch(
            grid_torch, aabb_min, aabb_max, pad_vox=0.0
        )

    # ---------- helpers ----------
    def build_rgb_from_masks(m_int: torch.Tensor, m_occ: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        m_int, m_occ: boolean masks (H,W) on CUDA/CPU.
        returns: (H,W,3) uint8 RGB tensor on same device
        """
        img = torch.full((H, W, 3), 220, dtype=torch.uint8, device=m_int.device)  # outside = light gray
        if m_occ.any():
            img[m_occ] = 0  # walls
        if m_int.any():
            img[m_int] = torch.tensor([255, 255, 255], dtype=torch.uint8, device=m_int.device)  # interior on top
        return img

    def draw_rect_world(ax, min_w: torch.Tensor, max_w: torch.Tensor, axes: str, color):
        """
        axes: 'xy', 'xz', or 'yz' — which 2D plane to draw the world-AABB on.
        """
        mn = min_w.detach().cpu().numpy()
        mx = max_w.detach().cpu().numpy()
        if axes == 'xy':
            x0, y0 = float(mn[0]), float(mn[1])
            x1, y1 = float(mx[0]), float(mx[1])
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
        elif axes == 'xz':
            x0, z0 = float(mn[0]), float(mn[2])
            x1, z1 = float(mx[0]), float(mx[2])
            x, y = min(x0, x1), min(z0, z1)
            w, h = abs(x1 - x0), abs(z1 - z0)
        elif axes == 'yz':
            y0, z0 = float(mn[1]), float(mn[2])
            y1, z1 = float(mx[1]), float(mx[2])
            x, y = min(y0, y1), min(z0, z1)
            w, h = abs(y1 - y0), abs(z1 - z0)
        else:
            return
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, lw=1.5, edgecolor=color))

    # XY (collapse Z) → image wants (ny, nx)
    occ_xy = occ.any(dim=2)
    int_xy = interior.any(dim=2)
    m_occ_xy = occ_xy.transpose(0, 1)
    m_int_xy = int_xy.transpose(0, 1)
    img_xy = build_rgb_from_masks(m_int_xy, m_occ_xy, H=ny, W=nx)
    extent_xy = (x_min, x_max, y_min, y_max)

    # XZ (collapse Y) → image wants (nz, nx)
    occ_xz = occ.any(dim=1)
    int_xz = interior.any(dim=1)
    m_occ_xz = occ_xz.transpose(0, 1)
    m_int_xz = int_xz.transpose(0, 1)
    img_xz = build_rgb_from_masks(m_int_xz, m_occ_xz, H=nz, W=nx)
    extent_xz = (x_min, x_max, z_min, z_max)

    # YZ (collapse X) → image wants (nz, ny)
    occ_yz = occ.any(dim=0)
    int_yz = interior.any(dim=0)
    m_occ_yz = occ_yz.transpose(0, 1)
    m_int_yz = int_yz.transpose(0, 1)
    img_yz = build_rgb_from_masks(m_int_yz, m_occ_yz, H=nz, W=ny)
    extent_yz = (y_min, y_max, z_min, z_max)

    # ---------- plot ----------
    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(3, 1, figsize=figsize)

    ax_xy.imshow(img_xy.detach().cpu().numpy(), origin='lower', extent=extent_xy, interpolation='nearest')
    ax_xy.set_title("Top (X–Y)")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_aspect('equal')

    ax_xz.imshow(img_xz.detach().cpu().numpy(), origin='lower', extent=extent_xz, interpolation='nearest')
    ax_xz.set_title("Middle (X–Z)")
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_aspect('equal')

    ax_yz.imshow(img_yz.detach().cpu().numpy(), origin='lower', extent=extent_yz, interpolation='nearest')
    ax_yz.set_title("Bottom (Y–Z)")
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.set_aspect('equal')

    if draw_boxes and (room_air_aabb is not None):
        air_min, air_max = room_air_aabb
        wall_min, wall_max = room_wall_aabb if room_wall_aabb is not None else (None, None)
        # draw "air" rect
        draw_rect_world(ax_xy, air_min, air_max, 'xy', (1.0, 0.0, 0.0))
        draw_rect_world(ax_xz, air_min, air_max, 'xz', (1.0, 0.0, 0.0))
        draw_rect_world(ax_yz, air_min, air_max, 'yz', (1.0, 0.0, 0.0))
        if wall_min is not None and wall_max is not None:
            # draw "walls" rect
            draw_rect_world(ax_xy, wall_min, wall_max, 'xy', (0.0, 0.8, 0.0))
            draw_rect_world(ax_xz, wall_min, wall_max, 'xz', (0.0, 0.8, 0.0))
            draw_rect_world(ax_yz, wall_min, wall_max, 'yz', (0.0, 0.8, 0.0))

    fig.tight_layout()
    if show:
        plt.show()

    imgs_cpu = {
        "xy": img_xy.detach().cpu().numpy(),
        "xz": img_xz.detach().cpu().numpy(),
        "yz": img_yz.detach().cpu().numpy(),
    }
    return fig, (ax_xy, ax_xz, ax_yz), imgs_cpu

# --- new: pure function that does what _debug_grid_and_plots did ---
def debug_scene_plots(grid_torch,
                      aabb_min, aabb_max,
                      *, pad_vox: float = 0.0,
                      draw_boxes: bool = True,
                      figsize=(8, 12),
                      show: bool = True):
    """
    Prepare/compute AABBs (air & wall-shell) from masks and draw the 3 projections.
    Returns: (room_air_aabb, room_wall_aabb, fig, (ax_xy, ax_xz, ax_yz), imgs_cpu)
    """
    if not IS_DEBUG_MODE or not IS_DEBUG_INTERACTIVE:
        # Both IS_DEBUG_MODE and IS_DEBUG_INTERACTIVE need to be True.
        return None, None, None, None, None
    room_air_aabb, room_wall_aabb = compute_room_aabbs_from_masks_torch(
        grid_torch, aabb_min, aabb_max, pad_vox=pad_vox
    )
    fig, axes, imgs = plot_three_projections_torch(
        grid_torch, aabb_min, aabb_max,
        room_air_aabb, room_wall_aabb,
        draw_boxes=draw_boxes, figsize=figsize, show=show
    )
    return room_air_aabb, room_wall_aabb, fig, axes, imgs
