# gs_coresets.utils/scene_utils.py
"""
Scene utilities: AABB defaults + grid/index<->world mappings + sampling helpers.

This module consolidates:
  - AABB normalization parameters
  - Grid/voxel mapping helpers
  - Fibonacci sphere sampling
  - A simple grid size heuristic (kept minimal to avoid surprises)

Conventions:
  • "Edge-based" indexing: voxel edges are at integer indices in [0, nx] (inclusive).
    - Centers are at i + 0.5 for i in [0, nx-1].
  • World<->index helpers are provided in NumPy and Torch variants.

All functions avoid importing heavy modules (e.g., CameraGenerator) to prevent circular deps.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np
import torch

# Import all low-level type/device helpers
from gs_coresets.utils.types_devices_utils import (
    as_vec3_numpy,
    as_vec3_torch,
    as_dims3,
    torch_ctx
)

"""
Scene params:
# Single source of truth for scene AABB normalization / trimming parameters.
#
# Anything that needs the "how do we shrink/expand the scene bbox to ignore outliers?"
# should import from here rather than pulling from camera_generator.
#
# This keeps low-level gs_coresets.utils independent and avoids heavy imports / circular deps.
"""

@dataclass(frozen=True)
class AABBParams:
    """Parameters controlling scene AABB normalization."""
    pct_bounds: Tuple[float, float]  # (low_pct, high_pct) in [0,1]
    mul_fact: float                  # multiplicative expansion factor (>0)

"""
# --- DEFAULTS ---
# pct_bounds[0] = Ignore outliers by looking at BBOX starting from 0.01% of Gaussian's mean placement.
# pct_bounds[1] = Ignore outliers by looking at BBOX ending at 99.99% of Gaussian's mean placement.
# mul_fact = Multiplication factor to increase BBOX size around its center.
"""
AABB_DEFAULTS = AABBParams(
    pct_bounds=(0.0001, 0.9999),
    mul_fact=1.20,
)


def clamp_percentiles(p: Tuple[float, float]) -> Tuple[float, float]:
    """Clamp a (low, high) percentile pair into [0,1] and keep them ordered."""
    lo, hi = p
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


"""
Grid gs_coresets.utils:
# Unified grid + geometry helpers shared by camera_generator and classify_scene_utils
#
# • We expose ONE canonical mapping pair that matches CameraGenerator precision:  --  (dims = (nx, ny, nz))
#     - world_to_index(...): edge-based, returns integer indices (rounding), uses (dims - 1).
#     - index_to_world(...): edge-based, returns Torch (default) or NumPy, uses (dims - 1).
#
# Debug/diagnostic code should switch to the edge-based pair. Center-based math (+0.5/dims) is
# intentionally removed to ensure identical behavior across gs_coresets.modules.
#
"""

# -----------------------------------------------------------------------------
# Robust AABB (quantile-based, symmetric expansion around mid)
# -----------------------------------------------------------------------------
@torch.no_grad()
def robust_aabb(
    xyz: torch.Tensor,
    pct_low: float = 0.005,
    pct_high: float = 0.995,
    mul_fact: float = 1.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust axis-aligned bounding box that ignores extreme outliers by
    taking per-axis quantiles and symmetrically expanding them around the midpoint.
    Returns (lo, hi) as float tensors on xyz.device/dtype.
    """
    pct_lo = torch.quantile(xyz, float(pct_low), dim=0)
    pct_hi = torch.quantile(xyz, float(pct_high), dim=0)
    center = (pct_lo + pct_hi) / 2.0
    expand_lo = center + (pct_lo - center) * float(mul_fact)
    expand_hi = center + (pct_hi - center) * float(mul_fact)
    lo = torch.minimum(expand_lo, expand_hi)
    hi = torch.maximum(expand_lo, expand_hi)
    return lo, hi


# -----------------------------------------------------------------------------
# Grid sizing (legacy log-scale rule used by CameraGenerator)
# -----------------------------------------------------------------------------
@torch.no_grad()
def grid_size_from_scene_logscale(
    N: int,
    min_dim: int = 64,
    max_dim: int = 256,
    log_mul: int = 8,
) -> tuple[int, int, int]:
    """Legacy sizing used by CameraGenerator:
        base = round(min_dim + log_mul * log10(max(N, 10)))
        base = clamp(base, 48 .. max_dim)
        return (base, base, base)
    """
    eff_N = max(10.0, float(N))
    eff_log_N = math.log10(eff_N)
    base = int(round(min_dim + log_mul * eff_log_N))  # e.g., 64 + 8*log10(N)
    base = int(min(max_dim, max(48, base)))
    return base, base, base


# -----------------------------------------------------------------------------
# Canonical edge-based world <-> index (exact math used by CameraGenerator)
# -----------------------------------------------------------------------------
@torch.no_grad()
def voxel_size_world(
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    dims: Tuple[int, int, int],
) -> torch.Tensor:
    """Per-axis voxel edge length in world units using (dims - 1)."""
    nx, ny, nz = dims
    dev, dt = aabb_min.device, aabb_min.dtype
    denom = torch.tensor([max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)], device=dev, dtype=dt)
    return (aabb_max - aabb_min) / denom


@torch.no_grad()
def world_to_index(
    pos_w: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    dims: Tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World -> integer index (edge-based):
        rel = (world - aabb_min) / (aabb_max - aabb_min)
        idx = round(rel * (dims - 1))
    Returns (ix, iy, iz) long tensors on aabb_min.device.
    """
    nx, ny, nz = dims
    dev, dt = aabb_min.device, aabb_min.dtype
    size = (aabb_max - aabb_min).to(device=dev, dtype=dt).clamp_min(1e-12)
    rel = (pos_w.to(device=dev, dtype=dt) - aabb_min) / size
    ix = (rel[..., 0] * (nx - 1)).round().long()
    iy = (rel[..., 1] * (ny - 1)).round().long()
    iz = (rel[..., 2] * (nz - 1)).round().long()
    return ix, iy, iz


@torch.no_grad()
def index_to_world(
    ix: torch.Tensor | int | np.ndarray,
    iy: Optional[torch.Tensor | int] = None,
    iz: Optional[torch.Tensor | int] = None,
    aabb_min: torch.Tensor = None,
    aabb_max: torch.Tensor = None,
    dims: Tuple[int, int, int] = (1, 1, 1),
    *,
    as_numpy: bool = True,
) -> torch.Tensor | np.ndarray:
    """Index -> world (edge-based) unified API.

    Accepts either:
      • (ix, iy, iz) as scalars/tensors, OR
      • ix as a tensor of shape (...,3) with iy=iz=None.

    Set as_numpy=True to return a NumPy array (for legacy camera_generator call-sites).
    Set as_numpy=False to return a PyTorch tensor (for legacy debug_utils call-sites).
    """
    nx, ny, nz = dims
    if iy is None and iz is None:
        # Vector-of-3 tensor path
        idx = torch.as_tensor(ix)
        dev, dt = (aabb_min.device if isinstance(aabb_min, torch.Tensor) else torch_ctx.device), (
            aabb_min.dtype if isinstance(aabb_min, torch.Tensor) else torch_ctx.dtype)
        idx = idx.to(device=dev, dtype=torch_ctx.dtype)
        denom = torch.tensor([max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)], device=dev, dtype=torch_ctx.dtype)
        rel = idx / denom
        out = aabb_min.to(device=dev, dtype=dt) + rel.to(dtype=dt) * (aabb_max - aabb_min)
        return out.detach().cpu().numpy() if as_numpy else out

    # Scalar/tensor triplet path
    if as_numpy:
        a_min = aabb_min.detach().cpu().numpy()
        a_max = aabb_max.detach().cpu().numpy()
        rel = np.array([
            float(ix) / max(nx - 1, 1),
            float(iy) / max(ny - 1, 1),
            float(iz) / max(nz - 1, 1),
        ], dtype=np.float32)
        return a_min + rel * (a_max - a_min)

    # Torch triplet
    dev, dt = aabb_min.device, aabb_min.dtype
    ix_t = torch.as_tensor(ix, device=dev, dtype=torch_ctx.dtype)
    iy_t = torch.as_tensor(iy, device=dev, dtype=torch_ctx.dtype)
    iz_t = torch.as_tensor(iz, device=dev, dtype=torch_ctx.dtype)
    denom = torch.tensor([max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)], device=dev, dtype=torch_ctx.dtype)
    rel = torch.stack([ix_t, iy_t, iz_t], dim=-1) / denom
    return aabb_min.to(device=dev, dtype=dt) + rel.to(dtype=dt) * (aabb_max - aabb_min)


# -----------------------------------------------------------------------------
# Simple index clamping helpers
# -----------------------------------------------------------------------------
def clamp_indices_np(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamp 3D integer grid indices into [0..dim-1] along each axis (NumPy)."""
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)
    return ix, iy, iz


@torch.no_grad()
def clamp_indices_torch(
    ix: torch.Tensor,
    iy: torch.Tensor,
    iz: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamp 3D integer grid indices into [0..dim-1] along each axis (Torch)."""
    ix = ix.clamp(0, nx - 1)
    iy = iy.clamp(0, ny - 1)
    iz = iz.clamp(0, nz - 1)
    return ix, iy, iz


# -----------------------------------------------------------------------------
# Shared geometry helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def fibonacci_sphere(n: int, device: torch.device, dtype: torch.dtype = torch_ctx.dtype) -> torch.Tensor:
    """Even-ish points on the unit sphere using a Fibonacci spiral.
    Returns (n,3) float tensor on the requested device.
    Matches prior implementations exactly (x,z from cos/sin; y linear).
    (Same sampler used by CameraGenerator.)
    """
    if n <= 0:
        return torch.zeros((0, 3), device=device, dtype=dtype)
    k = torch.arange(n, device=device, dtype=dtype) + 0.5
    phi = math.pi * (1.0 + 5 ** 0.5)
    z = 1.0 - 2.0 * k / float(n)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = phi * k
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    points = torch.stack([x, y, z], dim=-1)
    return points


@torch.no_grad()
def ray_aabb_exit_lengths(
    center: torch.Tensor,                       # (3,)
    dirs: torch.Tensor,                         # (B,3) unit-ish
    lo: torch.Tensor,                           # (3,)
    hi: torch.Tensor,                           # (3,)
) -> torch.Tensor:
    """Distance from ray origin `center` to exiting the AABB [lo,hi] along each dir.
    Returns (B,) with non-negative distances."""
    eps = 1e-9
    c = center.view(1, 3)
    d = dirs.clamp(min=-1.0, max=1.0)
    inv = torch.where(d.abs() < eps, torch.full_like(d, float('inf')), 1.0 / d)
    t1 = (lo.view(1, 3) - c) * inv
    t2 = (hi.view(1, 3) - c) * inv
    tmax = torch.maximum(t1, t2)  # (B,3)
    r_exit = torch.minimum(torch.minimum(tmax[:, 0], tmax[:, 1]), tmax[:, 2])
    return torch.clamp(r_exit, min=0.0)



# __all__ = [
#     # Scene params
#     "AABBParams",
#     "AABB_DEFAULTS",
#     "clamp_percentiles",
#     # AABB
#     "robust_aabb",
#     # Grid sizing
#     "grid_size_from_scene_logscale",
#     # Canonical edge-based transforms
#     "world_to_index",
#     "index_to_world",
#     "voxel_size_world",
#     # Back-compat shims (kept to avoid churn in callers)
#     "index_to_world",
#     # Misc geometry/gs_coresets.utils
#     "clamp_indices_np",
#     "clamp_indices_torch",
#     "fibonacci_sphere",
#     "ray_aabb_exit_lengths",
# ]

if __name__ == "__main__":
    def _world_to_index_vec(pos_w, aabb_min, aabb_max, dims):
        ix, iy, iz = world_to_index(pos_w, aabb_min, aabb_max, dims)
        return torch.stack([ix, iy, iz], dim=-1)

    def _self_test_edge_based():
        dev = torch_ctx.device
        dt = torch_ctx.dtype
        tol = torch_ctx.denom_eps

        a0 = torch.tensor([-3.0, 0.0, -8.0], device=dev, dtype=dt)
        a1 = torch.tensor([5.0, 6.0, 7.0], device=dev, dtype=dt)
        dims = (8, 12, 15)

        v = voxel_size_world(a0, a1, dims)  # uses (dims - 1)
        den = torch.tensor([d - 1 for d in dims], device=dev, dtype=dt)

        # 1) Edge-spacing identity
        assert torch.allclose(a0 + v * den, a1, rtol=tol, atol=tol)

        # 2) Min/max map to edge indices 0 and (dims-1)
        idx0 = _world_to_index_vec(a0, aabb_min=a0, aabb_max=a1, dims=dims)
        idx1 = _world_to_index_vec(a1, aabb_min=a0, aabb_max=a1, dims=dims)

        expect0 = torch.zeros(3, device=dev, dtype=idx0.dtype)
        expect1 = den.to(idx1.dtype)

        assert torch.equal(idx0, expect0)  # 0,0,0
        assert torch.equal(idx1, expect1)  # nx-1,ny-1,nz-1

        # 3) Round-trip check
        x0 = index_to_world(idx0, aabb_min=a0, aabb_max=a1, dims=dims, as_numpy=False)  # vector path (...,3)
        x1 = index_to_world(idx1, aabb_min=a0, aabb_max=a1, dims=dims, as_numpy=False)

        assert torch.allclose(x0, a0, rtol=tol, atol=tol)
        assert torch.allclose(x1, a1, rtol=tol, atol=tol)

        # 4) Type/device hygiene
        assert x0.dtype == dt and x1.dtype == dt
        assert x0.device == a0.device and x1.device == a1.device
        assert v.dtype == dt and v.device == a0.device

        # 5) Tie behavior (exactly midway between edges)
        # world_to_index should match your chosen policy inside world_to_index()
        mid = (a0 + a1) * 0.5
        idx_mid = _world_to_index_vec(mid, aabb_min=a0, aabb_max=a1, dims=dims)
        # If you use torch.round on edge coords: indices should be close to (dims-1)/2
        expect_mid = (den.to(idx_mid.dtype) / 2)
        # Allow one-off on odd sizes due to rounding semantics
        assert torch.all((idx_mid - expect_mid).abs() <= 0.5)

        # 6) Batched inputs path
        P = 5
        pts = torch.linspace(0, 1, P, device=dev, dtype=dt).unsqueeze(-1).repeat(1, 3)
        pts_w = a0 + pts * (a1 - a0)  # (P,3) in world
        idx_b = _world_to_index_vec(pts_w, aabb_min=a0, aabb_max=a1, dims=dims)  # (P,3)
        pts_w_rt = index_to_world(idx_b, aabb_min=a0, aabb_max=a1, dims=dims, as_numpy=False)
        # Round-trip should land on grid; allow small tol due to round
        assert torch.allclose(torch.clamp(pts_w_rt, a0, a1), pts_w_rt, rtol=tol, atol=tol)

        # 7) Degenerate axis (dims=1 → always min==max, only index 0 is valid)
        dims_deg = (1, 12, 1)
        v_deg = voxel_size_world(a0, a1, dims_deg)
        den_deg = torch.tensor([d - 1 for d in dims_deg], device=dev, dtype=dt).clamp_min(1)
        assert torch.allclose(a0 + v_deg * den_deg, a1, rtol=tol, atol=tol)
        idx0_deg = _world_to_index_vec(a0, aabb_min=a0, aabb_max=a1, dims=dims_deg)
        idx1_deg = _world_to_index_vec(a1, aabb_min=a0, aabb_max=a1, dims=dims_deg)
        assert torch.equal(idx0_deg[..., [0, 2]], torch.zeros(2, device=dev, dtype=dt))
        assert torch.equal(idx1_deg[..., [0, 2]], torch.zeros(2, device=dev, dtype=dt))


    _self_test_edge_based()
