#!/usr/bin/env python3
# gs_coresets.utils/metric_utils.py
# CUDA-friendly, production-ready image metrics for 3DGS pipelines.
#
# Tensors expected in CHW (single image) or NCHW (batch), float in [0,1].
# All metrics work on CPU or GPU, respect the input device, and are batched.

from __future__ import annotations
from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn.functional as F

from gs_coresets.utils.types_devices_utils import torch_ctx

Tensor = torch.Tensor

__all__ = [
    "to_tensor01",
    "rgb_to_y",
    "crop_border",
    "compute_psnr",
    "compute_ssim",
    "compute_dssim",
    "compute_lpips",
]

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _ensure_nchw(x: Tensor) -> Tuple[Tensor, bool]:
    """Ensure a tensor is NCHW. If CHW, add batch dim and return (x, added_batch=True)."""
    if x.ndim == 3:
        return x.unsqueeze(0), True
    elif x.ndim == 4:
        return x, False
    else:
        raise ValueError(f"Expected CHW or NCHW, got shape {tuple(x.shape)}")

def _to_float32(x: Tensor) -> Tensor:
    """Cast to default project dtype for numerically-stable stats, preserving device."""
    if x.dtype == torch_ctx.dtype:
        return x
    return x.to(dtype=torch_ctx.dtype)

def crop_border(x: Tensor, pixels: int) -> Tensor:
    """Crop N pixels from all sides (NCHW or CHW)."""
    if pixels <= 0:
        return x
    if x.ndim == 3:
        return x[:, pixels:-pixels, pixels:-pixels]
    elif x.ndim == 4:
        return x[:, :, pixels:-pixels, pixels:-pixels]
    else:
        raise ValueError(f"crop_border expects CHW or NCHW, got {x.ndim}D")

def to_tensor01(img) -> Tensor:
    """
    Convert PIL.Image / numpy array / torch.Tensor to float CHW tensor in [0,1].
    If already a tensor in [0,1], returns clamped copy.
    """
    if isinstance(img, torch.Tensor):
        t = img
        # Allow HxW or HxWxC too, convert to CHW
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[0] not in (1, 3):  # likely HWC
            t = t.permute(2, 0, 1)
        return t.to(dtype=torch_ctx.dtype).clamp(0.0, 1.0)

    try:
        import numpy as np
        if hasattr(img, "size"):  # PIL.Image
            t = torch.from_numpy(np.array(img))
        else:  # numpy array
            t = torch.from_numpy(img)
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        if t.shape[-1] in (1, 3):  # HWC -> CHW
            t = t.permute(2, 0, 1)
        t = t.to(dtype=torch_ctx.dtype) / 255.0
        return t.clamp(0.0, 1.0)
    except Exception as e:
        raise RuntimeError("to_tensor01 expects PIL.Image, numpy array, or torch.Tensor") from e

def rgb_to_y(x: Tensor, standard: str = "bt601") -> Tensor:
    """
    Convert RGB tensor (NCHW or CHW, C=3) to luma Y (N1HW or 1HW).
    Coefficients:
      - BT.601: Y = 0.299 R + 0.587 G + 0.114 B
      - BT.709: Y = 0.2126 R + 0.7152 G + 0.0722 B
    """
    coeffs = {
        "bt601": (0.299, 0.587, 0.114),
        "bt709": (0.2126, 0.7152, 0.0722),
    }
    if x.ndim not in (3, 4):
        raise ValueError("rgb_to_y expects CHW or NCHW")
    C = x.shape[-3]
    if C != 3:
        raise ValueError(f"rgb_to_y expects 3 channels, got {C}")
    a, b, c = coeffs.get(standard.lower(), coeffs["bt601"])
    if x.ndim == 3:
        R, G, B = x[0], x[1], x[2]
        y = a * R + b * G + c * B
        return y.unsqueeze(0)  # 1HW
    else:
        R, G, B = x[:, 0], x[:, 1], x[:, 2]
        y = a * R + b * G + c * B
        return y.unsqueeze(1)  # N1HW

def _apply_mask_reduce(values: Tensor, mask: Optional[Tensor], reduction: str) -> Tensor:
    """
    Reduce a per-pixel/per-sample map with optional mask.
    values: N...HW
    mask:   N1HW or broadcastable to values
    reduction in {"mean", "sum", "none", "batch"}:
      - "mean": scalar mean over N,H,W (channels already aggregated)
      - "batch": per-image mean (N,)
      - "sum": sum
      - "none": return map (no reduction)
    """
    if reduction not in {"mean", "sum", "none", "batch"}:
        raise ValueError("reduction must be one of {'mean','sum','none','batch'}")

    if mask is not None:
        # Broadcast mask to values
        mask = mask.to(dtype=values.dtype, device=values.device)
        # Sum over channel dims if present in values; mask must match last two dims (H,W)
        m = mask
        vals = values * m
        denom = m.sum(dim=(-2, -1))
        if values.ndim == 4:  # N1HW map
            num = vals.sum(dim=(-2, -1))
        elif values.ndim == 3:  # N..HW without channel dim
            num = vals.sum(dim=(-2, -1))
        else:
            raise ValueError("Unexpected values ndim for mask reduction")
        # Avoid division by zero
        denom = torch.clamp(denom, min=1e-8)
        per_img = num / denom
    else:
        # Average over H,W
        per_img = values.mean(dim=(-2, -1))

    if reduction == "batch":
        # If values had channel dim, average that too
        if per_img.ndim > 1:
            per_img = per_img.mean(dim=tuple(range(1, per_img.ndim)))
        return per_img
    elif reduction == "mean":
        return per_img.mean()
    elif reduction == "sum":
        return per_img.sum()
    else:  # "none"
        return values

# ------------------------------------------------------------------------------
# PSNR
# ------------------------------------------------------------------------------

def compute_psnr(
    pred: Tensor,
    gt: Tensor,
    *,
    data_range: float = 1.0,
    crop: int = 0,
    mask: Optional[Tensor] = None,   # N1HW (broadcastable) or None
    luma: Optional[str] = None,      # None | "bt601" | "bt709"
    reduction: str = "mean",         # "mean" | "batch" | "sum" | "none"
    eps: float = 1e-10,
    enable_grad: bool = True,
) -> Tensor:
    """
    PSNR (in dB). Supports CHW or NCHW, optional border crop, mask, and luma-only.
    """
    ctx = (torch.enable_grad() if enable_grad else torch.no_grad())
    with ctx:
        x, _ = _ensure_nchw(pred)
        y, _ = _ensure_nchw(gt)
        if x.shape != y.shape:
            raise ValueError(f"PSNR: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")

        if crop > 0:
            x = crop_border(x, crop)
            y = crop_border(y, crop)
            if mask is not None:
                mask = crop_border(mask, crop)

        if luma is not None:
            x = rgb_to_y(x, luma)
            y = rgb_to_y(y, luma)

        # Work in float32 for numeric stability
        x32 = _to_float32(x)
        y32 = _to_float32(y)
        diff2 = (x32 - y32) ** 2  # NCHW

        if mask is not None:
            if mask.ndim == 3:  # 1HW or CHW?
                mask = mask.unsqueeze(0)
            if mask.shape[-3] != 1:
                # if user provided NCHW, collapse to N1HW
                mask = mask.mean(dim=-3, keepdim=True)
            mse = _apply_mask_reduce(diff2.mean(dim=-3, keepdim=True), mask, reduction="batch")
        else:
            mse = diff2.mean(dim=(-1, -2, -3))  # per-image (N,)

        # Convert per-image MSE to PSNR
        C = (data_range ** 2) / torch.clamp(mse, min=eps)
        psnr_per_img = 10.0 * torch.log10(C)

        # Final reduction
        if reduction == "mean":
            return psnr_per_img.mean()
        elif reduction == "batch":
            return psnr_per_img
        elif reduction == "sum":
            return psnr_per_img.sum()
        elif reduction == "none":
            # return per-pixel map is not meaningful for PSNR; return per-image anyway
            return psnr_per_img
        else:
            raise ValueError("Invalid reduction")

# ------------------------------------------------------------------------------
# SSIM / D-SSIM
# ------------------------------------------------------------------------------

# Cache Gaussian kernels per (win_size, sigma, channels, device, dtype)
_kernel_cache: Dict[Tuple[int, float, int, str, Optional[int], torch.dtype], Tensor] = {}

def _gaussian_kernel(win_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (win_size, float(sigma), int(channels), device.type, device.index, dtype)
    if key in _kernel_cache:
        return _kernel_cache[key]

    # 1D Gaussian
    coords = torch.arange(win_size, device=device, dtype=torch_ctx.dtype) - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()

    # 2D separable -> outer product then expand to depthwise conv weights
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    kernel = kernel.repeat(channels, 1, 1, 1)  # Cx1xHxW for groups=C depthwise conv
    _kernel_cache[key] = kernel
    return kernel

def _conv2d_same(x: Tensor, weight: Tensor) -> Tensor:
    """Depthwise conv with reflect padding to mimic 'same' padding."""
    c = x.shape[-3]
    pad = weight.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, weight, groups=c)

def compute_ssim(
    pred: Tensor,
    gt: Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    crop: int = 0,
    mask: Optional[Tensor] = None,   # N1HW
    luma: Optional[str] = None,      # None | "bt601" | "bt709"
    reduction: str = "mean",         # "mean" | "batch" | "sum" | "none"
    return_map: bool = False,
    eps: float = 1e-12,
    enable_grad: bool = True,
) -> Tensor:
    """
    SSIM (Wang et al. 2004) with depthwise Gaussian filter. Returns scalar (mean),
    per-image (batch), sum, or full SSIM map ("none" with return_map=True).
    """
    ctx = (torch.enable_grad() if enable_grad else torch.no_grad())
    with ctx:
        x, _ = _ensure_nchw(pred)
        y, _ = _ensure_nchw(gt)
        if x.shape != y.shape:
            raise ValueError(f"SSIM: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")

        if crop > 0:
            x = crop_border(x, crop)
            y = crop_border(y, crop)
            if mask is not None:
                mask = crop_border(mask, crop)

        if luma is not None:
            x = rgb_to_y(x, luma)
            y = rgb_to_y(y, luma)

        # Compute in float32 for stability
        x32 = _to_float32(x)
        y32 = _to_float32(y)

        N, C, H, W = x32.shape
        device = x32.device
        dtype = x32.dtype

        # Gaussian kernel (depthwise)
        k = _gaussian_kernel(win_size, win_sigma, C, device, dtype)

        mu_x = _conv2d_same(x32, k)
        mu_y = _conv2d_same(y32, k)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = _conv2d_same(x32 * x32, k) - mu_x2
        sigma_y2 = _conv2d_same(y32 * y32, k) - mu_y2
        sigma_xy = _conv2d_same(x32 * y32, k) - mu_xy

        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

        ssim_map = num / torch.clamp(den, min=eps)  # NCHW

        if return_map or reduction == "none":
            if mask is not None:
                # Reduce channels, keep N1HW map for masking
                ssim_map_c = ssim_map.mean(dim=1, keepdim=True)  # N1HW
                return _apply_mask_reduce(ssim_map_c, mask, reduction="none")
            return ssim_map

        # Reduce over channels first
        ssim_c = ssim_map.mean(dim=1, keepdim=True)  # N1HW

        # Reduce with optional mask and requested reduction
        return _apply_mask_reduce(ssim_c, mask, reduction=reduction)

def compute_dssim(**kwargs) -> Tensor:
    """
    D-SSIM = (1 - SSIM) / 2. Accepts same kwargs as compute_ssim().
    Example:
        dssim = compute_dssim(pred=img1, gt=img2, luma="bt601", reduction="mean")
    """
    ssim_val = compute_ssim(**kwargs)
    # ssim_val may be scalar or tensor; ensure tensor math
    return (1.0 - ssim_val) / 2.0

# ------------------------------------------------------------------------------
# LPIPS (requires 'lpips' package)
# ------------------------------------------------------------------------------

_lpips_cache: Dict[Tuple[str, str, Optional[int]], torch.nn.Module] = {}

def _get_lpips_model(net: str, device: torch.device):
    key = (net, device.type, device.index)
    if key in _lpips_cache:
        return _lpips_cache[key]
    try:
        import lpips  # type: ignore
    except Exception as e:
        raise RuntimeError("LPIPS requires the 'lpips' package. Try: pip install lpips") from e
    m = lpips.LPIPS(net=net).to(device)
    m.eval()
    _lpips_cache[key] = m
    return m

def _to_rgb_nchw01(x: Tensor) -> Tensor:
    """Ensure NCHW, 3 channels, [0,1]. If 1 channel, repeat to 3."""
    x, added = _ensure_nchw(x)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError(f"LPIPS expects 1 or 3 channels, got {x.shape[1]}")
    return x.clamp(0.0, 1.0)

def compute_lpips(
    pred: Tensor,
    gt: Tensor,
    *,
    net: str = "vgg",                 # "vgg" | "alex" | "squeeze"
    device: Optional[torch.device] = None,
    crop: int = 0,
    mask: Optional[Tensor] = None,    # N1HW
    reduction: str = "mean",          # "mean" | "batch" | "sum" | "none"
    enable_grad: bool = False,        # LPIPS usually used without grad
) -> Tensor:
    """
    LPIPS via the 'lpips' package. Batched, GPU-accelerated. Inputs in [0,1].
    If grayscale, channels are repeated to 3.
    """
    # LPIPS model runs on its own device; inputs should be moved accordingly
    # If device not provided, infer from input tensors
    if device is None:
        device = pred.device

    ctx = (torch.enable_grad() if enable_grad else torch.no_grad())
    with ctx:
        x = _to_rgb_nchw01(pred).to(device=device, dtype=torch_ctx.dtype)
        y = _to_rgb_nchw01(gt).to(device=device, dtype=torch_ctx.dtype)
        if x.shape != y.shape:
            raise ValueError(f"LPIPS: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")

        if crop > 0:
            x = crop_border(x, crop)
            y = crop_border(y, crop)
            if mask is not None:
                mask = crop_border(mask, crop)

        # Scale to [-1,1] as LPIPS expects
        x_lp = x * 2.0 - 1.0
        y_lp = y * 2.0 - 1.0

        model = _get_lpips_model(net=net, device=device)
        # lpips returns (N,1,1,1) or (N,1,1,1)-like tensor
        val = model(x_lp, y_lp)  # already no-grad internally
        # Squeeze to (N,)
        val = val.view(val.shape[0])

        if mask is not None:
            # No standard masked LPIPS; approximate by weighting pixel-wise feature loss is not exposed.
            # Practical workaround: just compute per-image LPIPS then apply mask as binary weight = mean(mask) per image.
            # If you need true masked LPIPS, you'd have to modify lpips internals.
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            # mean mask value per image (N,)
            m = mask.to(device=device, dtype=torch_ctx.dtype).mean(dim=(-2, -1, -3))
            val = val * torch.clamp(m, min=1e-8)

        if reduction == "mean":
            return val.mean()
        elif reduction == "batch":
            return val
        elif reduction == "sum":
            return val.sum()
        elif reduction == "none":
            return val
        else:
            raise ValueError("Invalid reduction for LPIPS")
