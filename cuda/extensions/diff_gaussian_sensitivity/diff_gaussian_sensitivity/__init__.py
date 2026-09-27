"""Forward-only GraphDECO CUDA rasterization with coreset sensitivities."""

from __future__ import annotations

from typing import NamedTuple

import torch

from . import _C


class GaussianSensitivitySettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool = False
    antialiasing: bool = False
    debug: bool = False


class GaussianSensitivityResult(NamedTuple):
    """One-camera GraphDECO render stages and sensitivity partials."""

    rendered_tile_overlaps: int
    background_composited_color: torch.Tensor
    gaussian_only_color: torch.Tensor
    radii: torch.Tensor
    final_transmittance: torch.Tensor
    contributor_counts: torch.Tensor
    tile_ranges: torch.Tensor
    sorted_point_list: torch.Tensor
    feature_buffer: torch.Tensor
    channel_value: torch.Tensor
    pixel_value: torch.Tensor
    tile_value: torch.Tensor
    image_numerator: torch.Tensor
    channel_denominator: torch.Tensor
    pixel_denominator: torch.Tensor
    tile_denominator: torch.Tensor
    image_denominator: torch.Tensor
    candidate_visits: torch.Tensor
    accepted_contributions: torch.Tensor
    trace_means2D: torch.Tensor
    trace_conic_opacity: torch.Tensor
    trace_candidate: torch.Tensor
    trace_accepted: torch.Tensor
    trace_power: torch.Tensor
    trace_alpha: torch.Tensor
    trace_transmittance_before: torch.Tensor
    trace_weight: torch.Tensor
    trace_contribution: torch.Tensor


def rasterize_gaussians_sensitivity(
    *,
    means3D: torch.Tensor,
    sh: torch.Tensor,
    colors_precomp: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cov3D_precomp: torch.Tensor,
    settings: GaussianSensitivitySettings,
    sensitivity_flags: int,
    norm: str,
    reduction: str,
    nocolor: bool,
):
    """Rasterize one camera and return Gaussian-only sensitivity partials."""
    if norm not in {"l1", "l2", "l2-channel", "l2-agg"}:
        raise ValueError(
            "norm must be 'l1', 'l2', 'l2-channel', or 'l2-agg'"
        )
    if reduction != "max":
        raise ValueError("the corrected CUDA backend supports max reduction only")

    values = _C.rasterize_gaussians_sensitivity(
        settings.bg,
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        float(settings.scale_modifier),
        cov3D_precomp,
        settings.viewmatrix,
        settings.projmatrix,
        float(settings.tanfovx),
        float(settings.tanfovy),
        int(settings.image_height),
        int(settings.image_width),
        sh,
        int(settings.sh_degree),
        settings.campos,
        bool(settings.prefiltered),
        bool(settings.antialiasing),
        int(sensitivity_flags),
        norm in {"l2", "l2-channel", "l2-agg"},
        norm == "l2-agg",
        bool(nocolor),
        bool(settings.debug),
    )
    return GaussianSensitivityResult(*values)


__all__ = [
    "GaussianSensitivityResult",
    "GaussianSensitivitySettings",
    "rasterize_gaussians_sensitivity",
]
