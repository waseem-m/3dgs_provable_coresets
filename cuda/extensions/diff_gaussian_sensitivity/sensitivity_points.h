/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Modifications Copyright (C) 2026 Waseem Mousa and Alaa Maalouf.
 * See LICENSE.md.
 */

#pragma once

#include <torch/extension.h>
#include <tuple>

using RasterizeGaussiansSensitivityOutput = std::tuple<
    int,
    torch::Tensor,  // background-composited rasterizer output C^B
    torch::Tensor,  // Gaussian-only pixel accumulation C^G
    torch::Tensor,  // radii
    torch::Tensor,  // final transmittance
    torch::Tensor,  // GraphDECO per-pixel last-contributor counts (debug)
    torch::Tensor,  // GraphDECO tile ranges (debug)
    torch::Tensor,  // GraphDECO sorted point list (debug)
    torch::Tensor,  // exact feature buffer (debug mode)
    torch::Tensor,  // per-channel sensitivity
    torch::Tensor,  // per-pixel sensitivity
    torch::Tensor,  // per-tile sensitivity
    torch::Tensor,  // per-image/scene Gaussian numerators
    torch::Tensor,  // per-channel Gaussian-only denominators
    torch::Tensor,  // per-pixel Gaussian-only denominators
    torch::Tensor,  // per-tile Gaussian-only denominators
    torch::Tensor,  // per-image Gaussian-only denominator
    torch::Tensor,  // sparse candidate visits
    torch::Tensor,  // accepted Gaussian-pixel contributions
    torch::Tensor,  // bounded debug: GraphDECO means2D
    torch::Tensor,  // bounded debug: GraphDECO conic_opacity
    torch::Tensor,  // bounded debug: candidate mask (N,H,W)
    torch::Tensor,  // bounded debug: accepted mask (N,H,W)
    torch::Tensor,  // bounded debug: power (N,H,W)
    torch::Tensor,  // bounded debug: alpha (N,H,W)
    torch::Tensor,  // bounded debug: T_before (N,H,W)
    torch::Tensor,  // bounded debug: weight (N,H,W)
    torch::Tensor   // bounded debug: contribution a (N,3,H,W)
>;

RasterizeGaussiansSensitivityOutput RasterizeGaussiansSensitivityCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    float tan_fovx,
    float tan_fovy,
    int image_height,
    int image_width,
    const torch::Tensor& sh,
    int degree,
    const torch::Tensor& campos,
    bool prefiltered,
    bool antialiasing,
    int sensitivity_flags,
    bool use_l2,
    bool l2_rgb_aggregation,
    bool nocolor,
    bool debug);
