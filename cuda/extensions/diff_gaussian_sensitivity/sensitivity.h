/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Modifications Copyright (C) 2026 Waseem Mousa and Alaa Maalouf.
 * This derivative is available only for non-commercial research and
 * evaluation under the terms in LICENSE.md.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace SENSITIVITY
{
    void denominators(
        dim3 grid,
        dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int width,
        int height,
        const float2* means2D,
        const float* colors,
        const float4* conic_opacity,
        const uint32_t* n_contrib,
        bool use_l2,
        bool l2_rgb_aggregation,
        bool nocolor,
        float* gaussian_only_color,
        float* channel_denominators,
        float* pixel_denominators,
        float* tile_denominators,
        float* image_denominator,
        unsigned long long* candidate_visits,
        unsigned long long* accepted_contributions,
        unsigned char* trace_candidate,
        unsigned char* trace_accepted,
        float* trace_power,
        float* trace_alpha,
        float* trace_transmittance_before,
        float* trace_weight,
        float* trace_contribution);

    void replay(
        dim3 grid,
        dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int width,
        int height,
        const float2* means2D,
        const float* colors,
        const float4* conic_opacity,
        const uint32_t* n_contrib,
        const float* channel_denominators,
        const float* pixel_denominators,
        const float* tile_denominators,
        int sensitivity_flags,
        bool use_l2,
        bool l2_rgb_aggregation,
        bool nocolor,
        float* channel_values,
        float* pixel_values,
        float* tile_values,
        float* image_numerators);
}
