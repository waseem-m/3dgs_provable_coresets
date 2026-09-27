/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Modifications Copyright (C) 2026 Waseem Mousa and Alaa Maalouf.
 * This derivative is available only for non-commercial research and
 * evaluation under the terms in LICENSE.md.
 */

#include "sensitivity.h"

#include "cuda_rasterizer/config.h"
#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;

namespace
{
constexpr int kBlockSize = BLOCK_X * BLOCK_Y;
static_assert(kBlockSize > 0, "the sensitivity replay block must not be empty");
static_assert(
    (kBlockSize & (kBlockSize - 1)) == 0,
    "the sensitivity replay reduction requires a power-of-two block size");
static_assert(
    kBlockSize <= 1024,
    "the sensitivity replay block exceeds CUDA's threads-per-block limit");

constexpr int PER_CHANNEL = 1 << 0;
constexpr int PER_PIXEL = 1 << 1;
constexpr int PER_TILE = 1 << 2;
constexpr int PER_IMAGE = 1 << 3;
constexpr int PER_SCENE = 1 << 4;

__device__ inline void atomicMaxNonnegative(float* address, float value)
{
    atomicMax(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
}

__device__ inline float ratioOrZero(float numerator, float denominator)
{
    return denominator > 0.0f ? numerator / denominator : 0.0f;
}

__device__ inline bool acceptedContribution(
    uint32_t gaussian_id,
    const float2& pixel,
    const float2* means2D,
    const float* colors,
    const float4* conic_opacity,
    float& transmittance,
    float& power,
    float& alpha,
    float& weight,
    float contribution[3])
{
    const float2 center = means2D[gaussian_id];
    const float2 delta = {center.x - pixel.x, center.y - pixel.y};
    const float4 conic = conic_opacity[gaussian_id];
    power = -0.5f * (
        conic.x * delta.x * delta.x + conic.z * delta.y * delta.y)
        - conic.y * delta.x * delta.y;
    if (power > 0.0f)
        return false;

    alpha = fminf(0.99f, conic.w * expf(power));
    if (alpha < 1.0f / 255.0f)
        return false;

    const float next_transmittance = transmittance * (1.0f - alpha);
    if (next_transmittance < 0.0001f)
        return false;

    weight = alpha * transmittance;
    #pragma unroll
    for (int channel = 0; channel < 3; ++channel)
    {
        // The SH path passes geometry_state.rgb, i.e. the exact feature buffer
        // produced by GraphDECO's pinned computeColorFromSH implementation.
        const float feature = colors[gaussian_id * 3 + channel];
        contribution[channel] = weight * feature;
    }
    transmittance = next_transmittance;
    return true;
}

__global__ __launch_bounds__(BLOCK_X * BLOCK_Y) void denominatorReplayKernel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int width,
    int height,
    const float2* __restrict__ means2D,
    const float* __restrict__ colors,
    const float4* __restrict__ conic_opacity,
    const uint32_t* __restrict__ n_contrib,
    bool use_l2,
    bool l2_rgb_aggregation,
    bool nocolor,
    float* __restrict__ gaussian_only_color,
    float* __restrict__ channel_denominators,
    float* __restrict__ pixel_denominators,
    float* __restrict__ tile_denominators,
    float* __restrict__ image_denominator,
    unsigned long long* __restrict__ candidate_visits,
    unsigned long long* __restrict__ accepted_contributions,
    unsigned char* __restrict__ trace_candidate,
    unsigned char* __restrict__ trace_accepted,
    float* __restrict__ trace_power,
    float* __restrict__ trace_alpha,
    float* __restrict__ trace_transmittance_before,
    float* __restrict__ trace_weight,
    float* __restrict__ trace_contribution)
{
    auto block = cg::this_thread_block();
    const int grid_x = (width + BLOCK_X - 1) / BLOCK_X;
    const int tile_id = block.group_index().y * grid_x + block.group_index().x;
    const uint2 pixel_min = {
        block.group_index().x * BLOCK_X,
        block.group_index().y * BLOCK_Y
    };
    const uint2 pixel = {
        pixel_min.x + block.thread_index().x,
        pixel_min.y + block.thread_index().y
    };
    const bool inside = pixel.x < width && pixel.y < height;
    const int pixel_id = inside
        ? static_cast<int>(pixel.y * width + pixel.x)
        : 0;
    const int pixel_count = width * height;
    const float2 pixel_f = {
        static_cast<float>(pixel.x),
        static_cast<float>(pixel.y)
    };
    const int thread_id = block.thread_rank();
    const uint2 range = ranges[tile_id];
    const uint32_t last_contributor = inside ? n_contrib[pixel_id] : 0;

    float transmittance = 1.0f;
    float gaussian_sum[3] = {0.0f, 0.0f, 0.0f};
    float denominator_channel[3] = {0.0f, 0.0f, 0.0f};
    float l2_rgb_denominator = 0.0f;
    unsigned long long local_visits = 0;
    unsigned long long local_accepted = 0;

    // Mathematically this query sums contributions from every Gaussian.
    // GraphDECO's ranges/point_list is the exact sparse support for this tile;
    // Gaussians outside the range have zero contribution and are not inspected.
    for (
        uint32_t entry = 0;
        inside && range.x + entry < range.y && entry < last_contributor;
        ++entry)
    {
        ++local_visits;
        const uint32_t gaussian_id = point_list[range.x + entry];
        float contribution[3] = {0.0f, 0.0f, 0.0f};
        float power = 0.0f;
        float alpha = 0.0f;
        float weight = 0.0f;
        const float transmittance_before = transmittance;
        const bool accepted = acceptedContribution(
                gaussian_id,
                pixel_f,
                means2D,
                colors,
                conic_opacity,
                transmittance,
                power,
                alpha,
                weight,
                contribution);

        if (trace_candidate != nullptr)
        {
            const size_t trace_index =
                static_cast<size_t>(gaussian_id) * pixel_count + pixel_id;
            trace_candidate[trace_index] = 1;
            trace_power[trace_index] = power;
            trace_alpha[trace_index] = alpha;
            trace_transmittance_before[trace_index] =
                transmittance_before;
            if (accepted)
            {
                trace_accepted[trace_index] = 1;
                trace_weight[trace_index] = weight;
                #pragma unroll
                for (int channel = 0; channel < 3; ++channel)
                {
                    trace_contribution[
                        (static_cast<size_t>(gaussian_id) * 3 + channel)
                            * pixel_count
                        + pixel_id] = contribution[channel];
                }
            }
        }

        if (!accepted)
            continue;

        ++local_accepted;
        float pixel_channel_sum = 0.0f;
        #pragma unroll
        for (int channel = 0; channel < 3; ++channel)
        {
            gaussian_sum[channel] += contribution[channel];
            const float sensitivity_contribution = nocolor
                ? weight
                : contribution[channel];
            denominator_channel[channel] += use_l2
                ? sensitivity_contribution * sensitivity_contribution
                : sensitivity_contribution;
            if (l2_rgb_aggregation)
                pixel_channel_sum += sensitivity_contribution;
        }
        if (l2_rgb_aggregation)
            l2_rgb_denominator += pixel_channel_sum * pixel_channel_sum;
    }

    float denominator_pixel = 0.0f;
    if (inside)
    {
        #pragma unroll
        for (int channel = 0; channel < 3; ++channel)
        {
            gaussian_only_color[channel * pixel_count + pixel_id] =
                gaussian_sum[channel];
            channel_denominators[channel * pixel_count + pixel_id] =
                denominator_channel[channel];
            denominator_pixel += denominator_channel[channel];
        }
        if (l2_rgb_aggregation)
            denominator_pixel = l2_rgb_denominator;
        pixel_denominators[pixel_id] = denominator_pixel;
    }

    __shared__ float reduce_denominator[kBlockSize];
    __shared__ unsigned long long reduce_visits[kBlockSize];
    __shared__ unsigned long long reduce_accepted[kBlockSize];
    reduce_denominator[thread_id] = inside ? denominator_pixel : 0.0f;
    reduce_visits[thread_id] = local_visits;
    reduce_accepted[thread_id] = local_accepted;
    block.sync();

    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (thread_id < stride)
        {
            reduce_denominator[thread_id] +=
                reduce_denominator[thread_id + stride];
            reduce_visits[thread_id] += reduce_visits[thread_id + stride];
            reduce_accepted[thread_id] +=
                reduce_accepted[thread_id + stride];
        }
        block.sync();
    }

    if (thread_id == 0)
    {
        tile_denominators[tile_id] = reduce_denominator[0];
        atomicAdd(image_denominator, reduce_denominator[0]);
        atomicAdd(candidate_visits, reduce_visits[0]);
        atomicAdd(accepted_contributions, reduce_accepted[0]);
    }
}

__global__ __launch_bounds__(BLOCK_X * BLOCK_Y) void sensitivityReplayKernel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int width,
    int height,
    const float2* __restrict__ means2D,
    const float* __restrict__ colors,
    const float4* __restrict__ conic_opacity,
    const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ channel_denominators,
    const float* __restrict__ pixel_denominators,
    const float* __restrict__ tile_denominators,
    int sensitivity_flags,
    bool use_l2,
    bool l2_rgb_aggregation,
    bool nocolor,
    float* __restrict__ channel_values,
    float* __restrict__ pixel_values,
    float* __restrict__ tile_values,
    float* __restrict__ image_numerators)
{
    auto block = cg::this_thread_block();
    const int grid_x = (width + BLOCK_X - 1) / BLOCK_X;
    const int tile_id = block.group_index().y * grid_x + block.group_index().x;
    const uint2 pixel_min = {
        block.group_index().x * BLOCK_X,
        block.group_index().y * BLOCK_Y
    };
    const uint2 pixel = {
        pixel_min.x + block.thread_index().x,
        pixel_min.y + block.thread_index().y
    };
    const bool inside = pixel.x < width && pixel.y < height;
    const int pixel_id = inside
        ? static_cast<int>(pixel.y * width + pixel.x)
        : 0;
    const int pixel_count = width * height;
    const float2 pixel_f = {
        static_cast<float>(pixel.x),
        static_cast<float>(pixel.y)
    };
    const int thread_id = block.thread_rank();
    const uint2 range = ranges[tile_id];
    const int tile_candidate_count = static_cast<int>(range.y - range.x);
    const uint32_t last_contributor = inside ? n_contrib[pixel_id] : 0;

    __shared__ float reduce_channel[kBlockSize];
    __shared__ float reduce_pixel[kBlockSize];
    __shared__ float reduce_tile[kBlockSize];

    const bool want_channel = (sensitivity_flags & PER_CHANNEL) != 0;
    const bool want_pixel = (sensitivity_flags & PER_PIXEL) != 0;
    const bool want_tile = (sensitivity_flags & PER_TILE) != 0;
    const bool want_image =
        (sensitivity_flags & (PER_IMAGE | PER_SCENE)) != 0;
    float transmittance = 1.0f;

    // Mathematically this query sums contributions from every Gaussian.
    // GraphDECO's ranges/point_list is the exact sparse support for this tile;
    // Gaussians outside the range have zero contribution and are not inspected.
    for (int entry = 0; entry < tile_candidate_count; ++entry)
    {
        const uint32_t gaussian_id = point_list[range.x + entry];
        float contribution[3] = {0.0f, 0.0f, 0.0f};
        float power = 0.0f;
        float alpha = 0.0f;
        float weight = 0.0f;
        bool accepted = false;
        if (inside && static_cast<uint32_t>(entry) < last_contributor)
        {
            accepted = acceptedContribution(
                gaussian_id,
                pixel_f,
                means2D,
                colors,
                conic_opacity,
                transmittance,
                power,
                alpha,
                weight,
                contribution);
        }

        float channel_ratio = 0.0f;
        float pixel_numerator = 0.0f;
        if (accepted)
        {
            if (l2_rgb_aggregation)
            {
                float pixel_channel_sum = 0.0f;
                #pragma unroll
                for (int channel = 0; channel < 3; ++channel)
                {
                    const float raw = nocolor
                        ? weight
                        : contribution[channel];
                    const float channel_term = raw * raw;
                    pixel_channel_sum += raw;
                    channel_ratio = fmaxf(
                        channel_ratio,
                        ratioOrZero(
                            channel_term,
                            channel_denominators[
                                channel * pixel_count + pixel_id]));
                }
                // L2-agg squares the RGB sum independently at this pixel.
                // Tile/image/scene reductions sum these already-squared pixel
                // terms; they never square a spatially aggregated sum.
                pixel_numerator =
                    pixel_channel_sum * pixel_channel_sum;
            }
            else
            {
                #pragma unroll
                for (int channel = 0; channel < 3; ++channel)
                {
                    const float raw = nocolor
                        ? weight
                        : contribution[channel];
                    const float term = use_l2 ? raw * raw : raw;
                    pixel_numerator += term;
                    channel_ratio = fmaxf(
                        channel_ratio,
                        ratioOrZero(
                            term,
                            channel_denominators[
                                channel * pixel_count + pixel_id]));
                }
            }
        }
        const float pixel_ratio = inside
            ? ratioOrZero(pixel_numerator, pixel_denominators[pixel_id])
            : 0.0f;

        reduce_channel[thread_id] = channel_ratio;
        reduce_pixel[thread_id] = pixel_ratio;
        reduce_tile[thread_id] = accepted ? pixel_numerator : 0.0f;
        block.sync();

        for (int stride = kBlockSize / 2; stride > 0; stride >>= 1)
        {
            if (thread_id < stride)
            {
                reduce_channel[thread_id] = fmaxf(
                    reduce_channel[thread_id],
                    reduce_channel[thread_id + stride]);
                reduce_pixel[thread_id] = fmaxf(
                    reduce_pixel[thread_id],
                    reduce_pixel[thread_id + stride]);
                reduce_tile[thread_id] += reduce_tile[thread_id + stride];
            }
            block.sync();
        }

        if (thread_id == 0)
        {
            if (want_channel)
                atomicMaxNonnegative(
                    channel_values + gaussian_id,
                    reduce_channel[0]);
            if (want_pixel)
                atomicMaxNonnegative(
                    pixel_values + gaussian_id,
                    reduce_pixel[0]);
            if (want_tile)
                atomicMaxNonnegative(
                    tile_values + gaussian_id,
                    ratioOrZero(
                        reduce_tile[0],
                        tile_denominators[tile_id]));
            if (want_image)
                atomicAdd(image_numerators + gaussian_id, reduce_tile[0]);
        }
        block.sync();
    }
}
}

void SENSITIVITY::denominators(
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
    float* trace_contribution)
{
    denominatorReplayKernel<<<grid, block>>>(
        ranges,
        point_list,
        width,
        height,
        means2D,
        colors,
        conic_opacity,
        n_contrib,
        use_l2,
        l2_rgb_aggregation,
        nocolor,
        gaussian_only_color,
        channel_denominators,
        pixel_denominators,
        tile_denominators,
        image_denominator,
        candidate_visits,
        accepted_contributions,
        trace_candidate,
        trace_accepted,
        trace_power,
        trace_alpha,
        trace_transmittance_before,
        trace_weight,
        trace_contribution);
}

void SENSITIVITY::replay(
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
    float* image_numerators)
{
    sensitivityReplayKernel<<<grid, block>>>(
        ranges,
        point_list,
        width,
        height,
        means2D,
        colors,
        conic_opacity,
        n_contrib,
        channel_denominators,
        pixel_denominators,
        tile_denominators,
        sensitivity_flags,
        use_l2,
        l2_rgb_aggregation,
        nocolor,
        channel_values,
        pixel_values,
        tile_values,
        image_numerators);
}
