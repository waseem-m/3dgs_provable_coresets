/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Modifications Copyright (C) 2026 Waseem Mousa and Alaa Maalouf.
 * This derivative is available only for non-commercial research and
 * evaluation under the terms in LICENSE.md.
 */

#include "sensitivity_points.h"

#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "sensitivity.h"

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <functional>
#include <limits>
#include <string>

namespace
{
std::function<char*(size_t)> resizeFunctional(torch::Tensor& tensor)
{
    return [&tensor](size_t size) {
        tensor.resize_({static_cast<long long>(size)});
        return reinterpret_cast<char*>(tensor.contiguous().data_ptr());
    };
}

void requireCudaFloatContiguous(
    const torch::Tensor& tensor,
    const char* name,
    bool allow_empty = false)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32,
        name,
        " must use float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    if (!allow_empty)
        TORCH_CHECK(tensor.numel() > 0, name, " must not be empty");
}

void requireSameDevice(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name)
{
    TORCH_CHECK(
        tensor.device() == reference.device(),
        name,
        " must be on ",
        reference.device(),
        ", got ",
        tensor.device());
}

void checkCuda(const char* stage, bool synchronize)
{
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, stage, ": ", cudaGetErrorString(error));
    if (synchronize)
    {
        error = cudaDeviceSynchronize();
        TORCH_CHECK(error == cudaSuccess, stage, ": ", cudaGetErrorString(error));
    }
}
}

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
    bool debug)
{
    requireCudaFloatContiguous(background, "background");
    requireCudaFloatContiguous(means3D, "means3D", true);
    requireCudaFloatContiguous(opacity, "opacity", true);
    requireCudaFloatContiguous(viewmatrix, "viewmatrix");
    requireCudaFloatContiguous(projmatrix, "projmatrix");
    requireCudaFloatContiguous(campos, "campos");
    requireCudaFloatContiguous(colors, "colors", true);
    requireCudaFloatContiguous(scales, "scales", true);
    requireCudaFloatContiguous(rotations, "rotations", true);
    requireCudaFloatContiguous(cov3D_precomp, "cov3D_precomp", true);
    requireCudaFloatContiguous(sh, "sh", true);

    requireSameDevice(means3D, background, "background");
    requireSameDevice(means3D, opacity, "opacity");
    requireSameDevice(means3D, viewmatrix, "viewmatrix");
    requireSameDevice(means3D, projmatrix, "projmatrix");
    requireSameDevice(means3D, campos, "campos");
    requireSameDevice(means3D, colors, "colors");
    requireSameDevice(means3D, scales, "scales");
    requireSameDevice(means3D, rotations, "rotations");
    requireSameDevice(means3D, cov3D_precomp, "cov3D_precomp");
    requireSameDevice(means3D, sh, "sh");

    TORCH_CHECK(
        means3D.dim() == 2 && means3D.size(1) == 3,
        "means3D must have shape (N, 3)");
    TORCH_CHECK(background.dim() == 1 && background.numel() == 3,
        "background must have shape (3,)");
    TORCH_CHECK(
        viewmatrix.dim() == 2
            && viewmatrix.size(0) == 4
            && viewmatrix.size(1) == 4,
        "viewmatrix must have shape (4, 4)");
    TORCH_CHECK(
        projmatrix.dim() == 2
            && projmatrix.size(0) == 4
            && projmatrix.size(1) == 4,
        "projmatrix must have shape (4, 4)");
    TORCH_CHECK(campos.dim() == 1 && campos.numel() == 3,
        "campos must have shape (3,)");
    TORCH_CHECK(
        image_height > 0 && image_width > 0,
        "image dimensions must be positive");
    TORCH_CHECK(
        std::isfinite(scale_modifier) && scale_modifier > 0.0f,
        "scale_modifier must be finite and positive");
    TORCH_CHECK(
        std::isfinite(tan_fovx)
            && std::isfinite(tan_fovy)
            && tan_fovx > 0.0f
            && tan_fovy > 0.0f,
        "tangent FoVs must be finite and positive");
    TORCH_CHECK(degree >= 0 && degree <= 3, "SH degree must be in [0, 3]");
    TORCH_CHECK(
        sensitivity_flags >= 0 && (sensitivity_flags & ~31) == 0,
        "sensitivity_flags contains an unknown granularity bit");
    TORCH_CHECK(
        !l2_rgb_aggregation || use_l2,
        "l2_rgb_aggregation requires L2 sensitivity");
    TORCH_CHECK(!prefiltered,
        "the CUDA sensitivity backend requires prefiltered=False");
    TORCH_CHECK(!antialiasing,
        "the CUDA sensitivity backend requires antialiasing=False");

    TORCH_CHECK(
        means3D.size(0) <= std::numeric_limits<int>::max(),
        "the CUDA rasterizer supports at most INT_MAX Gaussians");
    const int gaussian_count = static_cast<int>(means3D.size(0));
    TORCH_CHECK(
        opacity.dim() == 2
            && opacity.size(0) == gaussian_count
            && opacity.size(1) == 1,
        "opacity must have shape (N, 1)");

    const bool has_colors = colors.dim() == 2;
    const bool has_sh = sh.dim() == 3;
    TORCH_CHECK(
        has_colors != has_sh,
        "provide exactly one of colors (N,3) or SH coefficients (N,M,3)");
    if (has_colors)
        TORCH_CHECK(
            colors.size(0) == gaussian_count && colors.size(1) == 3,
            "colors must have shape (N, 3)");
    if (has_sh)
        TORCH_CHECK(
            sh.size(0) == gaussian_count
                && sh.size(2) == 3
                && sh.size(1) >= (degree + 1) * (degree + 1),
            "SH coefficients must have shape (N, M, 3) with enough coefficients");

    const bool has_covariance = cov3D_precomp.dim() == 2;
    const bool has_scale_rotation =
        scales.dim() == 2 && rotations.dim() == 2;
    TORCH_CHECK(
        has_covariance != has_scale_rotation,
        "provide exactly one of covariance or scale/rotation inputs");
    if (has_covariance)
        TORCH_CHECK(
            cov3D_precomp.size(0) == gaussian_count
                && cov3D_precomp.size(1) == 6,
            "cov3D_precomp must have shape (N, 6)");
    if (has_scale_rotation)
    {
        TORCH_CHECK(
            scales.size(0) == gaussian_count && scales.size(1) == 3,
            "scales must have shape (N, 3)");
        TORCH_CHECK(
            rotations.size(0) == gaussian_count && rotations.size(1) == 4,
            "rotations must have shape (N, 4)");
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means3D));
    const auto float_options = means3D.options().dtype(torch::kFloat32);
    const auto int_options = means3D.options().dtype(torch::kInt32);
    const auto int64_options = means3D.options().dtype(torch::kInt64);
    const auto byte_options = means3D.options().dtype(torch::kByte);
    const int grid_x = (image_width + BLOCK_X - 1) / BLOCK_X;
    const int grid_y = (image_height + BLOCK_Y - 1) / BLOCK_Y;

    torch::Tensor background_composited_color = torch::zeros(
        {3, image_height, image_width},
        float_options);
    torch::Tensor gaussian_only_color = torch::zeros(
        {3, image_height, image_width},
        float_options);
    torch::Tensor rendered_depth = torch::zeros(
        {1, image_height, image_width},
        float_options);
    torch::Tensor radii = torch::zeros({gaussian_count}, int_options);
    torch::Tensor final_transmittance = torch::ones(
        {image_height, image_width},
        float_options);
    torch::Tensor contributor_counts = debug
        ? torch::zeros({image_height, image_width}, int_options)
        : torch::empty({0}, int_options);
    torch::Tensor tile_ranges = debug
        ? torch::zeros({grid_x * grid_y, 2}, int_options)
        : torch::empty({0}, int_options);
    torch::Tensor sorted_point_list = torch::empty({0}, int_options);
    torch::Tensor feature_buffer = debug
        ? torch::empty({gaussian_count, 3}, float_options)
        : torch::empty({0}, float_options);

    torch::Tensor channel_values = torch::zeros(
        {gaussian_count},
        float_options);
    torch::Tensor pixel_values = torch::zeros(
        {gaussian_count},
        float_options);
    torch::Tensor tile_values = torch::zeros(
        {gaussian_count},
        float_options);
    torch::Tensor image_numerators = torch::zeros(
        {gaussian_count},
        float_options);
    torch::Tensor channel_denominators = torch::zeros(
        {3, image_height, image_width},
        float_options);
    torch::Tensor pixel_denominators = torch::zeros(
        {image_height, image_width},
        float_options);
    torch::Tensor image_denominator = torch::zeros({1}, float_options);
    torch::Tensor candidate_visits = torch::zeros({1}, int64_options);
    torch::Tensor accepted_contributions = torch::zeros({1}, int64_options);

    torch::Tensor tile_denominators = torch::zeros(
        {grid_x * grid_y},
        float_options);
    // Full atomic traces are intentionally available only for tiny synthetic
    // debug fixtures. Real-model debug runs receive empty trace tensors and
    // cannot accidentally allocate Gaussian-by-pixel storage.
    constexpr int kTraceGaussianLimit = 64;
    constexpr int kTracePixelLimit = 4096;
    const int pixel_count = image_width * image_height;
    const bool trace_enabled =
        debug
        && gaussian_count <= kTraceGaussianLimit
        && pixel_count <= kTracePixelLimit;
    torch::Tensor trace_means2D = trace_enabled
        ? torch::zeros({gaussian_count, 2}, float_options)
        : torch::empty({0}, float_options);
    torch::Tensor trace_conic_opacity = trace_enabled
        ? torch::zeros({gaussian_count, 4}, float_options)
        : torch::empty({0}, float_options);
    torch::Tensor trace_candidate = trace_enabled
        ? torch::zeros(
            {gaussian_count, image_height, image_width},
            byte_options)
        : torch::empty({0}, byte_options);
    torch::Tensor trace_accepted = trace_enabled
        ? torch::zeros(
            {gaussian_count, image_height, image_width},
            byte_options)
        : torch::empty({0}, byte_options);
    torch::Tensor trace_power = trace_enabled
        ? torch::zeros(
            {gaussian_count, image_height, image_width},
            float_options)
        : torch::empty({0}, float_options);
    torch::Tensor trace_alpha = trace_enabled
        ? torch::zeros_like(trace_power)
        : torch::empty({0}, float_options);
    torch::Tensor trace_transmittance_before = trace_enabled
        ? torch::zeros_like(trace_power)
        : torch::empty({0}, float_options);
    torch::Tensor trace_weight = trace_enabled
        ? torch::zeros_like(trace_power)
        : torch::empty({0}, float_options);
    torch::Tensor trace_contribution = trace_enabled
        ? torch::zeros(
            {gaussian_count, 3, image_height, image_width},
            float_options)
        : torch::empty({0}, float_options);

    int num_rendered = 0;
    if (gaussian_count > 0)
    {
        torch::Tensor geometry_buffer = torch::empty({0}, byte_options);
        torch::Tensor binning_buffer = torch::empty({0}, byte_options);
        torch::Tensor image_buffer = torch::empty({0}, byte_options);
        auto geometry_function = resizeFunctional(geometry_buffer);
        auto binning_function = resizeFunctional(binning_buffer);
        auto image_function = resizeFunctional(image_buffer);

        const int sh_coefficients = has_sh
            ? static_cast<int>(sh.size(1))
            : 0;
        const float* sh_pointer = has_sh ? sh.data_ptr<float>() : nullptr;
        const float* color_pointer =
            has_colors ? colors.data_ptr<float>() : nullptr;
        const float* scale_pointer =
            has_scale_rotation ? scales.data_ptr<float>() : nullptr;
        const float* rotation_pointer =
            has_scale_rotation ? rotations.data_ptr<float>() : nullptr;
        const float* covariance_pointer =
            has_covariance ? cov3D_precomp.data_ptr<float>() : nullptr;

        num_rendered = CudaRasterizer::Rasterizer::forward(
            geometry_function,
            binning_function,
            image_function,
            gaussian_count,
            degree,
            sh_coefficients,
            background.data_ptr<float>(),
            image_width,
            image_height,
            means3D.data_ptr<float>(),
            sh_pointer,
            color_pointer,
            opacity.data_ptr<float>(),
            scale_pointer,
            scale_modifier,
            rotation_pointer,
            covariance_pointer,
            viewmatrix.data_ptr<float>(),
            projmatrix.data_ptr<float>(),
            campos.data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            background_composited_color.data_ptr<float>(),
            rendered_depth.data_ptr<float>(),
            antialiasing,
            radii.data_ptr<int>(),
            debug);

        char* geometry_pointer =
            reinterpret_cast<char*>(geometry_buffer.data_ptr());
        char* binning_pointer =
            reinterpret_cast<char*>(binning_buffer.data_ptr());
        char* image_pointer =
            reinterpret_cast<char*>(image_buffer.data_ptr());
        auto geometry_state = CudaRasterizer::GeometryState::fromChunk(
            geometry_pointer,
            gaussian_count);
        auto binning_state = CudaRasterizer::BinningState::fromChunk(
            binning_pointer,
            num_rendered);
        auto image_state = CudaRasterizer::ImageState::fromChunk(
            image_pointer,
            image_width * image_height);
        const float* features = has_colors
            ? colors.data_ptr<float>()
            : geometry_state.rgb;

        if (debug)
        {
            sorted_point_list = torch::empty({num_rendered}, int_options);
            const cudaError_t feature_copy_error = cudaMemcpy(
                feature_buffer.data_ptr<float>(),
                features,
                static_cast<size_t>(gaussian_count) * 3 * sizeof(float),
                cudaMemcpyDeviceToDevice);
            TORCH_CHECK(
                feature_copy_error == cudaSuccess,
                "copying GraphDECO feature buffer: ",
                cudaGetErrorString(feature_copy_error));
            const cudaError_t contributor_copy_error = cudaMemcpy(
                contributor_counts.data_ptr<int>(),
                image_state.n_contrib,
                static_cast<size_t>(image_width) * image_height
                    * sizeof(uint32_t),
                cudaMemcpyDeviceToDevice);
            TORCH_CHECK(
                contributor_copy_error == cudaSuccess,
                "copying GraphDECO contributor counts: ",
                cudaGetErrorString(contributor_copy_error));
            const cudaError_t range_copy_error = cudaMemcpy(
                tile_ranges.data_ptr<int>(),
                image_state.ranges,
                static_cast<size_t>(grid_x) * grid_y * sizeof(uint2),
                cudaMemcpyDeviceToDevice);
            TORCH_CHECK(
                range_copy_error == cudaSuccess,
                "copying GraphDECO tile ranges: ",
                cudaGetErrorString(range_copy_error));
            if (num_rendered > 0)
            {
                const cudaError_t point_list_copy_error = cudaMemcpy(
                    sorted_point_list.data_ptr<int>(),
                    binning_state.point_list,
                    static_cast<size_t>(num_rendered) * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice);
                TORCH_CHECK(
                    point_list_copy_error == cudaSuccess,
                    "copying GraphDECO sorted point list: ",
                    cudaGetErrorString(point_list_copy_error));
            }
            if (trace_enabled)
            {
                const cudaError_t means_copy_error = cudaMemcpy(
                    trace_means2D.data_ptr<float>(),
                    geometry_state.means2D,
                    static_cast<size_t>(gaussian_count) * 2 * sizeof(float),
                    cudaMemcpyDeviceToDevice);
                TORCH_CHECK(
                    means_copy_error == cudaSuccess,
                    "copying GraphDECO means2D trace: ",
                    cudaGetErrorString(means_copy_error));
                const cudaError_t conic_copy_error = cudaMemcpy(
                    trace_conic_opacity.data_ptr<float>(),
                    geometry_state.conic_opacity,
                    static_cast<size_t>(gaussian_count) * 4 * sizeof(float),
                    cudaMemcpyDeviceToDevice);
                TORCH_CHECK(
                    conic_copy_error == cudaSuccess,
                    "copying GraphDECO conic_opacity trace: ",
                    cudaGetErrorString(conic_copy_error));
            }
        }

        SENSITIVITY::denominators(
            dim3(grid_x, grid_y, 1),
            dim3(BLOCK_X, BLOCK_Y, 1),
            image_state.ranges,
            binning_state.point_list,
            image_width,
            image_height,
            geometry_state.means2D,
            features,
            geometry_state.conic_opacity,
            image_state.n_contrib,
            use_l2,
            l2_rgb_aggregation,
            nocolor,
            gaussian_only_color.data_ptr<float>(),
            channel_denominators.data_ptr<float>(),
            pixel_denominators.data_ptr<float>(),
            tile_denominators.data_ptr<float>(),
            image_denominator.data_ptr<float>(),
            reinterpret_cast<unsigned long long*>(
                candidate_visits.data_ptr<int64_t>()),
            reinterpret_cast<unsigned long long*>(
                accepted_contributions.data_ptr<int64_t>()),
            trace_enabled
                ? trace_candidate.data_ptr<unsigned char>()
                : nullptr,
            trace_enabled
                ? trace_accepted.data_ptr<unsigned char>()
                : nullptr,
            trace_enabled ? trace_power.data_ptr<float>() : nullptr,
            trace_enabled ? trace_alpha.data_ptr<float>() : nullptr,
            trace_enabled
                ? trace_transmittance_before.data_ptr<float>()
                : nullptr,
            trace_enabled ? trace_weight.data_ptr<float>() : nullptr,
            trace_enabled
                ? trace_contribution.data_ptr<float>()
                : nullptr);
        checkCuda("Gaussian-only sensitivity denominator replay", debug);

        SENSITIVITY::replay(
            dim3(grid_x, grid_y, 1),
            dim3(BLOCK_X, BLOCK_Y, 1),
            image_state.ranges,
            binning_state.point_list,
            image_width,
            image_height,
            geometry_state.means2D,
            features,
            geometry_state.conic_opacity,
            image_state.n_contrib,
            channel_denominators.data_ptr<float>(),
            pixel_denominators.data_ptr<float>(),
            tile_denominators.data_ptr<float>(),
            sensitivity_flags,
            use_l2,
            l2_rgb_aggregation,
            nocolor,
            channel_values.data_ptr<float>(),
            pixel_values.data_ptr<float>(),
            tile_values.data_ptr<float>(),
            image_numerators.data_ptr<float>());
        checkCuda("Gaussian-only sensitivity numerator replay", debug);

        const cudaError_t transmittance_copy_error = cudaMemcpy(
            final_transmittance.data_ptr<float>(),
            image_state.accum_alpha,
            static_cast<size_t>(image_width) * image_height * sizeof(float),
            cudaMemcpyDeviceToDevice);
        TORCH_CHECK(
            transmittance_copy_error == cudaSuccess,
            "copying final transmittance: ",
            cudaGetErrorString(transmittance_copy_error));
    }
    else
    {
        // GraphDECO's composited output for an empty scene is the background;
        // C^G and every Gaussian-only sensitivity denominator remain zero.
        background_composited_color.copy_(
            background.view({3, 1, 1}).expand_as(
                background_composited_color));
    }

    return std::make_tuple(
        num_rendered,
        background_composited_color,
        gaussian_only_color,
        radii,
        final_transmittance,
        contributor_counts,
        tile_ranges,
        sorted_point_list,
        feature_buffer,
        channel_values,
        pixel_values,
        tile_values,
        image_numerators,
        channel_denominators,
        pixel_denominators,
        tile_denominators,
        image_denominator,
        candidate_visits,
        accepted_contributions,
        trace_means2D,
        trace_conic_opacity,
        trace_candidate,
        trace_accepted,
        trace_power,
        trace_alpha,
        trace_transmittance_before,
        trace_weight,
        trace_contribution);
}
