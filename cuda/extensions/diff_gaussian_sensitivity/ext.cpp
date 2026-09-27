/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Modifications Copyright (C) 2026 Waseem Mousa and Alaa Maalouf.
 * See LICENSE.md.
 */

#include <torch/extension.h>

#include "sensitivity_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "rasterize_gaussians_sensitivity",
        &RasterizeGaussiansSensitivityCUDA,
        "GraphDECO forward rasterization and coreset sensitivity (CUDA)");
}
