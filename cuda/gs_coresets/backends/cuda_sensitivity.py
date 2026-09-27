"""GraphDECO-native CUDA sensitivity backend.

The extension processes one camera at a time.  Python combines the camera
partials into image and scene definitions without changing the mathematical
meaning of the optional camera processing chunks.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Sequence

import torch

from gs_coresets.utils.camera_utils import Camera
from gs_coresets.utils.gaussian_utils import GaussianPLY


PER_CHANNEL = 1
PER_PIXEL = 2
PER_TILE = 4
PER_IMAGE = 8
PER_SCENE = 16
SENSITIVITY_BOUND_TOLERANCE = 1.0e-6


class CudaSensitivityUnavailable(RuntimeError):
    """Raised when the compiled CUDA backend cannot be used."""


def _load_extension():
    try:
        from diff_gaussian_sensitivity import (
            GaussianSensitivitySettings,
            rasterize_gaussians_sensitivity,
        )
    except Exception as exc:
        raise CudaSensitivityUnavailable(
            "CUDA sensitivity backend is unavailable. Install the in-tree "
            "extension with `python -m pip install --no-build-isolation "
            "./extensions/diff_gaussian_sensitivity`, or rerun explicitly with "
            "`--backend pytorch`."
        ) from exc
    return GaussianSensitivitySettings, rasterize_gaussians_sensitivity


def _projection_matrix(
    fov_x: float,
    fov_y: float,
    *,
    z_near: float = 0.01,
    z_far: float = 100.0,
    device: torch.device,
) -> torch.Tensor:
    """Return the transposed projection matrix consumed by GraphDECO CUDA."""
    tan_x = math.tan(float(fov_x) * 0.5)
    tan_y = math.tan(float(fov_y) * 0.5)
    matrix = torch.zeros((4, 4), dtype=torch.float32, device="cpu")
    matrix[0, 0] = 1.0 / tan_x
    matrix[1, 1] = 1.0 / tan_y
    matrix[3, 2] = 1.0
    matrix[2, 2] = z_far / (z_far - z_near)
    matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return matrix.transpose(0, 1).to(device=device)


def _camera_settings(
    camera: Camera,
    background: torch.Tensor,
    settings_type,
    *,
    sh_degree: int,
    debug: bool = False,
):
    """Construct exactly the matrices supplied by a stock GraphDECO Camera."""
    device = background.device
    row_major_w2c = camera.world_view_transform.to(
        device="cpu",
        dtype=torch.float32,
    )
    view = row_major_w2c.transpose(0, 1).to(device=device)
    projection = _projection_matrix(
        camera.FoVx,
        camera.FoVy,
        device=device,
    )
    full_projection = (
        view.unsqueeze(0)
        .bmm(projection.unsqueeze(0))
        .squeeze(0)
    )
    camera_center = torch.linalg.inv(view)[3, :3].contiguous()
    view_for_extension = view.contiguous()
    return settings_type(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=math.tan(float(camera.FoVx) * 0.5),
        tanfovy=math.tan(float(camera.FoVy) * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view_for_extension,
        projmatrix=full_projection,
        sh_degree=int(sh_degree),
        campos=camera_center,
        prefiltered=False,
        antialiasing=False,
        debug=bool(debug),
    )


def _model_tensors(
    model: GaussianPLY,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Apply stock GraphDECO activations once, on the target CUDA device."""
    features = torch.cat(
        (model.features_dc, model.features_rest),
        dim=2,
    )
    features = (
        features.transpose(1, 2)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    required_coefficients = (int(model.sh_degree) + 1) ** 2
    if features.ndim != 3 or features.shape[2] != 3:
        raise ValueError("GraphDECO SH features must have shape (N, M, 3)")
    if features.shape[1] < required_coefficients:
        raise ValueError(
            f"active SH degree {model.sh_degree} requires "
            f"{required_coefficients} coefficients"
        )

    means = model.means.to(
        device=device,
        dtype=torch.float32,
    ).contiguous()
    raw_scales = model.scales.to(
        device=device,
        dtype=torch.float32,
    )
    raw_rotations = model.rotations_wxyz.to(
        device=device,
        dtype=torch.float32,
    )
    raw_opacities = model.opacity_logits.to(
        device=device,
        dtype=torch.float32,
    )
    tensors = {
        "means": means,
        "scales": torch.exp(raw_scales).contiguous(),
        # Stock GaussianModel stores the raw PLY quaternion and performs this
        # single normalization in get_rotation on the CUDA device.
        "rotations": torch.nn.functional.normalize(
            raw_rotations,
            dim=-1,
        ).contiguous(),
        "opacities": torch.sigmoid(raw_opacities).contiguous(),
        "sh": features,
        "empty": torch.empty((0,), dtype=torch.float32, device=device),
    }
    for name, tensor in tensors.items():
        if name != "empty" and not torch.isfinite(tensor).all():
            raise ValueError(f"prepared GraphDECO tensor {name} is non-finite")
    return tensors


def _ratio_or_zero(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def _validate_sensitivity(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values")
    if tensor.numel() == 0:
        return
    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())
    tolerance = SENSITIVITY_BOUND_TOLERANCE
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise RuntimeError(
            f"{name} violates Gaussian-contribution sensitivity bounds: "
            f"min={minimum:.9g}, max={maximum:.9g}"
        )


def compute_cuda_sensitivities(
    model: GaussianPLY,
    cameras: Sequence[Camera],
    *,
    device: torch.device,
    cams_per_batch: int,
    norm: str,
    reduction: str,
    nocolor: bool,
    per_channel: bool,
    per_pixel: bool,
    per_tile: bool,
    per_image: bool,
    per_scene: bool,
    background: Sequence[float],
    white_background: bool,
) -> tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise CudaSensitivityUnavailable(
            "The CUDA sensitivity backend requires an available CUDA GPU. "
            "Use `--backend pytorch` explicitly for the regular backend."
        )
    if norm == "l2-channel":
        norm = "l2"
    if norm not in {"l1", "l2", "l2-agg"}:
        raise ValueError(
            "CUDA sensitivity norm must be 'l1', 'l2', 'l2-channel', "
            "or 'l2-agg'"
        )
    if reduction != "max":
        raise ValueError("the corrected CUDA backend supports max reduction only")
    if not cameras:
        raise ValueError("at least one camera is required")
    if cams_per_batch <= 0:
        raise ValueError("cams_per_batch must be positive")
    if not any((per_channel, per_pixel, per_tile, per_image, per_scene)):
        raise ValueError("at least one CUDA sensitivity granularity is required")
    if len(background) != 3 or not all(math.isfinite(float(v)) for v in background):
        raise ValueError("background must contain three finite values")

    settings_type, rasterize = _load_extension()
    tensors = _model_tensors(model, device)
    count = int(tensors["means"].shape[0])
    dtype = torch.float32
    background_tensor = torch.tensor(
        background,
        dtype=dtype,
        device=device,
    )
    if white_background:
        background_tensor.fill_(1.0)

    flags = (
        int(per_channel) * PER_CHANNEL
        | int(per_pixel) * PER_PIXEL
        | int(per_tile) * PER_TILE
        | int(per_image) * PER_IMAGE
        | int(per_scene) * PER_SCENE
    )
    zero = lambda: torch.zeros(count, dtype=dtype, device=device)
    channel_value = zero() if per_channel else None
    pixel_value = zero() if per_pixel else None
    tile_value = zero() if per_tile else None
    image_value = zero() if per_image else None
    scene_numerator = zero() if per_scene else None
    scene_denominator = (
        torch.zeros((), dtype=dtype, device=device)
        if per_scene
        else None
    )

    timings: List[Dict[str, Any]] = []
    for batch_id, camera_start in enumerate(
        range(0, len(cameras), cams_per_batch)
    ):
        batch_cameras = list(
            cameras[camera_start : camera_start + cams_per_batch]
        )
        for local_camera, camera in enumerate(batch_cameras):
            settings = _camera_settings(
                camera,
                background_tensor,
                settings_type,
                sh_degree=int(model.sh_degree),
            )
            started = time.perf_counter()
            output = rasterize(
                means3D=tensors["means"],
                sh=tensors["sh"],
                colors_precomp=tensors["empty"],
                opacities=tensors["opacities"],
                scales=tensors["scales"],
                rotations=tensors["rotations"],
                cov3D_precomp=tensors["empty"],
                settings=settings,
                sensitivity_flags=flags,
                norm=norm,
                reduction=reduction,
                nocolor=nocolor,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            camera_denominator = output.image_denominator.reshape(())
            global_camera = camera_start + local_camera
            timings.append(
                {
                    "batch": batch_id,
                    "camera": global_camera,
                    "seconds": elapsed,
                    "rendered_tile_overlaps": int(
                        output.rendered_tile_overlaps
                    ),
                    "denominator_replay_candidate_visits": int(
                        output.candidate_visits.item()
                    ),
                    "accepted_gaussian_pixel_contributions": int(
                        output.accepted_contributions.item()
                    ),
                    "renderer_stages": {
                        "feature_buffer": {
                            "producer": "GraphDECO preprocess/computeColorFromSH",
                            "background_added": False,
                            "final_clamp_applied": False,
                        },
                        "gaussian_only_color": {
                            "producer": "sparse denominator replay sum_g(a_g)",
                            "background_added": False,
                            "final_clamp_applied": False,
                        },
                        "background_composited_color": {
                            "producer": "GraphDECO renderCUDA/out_color",
                            "background_added": True,
                            "final_clamp_applied": False,
                        },
                    },
                }
            )

            if per_channel:
                torch.maximum(
                    channel_value,
                    output.channel_value,
                    out=channel_value,
                )
            if per_pixel:
                torch.maximum(
                    pixel_value,
                    output.pixel_value,
                    out=pixel_value,
                )
            if per_tile:
                torch.maximum(
                    tile_value,
                    output.tile_value,
                    out=tile_value,
                )

            camera_ratio = _ratio_or_zero(
                output.image_numerator,
                camera_denominator,
            )
            if per_image:
                torch.maximum(
                    image_value,
                    camera_ratio,
                    out=image_value,
                )
            if per_scene:
                scene_numerator += output.image_numerator
                scene_denominator += camera_denominator

    results: Dict[str, torch.Tensor] = {}
    if per_channel:
        results["sens_channel"] = channel_value
    if per_pixel:
        results["sens_pixel"] = pixel_value
    if per_tile:
        results["sens_tile"] = tile_value
    if per_image:
        results["sens_image"] = image_value
    if per_scene:
        results["sens_scene"] = _ratio_or_zero(
            scene_numerator,
            scene_denominator,
        )

    for name, tensor in results.items():
        _validate_sensitivity(name, tensor)
    return results, timings


__all__ = [
    "CudaSensitivityUnavailable",
    "SENSITIVITY_BOUND_TOLERANCE",
    "compute_cuda_sensitivities",
]
