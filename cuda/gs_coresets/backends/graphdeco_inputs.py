"""GraphDECO-native model and camera preparation for the CUDA backend.

This module deliberately does not replace the regular package loaders.  The
released PyTorch implementation keeps using those historical loaders, while
the CUDA sensitivity backend uses the conventions implemented by stock
GraphDECO.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from plyfile import PlyData

from gs_coresets.utils.camera_utils import (
    Camera,
    fovs_from_intrinsics_dict,
    get_dims,
)
from gs_coresets.utils.gaussian_utils import GaussianPLY
from gs_coresets.utils.io_utils import frames_from_container, read_json
from gs_coresets.utils.types_devices_utils import as_4x4_np


CAMERA_CONVENTIONS = ("auto", "graphdeco", "w2c", "c2w")

_DIRECT_W2C_KEYS: Sequence[str] = (
    "world_to_camera",
    "w2c",
    "view_matrix",
    "world_view",
    "world_view_transform",
)
_DIRECT_C2W_KEYS: Sequence[str] = (
    "camera_to_world",
    "c2w",
    "transform_matrix",
    "pose",
    "transform",
    "matrix",
    "extrinsic_matrix",
)


def _first_present(frame: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in frame and frame[key] is not None:
            return frame[key]
    return None


def _matrix_from_rotation_translation(
    rotation: Any,
    translation: Any,
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return matrix


def _graphdeco_w2c(rotation: Any, translation: Any) -> np.ndarray:
    """Reproduce ``getWorld2View2(R, T)`` for translate=0 and scale=1."""
    # GraphDECO constructs Rt as NumPy float64, performs both inversions in
    # that precision, and casts only the returned W2C to float32.  Preserve
    # that ordering for extracted JSON values rather than rounding R/T first.
    stored_rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = stored_rotation.transpose()
    w2c[:3, 3] = np.asarray(
        translation,
        dtype=np.float64,
    ).reshape(3)

    # Keep the same conceptual construction as GraphDECO's getWorld2View2.
    # With its default translate/scale this round trip is mathematically a
    # no-op, but spelling it out makes the convention unambiguous.
    c2w = np.linalg.inv(w2c)
    return np.linalg.inv(c2w).astype(np.float32)


def graphdeco_w2c_from_frame(
    frame: Dict[str, Any],
    *,
    convention: str = "auto",
) -> tuple[np.ndarray, str]:
    """Return a row-major W2C matrix and the convention actually selected."""
    if convention not in CAMERA_CONVENTIONS:
        raise ValueError(
            f"camera convention must be one of {CAMERA_CONVENTIONS}, "
            f"got {convention!r}"
        )

    direct_w2c = _first_present(frame, _DIRECT_W2C_KEYS)
    direct_c2w = _first_present(frame, _DIRECT_C2W_KEYS)
    graphdeco_r = frame.get("R")
    graphdeco_t = frame.get("T")
    rotation = frame.get("rotation")
    position = frame.get("position")

    selected = convention
    if convention == "auto":
        if direct_w2c is not None:
            selected = "w2c"
        elif direct_c2w is not None:
            selected = "c2w"
        elif (
            graphdeco_r is not None
            and graphdeco_t is not None
            and rotation is not None
            and position is not None
        ):
            # This is the schema emitted by extract_cameras.py through
            # GraphDECO's camera_to_JSON: R/T retain GraphDECO's stored
            # convention and rotation/position redundantly describe C2W.
            selected = "graphdeco"
        elif graphdeco_r is not None and graphdeco_t is not None:
            raise ValueError(
                "bare R/T camera extrinsics are ambiguous; pass "
                "--camera-convention graphdeco, w2c, or c2w"
            )
        elif rotation is not None and position is not None:
            selected = "c2w"
        else:
            raise ValueError("camera frame does not contain recognized extrinsics")

    if selected == "graphdeco":
        if graphdeco_r is None or graphdeco_t is None:
            raise ValueError("GraphDECO camera convention requires R and T")
        w2c = _graphdeco_w2c(graphdeco_r, graphdeco_t)
    elif selected == "w2c":
        if direct_w2c is not None:
            w2c = as_4x4_np(direct_w2c).astype(np.float32)
        elif graphdeco_r is not None and graphdeco_t is not None:
            w2c = _matrix_from_rotation_translation(
                graphdeco_r,
                graphdeco_t,
            )
        else:
            raise ValueError("w2c convention requires a W2C matrix or R/T")
    else:
        if direct_c2w is not None:
            c2w = as_4x4_np(direct_c2w).astype(np.float32)
        elif rotation is not None and position is not None:
            c2w = _matrix_from_rotation_translation(rotation, position)
        elif graphdeco_r is not None and graphdeco_t is not None:
            c2w = _matrix_from_rotation_translation(
                graphdeco_r,
                graphdeco_t,
            )
        else:
            raise ValueError("c2w convention requires a C2W matrix or pose")
        w2c = np.linalg.inv(c2w).astype(np.float32)

    if not np.isfinite(w2c).all():
        raise ValueError("camera transform contains non-finite values")

    if selected == "graphdeco" and position is not None:
        reconstructed = np.linalg.inv(w2c)[:3, 3]
        expected = np.asarray(position, dtype=np.float32).reshape(3)
        if not np.allclose(reconstructed, expected, atol=1e-4, rtol=1e-5):
            raise ValueError(
                "GraphDECO R/T and redundant camera position disagree"
            )

    if selected == "graphdeco" and rotation is not None:
        reconstructed_rotation = np.linalg.inv(w2c)[:3, :3]
        expected_rotation = np.asarray(
            rotation,
            dtype=np.float32,
        ).reshape(3, 3)
        if not np.allclose(
            reconstructed_rotation,
            expected_rotation,
            atol=1e-4,
            rtol=1e-5,
        ):
            raise ValueError(
                "GraphDECO R/T and redundant camera rotation disagree"
            )
    return w2c, selected


def load_graphdeco_cameras_from_json(
    path: str | Path,
    *,
    convention: str = "auto",
) -> List[Camera]:
    """Load cameras without passing GraphDECO R/T through the PyTorch parser."""
    data = read_json(path)
    frames, defaults = frames_from_container(data)
    cameras: List[Camera] = []
    for frame in frames:
        width, height = get_dims(frame, defaults)
        fov_x, fov_y = fovs_from_intrinsics_dict(
            frame,
            width,
            height,
        )
        w2c, selected = graphdeco_w2c_from_frame(
            frame,
            convention=convention,
        )
        w2c_tensor = torch.tensor(w2c, dtype=torch.float32)
        camera = Camera.from_view_fov(
            w2c_tensor,
            width,
            height,
            fov_x,
            fov_y,
            near=0.01,
            far=100.0,
        )
        camera.camera_center = torch.linalg.inv(w2c_tensor)[:3, 3]
        camera.graphdeco_convention = selected
        camera.source_id = frame.get("id", frame.get("uid"))
        camera.image_name = frame.get(
            "image_name",
            frame.get("img_name"),
        )
        cameras.append(camera)
    return cameras


def _infer_sh_degree(rest_field_count: int) -> int:
    if rest_field_count % 3 != 0:
        raise ValueError(
            "GraphDECO f_rest_* field count must be divisible by three"
        )
    coefficients = rest_field_count // 3 + 1
    root = math.isqrt(coefficients)
    if root * root != coefficients:
        raise ValueError(
            "GraphDECO SH fields do not describe a complete coefficient set"
        )
    degree = root - 1
    if degree < 0 or degree > 3:
        raise ValueError(f"unsupported GraphDECO SH degree {degree}")
    return degree


def load_graphdeco_gaussian_ply(
    path: str | Path,
    *,
    active_sh_degree: int | None = None,
) -> GaussianPLY:
    """Load stored PLY parameters without changing GraphDECO activation order."""
    ply = PlyData.read(str(path))
    vertices = ply.elements[0]
    count = len(vertices.data)
    property_names = {prop.name for prop in vertices.properties}

    def sorted_fields(prefix: str) -> List[str]:
        return sorted(
            (name for name in property_names if name.startswith(prefix)),
            key=lambda name: int(name.split("_")[-1]),
        )

    xyz = np.stack(
        [
            np.asarray(vertices["x"], dtype=np.float32),
            np.asarray(vertices["y"], dtype=np.float32),
            np.asarray(vertices["z"], dtype=np.float32),
        ],
        axis=1,
    )
    opacity = np.asarray(
        vertices["opacity"],
        dtype=np.float32,
    )[:, np.newaxis]
    features_dc = np.stack(
        [
            np.asarray(vertices[f"f_dc_{channel}"], dtype=np.float32)
            for channel in range(3)
        ],
        axis=1,
    )[:, :, np.newaxis]

    rest_names = sorted_fields("f_rest_")
    maximum_degree = _infer_sh_degree(len(rest_names))
    coefficients_per_channel = (maximum_degree + 1) ** 2 - 1
    if rest_names:
        rest_flat = np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rest_names
            ],
            axis=1,
        )
        features_rest = rest_flat.reshape(
            count,
            3,
            coefficients_per_channel,
        )
    else:
        features_rest = np.empty((count, 3, 0), dtype=np.float32)

    scale_names = sorted_fields("scale_")
    rotation_names = sorted_fields("rot_")
    if len(scale_names) != 3:
        raise ValueError(f"expected three scale_* fields, found {scale_names}")
    if len(rotation_names) != 4:
        raise ValueError(
            f"expected four rot_* fields, found {rotation_names}"
        )
    scales = np.stack(
        [
            np.asarray(vertices[name], dtype=np.float32)
            for name in scale_names
        ],
        axis=1,
    )
    # Deliberately preserve the stored quaternion.  Stock GaussianModel keeps
    # this raw value and applies normalize() only in get_rotation on the GPU.
    rotations = np.stack(
        [
            np.asarray(vertices[name], dtype=np.float32)
            for name in rotation_names
        ],
        axis=1,
    )

    selected_degree = (
        maximum_degree
        if active_sh_degree is None
        else int(active_sh_degree)
    )
    if selected_degree < 0 or selected_degree > maximum_degree:
        raise ValueError(
            f"active SH degree must be in [0, {maximum_degree}], "
            f"got {selected_degree}"
        )

    for name, value in (
        ("means", xyz),
        ("opacity", opacity),
        ("features_dc", features_dc),
        ("features_rest", features_rest),
        ("scales", scales),
        ("rotations", rotations),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    model = GaussianPLY(
        means=torch.from_numpy(xyz),
        scales=torch.from_numpy(scales),
        rotations_wxyz=torch.from_numpy(rotations),
        opacity_logits=torch.from_numpy(opacity),
        features_dc=torch.from_numpy(features_dc),
        features_rest=torch.from_numpy(features_rest),
        sh_degree=selected_degree,
    )
    model.maximum_sh_degree = maximum_degree
    model.source_path = str(Path(path).expanduser().resolve())
    return model


__all__ = [
    "CAMERA_CONVENTIONS",
    "graphdeco_w2c_from_frame",
    "load_graphdeco_cameras_from_json",
    "load_graphdeco_gaussian_ply",
]
