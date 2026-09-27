# gs_coresets.utils/io_utils.py
# GraphDECO-accurate PLY & JSON loaders
from __future__ import annotations
import os
import json
from typing import Any, List, Optional, Union
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from gs_coresets.utils.types_devices_utils import torch_ctx
from gs_coresets.utils.camera_utils import Camera, camera_from_frame
from gs_coresets.utils.gaussian_utils import GaussianPLY


PathLike = Union[str, os.PathLike]


def expanduser_path(path: str) -> str:
    """Expand shell-style '~' in filesystem paths."""
    return os.path.expanduser(path)

def ensure_dir_for(path: PathLike, *, treat_as_file: bool | None = None) -> Path:
    """
    Create a directory so that `path` can be written without `FileNotFoundError`.

    If `path` looks like a file (e.g., ends with ".json"), we create its parent dir.
    If it looks like a directory path, we create that directory itself.

    Args:
        path: file or directory path.
        treat_as_file: override detection (True=path is a file, False=dir).

    Returns:
        The directory that was ensured/created.
    """
    p = Path(path)
    if treat_as_file is None:
        # Heuristic: if the last component has a dot and isn't a hidden name, treat as file.
        treat_as_file = ('.' in p.name) and (not p.name.startswith('.'))

    d = p.parent if treat_as_file else p
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path: PathLike, *, encoding: str = "utf-8") -> Any:
    """Read JSON data from `path` with UTF-8 encoding by default."""
    p = Path(path).expanduser()
    with p.open("r", encoding=encoding) as handle:
        return json.load(handle)


def write_json(
    path: PathLike,
    data: Any,
    *,
    encoding: str = "utf-8",
    indent: int | None = 2,
    ensure_ascii: bool = False,
    **dump_kwargs: Any,
) -> Path:
    """
    Write JSON data to `path`, creating parent directories automatically.

    Returns the resolved path that was written.
    """
    p = Path(path).expanduser()
    ensure_dir_for(p, treat_as_file=True)
    params: dict[str, Any] = {"indent": indent, "ensure_ascii": ensure_ascii, **dump_kwargs}
    with p.open("w", encoding=encoding) as handle:
        json.dump(data, handle, **params)
    return p


# ------------------------
# PLY loading via plyfile
# ------------------------
def load_gaussian_ply(path: str, max_sh_degree: int = 3, device: Optional[torch.device] = None) -> GaussianPLY:
    """
    GraphDECO-accurate PLY reader:
      - f_dc_0..2 → features_dc (N,3,1)
      - f_rest_0.. → features_rest (N,3,M-1) channel-major (R then G then B)
      - opacity kept as logits
      - scales kept as stored (log-scales)
      - rot_0..3 is WXYZ (normalized)
    """
    device = device or torch_ctx.device

    plydata = PlyData.read(path)
    v = plydata.elements[0]  # 'vertex'
    N = len(v.data)

    # xyz
    xyz = np.stack((
        np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])
    ), axis=1).astype(np.float32)  # (N,3)

    # opacity (logits)
    opacities = np.asarray(v["opacity"], dtype=np.float32)[..., np.newaxis]  # (N,1)

    # DC per channel -> (N,3,1)
    features_dc = np.zeros((N, 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.asarray(v["f_dc_0"], dtype=np.float32)
    features_dc[:, 1, 0] = np.asarray(v["f_dc_1"], dtype=np.float32)
    features_dc[:, 2, 0] = np.asarray(v["f_dc_2"], dtype=np.float32)

    # REST (channel-major) → collect and sort by numeric suffix
    M = (max_sh_degree + 1) ** 2              # total coeffs per channel
    need_rest = 3 * (M - 1)                   # 45 when degree=3
    extra_f_names = [p.name for p in v.properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

    # Read available and pad to expected count if needed (lower degree models)
    features_extra = np.zeros((N, max(need_rest, len(extra_f_names))), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(v[attr_name], dtype=np.float32)

    if features_extra.shape[1] < need_rest:
        # zero-pad up to 3*(M-1)
        pad = need_rest - features_extra.shape[1]
        features_extra = np.pad(features_extra, ((0, 0), (0, pad)), mode="constant")

    # Reshape (N, 3*(M-1)) → (N, 3, M-1)  [R block, then G, then B]
    features_rest = features_extra[:, :need_rest].reshape(N, 3, M - 1)

    # scales (stored log-scales): scale_0..2, sorted by numeric suffix
    scale_names = [p.name for p in v.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    if len(scale_names) != 3:
        raise ValueError(f"Expected 3 scale_* fields, found {scale_names}")
    scales = np.stack([np.asarray(v[n], dtype=np.float32) for n in scale_names], axis=1)  # (N,3)

    # rotations WXYZ: rot_0..3, sorted by numeric suffix, normalize
    rot_names = [p.name for p in v.properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    if len(rot_names) != 4:
        raise ValueError(f"Expected 4 rot_* fields (WXYZ), found {rot_names}")
    rots = np.stack([np.asarray(v[n], dtype=np.float32) for n in rot_names], axis=1)  # (N,4)
    norms = np.linalg.norm(rots, axis=1, keepdims=True).clip(min=1e-8)
    rots = rots / norms

    # Torchify
    return GaussianPLY(
        means=torch.from_numpy(xyz).to(device=device),
        scales=torch.from_numpy(scales).to(device=device),
        rotations_wxyz=torch.from_numpy(rots).to(device=device),
        opacity_logits=torch.from_numpy(opacities).to(device=device),
        features_dc=torch.from_numpy(features_dc).to(device=device),
        features_rest=torch.from_numpy(features_rest).to(device=device),
        sh_degree=int(max_sh_degree),
    )


# ------------------------
# PLY saver via plyfile
# ------------------------

def save_gaussian_ply(path: str, g: GaussianPLY, *, binary: bool = True) -> None:

    N = g.means.shape[0]
    deg = int(g.sh_degree)
    M = (deg + 1) ** 2
    assert g.features_dc.shape == (N, 3, 1)
    assert g.features_rest.ndim == 3 and g.features_rest.shape[0] == N

    xyz = g.means.detach().cpu().numpy().astype(np.float32)               # (N,3)
    normals = np.zeros_like(xyz, dtype=np.float32)                        # (N,3)  <-- ADD
    dc  = g.features_dc.detach().cpu().numpy().astype(np.float32)         # (N,3,1)
    rst = g.features_rest.detach().cpu().numpy().astype(np.float32)       # (N,3,M-1) or (N,M-1,3)
    op  = g.opacity_logits.detach().cpu().numpy().astype(np.float32)      # (N,1)
    sc  = g.scales.detach().cpu().numpy().astype(np.float32)              # (N,3)  (log-scales)
    rot = g.rotations_wxyz.detach().cpu().numpy().astype(np.float32)      # (N,4)  (WXYZ)
    rot /= np.clip(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8, None)

    # Make sure SH rest is channel-major [R..., G..., B...]
    if rst.shape[1] == 3:                 # (N,3,M-1)
        rest_flat = rst.reshape(N, 3*(M-1))
    elif rst.shape[2] == 3:               # (N,M-1,3) → transpose to (N,3,M-1)
        rest_flat = np.transpose(rst, (0,2,1)).reshape(N, 3*(M-1))
    else:
        raise ValueError("features_rest must be (N,3,M-1) or (N,M-1,3)")

    dtype = (
        [("x","f4"),("y","f4"),("z","f4"),
         ("nx","f4"),("ny","f4"),("nz","f4")] +                     # <-- ADD normals
        [("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4")] +
        [(f"f_rest_{i}","f4") for i in range(3*(M-1))] +
        [("opacity","f4"),
         ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
         ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]
    )

    v = np.empty(N, dtype=dtype)
    v["x"], v["y"], v["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    v["nx"], v["ny"], v["nz"] = normals[:,0], normals[:,1], normals[:,2]   # <-- ADD

    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = dc[:,0,0], dc[:,1,0], dc[:,2,0]
    for i in range(3*(M-1)):
        v[f"f_rest_{i}"] = rest_flat[:, i]

    v["opacity"] = op[:,0]
    v["scale_0"], v["scale_1"], v["scale_2"] = sc[:,0], sc[:,1], sc[:,2]
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = rot[:,0], rot[:,1], rot[:,2], rot[:,3]

    el = PlyElement.describe(v, "vertex")
    PlyData([el], text=not binary).write(path)


# ------------------------------------------------
# Returns individual camera frames from a JSON container
# ------------------------------------------------
def frames_from_container(data):
    """Normalize a camera JSON root to (frames, defaults). No math here."""
    if isinstance(data, list):
        return data, {}
    frames = (
        data.get("frames")
        or data.get("cameras")
        or data.get("train_cameras")
        or data.get("test_cameras")
        or []
    )
    # Allow single-camera dict at root
    if frames == [] and any(k in data for k in (
        "w","width","h","height",
        "FoVx","fov_x","fovX","FoVy","fov_y","fovY",
        "fx","fy","fl_x","fl_y","K","k","intrinsics",
        "world_to_camera","world_view_transform","view_matrix",
        "camera_to_world","transform_matrix","rotation","R","pose","transform"
    )):
        frames = [data]
    return frames, data


# ------------------------
# Public camera loader
# ------------------------
def _load_cameras_from_json_cpu(path: str) -> List[Camera]:
    """
    File-I/O wrapper only. Reads JSON, delegates parsing and construction
    to camera_class (GraphDECO-compatible).
    """
    device = torch.device("cpu")
    data = read_json(path)
    frames, defaults = frames_from_container(data)
    cams: List[Camera] = [camera_from_frame(fr, defaults, device=device) for fr in frames]
    return cams

def load_cameras_from_json(path: str, device: Optional[torch.device] = None) -> List[Camera]:
    """
    Wrapper of _load_cameras_from_json_cpu for backwards compatibility.
    Ignores device, always uses CPU.
    """
    return _load_cameras_from_json_cpu(path)
