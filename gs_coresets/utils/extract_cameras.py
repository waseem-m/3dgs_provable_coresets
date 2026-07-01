# gs_coresets.utils/extract_cameras.py
"""
Extract GraphDECO training/testing cameras to JSON files.

Usage (Python):
    from gs_coresets.utils.extract_cameras import extract_train_cameras_json, extract_test_cameras_json, extract_both_jsons

    extract_train_cameras_json(
        source_path="/path/to/dataset",
        model_path="/tmp/gs_model_dir",
        out_json="/path/to/extract_cameras_train.json",
        white_background=False,   # True for Blender white bg
        resolution=-1,             # 1 full-res, 2/4/8 downscale, -1 auto
        images_subdir="images",   # for COLMAP datasets
        data_device="cpu",        # "cpu" is fine; we only serialize metadata
        load_test_split=True,     # needed only for test extraction
    )

    extract_test_cameras_json(...)

    # Or both at once:
    extract_both_jsons(
        source_path="/path/to/dataset",
        model_path="/tmp/gs_model_dir",
        out_dir="/path/to/output_dir",
        white_background=False,
        resolution=-1,
        images_subdir="images",
        data_device="cpu",
        load_test_split=True,
        # filenames can be customized:
        train_filename="extract_cameras_train.json",
        test_filename="extract_cameras_test.json",
    )
"""

from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Tuple

# --- Ensure we can import from the GraphDECO submodule ------------------------
from gs_coresets.utils.default_paths import (
    resolve_graphdeco_dir,
)

_GD_PATH = resolve_graphdeco_dir()

try:
    from scene import Scene, GaussianModel  # type: ignore
    from utils.camera_utils import camera_to_JSON  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        f"Failed to import GraphDECO from '{_GD_PATH}'. "
        "Make sure the submodule exists and is checked out. "
        f"Original error: {e}"
    ) from e

from gs_coresets.utils.io_utils import ensure_dir_for


# --- Minimal, stable camera serialization ------------------------------------
def _to_list(x: Any) -> Any:
    """Best-effort: torch.Tensor -> Python list; otherwise return as-is."""
    try:
        # torch.Tensor path
        return x.detach().cpu().tolist()  # type: ignore[attr-defined]
    except Exception:
        to_list = getattr(x, "tolist", None)
        if callable(to_list):
            return to_list()
        return x

def _cam_to_dict(cam: Any) -> Dict[str, Any]:
    """
    Keep it compact and stable across COLMAP/Blender.
    (Add fields here if you need more later.)
    """
    width = int(getattr(cam, "width", getattr(cam, "image_width", 0)) or 0)
    height = int(getattr(cam, "height", getattr(cam, "image_height", 0)) or 0)
    if not hasattr(cam, "width"):
        setattr(cam, "width", width)
    if not hasattr(cam, "height"):
        setattr(cam, "height", height)
    if not hasattr(cam, "FovX"):
        setattr(cam, "FovX", getattr(cam, "FoVx", 0.0))
    if not hasattr(cam, "FovY"):
        setattr(cam, "FovY", getattr(cam, "FoVy", 0.0))

    base = camera_to_JSON(getattr(cam, "uid", 0), cam)
    fovx = float(getattr(cam, "FoVx", 0.0) or 0.0)
    fovy = float(getattr(cam, "FoVy", 0.0) or 0.0)
    fx = float(base.get("fx", 0.0) or 0.0)
    fy = float(base.get("fy", 0.0) or 0.0)
    cx, cy = 0.5 * width, 0.5 * height

    out: Dict[str, Any] = {
        "id": int(base.get("id", getattr(cam, "uid", -1))),
        "uid": int(getattr(cam, "uid", base.get("id", -1))),
        "image_name": base.get("img_name", getattr(cam, "image_name", None)),
        "width": width,
        "height": height,
        "FoVx": fovx,
        "FoVy": fovy,
        "R": _to_list(getattr(cam, "R", None)),
        "T": _to_list(getattr(cam, "T", None)),
        "position": base.get("position"),
        "rotation": base.get("rotation"),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }

    return {k: v for k, v in out.items() if v is not None}


# --- Scene construction -------------------------------------------------------
def _make_scene(
    source_path: str,
    model_path: str,
    *,
    resolution: int = -1,
    white_background: bool = False,
    images_subdir: str = "images",
    data_device: str = "cpu",
    load_test_split: bool = True,
) -> Scene:
    """
    Build a GraphDECO Scene that auto-detects COLMAP vs Blender from source_path.
    """
    args = SimpleNamespace(
        source_path=source_path,
        model_path=model_path,
        images=images_subdir,
        depths="",
        eval=bool(load_test_split),
        white_background=bool(white_background),
        resolution=int(resolution),
        data_device=str(data_device),
        train_test_exp=False,
    )
    gaussians = GaussianModel(sh_degree=3)
    # Only need scale 1.0; add others if you downscale/upsample elsewhere.
    return Scene(args, gaussians, shuffle=False, resolution_scales=[1.0])


def _extract_cameras(cameras: Iterable[Any], out_path: Path) -> int:
    payload = [_cam_to_dict(c) for c in cameras]
    ensure_dir_for(out_path, treat_as_file=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return len(payload)


# --- Extract functions (separate JSON files) ----------------------------------
def extract_train_cameras_json(
    *,
    source_path: str,
    model_path: str,
    out_json: str,
    resolution: int = -1,
    white_background: bool = False,
    images_subdir: str = "images",
    data_device: str = "cpu",
) -> int:
    """
    Save training cameras to a JSON file. Returns number of cameras saved.
    """
    scene = _make_scene(
        source_path=source_path,
        model_path=model_path,
        resolution=resolution,
        white_background=white_background,
        images_subdir=images_subdir,
        data_device=data_device,
        load_test_split=False,  # not needed for train
    )
    cams = scene.getTrainCameras(1.0) or []
    return _extract_cameras(cams, Path(out_json))


def extract_test_cameras_json(
    *,
    source_path: str,
    model_path: str,
    out_json: str,
    resolution: int = -1,
    white_background: bool = False,
    images_subdir: str = "images",
    data_device: str = "cpu",
) -> int:
    """
    Save testing/eval cameras to a JSON file (if present). Returns number of cameras saved.
    """
    scene = _make_scene(
        source_path=source_path,
        model_path=model_path,
        resolution=resolution,
        white_background=white_background,
        images_subdir=images_subdir,
        data_device=data_device,
        load_test_split=True,  # needed to load eval/test split
    )
    cams = scene.getTestCameras(1.0) or []
    return _extract_cameras(cams, Path(out_json))


# --- Convenience: extract both into a directory --------------------------------
def extract_both_jsons(
    *,
    source_path: str,
    model_path: str,
    out_dir: str,
    resolution: int = -1,
    white_background: bool = False,
    images_subdir: str = "images",
    data_device: str = "cpu",
    train_filename: str = "extract_cameras_train.json",
    test_filename: str = "extract_cameras_test.json",
) -> Tuple[int, int]:
    """
    Save both train and test cameras to separate JSON files in out_dir.
    Returns (num_train, num_test).
    """
    out_dir_path = Path(out_dir)
    ensure_dir_for(out_dir_path, treat_as_file=False)

    scene = _make_scene(
        source_path=source_path,
        model_path=model_path,
        resolution=resolution,
        white_background=white_background,
        images_subdir=images_subdir,
        data_device=data_device,
        load_test_split=True,  # needed to load eval/test split
    )
    cams_test = scene.getTestCameras(1.0) or []
    cams_train = scene.getTrainCameras(1.0) or []

    n_test = _extract_cameras(cams_test, out_dir_path / test_filename)
    n_train = _extract_cameras(cams_train, out_dir_path / train_filename)

    # combined = out_dir_path / "extract_cameras_all.json"
    # train_payload = json.loads((out_dir_path / train_filename).read_text(encoding="utf-8")) if n_train else []
    # test_payload = json.loads((out_dir_path / test_filename).read_text(encoding="utf-8")) if n_test else []
    # ensure_dir_for(combined, treat_as_file=True)
    # with combined.open("w", encoding="utf-8") as handle:
    #     json.dump(train_payload + test_payload, handle, indent=2)

    return n_train, n_test

def _detect_dataset_type(source_path: str) -> str:
    sp = Path(source_path)
    if (sp / "transforms_train.json").exists() or any(sp.glob("transforms_*.json")):
        return "blender"
    if (sp / "sparse").exists():
        return "colmap"
    return "unknown"

def _guess_images_dir(source_path: str) -> str:
    sp = Path(source_path)
    for name in ("images", "Images", "imgs", "rgb", "RGB"):
        if (sp / name).exists():
            return name
    return "images"  # sane default for COLMAP

def extract_both_jsons_auto(
    *,
    source_path: str,
    model_path: str,
    out_dir: str,
    resolution: int = -1,
    data_device: str = "cpu",
    train_filename: str = "extract_cameras_train.json",
    test_filename: str = "extract_cameras_test.json",
):
    """
    One unified call for both COLMAP and Blender. We probe the dataset and pick sensible defaults.
    """
    kind = _detect_dataset_type(source_path)
    white_background = (kind == "blender")  # NeRF-Synthetic convention; harmless for camera extraction
    images_subdir = _guess_images_dir(source_path)  # ignored by Blender; used by COLMAP

    return extract_both_jsons(
        source_path=source_path,
        model_path=model_path,
        out_dir=out_dir,
        resolution=resolution,
        white_background=white_background,
        images_subdir=images_subdir,
        data_device=data_device,
        train_filename=train_filename,
        test_filename=test_filename,
    )

"""
# A single unified call for both COLMAP and Blender:
from gs_coresets.utils.extract_cameras import extract_both_jsons_auto
extract_both_jsons_auto(
    source_path="/data/any_scene",   # COLMAP (has sparse/) or Blender (has transforms_*.json)
    model_path="/tmp/gs_model_dir",
    out_dir="/data/any_scene/cams",
)
"""
