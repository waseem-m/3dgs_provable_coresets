import argparse
import time
import torch

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets.utils.default_paths import ensure_all_on_syspath
ensure_all_on_syspath()

from gs_coresets.utils.default_paths import (
    resolve_cameras_path,
    resolve_model_dir,
    resolve_point_cloud_path,
    resolve_output_dir,
)
from gs_coresets.utils.io_utils import ensure_dir_for, load_cameras_from_json, load_gaussian_ply
from gs_coresets.utils.renderer_utils import iter_render_batches, save_render_batch_images
from gs_coresets.utils.debug_utils import IS_DEBUG_MODE, print_debug
from gs_coresets.utils.types_devices_utils import resolve_torch_device


def run_renderer_cli(
    model_dir: str | None = None,
    cam_json_path: str | None = None,
    cams_per_batch: int = 5,
    save_rgb: bool = True,
    save_alpha: bool = False,
    save_depth: bool = False,
    out_dir: str | None = None,
) -> None:
    model_dir_expanded = resolve_model_dir(model_dir)
    cam_json_path = resolve_cameras_path(cam_json_path, model_dir_expanded)
    ply_path = resolve_point_cloud_path(None, model_dir_expanded)
    device = resolve_torch_device()

    # --- Load assets ---
    ## Load model PLY
    t0 = time.perf_counter()
    ply_orig = load_gaussian_ply(ply_path, device=device)
    t1 = time.perf_counter()
    print_debug(f"PLY loaded from {ply_path}")
    print(f"PLY Loading time: {t1 - t0:.6f} seconds\n")
    ## Load cameras JSON
    t2 = time.perf_counter()
    cameras = load_cameras_from_json(cam_json_path, device=torch.device("cpu"))
    t3 = time.perf_counter()
    print_debug(f"Cameras JSON loaded from {cam_json_path}")
    print_debug(f"Camera load time: {t3 - t2:.6f} seconds")

    # --- Prepare outputs ---
    save_root = resolve_output_dir(out_dir)
    save_dir = save_root / "renderer_batched"
    ensure_dir_for(save_dir)

    # --- Batched rendering (optionally chunked over cameras) ---
    total_start = time.perf_counter()
    render_kwargs = {
        "per_channel_sensitivity": False,
        "per_pixel_sensitivity": False,
        "per_tile_sensitivity": False,
        "per_image_sensitivity": False,
        "per_batch_sensitivity": False,
        "per_scene_sensitivity": False,
    }
    for batch_id, cam_start, batch_cams, out, elapsed in iter_render_batches(
        ply_orig,
        cameras,
        batch_size=cams_per_batch,
        device=device,
        white_background=False,
        block_xy=16,
        render_kwargs=render_kwargs,
    ):
        batch_len = len(batch_cams)
        cam_end = cam_start + batch_len - 1
        print(f"Rendering batch {batch_id} with {batch_len} cameras... (cameras[{cam_start}:{cam_end}])")
        print(f"Batch {batch_id} rendering time: {elapsed:.6f} seconds")

        save_render_batch_images(
            out,
            str(save_dir),
            batch_id=batch_id,
            camera_start_index=cam_start,
            stem="renderer_batched",
            save_rgb=save_rgb,
            save_alpha=save_alpha,
            save_depth=save_depth,
        )

    total_end = time.perf_counter()
    print(f"Total time for rendering & saving: {total_end - total_start:.6f} seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render cameras from a Gaussian splatting scene.")
    parser.add_argument("--model-dir", dest="model_dir", default=None, help="Model directory containing point_cloud.")
    parser.add_argument("--cams-json", dest="cam_json_path", default=None, help="Camera JSON path (defaults to model dir).")
    parser.add_argument("--cams-per-batch", type=int, default=10, help="Number of cameras per render batch.")
    parser.add_argument("--no-rgb", action="store_true", help="Skip saving RGB outputs.")
    parser.add_argument("--save-alpha", action="store_true", help="Save alpha buffers.")
    parser.add_argument("--save-depth", action="store_true", help="Save depth buffers.")
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Base directory for renderer outputs (defaults to GS_CORESETS_OUTPUT_DIR).",
    )
    args = parser.parse_args(argv)
    run_renderer_cli(
        model_dir=args.model_dir,
        cam_json_path=args.cam_json_path,
        cams_per_batch=args.cams_per_batch,
        save_rgb=not args.no_rgb,
        save_alpha=args.save_alpha,
        save_depth=args.save_depth,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
