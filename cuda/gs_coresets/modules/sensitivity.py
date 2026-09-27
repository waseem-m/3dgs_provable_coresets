import argparse
import json
import os, sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets.utils.default_paths import ensure_all_on_syspath
ensure_all_on_syspath()

from gs_coresets.utils.camera_utils import Camera
from gs_coresets.utils.default_paths import (
    resolve_cameras_path,
    resolve_model_dir,
    resolve_point_cloud_path,
    resolve_sensitivity_output_dir,
    resolve_test_sensitivity_output_dir,
)
from gs_coresets.utils.gaussian_utils import GaussianPLY
from gs_coresets.utils.io_utils import ensure_dir_for, load_cameras_from_json, load_gaussian_ply
from gs_coresets.utils.renderer_utils import iter_render_batches
from gs_coresets.utils.types_devices_utils import resolve_torch_device, torch_ctx

# Optional: render cameras in chunks to control VRAM. Set to None for single batch.
  # e.g., 4 or 8 if you run out of memory


def save_sensitivity(sens : torch.Tensor = None, file_name : str = None, save_dir : str = None) -> None:
    if sens is None:
        return
    if file_name is None:
        file_name = "sensitivities.pt"

    target_dir = resolve_test_sensitivity_output_dir(save_dir)
    ensure_dir_for(target_dir, treat_as_file=False)

    # --- Save sensitivities
    sens_nz = int(torch.count_nonzero(sens).item())
    print(f"Total Non-zero sensitivity in {file_name}: {sens_nz}")
    out_pt = Path(target_dir) / file_name
    torch.save(sens, str(out_pt))
    # To load sensitivities tensor:
    # sens = torch.load(out_pt, map_location=device)

def _get_sensitivities_pytorch(ply_orig: GaussianPLY = None,
                      cameras: List[Camera] = None,
                      save_dir: str = None,
                      device: torch.device = None,
                      cams_per_batch = 5,
                      sensitivity_norm: str = "l1",
                      sensitivity_reduce: str = "max",
                      sensitivity_nocolor: bool = False) -> None:
    device = resolve_torch_device(device)
    norm = (sensitivity_norm or "l1").lower()
    if norm not in {"l1", "l2"}:
        raise ValueError(f"Unsupported sensitivity_norm '{sensitivity_norm}' (expected 'l1' or 'l2')")
    reduce_mode = (sensitivity_reduce or "max").lower()
    if reduce_mode not in {"max", "mean"}:
        raise ValueError(f"Unsupported sensitivity_reduce '{sensitivity_reduce}' (expected 'max' or 'mean')")
    use_mean = (reduce_mode == "mean")
    model_dir = None
    if ply_orig is None:
        model_dir = resolve_model_dir(None)
        ply_path = resolve_point_cloud_path(None, model_dir)
        ply_orig = load_gaussian_ply(ply_path, device=device)
    if cameras is None:
        base_dir = model_dir or resolve_model_dir(None)
        cams_path = resolve_cameras_path(None, base_dir)
        cameras = load_cameras_from_json(cams_path, device=device)
    target_dir = resolve_test_sensitivity_output_dir(save_dir)
    ensure_dir_for(target_dir, treat_as_file=False)
    target_dir_str = str(target_dir)

    # --- Batched rendering (optionally chunked over cameras) ---
    total_start = time.perf_counter()
    N = ply_orig.means.shape[0]
    if use_mean:
        sens_ch_sum = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_ch_count = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_pix_sum = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_pix_count = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_tile_sum = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_tile_count = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_img_sum = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_img_count = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_batch_sum = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_batch_count = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
    else:
        sens_ch = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_pix = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_tile = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_img = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
        sens_batch = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
    gauss_total_contrib = torch.zeros(N, dtype=torch_ctx.dtype, device=device)
    denom_total_raw = torch.zeros(N, dtype=torch_ctx.dtype, device=device)

    render_kwargs = {
        "per_channel_sensitivity": True,
        "per_pixel_sensitivity": True,
        "per_tile_sensitivity": True,
        "per_image_sensitivity": True,
        "per_batch_sensitivity": True,
        "per_scene_sensitivity": True,
        "sensitivity_norm": norm,
        "sensitivity_reduce": reduce_mode,
        "sensitivity_nocolor": sensitivity_nocolor,
    }

    print(f"[SENS] Calculating sensitivities with '{norm}' norm and '{reduce_mode}' reduce "
          f"({'nocolor' if sensitivity_nocolor else 'rgb-weighted'}):")
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
        print(f"Batch {batch_id} rendering & sensitivity time: {elapsed:.6f} seconds")
        if use_mean:
            ch = out["sens_channel"]
            ch_count = out.get("sens_channel_count")
            if ch_count is None:
                raise KeyError("Expected 'sens_channel_count' in renderer outputs for mean reduction.")
            sens_ch_sum += ch * ch_count
            sens_ch_count += ch_count

            pix = out["sens_pixel"]
            pix_count = out.get("sens_pixel_count")
            if pix_count is None:
                raise KeyError("Expected 'sens_pixel_count' in renderer outputs for mean reduction.")
            sens_pix_sum += pix * pix_count
            sens_pix_count += pix_count

            tile = out["sens_tile"]
            tile_count = out.get("sens_tile_count")
            if tile_count is None:
                raise KeyError("Expected 'sens_tile_count' in renderer outputs for mean reduction.")
            sens_tile_sum += tile * tile_count
            sens_tile_count += tile_count

            img = out["sens_image"]
            img_count = out.get("sens_image_count")
            if img_count is None:
                raise KeyError("Expected 'sens_image_count' in renderer outputs for mean reduction.")
            sens_img_sum += img * img_count
            sens_img_count += img_count

            batch_val = out["sens_batch"]
            sens_batch_sum += batch_val * img_count
            sens_batch_count += img_count
        else:
            sens_ch = torch.max(sens_ch, out["sens_channel"])
            sens_pix = torch.max(sens_pix, out["sens_pixel"])
            sens_tile = torch.max(sens_tile, out["sens_tile"])
            sens_img = torch.max(sens_img, out["sens_image"])
            sens_batch = torch.max(sens_batch, out["sens_batch"])
        gauss_total_contrib += out["nom_batch"]
        denom_total_raw += out["denom_batch_raw"]

    ## Per-scene (dataset cameras) sensitivity:
    denom_total = denom_total_raw.clamp_min(torch_ctx.denom_eps)
    sens_scene = gauss_total_contrib / denom_total
    sens_scene = sens_scene.clamp(0.0, 1.0)

    if use_mean:
        safe_ch_count = torch.where(sens_ch_count > 0, sens_ch_count, torch.ones_like(sens_ch_count))
        sens_ch = (sens_ch_sum / safe_ch_count).clamp(0.0, 1.0)

        safe_pix_count = torch.where(sens_pix_count > 0, sens_pix_count, torch.ones_like(sens_pix_count))
        sens_pix = (sens_pix_sum / safe_pix_count).clamp(0.0, 1.0)

        safe_tile_count = torch.where(sens_tile_count > 0, sens_tile_count, torch.ones_like(sens_tile_count))
        sens_tile = (sens_tile_sum / safe_tile_count).clamp(0.0, 1.0)

        safe_img_count = torch.where(sens_img_count > 0, sens_img_count, torch.ones_like(sens_img_count))
        sens_img = (sens_img_sum / safe_img_count).clamp(0.0, 1.0)

        safe_batch_count = torch.where(sens_batch_count > 0, sens_batch_count, torch.ones_like(sens_batch_count))
        sens_batch = (sens_batch_sum / safe_batch_count).clamp(0.0, 1.0)

    ## Save all sensitivities:
    ## sens_ch, sens_pix, sens_tile, sens_img, sens_batch, sens_scene
    suffix = f"_{norm}_{reduce_mode}{'_nocolor' if sensitivity_nocolor else ''}"
    save_sensitivity(sens=sens_ch, file_name=f"per_channel{suffix}.pt", save_dir=target_dir_str)
    save_sensitivity(sens=sens_pix, file_name=f"per_pixel{suffix}.pt", save_dir=target_dir_str)
    save_sensitivity(sens=sens_tile, file_name=f"per_tile{suffix}.pt", save_dir=target_dir_str)
    save_sensitivity(sens=sens_img, file_name=f"per_image{suffix}.pt", save_dir=target_dir_str)
    save_sensitivity(sens=sens_batch, file_name=f"per_batch{suffix}.pt", save_dir=target_dir_str)
    save_sensitivity(sens=sens_scene, file_name=f"per_scene{suffix}.pt", save_dir=target_dir_str)

    total_end = time.perf_counter()
    print(f"[SENS] Total time for rendering & saving sensitivities: {total_end - total_start:.6f} seconds")


def _runtime_metadata(
    *,
    backend: str,
    model: GaussianPLY,
    cameras: List[Camera],
    norm: str,
    reduction: str,
    nocolor: bool,
    modes: Dict[str, bool],
    elapsed_seconds: float,
    timings: List[Dict[str, Any]],
    device: torch.device,
    input_metadata: Dict[str, Any] | None = None,
    requested_norm: str | None = None,
    output_semantics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    gpu_name = None
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    metadata = {
        "schema_version": 2 if backend == "cuda" else 1,
        "backend": backend,
        "semantics": "graphdeco-9c5c202" if backend == "cuda" else "pytorch",
        "project_base_commit": "1bf6ac9261ae07941a3777d9102c27bf966787aa",
        "graphdeco_commit": "54c035f7834b564019656c3e3fcc3646292f727d",
        "rasterizer_commit": "9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0",
        "norm": norm,
        "reduction": reduction,
        "nocolor": bool(nocolor),
        "modes": modes,
        "num_gaussians": int(model.means.shape[0]),
        "num_cameras": len(cameras),
        "device": str(device),
        "gpu": gpu_name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "camera_timings": timings,
    }
    if backend == "cuda":
        metadata.update({
            "sensitivity_definition": {
                "rgb_compositing_contribution": (
                    "a_gipc = feature_gic * alpha_gip * "
                    "transmittance_before_gip"
                ),
                "atomic_sensitivity_term": (
                    "alpha_gip * transmittance_before_gip, replicated across "
                    "the three channel slots"
                    if nocolor
                    else "a_gipc"
                ),
                "feature_buffer": (
                    "geometry_state.rgb from pinned GraphDECO "
                    "computeColorFromSH"
                ),
                "denominator": "sum over Gaussian contributions only",
                "l1_order": "aggregate a_gipc",
                "l2_order": "square each a_gipc before every aggregation",
                "sparse_support": (
                    "ranges[tile_id]/point_list; off-list Gaussians are "
                    "implicit zero contributions"
                ),
                "background_included": False,
                "final_rgb_clamp_included": False,
                "headroom_attribution": False,
            },
            "graphdeco_sparse_diagnostics": {
                "contributor_counts": "ImageState.n_contrib",
                "tile_ranges": "ImageState.ranges",
                "sorted_point_list": "BinningState.point_list",
                "availability": "returned by the extension when debug=True",
            },
            "renderer_stages": {
                "feature_buffer": {
                    "symbol": "geometry_state.rgb",
                    "background_added": False,
                    "exposure_applied": False,
                    "final_clamp_applied": False,
                },
                "gaussian_only_color": {
                    "definition": "C^G = sum_g a_g",
                    "background_added": False,
                    "exposure_applied": False,
                    "final_clamp_applied": False,
                },
                "background_composited_color": {
                    "symbol": "GraphDECO renderCUDA/out_color",
                    "definition": "C^B = C^G + T_final * background",
                    "background_added": True,
                    "exposure_applied": False,
                    "final_clamp_applied": False,
                },
                "final_wrapper_color": {
                    "symbol": "gaussian_renderer.render()['render']",
                    "definition": "C^F = clamp(C^B, 0, 1)",
                    "background_added": True,
                    "exposure_applied": False,
                    "final_clamp_applied": True,
                },
            },
        })
        if requested_norm == "l2-agg":
            metadata.update({
                "schema_version": 3,
                "requested_norm": "l2-agg",
                "l2_aggregation": "pixel",
                "output_semantics": output_semantics or {},
            })
            metadata["sensitivity_definition"]["l2_order"] = (
                "per-channel uses a_gipc^2; pixel and coarser use "
                "(sum_c a_gipc)^2 at each pixel before spatial aggregation"
            )
    if backend == "cuda" and input_metadata:
        metadata["inputs"] = input_metadata
    return metadata


def get_sensitivities(
    ply_orig: GaussianPLY = None,
    cameras: List[Camera] = None,
    save_dir: str = None,
    device: torch.device = None,
    cams_per_batch: int = 5,
    sensitivity_norm: str = "l1",
    sensitivity_reduce: str = "max",
    sensitivity_nocolor: bool = False,
    *,
    backend: str = "cuda",
    per_channel: bool = True,
    per_pixel: bool = True,
    per_tile: bool = True,
    per_image: bool = True,
    per_batch: bool | None = None,
    per_scene: bool = True,
    block_xy: int = 16,
    bg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    white_background: bool = False,
    input_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute sensitivities with the CUDA default or explicit PyTorch backend."""
    device = resolve_torch_device(device)
    requested_norm = (sensitivity_norm or "l1").lower()
    reduction = (sensitivity_reduce or "max").lower()
    if backend not in {"cuda", "pytorch"}:
        raise ValueError("backend must be 'cuda' or 'pytorch'")
    if backend == "cuda":
        if requested_norm not in {
            "l1",
            "l2",
            "l2-channel",
            "l2-agg",
        }:
            raise ValueError(
                "CUDA sensitivity norm must be 'l1', 'l2', "
                "'l2-channel', or 'l2-agg'"
            )
        norm = "l2" if requested_norm == "l2-channel" else requested_norm
    else:
        if requested_norm not in {"l1", "l2"}:
            raise ValueError(
                "the historical PyTorch backend accepts only 'l1' or 'l2'"
            )
        norm = requested_norm
    if reduction not in {"max", "mean"}:
        raise ValueError("sensitivity reduction must be 'max' or 'mean'")
    if backend == "cuda":
        if block_xy != 16:
            raise ValueError("the CUDA backend requires --block-xy 16")
        if reduction != "max":
            raise ValueError(
                "the corrected CUDA backend supports max reduction only"
            )
        if per_batch:
            raise ValueError(
                "per-batch sensitivity is obsolete and unavailable for CUDA"
            )
    if ply_orig is None or cameras is None:
        raise ValueError("model and cameras are required")

    modes = {
        "per_channel": bool(per_channel),
        "per_pixel": bool(per_pixel),
        "per_tile": bool(per_tile),
        "per_image": bool(per_image),
        "per_scene": bool(per_scene),
    }
    if backend == "pytorch":
        modes["per_batch"] = True
    target_dir = resolve_test_sensitivity_output_dir(save_dir)
    ensure_dir_for(target_dir, treat_as_file=False)
    started = time.perf_counter()
    timings: List[Dict[str, Any]] = []
    output_semantics: Dict[str, Any] | None = None
    _initialize_cuda_memory_tracking(device)

    if backend == "pytorch":
        # This path preserves the regular PyTorch implementation, including
        # its semantics and its all-six-output behavior.
        _get_sensitivities_pytorch(
            ply_orig=ply_orig,
            cameras=cameras,
            save_dir=str(target_dir),
            device=device,
            cams_per_batch=cams_per_batch,
            sensitivity_norm=norm,
            sensitivity_reduce=reduction,
            sensitivity_nocolor=sensitivity_nocolor,
        )
    else:
        from gs_coresets.backends.cuda_sensitivity import compute_cuda_sensitivities

        results, timings = compute_cuda_sensitivities(
            ply_orig,
            cameras,
            device=device,
            cams_per_batch=cams_per_batch,
            norm=norm,
            reduction=reduction,
            nocolor=sensitivity_nocolor,
            per_channel=per_channel,
            per_pixel=per_pixel,
            per_tile=per_tile,
            per_image=per_image,
            per_scene=per_scene,
            background=bg,
            white_background=white_background,
        )
        nocolor_suffix = "_nocolor" if sensitivity_nocolor else ""
        if norm == "l2-agg":
            output_names = {
                "sens_channel": (
                    f"per_channel_l2_{reduction}{nocolor_suffix}.pt"
                ),
                "sens_pixel": (
                    f"per_pixel_l2_agg_{reduction}{nocolor_suffix}.pt"
                ),
                "sens_tile": (
                    f"per_tile_l2_agg_{reduction}{nocolor_suffix}.pt"
                ),
                "sens_image": (
                    f"per_image_l2_agg_{reduction}{nocolor_suffix}.pt"
                ),
                "sens_scene": (
                    f"per_scene_l2_agg_{reduction}{nocolor_suffix}.pt"
                ),
            }
            semantics_by_key = {
                "sens_channel": {
                    "granularity": "per_channel",
                    "norm": "l2",
                    "l2_aggregation": "channel",
                    "formula": "a_gipc^2",
                },
                "sens_pixel": {
                    "granularity": "per_pixel",
                    "norm": "l2-agg",
                    "l2_aggregation": "pixel",
                    "formula": "(sum_c a_gipc)^2",
                },
                "sens_tile": {
                    "granularity": "per_tile",
                    "norm": "l2-agg",
                    "l2_aggregation": "pixel",
                    "formula": "sum_p (sum_c a_gipc)^2",
                },
                "sens_image": {
                    "granularity": "per_image",
                    "norm": "l2-agg",
                    "l2_aggregation": "pixel",
                    "formula": "sum_p (sum_c a_gipc)^2",
                },
                "sens_scene": {
                    "granularity": "per_scene",
                    "norm": "l2-agg",
                    "l2_aggregation": "pixel",
                    "formula": "sum_i,p (sum_c a_gipc)^2",
                },
            }
            output_semantics = {
                semantics_by_key[key]["granularity"]: {
                    **semantics_by_key[key],
                    "filename": output_names[key],
                }
                for key in results
            }
        else:
            suffix = (
                f"_{norm}_{reduction}"
                f"{nocolor_suffix}"
            )
            output_names = {
                "sens_channel": f"per_channel{suffix}.pt",
                "sens_pixel": f"per_pixel{suffix}.pt",
                "sens_tile": f"per_tile{suffix}.pt",
                "sens_image": f"per_image{suffix}.pt",
                "sens_scene": f"per_scene{suffix}.pt",
            }
        for key, tensor in results.items():
            save_sensitivity(tensor, output_names[key], str(target_dir))

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metadata = _runtime_metadata(
        backend=backend,
        model=ply_orig,
        cameras=cameras,
        norm=norm,
        reduction=reduction,
        nocolor=sensitivity_nocolor,
        modes=modes,
        elapsed_seconds=elapsed,
        timings=timings,
        device=device,
        input_metadata=input_metadata,
        requested_norm=(
            requested_norm if requested_norm == "l2-agg" else None
        ),
        output_semantics=output_semantics,
    )
    metadata_path = Path(target_dir) / "sensitivity_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[SENS] Backend: {backend}; total time: {elapsed:.6f} seconds")
    print(f"[SENS] Metadata: {metadata_path}")
    return metadata


def _initialize_cuda_memory_tracking(device: torch.device) -> None:
    """Initialize the selected CUDA runtime before resetting peak counters."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    # PyTorch 2.0's reset_peak_memory_stats does not reliably perform lazy
    # CUDA initialization itself. Explicit initialization preserves the
    # existing measurement boundary without allocating model tensors first.
    torch.cuda.init()
    with torch.cuda.device(device):
        torch.cuda.reset_peak_memory_stats(device)

def core_from_sens(ply_orig : GaussianPLY = None, save_dir : str = None, device : torch.device = None) -> None:
    from gs_coresets.utils.coreset_utils import get_SH_coreset
    device = resolve_torch_device(device)
    model_dir = None
    if ply_orig is None:
        model_dir = resolve_model_dir(None)
        ply_path = resolve_point_cloud_path(None, model_dir)
        ply_orig = load_gaussian_ply(ply_path, device=device)
    target_dir = resolve_test_sensitivity_output_dir(save_dir)
    ensure_dir_for(target_dir, treat_as_file=False)

    save_dir_abs_path = str(target_dir)
    if not os.path.isdir(save_dir_abs_path):
        print(f"[core_from_sens] {save_dir_abs_path} not found or not a directory!")
        return

    try:
        dir_entries = os.listdir(save_dir_abs_path)
    except OSError as e:
        print(f"[core_from_sens] Error reading directory: {e}")
        return

    # Filter to top-level .pt files and sort for stable output
    pt_files = sorted(
        name for name in dir_entries
        if name.endswith(".pt") and os.path.isfile(os.path.join(save_dir_abs_path, name))
    )

    if not pt_files:
        print("[core_from_sens] No .pt files found in top-level directory.")
        return

    sens_uniform = None
    for thousands in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        m_keep = thousands * 1000

        # Create coresets from sensitivities
        for file_name in pt_files:
            # --- Load sensitivities
            file_path = os.path.join(save_dir_abs_path, file_name)
            escaped_name, _ = os.path.splitext(file_name)  # strip ".pt"
            sens_global = torch.load(file_path, map_location=device)
            if sens_uniform is None:
                sens_uniform = torch.zeros_like(sens_global)
            new_file_name = f"{thousands}K_{escaped_name}.ply"
            ply_path_new = os.path.join(save_dir_abs_path, new_file_name)
            ply_new, uniq_cnt, total_cnt = get_SH_coreset(
                model_ply=ply_orig, sensitivities=sens_global, coreset_size=m_keep, out_path=ply_path_new,
                apply_SS_weights=False, allow_duplicates=False)
            print(f"Created coreset {new_file_name} with {total_cnt.numel()} gaussians ({uniq_cnt.numel()} unique).")

        # Create coresets from uniform sampling
        new_file_name = f"{thousands}K_uniform.ply"
        ply_path_new = os.path.join(save_dir_abs_path, new_file_name)
        ply_new, uniq_cnt, total_cnt = get_SH_coreset(
            model_ply=ply_orig, sensitivities=sens_uniform, coreset_size=m_keep, out_path=ply_path_new,
            apply_SS_weights=False, allow_duplicates=False)
        print(f"Created coreset {new_file_name} with {total_cnt.numel()} gaussians ({uniq_cnt.numel()} unique).")

def run_sensitivity_cli(args: argparse.Namespace) -> Tuple[Dict[str, Any] | None, int]:
    """
    CLI handler for sensitivity computation.
    """
    device = resolve_torch_device(args.device)
    backend = getattr(args, "backend", "cuda")
    camera_paths: List[str] = []
    if getattr(args, "cameras_train", None):
        camera_paths.extend(args.cameras_train)
    if getattr(args, "cams", None):
        camera_paths.extend(args.cams)

    if not getattr(args, "use_train_only", True):
        camera_paths.extend(getattr(args, "cameras_test", []) or [])

    if not camera_paths:
        print("[sens] no cameras provided; supply --cams/--cams-train/--cams-test", file=sys.stderr)
        return None, 2

    try:
        if backend == "cuda":
            if getattr(args, "image_size", None) is not None:
                raise ValueError(
                    "the CUDA backend uses the exact GraphDECO camera JSON "
                    "resolution and does not accept --image-size"
                )
            from gs_coresets.backends.graphdeco_inputs import (
                load_graphdeco_cameras_from_json,
                load_graphdeco_gaussian_ply,
            )

            model = load_graphdeco_gaussian_ply(
                args.model,
                active_sh_degree=getattr(args, "active_sh_degree", None),
            )
            cameras = []
            camera_convention = getattr(
                args,
                "camera_convention",
                "auto",
            )
            for path in camera_paths:
                cameras.extend(load_graphdeco_cameras_from_json(
                    path,
                    convention=camera_convention,
                ))
            selected_conventions = sorted({
                getattr(camera, "graphdeco_convention", "unknown")
                for camera in cameras
            })
            input_metadata = {
                "model_path": str(Path(args.model).expanduser().resolve()),
                "camera_paths": [
                    str(Path(path).expanduser().resolve())
                    for path in camera_paths
                ],
                "camera_convention_request": camera_convention,
                "camera_conventions_selected": selected_conventions,
                "active_sh_degree": int(model.sh_degree),
                "maximum_sh_degree": int(
                    getattr(model, "maximum_sh_degree", model.sh_degree)
                ),
                "model_preparation": (
                    "raw PLY parameters; GraphDECO activations once on CUDA"
                ),
            }
        else:
            model = load_gaussian_ply(args.model, device=device)
            cameras = []
            for path in camera_paths:
                cameras.extend(load_cameras_from_json(path, device=device))
            input_metadata = {
                "model_path": str(Path(args.model).expanduser().resolve()),
                "camera_paths": [
                    str(Path(path).expanduser().resolve())
                    for path in camera_paths
                ],
                "model_preparation": "historical PyTorch loader",
            }

        if not cameras:
            raise ValueError("camera JSON files contained no cameras")
        out_dir = resolve_sensitivity_output_dir(
            getattr(args, "out_dir", None)
        )
        ensure_dir_for(out_dir, treat_as_file=False)
        metadata = get_sensitivities(
            ply_orig=model,
            cameras=cameras,
            save_dir=str(out_dir),
            device=device,
            cams_per_batch=getattr(args, "cams_per_batch", 5),
            sensitivity_norm=getattr(args, "sensitivity_norm", "l1"),
            sensitivity_reduce=getattr(args, "sensitivity_reduce", "max"),
            sensitivity_nocolor=getattr(args, "sensitivity_nocolor", False),
            backend=backend,
            per_channel=getattr(args, "per_channel", True),
            per_pixel=getattr(args, "per_pixel", True),
            per_tile=getattr(args, "per_tile", True),
            per_image=getattr(args, "per_image", True),
            per_batch=getattr(args, "per_batch", None),
            per_scene=getattr(args, "per_scene", True),
            block_xy=getattr(args, "block_xy", 16),
            bg=getattr(args, "bg", (0.0, 0.0, 0.0)),
            white_background=getattr(args, "white_background", False),
            input_metadata=input_metadata,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[sens] {exc}", file=sys.stderr)
        return None, 2
    return {
        "out_dir": str(out_dir),
        "num_cameras": len(cameras),
        "metadata": metadata,
    }, 0


def main(argv: list[str] | None = None) -> int:
    from gs_coresets.utils.args_utils import build_commands_parser

    parser = build_commands_parser(selected=("sens",))
    args = parser.parse_args(["sens", *(argv or [])])
    _, exit_code = run_sensitivity_cli(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
