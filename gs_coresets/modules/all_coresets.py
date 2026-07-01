#!/usr/bin/env python3
"""
Generate coresets for every sensitivity configuration plus uniform sampling.

This command reuses the baseline model and sensitivity tensors efficiently to
emit a directory tree that mirrors the baseline layout for each coreset.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets.utils.default_paths import ensure_all_on_syspath

ensure_all_on_syspath()

from gs_coresets.utils.coreset_utils import get_SH_coreset
from gs_coresets.utils.io_utils import ensure_dir_for, load_gaussian_ply
from gs_coresets.utils.types_devices_utils import resolve_torch_device

from gs_coresets.modules.coreset import (
    GRANULARITIES,
    SENSITIVITY_NORMS,
    SENSITIVITY_REDUCES,
)


SIDE_CAR_FILENAMES: Tuple[str, ...] = ("cameras.json", "cfg_args")


def _printerr(message: str) -> None:
    print(f"[all_coresets] {message}", file=sys.stderr)


def _maybe_flush_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            print("[all_coresets] Failed to flush CUDA device", file=sys.stderr)
            pass


def _validate_baseline_path(model_path: str) -> Tuple[Path, str, Path]:
    """
    Ensure the baseline path follows <parent>/point_cloud/iteration_<n>/point_cloud.ply.

    Returns
    -------
    ply_path:
        Resolved baseline PLY path.
    iteration_token:
        The iteration suffix (e.g. "iteration_30000").
    baseline_parent:
        Directory containing point_cloud/, cameras.json, cfg_args, etc.
    """
    ply_path = Path(model_path).expanduser().resolve()
    if ply_path.name != "point_cloud.ply":
        raise ValueError("expected filename 'point_cloud.ply'")

    iteration_dir = ply_path.parent
    iteration_name = iteration_dir.name
    if not iteration_name.startswith("iteration_"):
        raise ValueError(f"parent directory must start with 'iteration_'; got '{iteration_name}'")
    iteration_suffix = iteration_name[len("iteration_") :]
    if not iteration_suffix.isdigit():
        raise ValueError(f"iteration directory suffix must be numeric; got '{iteration_suffix}'")

    point_cloud_dir = iteration_dir.parent
    if point_cloud_dir.name != "point_cloud":
        raise ValueError("expected 'point_cloud' directory enclosing iteration directory")

    baseline_parent = point_cloud_dir.parent
    if baseline_parent == point_cloud_dir:
        raise ValueError("failed to resolve baseline parent directory")

    return ply_path, iteration_name, baseline_parent


def _copy_sidecars(sources: Dict[str, Path], destination: Path) -> None:
    for name, src in sources.items():
        dest = destination / name
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            ensure_dir_for(dest, treat_as_file=True)
            shutil.copy2(src, dest)


def _format_ratio(ratio: float) -> str:
    return f"{ratio:.2f}"


def run_all_coresets_cli(args: argparse.Namespace) -> Tuple[Dict[str, Any] | None, int]:
    """
    CLI handler for generating coresets across all sensitivity combinations.
    """
    device = resolve_torch_device(args.device)

    try:
        baseline_ply, iteration_token, baseline_parent = _validate_baseline_path(args.model)
    except ValueError as exc:
        _printerr(str(exc))
        return None, 2

    sidecars: Dict[str, Path] = {}
    for name in SIDE_CAR_FILENAMES:
        candidate = baseline_parent / name
        if not candidate.exists():
            _printerr(f"required artifact missing: {candidate}")
            return None, 2
        sidecars[name] = candidate

    sens_path = Path(args.sens_path).expanduser().resolve()
    if not sens_path.is_dir():
        _printerr(f"sensitivity path must be a directory containing tensors: {sens_path}")
        return None, 2

    out_root = Path(args.out_dir).expanduser().resolve()
    ensure_dir_for(out_root, treat_as_file=False)

    model = load_gaussian_ply(str(baseline_ply), device=device)
    total_gaussians = int(model.means.shape[0])
    if total_gaussians <= 0:
        _printerr("baseline contains no Gaussians.")
        return None, 2

    prune_ratio = float(args.prune_ratio)
    coreset_size = max(1, round((1.0 - prune_ratio) * total_gaussians))
    if coreset_size > total_gaussians:
        coreset_size = total_gaussians

    sampling_mode = getattr(args, "sampling_mode", "multinomial")
    sens_nocolor = bool(getattr(args, "sens_nocolor", False))
    if sampling_mode == "topk" and coreset_size > total_gaussians:
        _printerr(
            f"top-k selection requests {coreset_size} items but only {total_gaussians} gaussians exist."
        )
        return None, 2

    ratio_token = _format_ratio(prune_ratio)
    iteration_str = iteration_token  # already in 'iteration_XXXXX' format
    temperature = float(getattr(args, "temperature", 1.0))
    seed: Optional[int] = getattr(args, "seed", None)

    coreset_records: List[Dict[str, Any]] = []

    def _emit(label: str, tensor: torch.Tensor, dir_name: str) -> bool:
        try:
            tensor_device = tensor.to(device=device, dtype=model.means.dtype)
        except RuntimeError as exc:
            _printerr(f"failed to move sensitivity tensor '{label}' to device: {exc}")
            return False

        target_dir = out_root / dir_name
        out_ply = target_dir / "point_cloud" / iteration_str / "point_cloud.ply"
        ensure_dir_for(out_ply, treat_as_file=True)

        try:
            get_SH_coreset(
                model,
                tensor_device,
                coreset_size,
                str(out_ply),
                apply_SS_weights=False,
                normalize_weights=False,
                allow_duplicates=False,
                temperature=temperature,
                seed=seed,
                sampling_mode=sampling_mode,
            )
        except ValueError as exc:
            _printerr(f"{label}: {exc}")
            return False

        _copy_sidecars(sidecars, target_dir)

        coreset_records.append(
            {
                "label": label,
                "directory": str(target_dir),
                "point_cloud": str(out_ply),
            }
        )
        print(f"wrote {label} → {out_ply}")
        return True

    print(f"[all_coresets] Creating all coresets for {ratio_token} prune ratio.")
    uniform_dir = f"uniform_{ratio_token}prune"
    had_failures = False
    uniform_tensor = torch.ones(total_gaussians, dtype=model.means.dtype)
    if not _emit("uniform", uniform_tensor, uniform_dir):
        _printerr("uniform sampling failed; continuing with sensitivity coresets.")
        had_failures = True
    del uniform_tensor
    _maybe_flush_cuda(device)

    for granularity in sorted(gr for gr in GRANULARITIES if gr.startswith("per_")):
        for norm_kind in SENSITIVITY_NORMS:
            for reduce_kind in SENSITIVITY_REDUCES:
                suffix_nocolor = "_nocolor" if sens_nocolor else ""
                tensor_path = sens_path / f"{granularity}_{norm_kind}_{reduce_kind}{suffix_nocolor}.pt"
                if not tensor_path.is_file():
                    _printerr(f"missing sensitivity tensor: {tensor_path.name}; skipping.")
                    had_failures = True
                    continue
                tensor = torch.load(tensor_path, map_location="cpu")
                dir_name = f"{granularity}_{norm_kind}_{reduce_kind}{suffix_nocolor}_{ratio_token}prune"
                label = f"{granularity}{':nocolor' if sens_nocolor else ''}:{norm_kind}:{reduce_kind}"
                if not _emit(label, tensor, dir_name):
                    _printerr(f"{label} coreset generation failed; continuing.")
                    had_failures = True
                del tensor
                _maybe_flush_cuda(device)

    payload: Dict[str, Any] = {
        "baseline_parent": str(baseline_parent),
        "baseline_iteration": iteration_str,
        "total_gaussians": total_gaussians,
        "prune_ratio": prune_ratio,
        "coreset_size": coreset_size,
        "outputs": coreset_records,
    }
    exit_code = 0 if not had_failures else 1
    return payload, exit_code


__all__ = ["run_all_coresets_cli"]
