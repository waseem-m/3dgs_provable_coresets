#!/usr/bin/env python3
"""
Coreset sampling utilities for the `gs-coresets coreset` subcommand.
"""

from __future__ import annotations

import argparse
import math
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

SENSITIVITY_NORMS: Tuple[str, ...] = ("l1", "l2")
SENSITIVITY_REDUCES: Tuple[str, ...] = ("max", "mean")
GRANULARITIES: Tuple[str, ...] = (
    "per_channel",
    "per_pixel",
    "per_tile",
    "per_image",
    "per_batch",
    "per_scene",
)
def _build_sensitivity_filename(
    granularity: str,
    reduce_kind: str,
    norm_kind: str,
    *,
    nocolor: bool = False,
) -> str:
    suffix = "_nocolor" if nocolor else ""
    return f"{granularity}_{norm_kind}_{reduce_kind}{suffix}.pt"


SENSITIVITY_FILE_MAP: Dict[str, Tuple[str, ...]] = {
    granularity: tuple(
        _build_sensitivity_filename(granularity, reduce_kind, norm, nocolor=use_nocolor)
        for reduce_kind in SENSITIVITY_REDUCES
        for norm in SENSITIVITY_NORMS
        for use_nocolor in (False, True)
    )
    for granularity in GRANULARITIES
}


def _load_sensitivity_tensors(
    path: Path,
    device: torch.device,
    granularity: str,
    reduce_kind: str,
    norm_kind: str,
    *,
    nocolor: bool = False,
) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    if path.is_file():
        tensors.append(torch.load(path, map_location=device))
    elif path.is_dir():
        target_name = _build_sensitivity_filename(granularity, reduce_kind, norm_kind, nocolor=nocolor)
        target = path / target_name
        if not target.is_file():
            print(f"[coreset] expected sensitivity tensor '{target_name}' in {path}", file=sys.stderr)
            return []
        tensors = [torch.load(target, map_location=device)]
    else:
        print(f"[coreset] sensitivity path not found: {path}", file=sys.stderr)
    return tensors


def run_coreset_cli(
    args: argparse.Namespace,
    *,
    coreset_size: Optional[int] = None,
    prune_ratio: Optional[float] = None,
) -> Tuple[Dict[str, Any] | None, int]:
    """
    CLI handler for coreset sampling from sensitivity tensors.

    Parameters
    ----------
    args:
        Parsed argparse namespace from the unified CLI.
    coreset_size:
        Explicit target count when already known (takes precedence over ``prune_ratio``).
    prune_ratio:
        Optional pruning ratio forwarded verbatim from callers. When provided, the
        authoritative conversion ``coreset_size = max(1, round((1 - ratio) * original_count))``
        is performed here after loading the source PLY so all entry-points share the
        same logic. No other module should replicate this computation.
    """
    device = resolve_torch_device(args.device)
    model = load_gaussian_ply(args.model, device=device)
    original_count = int(model.means.shape[0])
    if original_count <= 0:
        print("[coreset] source model contains no Gaussians.", file=sys.stderr)
        return None, 2

    explicit_size = coreset_size if coreset_size is not None else getattr(args, "coreset_size", None)
    explicit_ratio = prune_ratio if prune_ratio is not None else getattr(args, "prune_ratio", None)
    if explicit_size is not None and explicit_ratio is not None:
        print("[coreset] received both coreset_size and prune_ratio; choose one.", file=sys.stderr)
        return None, 2
    if explicit_size is None and explicit_ratio is None:
        print("[coreset] missing target size; specify --coreset-size or --prune-ratio.", file=sys.stderr)
        return None, 2
    if explicit_ratio is not None:
        computed_size = max(1, round((1.0 - float(explicit_ratio)) * original_count))
        if computed_size > original_count:
            computed_size = original_count
    else:
        computed_size = int(explicit_size)
        if computed_size < 1:
            print("[coreset] coreset size must be positive.", file=sys.stderr)
            return None, 2
    resolved_coreset_size = computed_size

    sens_path_arg = getattr(args, "sens_path", None)
    uniform_sampling = bool(getattr(args, "uniform", False))
    weights_L1 = getattr(args, "weights_L1", None)
    sens_norm = getattr(args, "sens_norm", "l1")
    sens_reduce = getattr(args, "sens_reduce", "max")
    sens_nocolor = bool(getattr(args, "sens_nocolor", False))
    if uniform_sampling and sens_path_arg:
        print("[coreset] --uniform cannot be combined with --sens.", file=sys.stderr)
        return None, 2
    if not uniform_sampling and sens_path_arg is None:
        print("[coreset] missing sensitivity input; provide --sens or enable --uniform.", file=sys.stderr)
        return None, 2
    if uniform_sampling and weights_L1:
        print("[coreset] --weights-L1 is incompatible with uniform sampling.", file=sys.stderr)
        return None, 2

    sampling_mode = getattr(args, "sampling_mode", "multinomial")
    if sampling_mode not in ("multinomial", "topk"):  # pragma: no cover - parser restricts choices
        print(f"[coreset] unknown sampling mode '{sampling_mode}'.", file=sys.stderr)
        return None, 2
    if sampling_mode == "topk":
        if getattr(args, "allow_duplicates", False):
            print("[coreset] --allow-duplicates cannot be combined with top-k selection.", file=sys.stderr)
            return None, 2
        if resolved_coreset_size > original_count:
            print(
                f"[coreset] top-k selection needs at most {original_count} items (requested {resolved_coreset_size}).",
                file=sys.stderr,
            )
            return None, 2

    if uniform_sampling:
        combined = torch.ones(
            original_count,
            dtype=model.means.dtype,
            device=device,
        )
        num_tensors = 0
        effective_granularity = "uniform"
        effective_reduce: Optional[str] = None
        effective_norm: Optional[str] = None
    else:
        sens_path = Path(sens_path_arg)
        granularity = getattr(args, "sens_granularity", "per_image")
        if granularity not in SENSITIVITY_FILE_MAP:
            print(f"[coreset] unknown sensitivity granularity '{granularity}'", file=sys.stderr)
            return None, 2
        if sens_reduce not in SENSITIVITY_REDUCES:
            print(f"[coreset] unknown sensitivity reduction '{sens_reduce}'", file=sys.stderr)
            return None, 2
        if sens_norm not in SENSITIVITY_NORMS:
            print(f"[coreset] unknown sensitivity norm '{sens_norm}'", file=sys.stderr)
            return None, 2

        tensors = _load_sensitivity_tensors(
            sens_path,
            device,
            granularity,
            sens_reduce,
            sens_norm,
            nocolor=sens_nocolor,
        )
        num_tensors = len(tensors)
        if num_tensors == 0:
            return None, 2

        if num_tensors == 1:
            if weights_L1:
                print("[coreset] weights not supported when a single sensitivity tensor is supplied.", file=sys.stderr)
                return None, 2
            combined = tensors[0].to(device=device)
        else:
            if weights_L1:
                if len(weights_L1) != num_tensors:
                    print(f"[coreset] number of weights ({len(weights_L1)}) does not match tensors ({num_tensors}).", file=sys.stderr)
                    return None, 2
                weights = torch.tensor(weights_L1, dtype=tensors[0].dtype, device=device)
                weight_sum = float(weights.sum().item())
                if not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                    print(f"[coreset] weight sum must equal 1.0 (got {weight_sum:.6f}).", file=sys.stderr)
                    return None, 2
                weights = weights / weight_sum
            else:
                weights = torch.full((num_tensors,), 1.0 / num_tensors, dtype=tensors[0].dtype, device=device)

            combined = torch.zeros_like(tensors[0], device=device)
            for w, t in zip(weights, tensors):
                combined += w * t.to(device=device)
        effective_granularity = granularity
        effective_reduce = sens_reduce
        effective_norm = sens_norm
        if sens_nocolor:
            effective_granularity = f"{effective_granularity}_nocolor"

    out_path = Path(args.out_ply)
    ensure_dir_for(out_path, treat_as_file=True)

    get_SH_coreset(
        model_ply=model,
        sensitivities=combined,
        coreset_size=resolved_coreset_size,
        out_path=str(out_path),
        apply_SS_weights=args.apply_ss_weights,
        normalize_weights=args.normalize_weights,
        allow_duplicates=args.allow_duplicates,
        temperature=float(args.temperature),
        seed=args.seed,
        sampling_mode=sampling_mode,
    )
    print(f"[coreset] saved coreset PLY → {out_path}")
    resolved_ratio = float(explicit_ratio) if explicit_ratio is not None else max(
        0.0, min(1.0, 1.0 - (float(resolved_coreset_size) / float(original_count)))
    )
    return {
        "out_ply": str(out_path),
        "sens_granularity": effective_granularity,
        "sens_reduce": effective_reduce,
        "sens_norm": effective_norm,
        "num_sens_tensors": num_tensors,
        "coreset_size": resolved_coreset_size,
        "prune_ratio": resolved_ratio,
        "original_count": original_count,
    }, 0


def main(argv: Optional[List[str]] = None) -> int:
    from gs_coresets.utils.args_utils import build_commands_parser

    parser = build_commands_parser(selected=("coreset",))
    args = parser.parse_args(["coreset", *(argv or [])])
    _, exit_code = run_coreset_cli(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
