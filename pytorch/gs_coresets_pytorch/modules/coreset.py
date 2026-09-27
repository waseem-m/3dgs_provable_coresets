#!/usr/bin/env python3
"""
Coreset sampling utilities for the `gs-coresets-pytorch coreset` subcommand.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets_pytorch.utils.default_paths import ensure_all_on_syspath
ensure_all_on_syspath()

from gs_coresets_pytorch.utils.coreset_utils import get_SH_coreset, get_probabilities
from gs_coresets_pytorch.utils.io_utils import load_gaussian_ply
from gs_coresets_pytorch.utils.output_safety import selection_outputs
from gs_coresets_pytorch.utils.raw_ply_subset import validate_raw_ply_subset
from gs_coresets_pytorch.utils.selection_manifest import (
    probability_statistics,
    write_selection_manifest,
)
from gs_coresets_pytorch.utils.types_devices_utils import resolve_torch_device

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


def _resolved_sensitivity_file(
    path: Path,
    granularity: str,
    reduce_kind: str,
    norm_kind: str,
    *,
    nocolor: bool,
) -> Path:
    if path.is_file():
        return path
    return path / _build_sensitivity_filename(
        granularity,
        reduce_kind,
        norm_kind,
        nocolor=nocolor,
    )


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
    requested_manifest = getattr(args, "selection_manifest", None)
    if requested_manifest is not None and not getattr(args, "preserve_raw_parameters", False):
        print("[coreset] --selection-manifest requires --preserve-raw-parameters.", file=sys.stderr)
        return None, 2
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
    preserve_raw_parameters = bool(
        getattr(args, "preserve_raw_parameters", False)
    )
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
    if preserve_raw_parameters:
        incompatible = []
        if args.allow_duplicates:
            incompatible.append("--allow-duplicates")
        if args.apply_ss_weights:
            incompatible.append("--apply-ss-weights")
        if args.normalize_weights:
            incompatible.append("--normalize-weights")
        if incompatible:
            print(
                "[coreset] --preserve-raw-parameters is incompatible with "
                + ", ".join(incompatible),
                file=sys.stderr,
            )
            return None, 2
        if combined.ndim != 1 or int(combined.numel()) != original_count:
            print(
                "[coreset] strict raw selection requires one score per "
                f"Gaussian (got {tuple(combined.shape)}, expected "
                f"({original_count},)).",
                file=sys.stderr,
            )
            return None, 2
        if not bool(torch.isfinite(combined).all().item()):
            print(
                "[coreset] strict raw selection rejects non-finite scores.",
                file=sys.stderr,
            )
            return None, 2
        if bool((combined < 0).any().item()):
            print(
                "[coreset] strict raw selection rejects negative scores.",
                file=sys.stderr,
            )
            return None, 2
        support_count = int((combined > 0).sum().item())
        if support_count < resolved_coreset_size:
            print(
                "[coreset] strict raw selection cannot choose "
                f"{resolved_coreset_size} identities from positive support "
                f"{support_count}; no fallback is permitted.",
                file=sys.stderr,
            )
            return None, 2

    resolved_ratio = float(explicit_ratio) if explicit_ratio is not None else max(
        0.0, min(1.0, 1.0 - (float(resolved_coreset_size) / float(original_count)))
    )
    manifest_path = None
    if preserve_raw_parameters:
        manifest_path = (
            Path(requested_manifest) if requested_manifest
            else out_path.with_name(out_path.name + ".selection.json")
        )
    score_file = None
    if not uniform_sampling:
        score_file = _resolved_sensitivity_file(
            Path(sens_path_arg), granularity, sens_reduce, sens_norm,
            nocolor=sens_nocolor,
        )
    protected = [args.model]
    if score_file is not None:
        protected.append(score_file)
    try:
        with selection_outputs(out_path, manifest_path, inputs=protected) as (staged_ply, staged_manifest):
            _, unique_idx, counts = get_SH_coreset(
                model_ply=model,
                sensitivities=combined,
                coreset_size=resolved_coreset_size,
                out_path=str(staged_ply),
                apply_SS_weights=args.apply_ss_weights,
                normalize_weights=args.normalize_weights,
                allow_duplicates=args.allow_duplicates,
                temperature=float(args.temperature),
                seed=args.seed,
                sampling_mode=sampling_mode,
                preserve_raw_parameters=preserve_raw_parameters,
                source_ply_path=args.model,
            )
            if preserve_raw_parameters:
                selected_indices = unique_idx.detach().cpu().tolist()
                if len(selected_indices) != resolved_coreset_size:
                    raise ValueError("strict raw selection returned the wrong number of identities")
                if not bool((counts == 1).all().item()):
                    raise ValueError("strict raw selection returned non-unit counts")
                probabilities = get_probabilities(combined, temperature=float(args.temperature))
                raw_validation = validate_raw_ply_subset(args.model, staged_ply, selected_indices)
                write_selection_manifest(
                    staged_manifest,
                    source_ply=args.model,
                    output_ply=staged_ply,
                    published_output_ply=out_path,
                    score_path=score_file,
                    selected_indices=selected_indices,
                    original_count=original_count,
                    requested_count=resolved_coreset_size,
                    prune_ratio=resolved_ratio,
                    seed=args.seed if sampling_mode == "multinomial" else None,
                    sampling_mode=sampling_mode,
                    replacement=False,
                    temperature=float(args.temperature),
                    unit_weights=True,
                    raw_subset_validation=raw_validation,
                    probability_stats=probability_statistics(combined, probabilities, selected_indices),
                    extra={
                        "sensitivity_granularity": effective_granularity,
                        "sensitivity_reduce": effective_reduce,
                        "sensitivity_norm": effective_norm,
                        "topk_tie_break": "original_index_ascending" if sampling_mode == "topk" else None,
                    },
                )
    except (OSError, IndexError, ValueError) as exc:
        print(f"[coreset] selection failed: {exc}", file=sys.stderr)
        return None, 2
    print(f"[coreset] saved coreset PLY → {out_path}")
    if manifest_path is not None:
        print(f"[coreset] saved selection manifest → {manifest_path}")
    return {
        "out_ply": str(out_path),
        "sens_granularity": effective_granularity,
        "sens_reduce": effective_reduce,
        "sens_norm": effective_norm,
        "num_sens_tensors": num_tensors,
        "coreset_size": resolved_coreset_size,
        "prune_ratio": resolved_ratio,
        "original_count": original_count,
        "selection_manifest": (
            str(manifest_path) if manifest_path is not None else None
        ),
        "raw_parameters_preserved": preserve_raw_parameters,
    }, 0


def main(argv: Optional[List[str]] = None) -> int:
    from gs_coresets_pytorch.utils.args_utils import build_commands_parser

    parser = build_commands_parser(selected=("coreset",))
    args = parser.parse_args(["coreset", *(argv or [])])
    _, exit_code = run_coreset_cli(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
