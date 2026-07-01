#!/usr/bin/env python3
"""
Argument helpers for the unified GraphDECO + gs_coresets.utils CLI.

This module only defines argparse surfaces; actual command implementations live
in `gs-coresets` (or thin wrappers imported there). Each subparser attaches a
`func` callback so `gs-coresets` can simply dispatch via `args.func(args)`.
"""

from __future__ import annotations

import argparse
import textwrap
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
    RawDescriptionHelpFormatter,
)
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from gs_coresets.utils.io_utils import ensure_dir_for, expanduser_path
from gs_coresets.utils.classify_scene_utils import available_classifier_presets

__all__ = [
    "build_commands_parser",
    "add_train_subparser",
    "add_render_subparser",
    "add_metrics_subparser",
    "add_full_eval_subparser",
    "add_classify_subparser",
    "add_sens_cams_subparser",
    "add_sens_subparser",
    "add_all_coresets_subparser",
    "add_coreset_subparser",
    "add_finetune_subparser",
]


# ---------------------------------------------------------------------------
# Helpers

GraphdecoHandler = Callable[[argparse.Namespace], int]

COMMAND_ORDER: Tuple[str, ...] = (
    "train",
    "render",
    "metrics",
    "full_eval",
    "classify",
    "sens_cams",
    "sens",
    "all_coresets",
    "coreset",
    "finetune",
)

CORESET_DESCRIPTION = textwrap.dedent(
    """Sample a coreset from sensitivities or via uniform sampling and write a compact PLY.

    Coreset size can be specified absolutely (-k) or as a prune ratio (--prune-ratio).
    """
)

CORESET_EXAMPLES = textwrap.dedent(
    """Examples:
      gs-coresets coreset -m MODEL -sens sens.pt -o out.ply -k 300000
      gs-coresets coreset -m MODEL -sens sens.pt -o out.ply --prune-ratio 0.9
      gs-coresets coreset -m MODEL --uniform -o out.ply -k 150000
    """
)

PIPELINE_DESCRIPTION = textwrap.dedent(
    """
End-to-end pipeline:
    A) Train baseline
    B) Baseline metrics (render + metrics)
    C) Cameras: generate, extract/import, or reuse (skipped for uniform sampling or provided baseline)
    D) Compute sensitivities (skipped for uniform sampling or provided baseline sensitivities)
    E) Sample coreset
    F) Coreset metrics
    G) Fine-tune coreset model
    H) Fine-tune metrics

Coreset size can be specified absolutely (-k) or as a prune ratio (--prune-ratio).
    """
)

PIPELINE_EXAMPLES = textwrap.dedent(
    """Examples:
      gs-coresets -- -s DATASET -o outputs -k 300000
      gs-coresets -- -s DATASET -o outputs --prune-ratio 0.9 --uniform
      gs-coresets -- -s DATASET --baseline-dir PRETRAINED_BASELINE --sens-path SENS_DIR -o outputs -k 150000
    """
)

PIPELINE_ALL_DESCRIPTION = textwrap.dedent(
    """
Accelerated end-to-end pipeline that reuses expensive stages across multiple
coreset granularities and uniform sampling in a single run.
    """
)

PIPELINE_ALL_EXAMPLES = textwrap.dedent(
    """Examples:
      gs-coresets pipeline-all -- -s DATASET -o outputs -k 300000
      gs-coresets pipeline-all -- -s DATASET -o outputs --prune-ratio 0.9
      gs-coresets pipeline-all -- -s DATASET --baseline-dir PRETRAINED_BASELINE --sens-path SENS_DIR -k 150000
    """
)

PIPELINE_ALL_DEFAULT_GRANULARITIES: Tuple[str, ...] = (
    "per_channel",
    "per_pixel",
    "per_tile",
    "per_image",
    "per_batch",
    "per_scene",
    "uniform",
)

SENS_CAMS_DESCRIPTION = textwrap.dedent(
    """
Manage cameras for sensitivity workflows.

Default behavior extracts dataset cameras from the GraphDECO workspace pointed
to by --model and --source. Use --generate to synthesize ROI-sampled cameras
(requires --num). Use --import-json to copy an existing cameras JSON into the
output directory.
    """
)


class _HelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """Formatter combining default value display with raw text handling."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_string = super()._get_help_string(action)
        if help_string:
            if action.default is None:
                help_string = help_string.replace(" (default: %(default)s)", "")
                help_string = help_string.replace(" (default: None)", "")
            elif action.required:
                help_string = help_string.replace(" (default: %(default)s)", "")
        return help_string


class _StoreWithFlag(argparse.Action):
    """Keep track of which CLI knobs were explicitly overridden."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        seen: Set[str] = getattr(namespace, "_seen_args", set())
        seen.add(self.dest)
        setattr(namespace, "_seen_args", seen)


def _float3(txt: str) -> Tuple[float, float, float]:
    parts = [p for p in txt.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three floats: 'r g b' or 'r,g,b'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _float_list(txt: str) -> List[float]:
    parts = [p for p in txt.replace(",", " ").split() if p]
    try:
        return [float(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"Invalid float in list: {exc}") from exc


def _prune_ratio(value: str) -> float:
    """
    Parse a pruning ratio and ensure it lies in the open interval (0, 1).
    """
    try:
        ratio = float(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"Invalid prune ratio '{value}'") from exc
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError("Prune ratio must satisfy 0.0 < r < 1.0")
    return ratio


def _attach_handler(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    handler: Optional[GraphdecoHandler],
) -> argparse.ArgumentParser:
    if handler is None:
        def _missing_handler(_: argparse.Namespace) -> int:  # pragma: no cover - configuration guard
            raise RuntimeError(f"No handler registered for subcommand '{name}'")

        parser.set_defaults(func=_missing_handler, command=name)
    else:
        parser.set_defaults(func=handler, command=name)
    return parser


def _add_common_runtime(group: argparse._ArgumentGroup, *, include_seed: bool = True) -> None:
    group.add_argument(
        "--device",
        default="auto",
        help="Torch device policy: 'auto', 'cuda', 'cuda:0', 'cpu', etc.",
    )
    if include_seed:
        group.add_argument("--seed", type=int, default=None, help="Global random seed override.")
    group.add_argument(
        "--log",
        type=lambda p: str(ensure_dir_for(expanduser_path(p), treat_as_file=True)),
        default=None,
        metavar="PATH",
        help="Mirror stdout & stderr to this logfile (wrapper enables logging before dispatch).",
    )
    group.add_argument(
        "--debug",
        action=BooleanOptionalAction,
        default=False,
        help="Toggle debug-mode prints from gs_coresets.utils.debug_utils.",
    )


def _make_graphdeco_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    name: str,
    script: str,
    description: str,
    examples: Sequence[str],
    handler: Optional[GraphdecoHandler],
) -> argparse.ArgumentParser:
    desc = [description]
    if examples:
        formatted = "\n".join(f"  {ex}" for ex in examples)
        desc.append("Examples:\n" + formatted)
    parser = subparsers.add_parser(
        name,
        add_help=False,
        help=description,
        formatter_class=_HelpFormatter,
        description="\n\n".join(desc),
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="_show_help",
        help=f"Show help from external/gaussian_splatting/{script} and exit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the delegated GraphDECO command and exit without running it.",
    )
    parser.set_defaults(script_name=script, _forward_unknown=True, script_args=[])
    return _attach_handler(parser, name=name, handler=handler)


# ---------------------------------------------------------------------------
# GraphDECO entry-points

def add_train_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    return _make_graphdeco_subparser(
        subparsers,
        name="train",
        script="train.py",
        description="Wrap GraphDECO training (`external/gaussian_splatting/train.py`).",
        examples=("python gs-coresets train -s DATASET --eval",),
        handler=handler,
    )


def add_render_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    return _make_graphdeco_subparser(
        subparsers,
        name="render",
        script="render.py",
        description="Wrap GraphDECO rendering (`external/gaussian_splatting/render.py`).",
        examples=("python gs-coresets render -m MODEL -s DATASET",),
        handler=handler,
    )


def add_metrics_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    return _make_graphdeco_subparser(
        subparsers,
        name="metrics",
        script="metrics.py",
        description="Wrap GraphDECO image metric computation.",
        examples=("python gs-coresets metrics -m MODEL --pred images --gt gt_images",),
        handler=handler,
    )


def add_full_eval_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    return _make_graphdeco_subparser(
        subparsers,
        name="full_eval",
        script="full_eval.py",
        description="Wrap GraphDECO end-to-end evaluation script.",
        examples=("python gs-coresets full_eval -m360 SCENE -tat TAT -db DB",),
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Local utilities (gs_coresets.utils)

def add_finetune_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune an existing Gaussian scene without pruning.",
        formatter_class=_HelpFormatter,
    )

    try:
        from external.gaussian_splatting.arguments import (  # noqa: WPS433 - runtime import keeping CLI lightweight
            ModelParams,
            OptimizationParams,
            PipelineParams,
        )
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Failed to import GraphDECO argument helpers") from exc

    resume = parser.add_argument_group("Finetune Options")
    resume.add_argument(
        "--start_iteration",
        "--first_iter",
        dest="start_iteration",
        type=int,
        default=30_000,
        help="Iteration to resume finetuning from.",
    )
    resume.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[30_000],
        help="Iteration(s) at which to export the scene during finetuning.",
    )
    resume.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[],
        help="Iteration(s) at which to write finetune checkpoints.",
    )

    logging = parser.add_argument_group("Logging & checkpoints")
    logging.add_argument("--test_iterations", nargs="+", type=int, default=None)
    logging.add_argument("--quiet", action="store_true")

    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)

    network = parser.add_argument_group("Viewer & debugging")
    network.add_argument("--ip", type=str, default="127.0.0.1")
    network.add_argument("--port", type=int, default=6009)
    network.add_argument("--debug_from", type=int, default=-1)
    network.add_argument("--detect_anomaly", action="store_true", default=False)

    return _attach_handler(parser, name="finetune", handler=handler)


def add_classify_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "classify",
        help="Indoor/outdoor classifier for a scene or CSV manifest.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=expanduser_path,
        default=None,
        help="Scene PLY to classify (single-scene mode).",
    )
    parser.add_argument(
        "--csv",
        type=expanduser_path,
        default=None,
        help="CSV of 'path,label' pairs for batch evaluation (overrides --model).",
    )
    parser.add_argument(
        "--json-out",
        type=expanduser_path,
        default=None,
        help="Optional JSON path for detailed classifier output.",
    )
    parser.add_argument(
        "--uncertain-as-skip",
        action=BooleanOptionalAction,
        default=False,
        help="Do not count uncertain predictions when computing accuracy.",
    )
    parser.add_argument(
        "--mode",
        choices=available_classifier_presets(),
        default=None,
        help="Preset of heuristic thresholds; explicit CLI flags override preset values.",
    )

    heur = parser.add_argument_group("Classifier thresholds")
    heur.add_argument("--tau-shell-frac", type=float, default=0.03, action=_StoreWithFlag)
    heur.add_argument("--tau-topbot-frac", type=float, default=0.03, action=_StoreWithFlag)
    heur.add_argument("--thr-core-frac", type=float, default=0.12, action=_StoreWithFlag)
    heur.add_argument("--open-radius-frac", type=float, default=0.35, action=_StoreWithFlag)
    heur.add_argument("--open-bins", type=int, default=200, action=_StoreWithFlag)
    heur.add_argument("--open-cone-deg", type=float, default=15.0, action=_StoreWithFlag)
    heur.add_argument("--open-alpha-min", type=float, default=0.25, action=_StoreWithFlag)

    bounds = parser.add_argument_group("Robust AABB")
    bounds.add_argument("--aabb-pct-low", type=float, default=0.005, action=_StoreWithFlag)
    bounds.add_argument("--aabb-pct-high", type=float, default=0.995, action=_StoreWithFlag)
    bounds.add_argument("--aabb-mul", type=float, default=1.05, action=_StoreWithFlag)

    fallback = parser.add_argument_group("Fallback & uncertainty")
    fallback.add_argument(
        "--io-fallback",
        choices=["auto", "off", "force"],
        default="auto",
        action=_StoreWithFlag,
        help="Fallback mode when the classifier is uncertain.",
    )
    fallback.add_argument(
        "--io-uncertain-band",
        type=float,
        default=0.10,
        action=_StoreWithFlag,
        help="If |p_outdoor-0.5| < band classify as 'uncertain'.",
    )

    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "-n",
        "--num-cams",
        dest="num_cams",
        type=int,
        default=500,
        help="Number of cameras to generate for sensitivity (default: 500).",
    )
    camera_group.add_argument(
        "--cameras-json",
        dest="cameras_json",
        type=expanduser_path,
        default=None,
        help="Use an existing cameras.json instead of generating new cameras.",
    )

    _add_common_runtime(parser.add_argument_group("Runtime"))
    return _attach_handler(parser, name="classify", handler=handler)


def add_sens_cams_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "sens_cams",
        help="Manage cameras for sensitivity workflows (extract/import/generate).",
        description=SENS_CAMS_DESCRIPTION,
        formatter_class=_HelpFormatter,
    )
    mode_group = parser.add_argument_group("Mode selection")
    mode_flags = mode_group.add_mutually_exclusive_group()
    mode_flags.add_argument(
        "--generate",
        dest="mode",
        action="store_const",
        const="generate",
        help="Generate new cameras via the ROI sampler (requires --num).",
    )
    mode_flags.add_argument(
        "--import-json",
        dest="import_json",
        type=expanduser_path,
        help="Import a precomputed cameras JSON file instead of using dataset assets.",
    )

    paths = parser.add_argument_group("Required paths")
    paths.add_argument(
        "-m",
        "--model",
        type=expanduser_path,
        required=True,
        help="Path to baseline point_cloud PLY.",
    )
    paths.add_argument(
        "-o",
        "--out",
        type=lambda p: str(ensure_dir_for(expanduser_path(p))),
        required=True,
        help="Directory for outputs (JSON + optional debug artifacts).",
    )
    extract_group = parser.add_argument_group("Extraction options")
    extract_group.add_argument(
        "--source",
        dest="source",
        type=expanduser_path,
        default=None,
        help="Dataset root used when extracting existing cameras.",
    )
    extract_group.add_argument(
        "--model-dir",
        dest="extract_model_dir",
        type=expanduser_path,
        default=None,
        help="GraphDECO model directory when extraction inference from --model fails (custom directory layouts, symlinks, etc.).",
    )
    extract_group.add_argument(
        "--extract-resolution",
        dest="extract_resolution",
        type=int,
        default=-1,
        help="Resolution downscale factor for extract mode (-1 keeps train resolution).",
    )
    extract_group.add_argument(
        "--extract-data-device",
        dest="extract_data_device",
        default="cuda",
        help="Device hint passed to the extractor Scene builder.",
    )
    extract_group.add_argument(
        "--extract-target",
        choices=("train", "test", "all"),
        default="train",
        dest="extract_target",
        help="Which extracted JSON to treat as primary (train/test/all).",
    )

    generate_group = parser.add_argument_group("Generation options")
    generate_group.add_argument(
        "-num",
        "--num",
        "-n",
        dest="num_cameras",
        type=int,
        default=None,
        help="Number of cameras to generate (required with --generate).",
    )
    generate_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation mode.",
    )
    generate_group.add_argument(
        "--target-coverage",
        type=float,
        default=0.99,
        help="Desired scene coverage fraction before early-stop.",
    )

    import_group = parser.add_argument_group("Import options")
    import_group.add_argument(
        "--json-name",
        default="import_cameras.json",
        help="Filename to use when importing cameras.",
    )
    _add_common_runtime(parser.add_argument_group("Runtime"), include_seed=False)
    return _attach_handler(parser, name="sens_cams", handler=handler)


def add_sens_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "sens",
        help="Compute sensitivities from cameras and export sens_*.pt tensors.",
    )
    io = parser.add_argument_group("I/O")
    io.add_argument("-m", "--model", dest="model", type=expanduser_path, required=True,
                    help="Input model PLY.")
    io.add_argument(
        "-cams",
        "--cams",
        dest="cams",
        type=expanduser_path,
        nargs="+",
        default=None,
        help="Shortcut for --cams-train; accepts one or more camera JSON files.",
    )
    io.add_argument(
        "--cams-train",
        dest="cameras_train",
        type=expanduser_path,
        nargs="+",
        default=None,
        help="Camera JSON(s) used for training views.",
    )
    io.add_argument(
        "--cams-test",
        dest="cameras_test",
        type=expanduser_path,
        nargs="+",
        default=None,
        help="Camera JSON(s) used for evaluation views.",
    )
    io.add_argument(
        "-o",
        "--out",
        dest="out_dir",
        type=lambda p: str(ensure_dir_for(expanduser_path(p))),
        default=None,
        help="Directory for sensitivity tensors (default: <project>/outputs/sens).",
    )
    io.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Override camera image size if JSON lacks it.",
    )
    io.add_argument(
        "--use-train-only",
        action=BooleanOptionalAction,
        default=True,
        help="Ignore test cameras when aggregating sensitivities.",
    )

    toggles = parser.add_argument_group("Sensitivity toggles")
    toggles.add_argument("--per-channel", action=BooleanOptionalAction, default=True)
    toggles.add_argument("--per-pixel", action=BooleanOptionalAction, default=True)
    toggles.add_argument("--per-tile", action=BooleanOptionalAction, default=True)
    toggles.add_argument("--per-image", action=BooleanOptionalAction, default=True)
    toggles.add_argument("--per-batch", action=BooleanOptionalAction, default=True)
    toggles.add_argument("--per-scene", action=BooleanOptionalAction, default=True)
    toggles.add_argument(
        "--sensitivity-norm",
        choices=("l1", "l2"),
        default="l1",
        help="Choose L1 (default) or L2 sensitivity aggregation.",
    )
    toggles.add_argument(
        "--sensitivity-reduce",
        choices=("max", "mean"),
        default="max",
        help="Aggregate per-gaussian sensitivities via max (default) or mean.",
    )
    toggles.add_argument(
        "--sensitivity-nocolor",
        action=BooleanOptionalAction,
        default=False,
        help="Ignore Gaussian RGB when computing sensitivities (saves *_nocolor tensors).",
    )

    parser.add_argument("--block-xy", type=int, default=16, help="Tile size (BLOCK_X = BLOCK_Y).")
    parser.add_argument("--bg", type=_float3, default=(0.0, 0.0, 0.0), help="Background RGB (linear).")
    parser.add_argument(
        "--white-background",
        action=BooleanOptionalAction,
        default=False,
        help="Render with white background regardless of --bg.",
    )

    _add_common_runtime(parser.add_argument_group("Runtime"))
    return _attach_handler(parser, name="sens", handler=handler)


def add_coreset_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "coreset",
        help="Sample a coreset from sensitivities and save a compact PLY.",
        description=CORESET_DESCRIPTION,
        epilog=CORESET_EXAMPLES,
        formatter_class=_HelpFormatter,
    )
    io = parser.add_argument_group("Required I/O")
    io.add_argument("-m", "--model", dest="model", type=expanduser_path, required=True,
                    help="Input Gaussian PLY to subsample.")
    io.add_argument(
        "-o",
        "--out",
        dest="out_ply",
        type=lambda p: str((ensure_dir_for(expanduser_path(p), treat_as_file=True) and expanduser_path(p))),
        required=True,
        help="Output coreset PLY path.",
    )

    sens_group = parser.add_argument_group("Sensitivity input")
    sens_exclusive = sens_group.add_mutually_exclusive_group(required=True)
    sens_exclusive.add_argument(
        "-sens",
        "--sens",
        dest="sens_path",
        type=expanduser_path,
        default=None,
        help="Sensitivity tensor (.pt) or directory containing tensors.",
    )
    sens_exclusive.add_argument(
        "--uniform",
        action="store_true",
        help="Sample uniformly without requiring sensitivity tensors.",
    )

    sizing_group = parser.add_argument_group("Coreset sizing (choose one)")
    size_group = sizing_group.add_mutually_exclusive_group(required=True)
    size_group.add_argument("-k", "--coreset-size", type=int, help="Number of draws/items.")
    size_group.add_argument(
        "--prune-ratio",
        type=_prune_ratio,
        help="Fraction of Gaussians removed (0.9 prunes 90%% and keeps 10%%).",
    )
    sampling = parser.add_argument_group("Sampling options")
    sampling.add_argument("--temperature", type=float, default=1.0, help="Flatten sensitivity distribution.")
    sampling.add_argument(
        "--allow-duplicates",
        action=BooleanOptionalAction,
        default=False,
        help="Sample with replacement when True.",
    )
    sampling.add_argument(
        "--apply-ss-weights",
        action=BooleanOptionalAction,
        default=False,
        help="Apply sensitivity-sampling weights to SH colors.",
    )
    sampling.add_argument(
        "--normalize-weights",
        action=BooleanOptionalAction,
        default=False,
        help="Clip/normalize weights post sampling.",
    )
    sampling.add_argument(
        "--weights-L1",
        type=_float_list,
        default=None,
        metavar="W0 W1 ...",
        help="Optional linear weights to mix multiple sensitivity tensors.",
    )
    sampling.add_argument("--seed", type=int, default=None, help="Sampling seed override.")
    sampling.add_argument(
        "--sampling-mode",
        choices=("multinomial", "topk"),
        default="multinomial",
        help="Choose stochastic multinomial draws (default) or deterministic top-k selection.",
    )

    sens_group = parser.add_argument_group("Sensitivity options")
    sens_group.add_argument(
        "--sens-granularity",
        choices=("per_channel", "per_pixel", "per_tile", "per_image", "per_batch", "per_scene"),
        default="per_image",
        help="Sensitivity tensor granularity to sample when --sens points to a directory.",
    )
    sens_group.add_argument(
        "--sens-norm",
        choices=("l1", "l2"),
        default="l1",
        help="Norm used when saving sensitivities (determines filename suffix).",
    )
    sens_group.add_argument(
        "--sens-reduce",
        choices=("max", "mean"),
        default="max",
        help="Reduction used when saving sensitivities (determines filename suffix).",
    )
    sens_group.add_argument(
        "--sens-nocolor",
        action=BooleanOptionalAction,
        default=False,
        help="Load *_nocolor sensitivity tensors when --sens points to a directory.",
    )

    _add_common_runtime(parser.add_argument_group("Runtime"), include_seed=False)
    return _attach_handler(parser, name="coreset", handler=handler)


def add_all_coresets_subparser(
    subparsers: argparse._SubParsersAction,
    handler: Optional[GraphdecoHandler] = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "all_coresets",
        help="Create coresets for uniform sampling and all sensitivity combinations.",
        formatter_class=_HelpFormatter,
    )

    required = parser.add_argument_group("Required")
    required.add_argument(
        "-m",
        "--model",
        dest="model",
        type=expanduser_path,
        required=True,
        help="Input Gaussian PLY to subsample.",
    )
    required.add_argument(
        "-o",
        "--out",
        dest="out_dir",
        type=expanduser_path,
        required=True,
        help="Path to output dir for all coresets.",
    )
    required.add_argument(
        "-sens",
        "--sens",
        dest="sens_path",
        type=expanduser_path,
        required=True,
        help="Path to sensitivity directory containing all tensors.",
    )
    required.add_argument(
        "--prune-ratio",
        type=_prune_ratio,
        required=True,
        help="Fraction of Gaussians removed (0.9 prunes 90%% and keeps 10%%).",
    )

    sampling = parser.add_argument_group("Optional sampling options")
    sampling.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Flatten sensitivity distribution. (default: 1.0)",
    )
    sampling.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed override.",
    )
    sampling.add_argument(
        "--sampling-mode",
        choices=("multinomial", "topk"),
        default="multinomial",
        help="Choose stochastic multinomial draws (default) or deterministic top-k selection.",
    )
    sampling.add_argument(
        "--sens-nocolor",
        action=BooleanOptionalAction,
        default=False,
        help="Consume *_nocolor sensitivity tensors and emit nocolor-tagged coreset directories.",
    )

    _add_common_runtime(parser.add_argument_group("Runtime"), include_seed=False)
    return _attach_handler(parser, name="all_coresets", handler=handler)

# ---------------------------------------------------------------------------
# Commands: Parser factory

def build_commands_parser(
    selected: Optional[Sequence[str]] = None,
    *,
    handlers: Optional[Mapping[str, GraphdecoHandler]] = None,
    prog: str = "gs-coresets",
) -> argparse.ArgumentParser:
    """
    Build the unified CLI parser.

    Parameters
    ----------
    selected:
        Optional ordered subset of subcommands to register. Defaults to all gs_coresets.modules.
    handlers:
        Mapping from subcommand name to callback. Missing entries produce a RuntimeError
        when the command is invoked (useful for partial CLI construction in tests).
    """
    handlers = handlers or {}

    # Top-level commands parser; prog is overridden by main.py so usage strings
    # read “gs-coresets …” when invoked through the umbrella CLI.
    parser = ArgumentParser(
        prog=prog,
        description="Unified CLI for GraphDECO scripts and gs_coresets.utils tooling.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Echo delegated modules and additional diagnostics.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    available = {
        "train": add_train_subparser,
        "render": add_render_subparser,
        "metrics": add_metrics_subparser,
        "full_eval": add_full_eval_subparser,
        "classify": add_classify_subparser,
        "sens_cams": add_sens_cams_subparser,
        "sens": add_sens_subparser,
        "all_coresets": add_all_coresets_subparser,
        "coreset": add_coreset_subparser,
        "finetune": add_finetune_subparser,
    }

    if selected is None:
        selected_iterable = COMMAND_ORDER
    else:
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(f"Unknown subcommand(s) requested: {', '.join(missing)}")
        selected_iterable = selected

    for name in selected_iterable:
        builder = available[name]
        builder(subparsers, handler=handlers.get(name))

    return parser
