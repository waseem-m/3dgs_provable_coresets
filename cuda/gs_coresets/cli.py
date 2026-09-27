#!/usr/bin/env python3
"""
Unified CLI that orchestrates GraphDECO scripts together with gs_coresets.utils tooling.

GraphDECO commands are delegated via subprocess (respecting the active Python
interpreter) while local utilities execute in-process.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Sequence

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets.utils.default_paths import ensure_all_on_syspath
ensure_all_on_syspath()

from gs_coresets.utils import debug_utils as dbg
from gs_coresets.utils.log_utils import LoggingContext, maybe_enable_logging, run_tee
from gs_coresets.utils.default_paths import resolve_graphdeco_dir, resolve_project_dir

__all__ = [
    "build_commands_parser_entry",
    "main"
]


ROOT = resolve_project_dir()
GRAPHDECO_ROOT = resolve_graphdeco_dir()


# ---------------------------------------------------------------------------
# Shared helpers

def _quote_cmd(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _graphdeco_script_path(script: str) -> Path:
    return GRAPHDECO_ROOT / script


def _graphdeco_help_text(script: str) -> str | None:
    try:
        if script == "train.py":
            from external.gaussian_splatting.arguments import ModelParams, OptimizationParams, PipelineParams

            parser = argparse.ArgumentParser(
                prog=f"python {script}",
                description="Training script parameters",
            )
            ModelParams(parser)
            OptimizationParams(parser)
            PipelineParams(parser)
            parser.add_argument("--ip", type=str, default="127.0.0.1")
            parser.add_argument("--port", type=int, default=6009)
            parser.add_argument("--debug_from", type=int, default=-1)
            parser.add_argument("--detect_anomaly", action="store_true", default=False)
            parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
            parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
            parser.add_argument("--quiet", action="store_true")
            parser.add_argument("--disable_viewer", action="store_true", default=False)
            parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
            parser.add_argument("--start_checkpoint", type=str, default=None)
            return parser.format_help()

        if script == "render.py":
            from external.gaussian_splatting.arguments import ModelParams, PipelineParams

            parser = argparse.ArgumentParser(
                prog=f"python {script}",
                description="Testing script parameters",
            )
            ModelParams(parser, sentinel=True)
            PipelineParams(parser)
            parser.add_argument("--iteration", default=-1, type=int)
            parser.add_argument("--skip_train", action="store_true")
            parser.add_argument("--skip_test", action="store_true")
            parser.add_argument("--quiet", action="store_true")
            return parser.format_help()

        if script == "metrics.py":
            parser = argparse.ArgumentParser(
                prog=f"python {script}",
                description="Metrics script parameters",
            )
            parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
            return parser.format_help()

        if script == "full_eval.py":
            parser = argparse.ArgumentParser(
                prog=f"python {script}",
                description="Full evaluation script parameters",
            )
            parser.add_argument("--skip_training", action="store_true")
            parser.add_argument("--skip_rendering", action="store_true")
            parser.add_argument("--skip_metrics", action="store_true")
            parser.add_argument("--output_path", "-o", default="./eval")
            parser.add_argument("--use_depth", action="store_true")
            parser.add_argument("--use_expcomp", action="store_true")
            parser.add_argument("--fast", action="store_true")
            parser.add_argument("--aa", action="store_true")
            # Always show dataset arguments so they appear in help regardless of skip flags.
            parser.add_argument("--mipnerf360", "-m360", required=False, type=str)
            parser.add_argument("--tanksandtemples", "-tat", required=False, type=str)
            parser.add_argument("--deepblending", "-db", required=False, type=str)
            return parser.format_help()
    except Exception:
        return None
    return None


def _show_graphdeco_help(script: str, args: argparse.Namespace) -> int:
    script_path = _graphdeco_script_path(script)
    if not script_path.is_file():  # pragma: no cover - configuration guard
        print(f"[cli] missing GraphDECO script: {script_path}", file=sys.stderr)
        return 2

    help_text = _graphdeco_help_text(script)
    if help_text is None:
        print(f"[cli] GraphDECO help → {script_path}")
        cmd = [sys.executable, str(script_path), "--help"]
        if getattr(args, "verbose", False):
            print("[cli]", _quote_cmd(cmd))
        proc = subprocess.run(cmd)
        return proc.returncode

    print(f"[cli] Wrapper flags: --dry_run (prints command), -h/--help (shows this text)\n")
    print(help_text)
    return 0


def _run_graphdeco(script: str, args: argparse.Namespace) -> int:
    script_path = _graphdeco_script_path(script)
    if not script_path.is_file():
        print(f"[cli] missing GraphDECO script: {script_path}", file=sys.stderr)
        return 2

    forwarded = list(getattr(args, "script_args", []))
    if script == "full_eval.py":
        forwarded = _normalize_full_eval_args(forwarded)
    cmd: List[str] = [sys.executable, "-u", str(script_path), *forwarded]
    verbose = bool(getattr(args, "verbose", False) or getattr(args, "dry_run", False))
    if verbose:
        print("[cli]", _quote_cmd(cmd))
    if getattr(args, "dry_run", False):
        return 0

    return run_tee(cmd)


def _normalize_full_eval_args(tokens: List[str]) -> List[str]:
    rewritten: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "-o":
            rewritten.append("--output_path")
            if idx + 1 < len(tokens):
                rewritten.append(tokens[idx + 1])
                idx += 2
                continue
        elif token.startswith("-o="):
            rewritten.append("--output_path=" + token.split("=", 1)[1])
            idx += 1
            continue

        rewritten.append(token)
        idx += 1
    return rewritten


def _dispatch_graphdeco(args: argparse.Namespace) -> int:
    script = getattr(args, "script_name", None)
    if not script:
        print("[cli] internal error: no script_name bound to command", file=sys.stderr)
        return 2
    if getattr(args, "_show_help", False):
        return _show_graphdeco_help(script, args)
    return _run_graphdeco(script, args)


# ---------------------------------------------------------------------------
# Local command implementations

def run_classify(args: argparse.Namespace) -> int:
    from gs_coresets.modules.classify_scene import run_classify_cli

    _, exit_code = run_classify_cli(args)
    return exit_code


def run_sens_cams(args: argparse.Namespace) -> int:
    from gs_coresets.modules.sens_cams import run_sens_cams_cli

    _, exit_code = run_sens_cams_cli(args)
    return exit_code


def run_sens(args: argparse.Namespace) -> int:
    from gs_coresets.modules.sensitivity import run_sensitivity_cli

    _, exit_code = run_sensitivity_cli(args)
    return exit_code


def run_coreset(args: argparse.Namespace) -> int:
    from gs_coresets.modules.coreset import run_coreset_cli

    _, exit_code = run_coreset_cli(
        args,
        coreset_size=getattr(args, "coreset_size", None),
        prune_ratio=getattr(args, "prune_ratio", None),
    )
    return exit_code


def run_all_coresets(args: argparse.Namespace) -> int:
    from gs_coresets.modules.all_coresets import run_all_coresets_cli

    _, exit_code = run_all_coresets_cli(args)
    return exit_code


def run_finetune(args: argparse.Namespace) -> int:
    from gs_coresets.modules.finetune import run_finetune_cli

    _, exit_code = run_finetune_cli(args)
    return exit_code


# ---------------------------------------------------------------------------
# Entry-point

def build_commands_parser_entry(prog: str = "gs-coresets") -> ArgumentParser:
    from gs_coresets.utils.args_utils import build_commands_parser
    handlers = {
        "train": _dispatch_graphdeco,
        "render": _dispatch_graphdeco,
        "metrics": _dispatch_graphdeco,
        "full_eval": _dispatch_graphdeco,
        "classify": run_classify,
        "sens_cams": run_sens_cams,
        "sens": run_sens,
        "all_coresets": run_all_coresets,
        "coreset": run_coreset,
        "finetune": run_finetune,
    }
    return build_commands_parser(handlers=handlers, prog=prog)

def main(argv: Sequence[str] | None = None, parser: ArgumentParser | None = None) -> int:
    if parser is None:
        parser = build_commands_parser_entry(prog="gs-coresets")
    args, extras = parser.parse_known_args(argv)

    log_ctx: LoggingContext | None = maybe_enable_logging(getattr(args, "log", None))
    debug_ctx = dbg.maybe_enable_debugging(getattr(args, "debug", False))
    if log_ctx or debug_ctx:
        target = log_ctx.log_path if log_ctx else "stdout only"
        print(f"[cli] logging -> {target} ; DEBUG={dbg.IS_DEBUG_MODE}")
    try:
        if extras:
            if getattr(args, "_forward_unknown", False):
                args.script_args.extend(extras)
            else:
                parser.error(f"unrecognized arguments: {' '.join(extras)}")
        return args.func(args)
    finally:
        if log_ctx is not None:
            log_ctx.close()
        if debug_ctx is not None:
            debug_ctx.close()


if __name__ == "__main__":
    raise SystemExit(main())
