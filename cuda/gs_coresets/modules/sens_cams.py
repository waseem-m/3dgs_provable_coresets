#!/usr/bin/env python3
"""
Camera utilities for the `gs-coresets sens_cams` subcommand.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets.utils.default_paths import ensure_all_on_syspath
ensure_all_on_syspath()

from gs_coresets.utils.camera_generator import CameraGenerator
from gs_coresets.utils import extract_cameras
from gs_coresets.utils.io_utils import ensure_dir_for


def run_sens_cams_cli(args: argparse.Namespace) -> Tuple[Dict[str, Any] | None, int]:
    """
    Handles the cameras used for sensitivity calculations:
        - Generate cameras using the CameraGenerator helper & return JSON.
        - Import cameras from JSON, return same JSON.
        - Extract cameras using extract_cameras.py, return same JSON.
    """
    mode = getattr(args, "mode", "extract")
    import_path = getattr(args, "import_json", None)
    if import_path:
        mode = "import"
    out_dir = Path(args.out)
    ensure_dir_for(out_dir, treat_as_file=False)

    if mode == "generate":
        if args.num_cameras is None:
            print("[sens_cams] --num is required when using --generate", file=sys.stderr)
            return None, 2
        out_json = out_dir / f"generate_cameras_{args.num_cameras}.json"
        generator = CameraGenerator(args.model, str(out_json), seed=args.seed)
        cameras = generator.generate(num_cameras=args.num_cameras, target_coverage=args.target_coverage)
        print(f"[sens_cams] generated {len(cameras)} cameras → {out_json}")
        return {
            "mode": "generate",
            "out_json": str(out_json),
            "num_cameras": len(cameras),
        }, 0

    if mode == "import":
        src = Path(import_path).expanduser().resolve()
        if not src.is_file():
            print(f"[sens_cams] import JSON not found: {src}", file=sys.stderr)
            return None, 2
        dest = out_dir / (args.json_name or src.name)
        if dest.resolve() != src:
            ensure_dir_for(dest, treat_as_file=True)
            dest.write_bytes(src.read_bytes())
        count = len(json.loads(dest.read_text(encoding="utf-8")))
        print(f"[sens_cams] imported {count} cameras → {dest}")
        return {
            "mode": "import",
            "cameras_json": str(dest),
            "original_json": str(src),
            "num_cameras": count,
        }, 0

    # extract mode
    source_path = getattr(args, "source", None)
    if not source_path:
        print("[sens_cams] --source is required for extract mode", file=sys.stderr)
        return None, 2

    model_dir_arg = getattr(args, "extract_model_dir", None)
    if model_dir_arg:
        model_dir_path = Path(model_dir_arg).resolve()
    else:
        model_path = Path(args.model).resolve()
        if model_path.is_file() and len(model_path.parents) >= 3:
            model_dir_path = model_path.parents[2]
        elif model_path.is_file() and len(model_path.parents) >= 1:
            model_dir_path = model_path.parent
        else:
            model_dir_path = model_path
    model_dir = str(model_dir_path)
    train_name = "extract_cameras_train.json"
    test_name = "extract_cameras_test.json"

    train_count, test_count = extract_cameras.extract_both_jsons_auto(
        source_path=source_path,
        model_path=model_dir,
        out_dir=str(out_dir),
        resolution=getattr(args, "extract_resolution", -1),
        data_device=getattr(args, "extract_data_device", "cpu"),
        train_filename=train_name,
        test_filename=test_name,
    )

    all_path = out_dir / "extract_cameras_all.json"
    if all_path.exists():
        try:
            all_count = len(json.loads(all_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:  # pragma: no cover - defensive
            print(f"[sens_cams] failed to parse {all_path} for camera count.", file=sys.stderr)
            all_count = train_count + test_count
    else:
        all_count = train_count + test_count

    payload = {
        "mode": "extract",
        "train_cameras": str(out_dir / train_name),
        "test_cameras": str(out_dir / test_name),
        "all_cameras": str(out_dir / "extract_cameras_all.json"),
        "num_train": train_count,
        "num_test": test_count,
        "num_all": all_count,
        "num_cameras": all_count,
    }
    print(
        "[sens_cams] extracted cameras →"
        f" train={payload['train_cameras']} ({train_count})"
        f", test={payload['test_cameras']} ({test_count})"
        f", all={payload['all_cameras']} ({all_count})"
    )
    return payload, 0


def main(argv: list[str] | None = None) -> int:
    from gs_coresets.utils.args_utils import build_commands_parser

    parser = build_commands_parser(selected=("sens_cams",))
    args = parser.parse_args(["sens_cams", *(argv or [])])
    _, exit_code = run_sens_cams_cli(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
