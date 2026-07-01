#!/usr/bin/env python3
"""
Scene classification utilities for the indoor/outdoor heuristic.

This module powers the `gs-coresets classify` subcommand and can also be used
programmatically to evaluate a CSV of scenes.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple
import torch

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets.utils.default_paths import ensure_all_on_syspath
ensure_all_on_syspath()

from gs_coresets.utils.classify_scene_utils import (
    CLASSIFIER_PRESETS,
    IndoorOutdoorResult,
    classify_indoor_outdoor,
    unwrap_io_result,
)
from gs_coresets.utils.io_utils import write_json
from gs_coresets.utils.types_devices_utils import resolve_device_name


def parse_csv(csv_path: Path) -> Iterator[Tuple[Path, str]]:
    """
    Yield (scene_path, label) pairs from a simple CSV without headers.
    Lines beginning with '#' and blank lines are skipped.
    """
    with csv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [token.strip() for token in stripped.split(",")]
            if len(parts) < 2:
                print(f"[WARN] Skipping malformed line: {raw_line.strip()}")
                continue
            ply_path, label = Path(parts[0]), parts[1].lower()
            if label not in {"indoor", "outdoor"}:
                print(f"[WARN] Unknown label '{label}' on line: {raw_line.strip()}")
                continue
            yield ply_path, label


def resolve_classifier_params(
    args: argparse.Namespace,
    *,
    default_mode: str = "safe",
) -> Tuple[Dict[str, Any], str]:
    """
    Merge CLI overrides with preset defaults while tracking which values the user set.
    """
    overrides = set(getattr(args, "_seen_args", set()))
    params: Dict[str, Any] = {
        "tau_shell_frac": args.tau_shell_frac,
        "tau_topbot_frac": args.tau_topbot_frac,
        "thr_core_frac": args.thr_core_frac,
        "open_radius_frac": args.open_radius_frac,
        "open_bins": args.open_bins,
        "open_cone_deg": args.open_cone_deg,
        "open_alpha_min": args.open_alpha_min,
        "aabb_pct_low": args.aabb_pct_low,
        "aabb_pct_high": args.aabb_pct_high,
        "aabb_mul": args.aabb_mul,
        "io_fallback": args.io_fallback,
        "io_uncertain_band": args.io_uncertain_band,
    }

    mode = getattr(args, "mode", None) or default_mode
    preset = CLASSIFIER_PRESETS.get(mode)
    if preset:
        for key, value in preset.items():
            if key not in overrides:
                params[key] = value

    params["tau_shell_frac"] = float(params["tau_shell_frac"])
    params["tau_topbot_frac"] = float(params["tau_topbot_frac"])
    params["thr_core_frac"] = float(params["thr_core_frac"])
    params["open_radius_frac"] = float(params["open_radius_frac"])
    params["open_bins"] = int(params["open_bins"])
    params["open_cone_deg"] = float(params["open_cone_deg"])
    params["open_alpha_min"] = float(params["open_alpha_min"])
    params["aabb_pct_low"] = float(params["aabb_pct_low"])
    params["aabb_pct_high"] = float(params["aabb_pct_high"])
    params["aabb_mul"] = float(params["aabb_mul"])
    params["io_fallback"] = str(params["io_fallback"])
    params["io_uncertain_band"] = float(params["io_uncertain_band"])
    return params, mode


def evaluate_indoor_outdoor(
    scene_path: str | Path,
    *,
    device: str | None,
    tau_shell_frac: float,
    tau_topbot_frac: float,
    thr_core_frac: float,
    open_radius_frac: float,
    open_bins: int,
    open_cone_deg: float,
    open_alpha_min: float,
    aabb_pct_low: float,
    aabb_pct_high: float,
    aabb_mul: float,
    io_fallback: str,
    io_uncertain_band: float,
) -> Dict[str, Any]:
    """
    Classify a single scene and return its label, confidence, and raw metrics.
    """
    torch_device = torch.device(resolve_device_name(device)) if device is not None else None

    t0 = time.time()
    result = classify_indoor_outdoor(
        str(scene_path),
        device=torch_device,
        tau_shell_frac=tau_shell_frac,
        tau_topbot_frac=tau_topbot_frac,
        thr_core_frac=thr_core_frac,
        openness_radius_frac=open_radius_frac,
        openness_bins=open_bins,
        openness_cone_deg=open_cone_deg,
        openness_alpha_min=open_alpha_min,
        pct_bounds=(aabb_pct_low, aabb_pct_high),
        mul_fact=aabb_mul,
        fallback_mode=io_fallback,
        uncertain_band=io_uncertain_band,
    )
    label, confidence, metrics = unwrap_io_result(result)
    elapsed = time.time() - t0

    return {
        "path": str(scene_path),
        "label": label,
        "confidence": float(confidence),
        "metrics": metrics,
        "time_sec": elapsed,
    }


def run_classify_cli(args: argparse.Namespace) -> Tuple[Dict[str, Any] | None, int]:
    """
    Dispatch CSV or single-scene classification based on parsed CLI arguments.

    Returns (payload, exit_code) where payload is a per-scene record (single mode)
    or the aggregate summary (CSV mode).
    """
    params, mode_used = resolve_classifier_params(args)
    device_name = args.device

    if args.csv:
        csv_path = Path(args.csv)
        json_path = Path(args.json_out) if args.json_out else None
        rows: list[Dict[str, Any]] = []
        cm: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total = 0
        correct = 0
        skipped = 0
        outdoor_fp_paths: list[str] = []
        outdoor_used_fallback: list[str] = []

        t0 = time.time()
        for scene_path, truth in parse_csv(csv_path):
            record = evaluate_indoor_outdoor(
                scene_path,
                device=device_name,
                tau_shell_frac=params["tau_shell_frac"],
                tau_topbot_frac=params["tau_topbot_frac"],
                thr_core_frac=params["thr_core_frac"],
                open_radius_frac=params["open_radius_frac"],
                open_bins=params["open_bins"],
                open_cone_deg=params["open_cone_deg"],
                open_alpha_min=params["open_alpha_min"],
                aabb_pct_low=params["aabb_pct_low"],
                aabb_pct_high=params["aabb_pct_high"],
                aabb_mul=params["aabb_mul"],
                io_fallback=params["io_fallback"],
                io_uncertain_band=params["io_uncertain_band"],
            )

            pred = record["label"]
            confidence = record["confidence"]
            metrics = record["metrics"]
            print(f"{scene_path}: truth={truth:7s} pred={pred:9s} conf={confidence:.3f} ({record['time_sec']:.2f}s)")

            cm[truth][pred] += 1
            if pred == "uncertain" and bool(args.uncertain_as_skip):
                skipped += 1
            else:
                total += 1
                correct += int(pred == truth)

            if truth == "outdoor":
                if pred != "outdoor":
                    outdoor_fp_paths.append(str(scene_path))
                if float(metrics.get("fallback_used", 0.0)) >= 0.5:
                    outdoor_used_fallback.append(str(scene_path))

            rows.append({
                "path": record["path"],
                "truth": truth,
                "pred": pred,
                "confidence": confidence,
                "p_outdoor": metrics.get("p_outdoor"),
                "time_sec": record["time_sec"],
                "fallback_used": float(metrics.get("fallback_used", 0.0)),
            })

        elapsed = time.time() - t0
        accuracy = (correct / total) if total else 0.0

        print("\nConfusion matrix (truth → pred counts):")
        labels = ["indoor", "outdoor", "uncertain"]
        header = "           " + " ".join(f"{label:>10s}" for label in labels)
        print(header)
        for truth_label in ["indoor", "outdoor"]:
            row_counts = [cm[truth_label][label] for label in labels]
            print(f"{truth_label:>10s} " + " ".join(f"{count:10d}" for count in row_counts))

        print(f"\nAccuracy: {accuracy*100:.2f}% (correct={correct}, total={total}, skipped={skipped})")
        print(f"Scenes evaluated: {len(rows)}  |  elapsed: {elapsed:.2f}s")

        if outdoor_fp_paths:
            print(f"[WARN] Outdoor scenes misclassified: {len(outdoor_fp_paths)}")
            for path in outdoor_fp_paths:
                print(f"  - {path}")
        if outdoor_used_fallback:
            print(f"[WARN] Outdoor scenes where fallback engaged: {len(outdoor_used_fallback)}")
            for path in outdoor_used_fallback:
                print(f"  - {path}")

        summary = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "skipped": skipped,
            "mode": mode_used,
            "confusion": {truth: dict(preds) for truth, preds in cm.items()},
            "rows": rows,
            "params": {
                "tau_shell_frac": params["tau_shell_frac"],
                "tau_topbot_frac": params["tau_topbot_frac"],
                "thr_core_frac": params["thr_core_frac"],
                "open_radius_frac": params["open_radius_frac"],
                "open_bins": params["open_bins"],
                "open_cone_deg": params["open_cone_deg"],
                "open_alpha_min": params["open_alpha_min"],
                "aabb_pct_low": params["aabb_pct_low"],
                "aabb_pct_high": params["aabb_pct_high"],
                "aabb_mul": params["aabb_mul"],
                "io_fallback": params["io_fallback"],
                "io_uncertain_band": params["io_uncertain_band"],
                "uncertain_as_skip": bool(args.uncertain_as_skip),
            },
            "regression": {
                "outdoor_false_positives": outdoor_fp_paths,
                "outdoor_used_fallback": outdoor_used_fallback,
            },
        }

        if json_path is not None:
            target = write_json(json_path, summary, indent=2)
            print(f"Wrote JSON: {target}")
        print(f"[classify] accuracy={accuracy:.4f} (total={total}, skipped={skipped})")
        return summary, 0

    if not args.model:
        print("[classify] supply either --csv or --model", file=sys.stderr)
        return None, 2

    record = evaluate_indoor_outdoor(
        args.model,
        device=device_name,
        tau_shell_frac=params["tau_shell_frac"],
        tau_topbot_frac=params["tau_topbot_frac"],
        thr_core_frac=params["thr_core_frac"],
        open_radius_frac=params["open_radius_frac"],
        open_bins=params["open_bins"],
        open_cone_deg=params["open_cone_deg"],
        open_alpha_min=params["open_alpha_min"],
        aabb_pct_low=params["aabb_pct_low"],
        aabb_pct_high=params["aabb_pct_high"],
        aabb_mul=params["aabb_mul"],
        io_fallback=params["io_fallback"],
        io_uncertain_band=params["io_uncertain_band"],
    )
    record["mode"] = mode_used

    print(f"[classify] mode={mode_used} label={record['label']} confidence={record['confidence']:.4f}")

    if args.json_out:
        payload = {
            "label": record["label"],
            "confidence": record["confidence"],
            "metrics": record["metrics"],
        }
        target = write_json(args.json_out, payload, indent=2)
        print(f"Wrote JSON: {target}")

    return record, 0


def main(argv: Iterable[str] | None = None) -> int:
    """
    Allow the module to be executed directly while reusing the shared CLI parser.
    """
    from gs_coresets.utils.args_utils import build_commands_parser

    parser = build_commands_parser()
    cli_args = parser.parse_args(["classify", *(argv or sys.argv[1:])])
    _, exit_code = run_classify_cli(cli_args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
