#!/usr/bin/env python3
# metrics.py — directory-level evaluator that uses gs_coresets.utils/metrics_utils.py

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv
import glob
import os
import torch
from PIL import Image

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets.utils.default_paths import ensure_all_on_syspath
ensure_all_on_syspath()

from gs_coresets.utils.io_utils import ensure_dir_for
from gs_coresets.utils.metrics_utils import (
    compute_psnr,
    compute_dssim,
    compute_lpips,
    to_tensor01,  # small helper
)

@dataclass
class Pair:
    pred: str
    gt: str
    key: str  # paired key (stem or index)
    def name(self) -> str:
        return os.path.basename(self.pred)

def _list_images(folder: str, pattern: str, recursive: bool) -> List[str]:
    if recursive and "**" not in pattern:
        pattern = os.path.join(folder, "**", pattern)
    else:
        pattern = os.path.join(folder, pattern)
    return sorted(glob.glob(pattern, recursive=recursive))

def _stem(path: str) -> str:
    return Path(path).stem

def _strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if prefix and name.startswith(prefix) else name

def _pair_sorted(preds: List[str], gts: List[str]) -> List[Pair]:
    if len(preds) != len(gts):
        raise RuntimeError(f"sorted pairing requires same counts: {len(preds)} preds vs {len(gts)} gts")
    return [Pair(pred=p, gt=g, key=str(i)) for i, (p, g) in enumerate(zip(preds, gts))]

def _pair_by_stem(preds: List[str], gts: List[str], strip_pred_prefix: str, strip_gt_prefix: str) -> List[Pair]:
    gt_map: Dict[str, str] = {}
    for g in gts:
        key = _strip_prefix(_stem(g), strip_gt_prefix)
        gt_map[key] = g
    pairs: List[Pair] = []
    misses: List[str] = []
    for p in preds:
        key = _strip_prefix(_stem(p), strip_pred_prefix)
        if key in gt_map:
            pairs.append(Pair(pred=p, gt=gt_map[key], key=key))
        else:
            misses.append(p)
    if not pairs:
        raise RuntimeError("No stem matches found; check prefixes or filenames.")
    if misses:
        print(f"[metrics] warning: {len(misses)} predictions had no GT match; first: {misses[0]}")
    return sorted(pairs, key=lambda x: x.key)

def _resize_to(img: Image.Image, w: int, h: int) -> Image.Image:
    if img.size == (w, h):
        return img
    return img.resize((w, h), Image.BICUBIC)

def run_metrics_cli(
    *,
    pred_dir: str,
    gt_dir: str,
    pred_glob: str,
    gt_glob: str,
    recursive: bool,
    match_mode: str,  # "sorted" or "stem"
    strip_pred_prefix: str,
    strip_gt_prefix: str,
    resize_pred_to_gt: bool,
    crop: int,
    want_psnr: bool,
    want_dssim: bool,
    want_lpips: bool,
    lpips_net: str,
    csv_out: Optional[str],
    aggregate_only: bool,
    device: str,
    seed: Optional[int],
) -> int:
    torch.manual_seed(seed or 0)

    preds = _list_images(pred_dir, pred_glob, recursive)
    gts   = _list_images(gt_dir,   gt_glob,   recursive)
    if not preds:
        raise RuntimeError(f"No predictions found in {pred_dir!r} with pattern {pred_glob!r}")
    if not gts:
        raise RuntimeError(f"No ground-truth images found in {gt_dir!r} with pattern {gt_glob!r}")

    if match_mode == "sorted":
        pairs = _pair_sorted(preds, gts)
    else:
        pairs = _pair_by_stem(preds, gts, strip_pred_prefix, strip_gt_prefix)

    dev = torch.device(device)

    rows: List[Dict[str, float]] = []
    sums = {"psnr": 0.0, "dssim": 0.0, "lpips": 0.0}
    counts = {"psnr": 0, "dssim": 0, "lpips": 0}

    for pr in pairs:
        im_p = Image.open(pr.pred).convert("RGB")
        im_g = Image.open(pr.gt).convert("RGB")
        if resize_pred_to_gt:
            im_p = _resize_to(im_p, *im_g.size)
        if im_p.size != im_g.size:
            raise RuntimeError(f"Size mismatch (pred vs gt): {im_p.size} vs {im_g.size} for {pr.name()}")

        p_t = to_tensor01(im_p).to(dev)
        g_t = to_tensor01(im_g).to(dev)

        row = {"file": pr.name()}
        if crop > 0:
            p_t = p_t[..., crop:-crop, crop:-crop]
            g_t = g_t[..., crop:-crop, crop:-crop]

        if want_psnr:
            v = float(compute_psnr(p_t, g_t))
            row["psnr"] = v; sums["psnr"] += v; counts["psnr"] += 1

        if want_dssim:
            try:
                v = float(compute_dssim(p_t, g_t))
                row["dssim"] = v; sums["dssim"] += v; counts["dssim"] += 1
            except Exception as e:
                print(f"[metrics] D-SSIM skipped for {pr.name()}: {e}")

        if want_lpips:
            try:
                v = float(compute_lpips(p_t, g_t, net=lpips_net, device=dev))
                row["lpips"] = v; sums["lpips"] += v; counts["lpips"] += 1
            except Exception as e:
                print(f"[metrics] LPIPS skipped for {pr.name()}: {e}")

        if not aggregate_only:
            rows.append(row)

    # CSV output (per-image)
    if not aggregate_only:
        out_csv = csv_out or str(Path(pred_dir, "metrics.csv"))
        ensure_dir_for(out_csv, treat_as_file=True)
        fieldnames = ["file"] + [k for k in ["psnr", "dssim", "lpips"] if k in rows[0] or rows]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[metrics] wrote {len(rows)} rows → {out_csv}")

    # Aggregate
    def _avg(key: str) -> Optional[float]:
        return (sums[key] / counts[key]) if counts[key] else None

    avg_psnr = _avg("psnr")
    avg_dssim = _avg("dssim")
    avg_lpips = _avg("lpips")

    msg = "[metrics] aggregate:"
    if avg_psnr is not None:  msg += f" PSNR={avg_psnr:.4f}"
    if avg_dssim is not None: msg += f" D-SSIM={avg_dssim:.6f}"
    if avg_lpips is not None: msg += f" LPIPS={avg_lpips:.6f}"
    print(msg)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute image metrics between prediction and ground truth folders.")
    parser.add_argument("pred", help="Directory containing predicted images.")
    parser.add_argument("gt", help="Directory containing ground-truth images.")
    parser.add_argument("--pred-glob", default="*.png", help="Glob for predictions (default: *.png).")
    parser.add_argument("--gt-glob", default="*.png", help="Glob for ground-truth images (default: *.png).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when matching.")
    parser.add_argument(
        "--match-mode",
        choices=["sorted", "stem"],
        default="stem",
        help="Pairing strategy between predictions and ground truth.",
    )
    parser.add_argument("--strip-pred-prefix", default="", help="Prefix to strip from prediction stems.")
    parser.add_argument("--strip-gt-prefix", default="", help="Prefix to strip from ground-truth stems.")
    parser.add_argument("--resize-pred-to-gt", action="store_true", help="Resize predictions to GT resolution.")
    parser.add_argument("--crop", type=int, default=0, help="Number of border pixels to crop before metrics.")
    parser.add_argument("--no-psnr", dest="want_psnr", action="store_false", help="Disable PSNR metric.")
    parser.add_argument("--dssim", dest="want_dssim", action="store_true", help="Enable D-SSIM metric.")
    parser.add_argument("--lpips", dest="want_lpips", action="store_true", help="Enable LPIPS metric.")
    parser.add_argument("--lpips-net", default="alex", help="LPIPS network to use (alex/vgg/squeeze).")
    parser.add_argument("--csv-out", default=None, help="Optional CSV path for per-image metrics.")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip per-image CSV output.")
    parser.add_argument("--device", default="cpu", help="Torch device to use (default: cpu).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.set_defaults(want_psnr=True, want_dssim=False, want_lpips=False)
    args = parser.parse_args(argv)

    return run_metrics_cli(
        pred_dir=args.pred,
        gt_dir=args.gt,
        pred_glob=args.pred_glob,
        gt_glob=args.gt_glob,
        recursive=args.recursive,
        match_mode=args.match_mode,
        strip_pred_prefix=args.strip_pred_prefix,
        strip_gt_prefix=args.strip_gt_prefix,
        resize_pred_to_gt=args.resize_pred_to_gt,
        crop=args.crop,
        want_psnr=args.want_psnr,
        want_dssim=args.want_dssim,
        want_lpips=args.want_lpips,
        lpips_net=args.lpips_net,
        csv_out=args.csv_out,
        aggregate_only=args.aggregate_only,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
