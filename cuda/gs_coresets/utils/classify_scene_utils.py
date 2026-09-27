"""
Indoor vs Outdoor classifier for 3D Gaussian Splatting (3DGS) PLY models.

v1.3 — Solid-angle openness (global & upward)
--------------------------------------------
Adds directional openness using a fast Fibonacci-sphere binning:
- From a robust center, we sample ~200 view directions.
- For each direction we ask: is there any reasonably opaque Gaussian within
  a short radial range along (roughly) that direction?
- We compute the fraction of open directions globally and for the upper
  hemisphere (y>0), which acts like a "sky openness" proxy.

This strengthens generalization across datasets:
- Outdoor: high upward openness, often high global openness.
- Indoor: low openness in all directions; even when ceilings are missing,
  walls reduce global openness.

Also keeps v1.2 improvements (paired walls, enclosure-gap, sparse-top gating).

Outputs
-------
- label: "indoor" | "outdoor" | "uncertain"
- confidence: float in [0,1]
- metrics: dict with intermediate measures

Usage
-----
>>> from gs_coresets.utils.io_utils import load_gaussian_ply
>>> from gs_coresets.utils.classify_scene_utils import classify_indoor_outdoor
>>> g = load_gaussian_ply("scene.ply")
>>> label, conf, metrics = classify_indoor_outdoor(g, openness_radius_frac=0.35)
"""
# ------------------------ helpers ------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math
import torch

from gs_coresets.utils.io_utils import load_gaussian_ply
from gs_coresets.utils.debug_utils import print_debug, IS_DEBUG_MODE
from gs_coresets.utils.scene_utils import (
    fibonacci_sphere,
    ray_aabb_exit_lengths,
    robust_aabb,
    AABB_DEFAULTS,
)


CLASSIFIER_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "tau_shell_frac": 0.03,
        "tau_topbot_frac": 0.03,
        "thr_core_frac": 0.12,
        "open_radius_frac": 0.35,
        "open_bins": 200,
        "open_cone_deg": 15.0,
        "open_alpha_min": 0.25,
        "aabb_pct_low": 0.005,
        "aabb_pct_high": 0.995,
        "aabb_mul": 1.05,
        "io_uncertain_band": 0.10,
        "io_fallback": "auto",
    },
    "smart": {
        "aabb_pct_low": 0.005,
        "aabb_pct_high": 0.995,
        "aabb_mul": 1.05,
        "tau_shell_frac": 0.085,
        "tau_topbot_frac": 0.050,
        "thr_core_frac": 0.14,
        "open_radius_frac": 0.37,
        "open_bins": 240,
        "open_cone_deg": 20.0,
        "open_alpha_min": 0.20,
    },
    "safe": {
        "aabb_pct_low": 0.005,
        "aabb_pct_high": 0.995,
        "aabb_mul": 1.05,
        "tau_shell_frac": 0.085,
        "tau_topbot_frac": 0.050,
        "thr_core_frac": 0.14,
        "open_radius_frac": 0.37,
        "open_bins": 240,
        "open_cone_deg": 20.0,
        "open_alpha_min": 0.20,
        "io_fallback": "auto",
        "io_uncertain_band": 0.10,
    },
    "wide": {
        "tau_shell_frac": 0.10,
        "tau_topbot_frac": 0.06,
        "thr_core_frac": 0.14,
        "open_radius_frac": 0.37,
        "open_bins": 240,
        "open_cone_deg": 22.0,
        "open_alpha_min": 0.20,
    },
    "tight": {
        "aabb_pct_low": 0.005,
        "aabb_pct_high": 0.995,
        "aabb_mul": 1.05,
        "tau_shell_frac": 0.038,
        "tau_topbot_frac": 0.03,
        "thr_core_frac": 0.16,
        "open_radius_frac": 0.37,
        "open_bins": 240,
        "open_cone_deg": 22.0,
        "open_alpha_min": 0.20,
    },
    "balanced": {
        "tau_shell_frac": 0.035,
        "tau_topbot_frac": 0.03,
        "thr_core_frac": 0.145,
        "open_radius_frac": 0.37,
        "open_bins": 240,
        "open_cone_deg": 18.0,
        "open_alpha_min": 0.23,
    },
    "skew-in": {
        "tau_shell_frac": 0.04,
        "tau_topbot_frac": 0.02,
        "thr_core_frac": 0.14,
        "open_radius_frac": 0.40,
        "open_bins": 300,
        "open_cone_deg": 20.0,
        "open_alpha_min": 0.20,
    },
    "skew-out": {
        "tau_shell_frac": 0.03,
        "tau_topbot_frac": 0.03,
        "thr_core_frac": 0.12,
        "open_radius_frac": 0.35,
        "open_bins": 300,
        "open_cone_deg": 15.0,
        "open_alpha_min": 0.25,
    },
}


def available_classifier_presets() -> Tuple[str, ...]:
    return tuple(CLASSIFIER_PRESETS.keys())


def unwrap_io_result(
    result: "IndoorOutdoorResult | Tuple[str, float, Dict[str, Any]]",
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Accept both (label, confidence, metrics) tuples and IndoorOutdoorResult objects.
    """
    if isinstance(result, tuple) and len(result) == 3:
        label, conf, metrics = result
        return label, conf, metrics

    label = getattr(result, "label", None)
    conf = getattr(result, "confidence", None)
    metrics = getattr(result, "metrics", None)
    if label is None or conf is None or metrics is None:
        raise TypeError(f"Unexpected return type from classify_indoor_outdoor: {type(result)}")
    return label, conf, metrics


# ---------- small utilities ----------

def _dc_rgb01(features_dc: torch.Tensor) -> torch.Tensor:
    """
    Convert SH-DC coefficients to approximate RGB in [0,1].
    Expect shape (N,3,1) as in GraphDECO.
    """
    if features_dc.ndim != 3 or features_dc.shape[1] != 3:
        raise ValueError(f"Expected features_dc (N,3,1), got {tuple(features_dc.shape)}")
    rgb = features_dc.squeeze(-1) + 0.5
    return rgb.clamp(0.0, 1.0)


# ---------- rewritten: directional openness ----------

@torch.no_grad()
def _directional_openness(
    xyz: torch.Tensor,
    alpha: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    *,
    lo: torch.Tensor,
    hi: torch.Tensor,
    num_bins: int = 200,
    cone_cos: float = 0.965,          # cos(15°) by default
    alpha_min: float = 0.25,
) -> Dict[str, float]:
    """
    Approximate solid-angle coverage around `center`.
    For each direction bin v, we test if there exists a point whose direction
    from center is within the cone around v AND closer than `radius` (and also
    closer than the AABB exit along v, to avoid counting "outside air").
    """
    dev = xyz.device
    dirs = fibonacci_sphere(num_bins, dev)                 # (B,3)
    r_wall = ray_aabb_exit_lengths(center, dirs, lo, hi)   # (B,)

    v = xyz - center.view(1, 3)          # (N,3)
    dist = torch.linalg.norm(v, dim=-1)  # (N,)
    good = alpha >= alpha_min
    if good.sum() == 0:
        return {"open_frac_global": 1.0, "open_frac_up": 1.0}

    v = v[good]; dist = dist[good]
    u = torch.nn.functional.normalize(v, dim=-1, eps=1e-8)  # (Ng,3)

    dots = dirs @ u.t()                  # (B,Ng)
    within_cone = dots >= cone_cos

    r_eff = torch.minimum(
        torch.full((dirs.shape[0], 1), radius, device=dev),
        r_wall.view(-1, 1),
    )                                    # (B,1)
    within_r = dist.view(1, -1) <= r_eff # (B,Ng)

    covered = (within_cone & within_r).any(dim=1)  # (B,)
    open_dirs = ~covered

    open_frac_global = open_dirs.float().mean().item()
    up_mask = dirs[:, 1] > 0.0
    open_frac_up = open_dirs[up_mask].float().mean().item() if up_mask.any() else open_frac_global

    return {"open_frac_global": open_frac_global, "open_frac_up": open_frac_up}


# ---------- rewritten: main classifier with safe fallback ----------

@dataclass
class IndoorOutdoorResult:
    label: str
    confidence: float
    metrics: Dict[str, float]
    def __init__(self, label, confidence, metrics) -> None:
        self.label = label
        self.confidence = confidence
        self.metrics = metrics

@torch.no_grad()
def classify_indoor_outdoor(
    model_or_path,
    device: Optional[torch.device] = None,
    *,
    # Robust AABB
    pct_bounds: Tuple[float, float] = AABB_DEFAULTS.pct_bounds,
    mul_fact: float = AABB_DEFAULTS.mul_fact,
    # Shell / strips / core
    tau_shell_frac: float = 0.03,
    tau_topbot_frac: float = 0.03,
    thr_core_frac: float = 0.12,
    # Openness
    openness_radius_frac: float = 0.35,
    openness_bins: int = 200,
    openness_cone_deg: float = 15.0,
    openness_alpha_min: float = 0.25,
    # NEW: controls exposed to CLI
    fallback_mode: str = "auto",       # one of {"auto","off","force"}
    uncertain_band: float = 0.10,      # |p_out-0.5| < band -> "uncertain"
) -> IndoorOutdoorResult:
    """
    Indoor/Outdoor classifier with a *conservative* fat-band fallback that will not
    create outdoor→indoor false positives; uncertainty band is configurable.
    """
    from gs_coresets.utils.gaussian_utils import GaussianPLY

    # ---- Load / unwrap
    if not isinstance(model_or_path, GaussianPLY):
        g = load_gaussian_ply(str(model_or_path), device=device or torch.device("cpu"))
    else:
        g = model_or_path

    xyz = g.means.detach()
    dc_rgb = _dc_rgb01(g.features_dc.detach())
    alpha = torch.sigmoid(g.opacity_logits.view(-1)).detach()

    # ---- Robust AABB & scales
    lo, hi = robust_aabb(xyz=xyz, pct_low=pct_bounds[0], pct_high=pct_bounds[1], mul_fact=mul_fact)
    center = 0.5 * (lo + hi)
    diag = float(torch.linalg.norm(hi - lo).clamp_min(1e-8).item())

    thr_shell = tau_shell_frac * diag
    thr_tb    = tau_topbot_frac * diag
    thr_core  = thr_core_frac * diag
    vertical_extent = float((hi[1] - lo[1]).item())
    v_ratio = vertical_extent / diag

    if IS_DEBUG_MODE:
        print_debug(f"AABB lo={tuple(float(x) for x in lo)}, hi={tuple(float(x) for x in hi)}, diag={diag:.5f}")
        print_debug(f"thr_shell={thr_shell:.5f}, thr_tb={thr_tb:.5f}, thr_core={thr_core:.5f}, v_ratio={v_ratio:.3f}")

    # ---- Distances to AABB planes
    dx_min = (xyz[:, 0] - lo[0]).abs(); dx_max = (hi[0] - xyz[:, 0]).abs()
    dy_min = (xyz[:, 1] - lo[1]).abs(); dy_max = (hi[1] - xyz[:, 1]).abs()
    dz_min = (xyz[:, 2] - lo[2]).abs(); dz_max = (hi[2] - xyz[:, 2]).abs()

    # helper to compute “near” masks/fractions for a given band thickness
    def _near_fracs(th_shell_abs: float, th_tb_abs: float):
        near_x_min = dx_min <= th_shell_abs; near_x_max = dx_max <= th_shell_abs
        near_y_min = dy_min <= th_shell_abs; near_y_max = dy_max <= th_shell_abs
        near_z_min = dz_min <= th_shell_abs; near_z_max = dz_max <= th_shell_abs

        near_x = near_x_min | near_x_max
        near_y = near_y_min | near_y_max
        near_z = near_z_min | near_z_max

        near_any = near_x | near_y | near_z
        near_two = (near_x.int() + near_y.int() + near_z.int()) >= 2

        frac_shell_any = near_any.float().mean().item()
        frac_shell_two = near_two.float().mean().item()

        frac_x = near_x.float().mean().item()
        frac_y = near_y.float().mean().item()
        frac_z = near_z.float().mean().item()

        frac_x_min = near_x_min.float().mean().item()
        frac_x_max = near_x_max.float().mean().item()
        frac_z_min = near_z_min.float().mean().item()
        frac_z_max = near_z_max.float().mean().item()
        paired_x = min(frac_x_min, frac_x_max)
        paired_z = min(frac_z_min, frac_z_max)

        # vertical strips
        top_strip    = dy_max <= th_tb_abs
        bottom_strip = dy_min <= th_tb_abs
        frac_top     = top_strip.float().mean().item()
        frac_bottom  = bottom_strip.float().mean().item()

        return {
            "near_any": near_any, "near_x": near_x, "near_z": near_z,
            "top_strip": top_strip, "bottom_strip": bottom_strip,
            "frac_shell_any": frac_shell_any, "frac_shell_two": frac_shell_two,
            "frac_x": frac_x, "frac_y": frac_y, "frac_z": frac_z,
            "paired_x": paired_x, "paired_z": paired_z,
            "frac_top": frac_top, "frac_bottom": frac_bottom,
            "frac_x_min": frac_x_min, "frac_x_max": frac_x_max,
            "frac_z_min": frac_z_min, "frac_z_max": frac_z_max,
        }

    # Initial shell/top/bottom read
    fr = _near_fracs(thr_shell, thr_tb)
    near_any, near_x, near_z = fr["near_any"], fr["near_x"], fr["near_z"]
    top_strip, bottom_strip  = fr["top_strip"], fr["bottom_strip"]
    frac_shell_any = fr["frac_shell_any"]; frac_shell_two = fr["frac_shell_two"]
    frac_x, frac_y, frac_z = fr["frac_x"], fr["frac_y"], fr["frac_z"]
    paired_x, paired_z = fr["paired_x"], fr["paired_z"]
    frac_top, frac_bottom = fr["frac_top"], fr["frac_bottom"]

    # Top color/opacity cues
    if top_strip.any():
        top_rgb = dc_rgb[top_strip]
        blue_top_score = (top_rgb[:, 2] - torch.maximum(top_rgb[:, 0], top_rgb[:, 1])).clamp_min(0.0).mean().item()
        top_alpha = alpha[top_strip].mean().item()
    else:
        blue_top_score = 0.0
        top_alpha = 1.0

    # Core occupancy & enclosure
    in_core = (dx_min > thr_core) & (dx_max > thr_core) & (dy_min > thr_core) & (dy_max > thr_core) & (dz_min > thr_core) & (dz_max > thr_core)
    frac_core = in_core.float().mean().item()
    enclosure_gap = max(0.0, frac_shell_any - frac_core)

    # Directional openness (choose a robust center)
    if in_core.any():
        center_for_open = xyz[in_core].median(dim=0).values
    elif (~near_any).any():
        center_for_open = xyz[~near_any].median(dim=0).values
    else:
        center_for_open = center

    open_metrics = _directional_openness(
        xyz, alpha, center_for_open, radius=openness_radius_frac * diag,
        lo=lo, hi=hi, num_bins=openness_bins,
        cone_cos=math.cos(math.radians(openness_cone_deg)),
        alpha_min=openness_alpha_min
    )
    open_frac_global = float(open_metrics["open_frac_global"])
    open_frac_up     = float(open_metrics["open_frac_up"])

    # ---- Conservative fallback controller
    OUTDOOR_LOCK = (open_frac_up >= 0.45) or (open_frac_global >= 0.45)

    def _needs_fallback_conservative() -> bool:
        return (
            (frac_shell_any < 0.40) and
            ((paired_x + paired_z) < 0.05) and
            ((frac_x + frac_z)   < 0.18) and
            (open_frac_up        < 0.25) and
            (open_frac_global    < 0.30) and
            (blue_top_score      < 0.06) and
            (top_alpha           >= 0.15) and
            ((enclosure_gap >= 0.25) or (frac_core >= 0.04) or (v_ratio <= 0.45))
        )

    want_fallback = (fallback_mode == "force") or ((fallback_mode == "auto") and (not OUTDOOR_LOCK) and _needs_fallback_conservative())
    fallback_used = False
    if want_fallback:
        for mul in (1.5, 2.0):
            thr_shell_try = min(thr_shell * mul, 0.12 * diag)
            thr_tb_try    = min(thr_tb    * mul, 0.12 * diag)
            fr_try = _near_fracs(thr_shell_try, thr_tb_try)
            cond_recover = (
                (fr_try["frac_shell_any"] >= 0.40) or
                ((fr_try["paired_x"] + fr_try["paired_z"]) >= 0.05) or
                ((fr_try["frac_x"] + fr_try["frac_z"]) >= 0.18)
            )
            if cond_recover:
                thr_shell, thr_tb = thr_shell_try, thr_tb_try
                # adopt recovered masks/fractions
                near_any, near_x, near_z = fr_try["near_any"], fr_try["near_x"], fr_try["near_z"]
                top_strip, bottom_strip  = fr_try["top_strip"], fr_try["bottom_strip"]
                frac_shell_any, frac_shell_two = fr_try["frac_shell_any"], fr_try["frac_shell_two"]
                frac_x, frac_y, frac_z = fr_try["frac_x"], fr_try["frac_y"], fr_try["frac_z"]
                paired_x, paired_z = fr_try["paired_x"], fr_try["paired_z"]
                frac_top, frac_bottom = fr_try["frac_top"], fr_try["frac_bottom"]
                fallback_used = True
                break

    # ---- Scoring
    OUT_W = {
        "bottom_vs_top": 1.4,
        "blue_top":      0.9,
        "low_top_alpha": 0.7,
        "weak_shell":    0.5,
        "weak_edges":    0.4,
        "open_global":   0.8,
        "open_up":       1.1,
    }
    IN_W = {
        "shell_any":     0.9,
        "shell_edges":   0.7,
        "paired_x":      1.0,
        "paired_z":      1.0,
        "xz_planes":     0.8,
        "top_plus_bot":  0.5,
        "enclosure_gap": 1.0,
        "sparse_top_walls_bonus": 0.2,
    }

    walls_evidence = (paired_x + paired_z) >= 0.05 or (frac_x + frac_z) >= 0.18
    bvt_gain = min(1.0, max(0.0, v_ratio / 0.25))

    outdoor_score  = 0.0
    outdoor_score += OUT_W["bottom_vs_top"] * bvt_gain * max(0.0, frac_bottom - frac_top)

    enough_top = frac_top >= 0.08
    if enough_top:
        outdoor_score += OUT_W["blue_top"]      * max(0.0, blue_top_score)
        outdoor_score += OUT_W["low_top_alpha"] * max(0.0, 0.35 - top_alpha)

    outdoor_score += OUT_W["weak_shell"] * max(0.0, 0.25 - frac_shell_any)
    outdoor_score += OUT_W["weak_edges"] * max(0.0, 0.12 - frac_shell_two)

    # Ceiling-hole robustness when walls exist
    if (not enough_top) and walls_evidence:
        OUT_W["open_up"]     *= 0.6
        OUT_W["open_global"] *= 0.7
        IN_W["sparse_top_walls_bonus"] = max(IN_W.get("sparse_top_walls_bonus", 0.0), 0.35)

    outdoor_score += OUT_W["open_global"] * max(0.0, open_frac_global - 0.35)
    outdoor_score += OUT_W["open_up"]     * max(0.0, open_frac_up     - 0.35)

    indoor_score  = 0.0
    indoor_score += IN_W["shell_any"]     * max(0.0, frac_shell_any - 0.28)
    indoor_score += IN_W["shell_edges"]   * max(0.0, frac_shell_two - 0.07)
    indoor_score += IN_W["paired_x"]      * paired_x
    indoor_score += IN_W["paired_z"]      * paired_z
    indoor_score += IN_W["xz_planes"]     * min(frac_x, frac_z)
    indoor_score += IN_W["top_plus_bot"]  * max(0.0, (frac_top + frac_bottom) - 0.18)
    indoor_score += IN_W["enclosure_gap"] * enclosure_gap
    if (not enough_top) and walls_evidence:
        indoor_score += IN_W["sparse_top_walls_bonus"]

    net  = outdoor_score - indoor_score
    p_out = float(torch.sigmoid(torch.tensor(2.0 * net)).item())
    p_in  = 1.0 - p_out

    # Base label
    label = "outdoor" if p_out > 0.5 else "indoor"
    confidence = max(p_out, p_in)

    # Monotonic safeguard: if OUTDOOR_LOCK was triggered, don't allow a flip to "indoor".
    if OUTDOOR_LOCK and (label == "indoor"):
        label = "outdoor" if p_out >= 0.5 else "uncertain"

    # Uncertainty band (configurable; replaces fixed ±0.10)
    if abs(p_out - 0.5) < float(uncertain_band):
        label = "uncertain"

    metrics = {
        "p_outdoor": p_out, "p_indoor": p_in, "score_net": float(net),
        "diag_scene": diag, "thr_shell_abs": float(thr_shell),
        "thr_topbot_abs": float(thr_tb), "thr_core_abs": float(thr_core),
        "v_ratio": float(v_ratio),
        "frac_shell_any": frac_shell_any, "frac_shell_two_plus": frac_shell_two,
        "frac_x_planes": frac_x, "frac_y_planes": frac_y, "frac_z_planes": frac_z,
        "frac_x_min": fr.get("frac_x_min", 0.0), "frac_x_max": fr.get("frac_x_max", 0.0),
        "frac_z_min": fr.get("frac_z_min", 0.0), "frac_z_max": fr.get("frac_z_max", 0.0),
        "paired_x": paired_x, "paired_z": paired_z,
        "frac_top": frac_top, "frac_bottom": frac_bottom,
        "blue_top_score": blue_top_score, "top_alpha_mean": top_alpha,
        "enough_top": float(enough_top), "bvt_gain": float(bvt_gain),
        "enclosure_gap": enclosure_gap, "frac_core": frac_core,
        "open_frac_global": open_frac_global, "open_frac_up": open_frac_up,
        "outdoor_score": float(outdoor_score), "indoor_score": float(indoor_score),
        "walls_evidence": float(walls_evidence),
        "fallback_used": float(1.0 if fallback_used else 0.0),
    }

    if IS_DEBUG_MODE:
        print_debug(f"[Indoor/Outdoor] OUT={outdoor_score:.3f}, IN={indoor_score:.3f}, net={net:.3f}")
        print_debug(f"[Indoor/Outdoor] openness: global={open_frac_global:.3f}, up={open_frac_up:.3f}")
        print_debug(f"[Indoor/Outdoor] shell_any={frac_shell_any:.3f}, shell2+={frac_shell_two:.3f}, paired_x={paired_x:.3f}, paired_z={paired_z:.3f}")
        print_debug(f"[Indoor/Outdoor] top={frac_top:.3f}, bottom={frac_bottom:.3f}, blue_top={blue_top_score:.3f}, top_alpha={top_alpha:.3f}, enough_top={enough_top}")
        print_debug(f"[Indoor/Outdoor] v_ratio={v_ratio:.3f}, bvt_gain={bvt_gain:.3f}, enclosure_gap={enclosure_gap:.3f}, core={frac_core:.3f}")
        print_debug(f"[Indoor/Outdoor] walls_evidence={walls_evidence} → label={label}, conf={confidence:.3f}")
        print_debug(f"[Indoor/Outdoor] fallback_used={fallback_used}")

    return IndoorOutdoorResult(label, confidence, metrics)
