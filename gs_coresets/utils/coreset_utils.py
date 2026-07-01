# coreset_utils.py
import torch
from typing import Literal, Optional, Tuple
from gs_coresets.utils.gaussian_utils import GaussianPLY, scale_colors_inplace
from gs_coresets.utils.io_utils import save_gaussian_ply
from gs_coresets.utils.debug_utils import print_debug, IS_DEBUG_MODE

def get_probabilities(sensitivities: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Turn nonnegative 'sensitivities' (N,) into a probability vector p (N,) on the same device.
    Applies temperature: <1 → peakier, >1 → flatter. Robust to all-zero input.
    """
    s = torch.nan_to_num(sensitivities, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if temperature != 1.0:
        s = s.pow(1.0 / temperature)
    if s.sum() <= 0:
        p = torch.full_like(s, 1.0 / max(1, s.numel()))
    else:
        p = s / s.sum()
        # Normalize for safety, p.sum() should be 1.
        p = p / p.sum()
    return p

def sample_with_replacement(
    p: torch.Tensor, sample_size: int, seed: Optional[int] = None
) -> torch.Tensor:
    """
    Multinomial sampling with replacement. Returns 'draws' ('sample_size',) of global gaussian ids.
    'p' must sum to 1; shape (N,). Respects device; optional seed.
    """
    if p.dim() != 1:
        raise ValueError(f"p must be 1D, got shape {tuple(p.shape)}")
    gen = None
    if seed is not None:
        # Torch versions differ on generator device; try matching p.device first, then CPU.
        try:
            gen = torch.Generator(device=p.device).manual_seed(int(seed))
        except TypeError:
            gen = torch.Generator().manual_seed(int(seed))
    return torch.multinomial(p, num_samples=sample_size, replacement=True, generator=gen)  # (m,)

def sample_without_replacement(
    p: torch.Tensor, sample_size: int, seed: Optional[int] = None
) -> torch.Tensor:
    if p.dim() != 1:
        raise ValueError(f"p must be 1D, got shape {tuple(p.shape)}")
    support = (p > 0)
    num_support = int(support.sum().item())
    if sample_size > num_support:
        raise ValueError(
            f"Cannot sample {sample_size} unique items: only {num_support} indices have non-zero probability. "
            f"Consider lowering temperature or ensuring scores>0 for at least 'sample_size' items."
        )
    gen = None
    if seed is not None:
        # Torch versions differ on generator device; try matching p.device first, then CPU.
        try:
            gen = torch.Generator(device=p.device).manual_seed(int(seed))
        except TypeError:
            gen = torch.Generator().manual_seed(int(seed))
    return torch.multinomial(p, sample_size, replacement=False, generator=gen)  # (m,)

if IS_DEBUG_MODE:
    from gs_coresets.utils.stats_utils import *

## If gaussians repicked, saves PLY with weighted SH.
def get_SH_coreset(
    model_ply: GaussianPLY,
    sensitivities: torch.Tensor,               # (N,) nonnegative (e.g., sensitivity)
    coreset_size: int,
    out_path: str,
    *,
    apply_SS_weights: bool = False,
    normalize_weights: bool = False,
    allow_duplicates: bool = False,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    sampling_mode: Literal["multinomial", "topk"] = "multinomial",
) -> Tuple[GaussianPLY, torch.Tensor, torch.Tensor]:
    """
    Sampling w/ replacement ('coreset_size' draws) or deterministic top-k selection, return a
    COMPACT PLY (unique ids only) with colors scaled by w_i = count_i / ('coreset_size' * p_i).
    Also writes to 'out_path'. Returns: (ply_sh, unique_idx, counts)
    """
    device = sensitivities.device
    p = get_probabilities(sensitivities, temperature=temperature)            # (N,)
    print_debug(f"probabilities.sum = {torch.sum(p):,} == 1")
    if sampling_mode not in ("multinomial", "topk"):
        raise ValueError(f"Unknown sampling mode '{sampling_mode}'.")

    if sampling_mode == "topk":
        if allow_duplicates:
            raise ValueError("Top-k selection is incompatible with --allow-duplicates.")
        total_gaussians = sensitivities.shape[0]
        if coreset_size > total_gaussians:
            raise ValueError(
                f"Top-k selection requested {coreset_size} items but only {total_gaussians} gaussians exist."
            )
        # Use the same normalized probabilities for ordering; equivalent to ranking raw scores.
        _, topk_idx = torch.topk(p, k=coreset_size, largest=True, sorted=False)
        # Deterministically sort indices so downstream subset ordering remains stable.
        unique_idx = torch.sort(topk_idx).values
        counts = torch.ones_like(unique_idx, dtype=torch.long, device=unique_idx.device)
    else:
        if allow_duplicates:
            draws = sample_with_replacement(p, sample_size=coreset_size, seed=seed)           # (m,)
        else:
            draws = sample_without_replacement(p, sample_size=coreset_size, seed=seed)
        # Get unique; sorted=True guarantees sorted unique_idx
        unique_idx, counts = torch.unique(draws, sorted=True, return_counts=True)  # (K,), (K,)

    # Build compact PLY of unique ids
    ply_sh = model_ply.subset(unique_idx)

    # Compute weights w_i = counts / (m * p_i) on the unique subset
    nom = counts.to(p.dtype)
    print_debug(f"unique_idx.length = {unique_idx.size(0):,}")
    print_debug(f"counts.length = {nom.size(0):,} == unique_idx.length")
    print_debug(f"counts.sum = {int(torch.sum(nom).item()) : ,}")
    if apply_SS_weights:
        denom = (coreset_size * p[unique_idx]).clamp_min(1e-12)
        weights = (nom / denom)                               # (K,)
    else:
        weights = nom                              # equals 1 if allow_duplicates=False

    ## Optional: Post-normalize weights to avoid pixel saturation
    if normalize_weights:
        weights = clamp_weights(weights, p[unique_idx])
    # Reweight colors only (leave opacity/scales/poses unchanged)
    # idx into ply_sh is [0..K-1]
    scale_colors_inplace(ply_sh, torch.arange(unique_idx.numel(), device=device), weights)
    if IS_DEBUG_MODE:
        print_debug(f"coreset.size = {coreset_size:,} == counts.sum")
        print_debug(f"probabilities.min_sampled = {torch.min(p[unique_idx]):,}")
        print_debug(f"coreset.size * probabilities.min_sampled = {coreset_size * torch.min(p[unique_idx]):,}")
        print_debug(f"(coreset.size * probabilities.min_sampled).clamp_min(1e-12) = "
                    f"{(coreset_size * torch.min(p[unique_idx])).clamp_min(1e-12) :,}")
        print_debug(f"1/(coreset_size * probabilities.min_sampled).clamp_min(1e-12)) "
                    f"{1 / (torch.min(coreset_size * p[unique_idx]).clamp_min(1e-12)) : ,}")
        print_debug(f"weights.max = {torch.max(weights):,} ("
                    f"nom.max = {torch.max(nom):,}, "
                    f"denom.min = {torch.min(denom) if 'denom' in dir() else 1:,})")
        print_debug(f"weights.min = {torch.min(weights):,} ("
                    f"nom.min = {torch.min(nom):,}, "
                    f"denom.max = {torch.max(denom) if 'denom' in dir() else 1:,}")
        print_debug(f"weights.sum = {torch.sum(weights):,}")
        if apply_SS_weights and allow_duplicates:
            print_debug(f"ply.length = {model_ply.means.size(0):,} == weights.sum")
        ## PRINT_STATS:
        p_norm = p / p.sum()
        plot_distribution_large(p, top_k=2000, steps_mass_axis=200, tail_bins=64)
        stats = plot_estimated_counts_large(p, m=400_000, top_k=2000, seed=0)

    # Save GraphDECO-compatible PLY
    save_gaussian_ply(out_path, ply_sh, binary=True)
    return ply_sh, unique_idx, counts

## If gaussians repicked, saves PLY with duplicate entries per-gaussians.
## Only valid for with-replacement sampling --> allow_duplicates=True.
def get_duplications_coreset(
    model_ply: GaussianPLY,
    sensitivities: torch.Tensor,
    coreset_size: int,
    out_path: str,
    *,
    temperature: float = 1.0,
    keep_sorted: bool = True,
    seed: Optional[int] = None,
) -> Tuple[GaussianPLY, torch.Tensor]:
    """
    Sampling w/ replacement ('coreset_size' draws), return a PLY with EXACTLY 'coreset_size' rows
    (duplicates kept), without reweighting. Also writes to 'out_path'.
    """
    p = get_probabilities(sensitivities, temperature=temperature)  # (N,)
    draws_init = sample_with_replacement(p, sample_size=coreset_size, seed=seed)
    draws = draws_init
    if keep_sorted:
        # Sort draws to keep gaussians ordered
        draws, perm = torch.sort(draws_init)
    ply_out = model_ply.subset(draws)  # duplicates preserved
    save_gaussian_ply(out_path, ply_out, binary=True)
    return ply_out, draws


#################################################################
### Optional: Post-normalize weights to avoid pixel saturation ##
#################################################################
# Use it right before scale_colors_inplace(...).
# It’s optional - the unbiased weights w_i = count_i / (m * p_i) are already correct in expectation,
# but this can tame rare “spikes” that cause visible saturations in a single sample.
# Args:
# - weights:
#       * Calculated re-weights of gaussians.
# - p:
#       * Calculated probabilities of gaussians' sampling.
# - q (quantile clip):
#       * A value in [0, 1).
#       * We compute clip_q = weights.quantile(q) and clamp every weight above that to clip_q.
#       * With the default q=0.999, only the top 0.1% largest weights are trimmed
#         (very gentle; just shaves the extreme tail).
# - max_factor (absolute cap):
#       * A hard ceiling on any single weight, relative to the “neutral” weight of 1.0
#         (for unbiased coreset weights, E_p[w]=1).
#       * With the default max_factor=4.0, no individual Gaussian is allowed to be scaled by more than 4×.
# ===>  Put simply:
#       * q controls how aggressively you trim the tail
#       * max_factor sets an absolute “never go above this” cap.
def clamp_weights(weights: torch.Tensor, p: torch.Tensor, q: float = 0.999, max_factor: float = 4.0):
    """
    Optionally limit extreme weights to reduce saturation risk.
    q: clip at q-quantile; also clip to <= max_factor * mean importance 1/p-avg.
    """
    # quantile clip
    clip_q = weights.quantile(q)
    w = torch.minimum(weights, clip_q)
    # global cap vs. mean
    mean_w = (w * p).sum()  # expectation wrt p should be ~1 in expectation; keep near it
    if mean_w > 0:
        w = torch.minimum(w, max_factor * (w / mean_w).mean() * torch.ones_like(w))
    return w
