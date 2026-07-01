import torch
import numpy as np
import matplotlib.pyplot as plt

# ---------- shared gs_coresets.utils ----------
def _sanitize_p(p: torch.Tensor) -> torch.Tensor:
    p = p.detach().flatten().float()
    s = p.sum()
    if s <= 0:
        raise ValueError("Probabilities sum to 0.")
    return p / s

@torch.inference_mode()
def topk_and_tail(p: torch.Tensor, top_k: int = 2000):
    p = _sanitize_p(p)
    N = p.numel()
    k = min(int(top_k), N)
    top_vals, top_idx = torch.topk(p, k=k, largest=True, sorted=True)
    tail_mass = float(1.0 - top_vals.sum())
    tail_count = int(N - k)
    return top_vals, top_idx, tail_mass, tail_count

def _nice_ylim(ymin: float, ymax: float, pad_frac: float = 0.03):
    if ymax == ymin:
        pad = max(1e-12, abs(ymax) * 0.05)
        return ymin - pad, ymax + pad
    pad = (ymax - ymin) * pad_frac
    return ymin - pad, ymax + pad

@torch.inference_mode()
def _subsample_topk_by_mass(top_vals: torch.Tensor, steps: int = 200):
    """
    Represent the top-k probabilities along a linear *probability-mass* x-axis.
    Produces at most `steps` points by taking values at evenly spaced mass levels.
    Returns x (cumulative mass) and y (corresponding p_i).
    """
    if top_vals.numel() == 0:
        return np.array([0.0]), np.array([0.0])
    c = top_vals.cumsum(0)
    total = float(c[-1])
    if total <= 0:
        return np.array([0.0]), np.array([0.0])
    steps = max(2, int(steps))
    grid = torch.linspace(0, total, steps, device=top_vals.device)
    idx = torch.searchsorted(c, grid, right=True).clamp(max=top_vals.numel() - 1)
    y = top_vals[idx]
    return grid.cpu().numpy(), y.cpu().numpy()

@torch.inference_mode()
def tail_prob_histogram(
    p: torch.Tensor,
    exclude_idx: torch.Tensor,
    bins: int = 64,
    log_bins: bool = True,
):
    p = _sanitize_p(p)
    N = p.numel()
    mask = torch.ones(N, dtype=torch.bool, device=p.device)
    mask[exclude_idx] = False
    tail = p[mask]

    zero_ct = int((tail == 0).sum())
    tail_nz = tail[tail > 0]
    if tail_nz.numel() == 0:
        counts = torch.zeros(bins, dtype=torch.int64)
        edges = torch.linspace(1e-12, 1.0, bins + 1)
        return zero_ct, counts.cpu(), edges.cpu()

    if log_bins:
        lo = torch.log10(tail_nz.min())
        hi = torch.log10(tail_nz.max())
        if torch.isclose(lo, hi):
            hi = lo + 1e-6
        edges_log = torch.linspace(lo, hi, bins + 1, device=p.device)
        b = torch.bucketize(torch.log10(tail_nz), edges_log) - 1
        b = b.clamp(0, bins - 1)
        counts = torch.bincount(b, minlength=bins)
        edges = 10 ** edges_log
    else:
        edges, counts = torch.histogram(tail_nz, bins=bins)

    return zero_ct, counts.cpu(), edges.cpu()

# ---------- 1) Distribution (x = probability mass; y = p_i; + bars+tail; + tail hist) ----------
@torch.inference_mode()
def plot_distribution_large(
    p: torch.Tensor,
    top_k: int = 2000,
    tail_bins: int = 64,
    steps_mass_axis: int = 200,  # controls the "jumps" on x
    show_tail_hist: bool = True,
):
    p = _sanitize_p(p)
    top_vals, top_idx, tail_mass, tail_count = topk_and_tail(p, top_k=top_k)
    k = top_vals.numel()

    # Plot 1: top-k profile over linear probability-mass x-axis (subsampled)
    x_mass, y_vals = _subsample_topk_by_mass(top_vals, steps=steps_mass_axis)
    ymin, ymax = float(np.min(y_vals)), float(np.max(y_vals))
    plt.figure()
    plt.step(x_mass, y_vals, where="post")
    plt.xlabel("cumulative probability mass over top-k")
    plt.ylabel("p_i")
    plt.ylim(*_nice_ylim(ymin, ymax))
    plt.xlim(0.0, float(x_mass[-1]) if len(x_mass) else 1.0)
    plt.title(f"Top-{k} probabilities over probability-mass axis")
    plt.tight_layout()
    plt.show()

    # Plot 2: bars for top-k + one bar for tail mass (if any)
    top_np = top_vals.cpu().numpy().astype(float)
    if tail_count > 0:
        bars = np.concatenate([top_np, np.array([tail_mass], dtype=float)])
        labels = [f"{i}" for i in range(1, k + 1)] + [f"tail ({tail_count})"]
    else:
        bars = top_np
        labels = [f"{i}" for i in range(1, k + 1)]
    plt.figure()
    x = np.arange(len(bars))
    plt.bar(x, bars.astype(float))
    ymin2, ymax2 = float(bars.min()), float(bars.max())
    plt.ylim(*_nice_ylim(ymin2, ymax2))
    plt.xlim(-0.5, len(bars) - 0.5)
    # Keep ticks sparse
    if len(bars) > 10:
        tick_positions = [0, min(4, len(bars) - 1), len(bars) - 1]
    else:
        tick_positions = list(range(len(bars)))
    plt.xticks(tick_positions, [labels[i] for i in tick_positions])
    plt.ylabel("probability")
    plt.title(f"Top-{k} bars + tail mass")
    plt.tight_layout()
    plt.show()

    # Plot 3: tail histogram (clip zero-count bins)
    if show_tail_hist and tail_count > 0:
        zero_ct, counts, edges = tail_prob_histogram(p, top_idx, bins=tail_bins, log_bins=True)
        centers = np.sqrt(edges[:-1].numpy() * edges[1:].numpy())
        counts = counts.numpy()
        mask = counts > 0
        if mask.any():
            plt.figure()
            plt.plot(centers[mask], counts[mask], marker="o", linewidth=1)
            plt.xscale("log")
            plt.xlabel("p in tail (log scale)")
            plt.ylabel("count of items")
            plt.title(f"Tail probability histogram (zeros: {zero_ct}, shown bins: {int(mask.sum())})")
            plt.tight_layout()
            plt.show()

# ---------- 2) m draws: bars+tail + calibration scatter ----------
@torch.inference_mode()
def plot_estimated_counts_large(
    p: torch.Tensor,
    m: int,
    top_k: int = 2000,
    seed: int | None = None,
    show_expectation: bool = True,
    show_ci: bool = True,
):
    if m <= 0:
        raise ValueError("m must be positive.")
    p = _sanitize_p(p)
    dev = p.device

    top_vals, top_idx, tail_mass, tail_count = topk_and_tail(p, top_k=top_k)
    k = top_vals.numel()

    # Draw
    g = None
    if seed is not None:
        g = torch.Generator(device=dev).manual_seed(seed)
    draws = torch.multinomial(p, num_samples=int(m), replacement=True, generator=g)

    # Count only top-k via index map; rest = tail
    idx_map = torch.full((p.numel(),), -1, device=dev, dtype=torch.int64)
    idx_map[top_idx] = torch.arange(k, device=dev, dtype=torch.int64)
    mapped = idx_map[draws]
    valid = mapped >= 0
    counts_topk = torch.bincount(mapped[valid], minlength=k)
    tail_draws = int(m - int(counts_topk.sum().item()))

    # Plot 2: bars (top-k + tail)
    bars = counts_topk.cpu().numpy().astype(float)
    if tail_count > 0:
        bars = np.concatenate([bars, np.array([tail_draws], dtype=float)])
    plt.figure()
    x = np.arange(len(bars))
    plt.bar(x, bars, alpha=0.7, label="sample counts")
    ymin3, ymax3 = float(bars.min()), float(bars.max())
    plt.ylim(*_nice_ylim(ymin3, ymax3))
    plt.xlim(-0.5, len(bars) - 0.5)
    plt.ylabel("count")

    if show_expectation:
        exp_topk = (m * top_vals).cpu().numpy().astype(float)
        if k:
            plt.plot(np.arange(k), exp_topk, linewidth=2, label="E[count] = m·p_i")
        if tail_count > 0:
            exp_tail = float(m * tail_mass)
            plt.plot([len(bars) - 1], [exp_tail], marker="o", label="E[tail]")

    if show_ci and k:
        sd_topk = torch.sqrt(m * top_vals * (1 - top_vals)).cpu().numpy().astype(float)
        lo = (m * top_vals).cpu().numpy().astype(float) - 1.96 * sd_topk
        hi = (m * top_vals).cpu().numpy().astype(float) + 1.96 * sd_topk
        plt.fill_between(np.arange(k), lo, hi, alpha=0.2, label="≈95% CI (top-k)")
        if tail_count > 0 and 0.0 < tail_mass < 1.0:
            sd_tail = float(np.sqrt(m * tail_mass * (1 - tail_mass)))
            x_tail = len(bars) - 1
            plt.vlines(x_tail, exp_tail - 1.96 * sd_tail, exp_tail + 1.96 * sd_tail, linestyles="dashed")

    # sparse ticks: first, maybe middle, tail
    if tail_count > 0:
        xt = [0, len(bars) - 1] if len(bars) > 1 else [0]
        xl = ["rank 1", f"tail ({tail_count})"]
        plt.xticks(xt, xl)
    else:
        if len(bars) > 1:
            plt.xticks([0, len(bars) - 1], ["rank 1", f"rank {k}"])
        else:
            plt.xticks([0], ["rank 1"])

    plt.title(f"m = {m} iid draws; top-{k} shown")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 4: calibration scatter (observed vs expected for top-k)
    if k:
        exp = (m * top_vals).cpu().numpy().astype(float)
        obs = counts_topk.cpu().numpy().astype(float)
        plt.figure()
        plt.scatter(exp, obs, s=10, alpha=0.6)
        lim_lo = float(min(exp.min(), obs.min()))
        lim_hi = float(max(exp.max(), obs.max()))
        lo, hi = _nice_ylim(lim_lo, lim_hi)
        lo = max(0.0, lo)
        plt.plot([lo, hi], [lo, hi], linestyle="--")  # y = x
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel("expected count (m·p_i)")
        plt.ylabel("observed count")
        plt.title("Calibration: observed vs expected (top-k)")
        plt.tight_layout()
        plt.show()

    return {
        "top_idx": top_idx,               # (k,)
        "top_p": top_vals,                # (k,)
        "counts_topk": counts_topk,       # (k,)
        "tail_draws": tail_draws,
        "tail_mass": tail_mass,
        "tail_count": tail_count,
    }
