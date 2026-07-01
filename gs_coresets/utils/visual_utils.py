from __future__ import annotations

import csv
import json
import math
import re
import textwrap
from pathlib import Path, PurePath
from collections import Counter
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

import default_paths
from gs_coresets.utils.io_utils import ensure_dir_for


_METRIC_PRIORITIES = ("psnr", "ssim", "lpips")
_MAX_CSV_FIG_WIDTH = 36.0  # inches at dpi=150 → 5400px, well below Matplotlib limits
_TICK_WRAP_WIDTH = 28
_METRIC_TITLES = {
    "psnr": "PSNR ↑",
    "ssim": "SSIM ↑",
    "lpips": "LPIPS ↓",
}
_PROTECTED_PNG_STEMS = {
    "metrics_per_model_summary_pre",
    "metrics_per_model_summary_post",
    "metrics_per_view_best_best_pre",
    "metrics_per_view_best_best_post",
    "metrics_per_view_best_summary_pre",
    "metrics_per_view_best_summary_post",
    "metrics_per_view_worst_best_pre",
    "metrics_per_view_worst_best_post",
    "metrics_per_view_worst_summary_pre",
    "metrics_per_view_worst_summary_post",
}


def resolve_parent_output_dir(path: str | Path) -> Path:
    """
    Ascend from `path` to locate the nearest directory that owns a `metrics/` folder.

    Args:
        path: File or directory within an experiment output tree.

    Returns:
        Path pointing to the parent output directory that contains the `metrics` subdirectory.

    Raises:
        FileNotFoundError: if no parent with a `metrics/` directory is found.
    """
    p = Path(path).expanduser()
    candidate = p if p.is_dir() else p.parent
    candidate = candidate.resolve()
    search_chain = (candidate,) + tuple(candidate.parents)
    for current in search_chain:
        metrics_dir = current / "metrics"
        if metrics_dir.is_dir():
            return current
        if current.name == "metrics":
            return current.parent
    raise FileNotFoundError(
        f"Could not find a parent output directory with a 'metrics' folder for: {path}"
    )


def _to_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
    else:
        text = str(value)
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _series_sort_key(name: str) -> Tuple[int, int, str]:
    lower = name.lower()
    base_rank = next(
        (idx for idx, metric in enumerate(_METRIC_PRIORITIES) if metric in lower),
        len(_METRIC_PRIORITIES),
    )
    prefix_rank = 0
    if lower.startswith("avg_"):
        prefix_rank = 1
    if "rank" in lower:
        prefix_rank = 2
    return (base_rank, prefix_rank, lower)


def _humanize_series_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _wrap_label_text(label: str) -> str:
    segments = []
    for segment in label.splitlines() or [""]:
        if not segment:
            segments.append("")
            continue
        wrapped = textwrap.wrap(segment, width=_TICK_WRAP_WIDTH) or [segment]
        segments.extend(wrapped)
    return "\n".join(segments)


def _is_baseline(row: Dict[str, str]) -> bool:
    granularity = (row.get("granularity") or "").strip().lower()
    return granularity == "baseline"


def _preserve_three_panel_layout(stem: str) -> bool:
    return stem in _PROTECTED_PNG_STEMS


def _compose_label(row: Dict[str, str], index: int) -> str:
    dataset = (row.get("dataset") or "").strip()
    category = (row.get("category") or "").strip()
    level = (row.get("level") or "").strip()
    model = (row.get("model") or "").strip()
    granularity = (row.get("granularity") or "").strip()
    coreset_k = (row.get("coreset_k") or "").strip()
    finetuned_flag = (row.get("finetuned") or "").strip()

    top_parts: List[str] = []
    if dataset:
        top_parts.append(dataset)
    if category:
        top_parts.append(category)
    if level and level not in top_parts:
        top_parts.append(level)

    bottom_parts: List[str] = []
    if model:
        bottom_parts.append(model)
    if granularity and granularity not in bottom_parts:
        bottom_parts.append(granularity)
    if coreset_k:
        bottom_parts.append(f"k={coreset_k}")
    if finetuned_flag in {"1", "true", "True"}:
        bottom_parts.append("finetuned")

    if top_parts and bottom_parts:
        return f"{' · '.join(top_parts)}\n{' · '.join(bottom_parts)}"
    if top_parts:
        return " · ".join(top_parts)
    if bottom_parts:
        return " · ".join(bottom_parts)
    return f"row-{index + 1}"


def _compose_tick_label(row: Dict[str, str], index: int) -> str:
    """Tick labels only show granularity; fall back to row index when missing."""
    granularity = (row.get("granularity") or "").strip()
    base = granularity if granularity else f"row-{index + 1}"
    if len(base) > 18:
        base = base[:15] + "…"
    return base


def _collect_numeric_columns(rows: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
    numeric: Dict[str, np.ndarray] = {}
    if not rows:
        return numeric
    keys = rows[0].keys()
    for key in keys:
        values: List[float] = []
        has_value = False
        for row in rows:
            num = _to_float(row.get(key))
            if num is None:
                values.append(np.nan)
            else:
                values.append(num)
                has_value = True
        if has_value:
            numeric[key] = np.array(values, dtype=float)
    return numeric


def _row_sort_key(row: Dict[str, str]) -> Tuple[float, float, float, float, float]:
    """Lower tuple is better."""
    rank_ovr = _to_float(row.get("rank_ovr"))
    rank_psnr = _to_float(row.get("rank_psnr"))
    rank_ssim = _to_float(row.get("rank_ssim"))
    rank_lpips = _to_float(row.get("rank_lpips"))

    var_rank_ovr = _to_float(row.get("rank_var_ovr"))
    var_rank_psnr = _to_float(row.get("rank_var_psnr"))
    var_rank_ssim = _to_float(row.get("rank_var_ssim"))
    var_rank_lpips = _to_float(row.get("rank_var_lpips"))

    if rank_ovr is None:
        rank_ovr = var_rank_ovr
    if rank_psnr is None:
        rank_psnr = var_rank_psnr
    if rank_ssim is None:
        rank_ssim = var_rank_ssim
    if rank_lpips is None:
        rank_lpips = var_rank_lpips

    psnr = _to_float(row.get("psnr"))
    psnr_is_var = False
    if psnr is None:
        psnr = _to_float(row.get("var_PSNR"))
        psnr_is_var = psnr is not None

    ssim = _to_float(row.get("ssim"))
    ssim_is_var = False
    if ssim is None:
        ssim = _to_float(row.get("var_SSIM"))
        ssim_is_var = ssim is not None

    lpips = _to_float(row.get("lpips"))
    if lpips is None:
        lpips = _to_float(row.get("var_LPIPS"))

    def safe(val: float | None, default: float) -> float:
        return val if val is not None else default

    avg_rank = None
    ranks = [v for v in (rank_psnr, rank_ssim, rank_lpips) if v is not None]
    if ranks:
        avg_rank = sum(ranks) / len(ranks)

    primary = safe(rank_ovr, safe(avg_rank, float("inf")))
    secondary = safe(rank_psnr, float("inf"))
    tertiary = safe(rank_ssim, float("inf"))
    quaternary = safe(rank_lpips, float("inf"))

    def make_key(value: float | None, prefer_high: bool) -> float:
        if value is None:
            return float("inf")
        return -value if prefer_high else value

    psnr_key = make_key(psnr, prefer_high=not psnr_is_var)
    ssim_key = make_key(ssim, prefer_high=not ssim_is_var)
    lpips_key = make_key(lpips, prefer_high=False)

    return (primary, secondary, tertiary, quaternary, psnr_key + ssim_key + lpips_key)


def _select_top_rows(
    rows: List[Dict[str, str]],
    *,
    max_items: int = 3,
    include_baseline: bool = False,
) -> List[Dict[str, str]]:
    if include_baseline:
        filtered = list(rows)
    else:
        filtered = [row for row in rows if not _is_baseline(row)]
    filtered.sort(key=_row_sort_key)
    return filtered[:max_items]


def _metric_value(row: Dict[str, str], metric: str) -> float | None:
    metric_lower = metric.lower()
    candidates = (
            metric_lower,
            f"avg_{metric_lower}",
            f"mean_{metric_lower}",
            f"{metric_lower}_avg",
    )

    for key, value in row.items():
        if not key:
            continue
        k = key.strip().lower()
        if k in candidates:
            return _to_float(value)

    return None


def _topk_rows_by_metric(
    rows: List[Dict[str, str]],
    metric: str,
    *,
    max_items: int | None = 3,
) -> List[Tuple[float, Dict[str, str]]]:
    prefer_high = metric.lower() != "lpips"
    scored: List[Tuple[float, int, Dict[str, str]]] = []
    for idx, row in enumerate(rows):
        if _is_baseline(row):
            continue
        value = _metric_value(row, metric)
        if value is None or math.isnan(value):
            continue
        scored.append((value, idx, row))
    if not scored:
        return []
    scored.sort(
        key=lambda item: (-item[0], item[1])
        if prefer_high
        else (item[0], item[1])
    )
    if max_items is None or max_items <= 0:
        selected = scored
    else:
        selected = scored[:max_items]
    return [(value, row) for value, _, row in selected]


def _baseline_row(rows: List[Dict[str, str]]) -> Dict[str, str] | None:
    for row in rows:
        if _is_baseline(row):
            return row
    return None


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _granularity_from_row(row: Dict[str, str]) -> str:
    granularity = (row.get("granularity") or "").strip()
    if granularity:
        return granularity
    model = (row.get("model") or "").strip()
    if model:
        model_norm = model.replace("\\", "/")
        basename = model_norm.rsplit("/", 1)[-1]
        return basename or model
    return "Unknown"


def _plot_metric_topk(
    ax: plt.Axes,
    entries: List[Tuple[float, Dict[str, str]]],
    metric: str,
) -> None:
    if not entries:
        ax.axis("off")
        return

    values = np.array([value for value, _ in entries], dtype=float)
    x = np.arange(len(entries))

    metric_lower = metric.lower()
    prefer_high = metric_lower != "lpips"
    baseline_indices = [
        idx for idx, (_, row) in enumerate(entries) if _is_baseline(row)
    ]
    candidate_indices = [
        idx for idx in range(len(entries)) if idx not in baseline_indices
    ]
    best_idx = None
    if candidate_indices:
        candidate_values = values[candidate_indices]
        selector = np.argmax if prefer_high else np.argmin
        best_idx = candidate_indices[selector(candidate_values)]
    colors: List[str] = []
    for idx in range(len(entries)):
        if idx in baseline_indices:
            colors.append("0.6")  # gray for baseline
        elif best_idx is not None and idx == best_idx:
            colors.append("tab:orange")
        else:
            colors.append("tab:blue")

    ax.bar(x, values, width=0.65, color=colors, edgecolor="black", linewidth=0.6)
    for idx, val in enumerate(values):
        ax.text(x[idx], val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    tick_labels = [
        _compose_tick_label(row, idx) for idx, (_, row) in enumerate(entries)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Y-limit handled by caller to avoid fixed bounds when comparing counts or summaries.


def _plot_granularity_counts(
    ax: plt.Axes,
    counts: Counter[str],
    metric: str,
    *,
    show_ylabel: bool,
    max_items: int | None = 3,
) -> None:
    title = _METRIC_TITLES.get(metric, metric.upper())
    if not counts:
        ax.set_title(title, fontsize=10)
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.axis("off")
        return

    if max_items is None or max_items <= 0:
        ordered_items = counts.most_common()
    else:
        ordered_items = counts.most_common(max_items)

    labels = [label for label, _ in ordered_items]
    values = np.array([count for _, count in ordered_items], dtype=int)
    x = np.arange(len(labels))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(values))]

    ax.bar(x, values, width=0.6, color=colors, edgecolor="black", linewidth=0.6)
    for idx, count in enumerate(values):
        ax.text(
            x[idx],
            count,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    tick_labels = [_wrap_label_text(label) for label in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    if show_ylabel:
        ax.set_ylabel("Number of datasets")
    else:
        ax.set_ylabel("")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2 if values.size else 1.0)

    for rank, (label, count) in enumerate(ordered_items, start=1):
        ax.text(
            0.98,
            0.92 - (rank - 1) * 0.14,
            f"{rank} — {label}: {int(count)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )


def _plot_average_metric(
    ax: plt.Axes,
    metric: str,
    baseline_value: float | None,
    top_items: List[Tuple[str, float]],
    *,
    show_ylabel: bool,
) -> bool:
    title = _METRIC_TITLES.get(metric, metric.upper())
    entries: List[Tuple[str, float, str]] = []

    if baseline_value is not None and not math.isnan(baseline_value):
        entries.append(("Baseline", baseline_value, "baseline"))

    for idx, (granularity, value) in enumerate(top_items):
        role = "top1" if idx == 0 else "top"
        entries.append((granularity, value, role))

    if not entries:
        ax.set_title(title, fontsize=10)
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.axis("off")
        return False

    labels = [label for label, _, _ in entries]
    bars = np.array([value for _, value, _ in entries], dtype=float)
    colors: List[str] = []
    for _, _, role in entries:
        if role == "baseline":
            colors.append("0.6")
        elif role == "top1":
            colors.append("tab:orange")
        else:
            colors.append("tab:blue")

    x = np.arange(len(bars))
    ax.bar(x, bars, width=0.65, color=colors, edgecolor="black", linewidth=0.6)
    for idx, value in enumerate(bars):
        ax.text(x[idx], value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    tick_labels = [_wrap_label_text(label) for label in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=8)
    if show_ylabel:
        ax.set_ylabel("Average value")
    else:
        ax.set_ylabel("")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    return True


def _plot_single_series(
    ax: plt.Axes,
    labels: List[str],
    tick_labels: List[str],
    values: np.ndarray,
    series_name: str,
    *,
    apply_metric_ylim: bool,
) -> None:
    x = np.arange(len(labels))
    finite_mask = np.isfinite(values)
    heights = np.where(finite_mask, values, 0.0)

    lower = series_name.lower()
    prefer_high = "lpips" not in lower and "rank" not in lower and "var" not in lower
    valid_indices = np.where(finite_mask)[0]
    bar_colors: List[str] = []
    highlight_idx = None
    if valid_indices.size > 0:
        if prefer_high:
            highlight_idx = valid_indices[np.argmax(values[valid_indices])]
        else:
            highlight_idx = valid_indices[np.argmin(values[valid_indices])]
    for idx in range(len(labels)):
        if not finite_mask[idx]:
            bar_colors.append("0.85")
        elif highlight_idx is not None and idx == highlight_idx:
            bar_colors.append("tab:orange")
        else:
            bar_colors.append("tab:blue")

    ax.bar(x, heights, width=0.65, color=bar_colors, edgecolor="black", linewidth=0.6)
    for idx in valid_indices:
        ax.text(
            x[idx],
            heights[idx],
            f"{heights[idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    if len(labels) > 200:
        ax.tick_params(axis="x", labelsize=4)
    elif len(labels) > 60:
        ax.tick_params(axis="x", labelsize=6)
    else:
        ax.tick_params(axis="x", labelsize=8)
    ax.set_title(_humanize_series_name(series_name), fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    if apply_metric_ylim:
        metric = next(
            (m.upper() for m in _METRIC_PRIORITIES if m in lower),
            None,
        )
        if metric:
            ylim = _metric_ylim(metric)
            if ylim:
                ax.set_ylim(*ylim)
    if "rank" in lower:
        ax.set_ylabel("Rank (lower is better)")
    elif "lpips" in lower:
        ax.set_ylabel("LPIPS (lower is better)")
    else:
        ax.set_ylabel("Score (higher is better)")


def _stage_from_filename(csv_path: Path) -> str:
    name = csv_path.stem
    match = re.search(r"_(pre|post)$", name)
    if match:
        return match.group(1)
    raise ValueError(
        f"CSV filename does not encode stage suffix '_pre' or '_post': {csv_path.name}"
    )


def _stage_label(stage: str) -> str:
    return "Post-Finetune" if stage == "post" else "Pre-Finetune"


def _variance_kind_from_stem(stem: str) -> str:
    lower = stem.lower()
    match = re.search(r"metrics_per_model_variance_(min|max|mean)(?:_|$)", lower)
    if match:
        return match.group(1)
    raise ValueError(f"Unable to extract variance kind from filename stem: {stem}")


def _collect_metric_variance_entries(
    rows: List[Dict[str, str]],
    metric: str,
    variance_kind: str,
    *,
    max_items: int | None = 3,
) -> Tuple[Tuple[str, str, float] | None, List[Tuple[str, str, float]]]:
    var_key = f"{variance_kind}_var_{metric}"
    dataset_key = f"{metric}_dataset"
    rank_key = f"rank_{variance_kind}_var_{metric}"

    baseline_entry: Tuple[str, str, float] | None = None
    candidates: List[Tuple[float, float, str, str]] = []
    for row in rows:
        value = _to_float(row.get(var_key))
        if value is None or math.isnan(value):
            continue
        dataset = (row.get(dataset_key) or "").strip()
        granularity = _granularity_from_row(row)
        if _is_baseline(row):
            baseline_entry = ("Baseline", dataset, value)
            continue
        rank_value = _to_float(row.get(rank_key))
        candidates.append(
            (
                rank_value if rank_value is not None else float("inf"),
                value,
                granularity,
                dataset,
            )
        )

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    if max_items is None or max_items <= 0:
        selected = candidates
    else:
        selected = candidates[:max_items]
    top_entries = [
        (granularity, dataset, value)
        for _, value, granularity, dataset in selected
    ]
    return baseline_entry, top_entries


def _format_variance_label(
    granularity: str,
    dataset: str,
    *,
    is_baseline: bool,
) -> str:
    name = "Baseline" if is_baseline or granularity.lower() == "baseline" else granularity
    label = name.strip() if name else "Unknown"
    # Keep only the leaf (e.g., "tandt_db/db/playroom" -> "playroom")
    short_ds = PurePath(dataset.replace("\\", "/")).name

    return f"{label}\n{short_ds}" if short_ds else label


def _plot_metric_variance_panel(
    ax: plt.Axes,
    metric: str,
    variance_kind: str,
    baseline_entry: Tuple[str, str, float] | None,
    top_entries: List[Tuple[str, str, float]],
    *,
    show_ylabel: bool,
) -> bool:
    entries: List[Tuple[str, str, float, str]] = []
    if baseline_entry is not None:
        granularity, dataset, value = baseline_entry
        entries.append(("baseline", granularity, value, dataset))

    for idx, (granularity, dataset, value) in enumerate(top_entries):
        entries.append(("top1" if idx == 0 else "top", granularity, value, dataset))

    if not entries:
        ax.axis("off")
        return False

    values = np.array([value for _, _, value, _ in entries], dtype=float)
    x = np.arange(len(entries))

    colors: List[str] = []
    labels: List[str] = []
    for role, granularity, value, dataset in entries:
        if role == "baseline":
            colors.append("0.6")
        elif role == "top1":
            colors.append("tab:orange")
        else:
            colors.append("tab:blue")
        labels.append(
            _format_variance_label(
                granularity,
                dataset,
                is_baseline=(role == "baseline"),
            )
        )

    ax.bar(x, values, width=0.6, color=colors, edgecolor="black", linewidth=0.6)
    for idx, value in enumerate(values):
        ax.text(
            x[idx],
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    wrapped_labels = [_wrap_label_text(label) for label in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    metric_title = _METRIC_TITLES.get(metric, metric.upper())
    ax.set_title(f"{metric_title} variance ({variance_kind})", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Variance")
    else:
        ax.set_ylabel("")
    return True


def _variance_sort_key(row: Dict[str, str]) -> Tuple[float, float, float, float, float, float, str]:
    def fetch(key: str) -> float:
        val = _to_float(row.get(key))
        return val if val is not None else float("inf")

    return (
        fetch("rank_var_ovr"),
        fetch("rank_var_psnr"),
        fetch("rank_var_ssim"),
        fetch("rank_var_lpips"),
        fetch("var_PSNR"),
        fetch("var_SSIM"),
        fetch("var_LPIPS"),
        (row.get("model") or ""),
    )


def create_metrics_csv_visualizations(path: str | Path) -> List[str]:
    """
    Locate `metrics/metrics_per_*.csv` relative to `path`, generate summary plots, and save PNGs.

    Args:
        path: A file or directory within an experiment output tree.

    Returns:
        List of saved PNG file paths.
    """
    output_dir = resolve_parent_output_dir(path)
    metrics_dir = output_dir / "metrics"
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"No metrics directory present under: {output_dir}")

    csv_files = sorted(metrics_dir.glob("metrics_per_*.csv"))
    variance_files = sorted(metrics_dir.glob("metrics_per_model_variance_*.csv"))
    if not csv_files and not variance_files:
        return []

    plot_dir = metrics_dir / "plots"
    ensure_dir_for(plot_dir)

    saved: List[str] = []
    for csv_path in csv_files:
        try:
            stage = _stage_from_filename(csv_path)
        except ValueError as exc:
            warn = f"[visual_utils] Skipping CSV without stage suffix: {csv_path.name} ({exc})"
            print(warn)
            continue

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                for row in reader
            ]

        if not rows:
            continue

        stem = csv_path.stem
        stem_lower = stem.lower()
        preserve_three_panel = _preserve_three_panel_layout(stem)
        if stem_lower.startswith("metrics_per_model_variance_"):
            # Handled by the variance branch
            continue
        if stem_lower.startswith("metrics_per_model_average"):
            baseline_metrics: Dict[str, List[float]] = {
                metric: [] for metric in _METRIC_PRIORITIES
            }
            granularity_metrics: Dict[str, Dict[str, List[float]]] = {
                metric: {} for metric in _METRIC_PRIORITIES
            }

            for row in rows:
                granularity = _granularity_from_row(row)
                for metric in _METRIC_PRIORITIES:
                    value = _metric_value(row, metric)
                    if value is None or math.isnan(value):
                        continue
                    if _is_baseline(row):
                        baseline_metrics[metric].append(value)
                    else:
                        gran_map = granularity_metrics[metric]
                        gran_map.setdefault(granularity, []).append(value)

            n_metrics = len(_METRIC_PRIORITIES)
            if preserve_three_panel:
                fig_width = max(9.0, 3.2 * n_metrics)
                fig_height = 3.6
                fig, axes = plt.subplots(
                    1,
                    n_metrics,
                    figsize=(fig_width, fig_height),
                    squeeze=False,
                )
                axes_iter = axes.flatten()
            else:
                fig_width = max(8.0, 6.0)
                fig_height = max(9.0, 3.2 * n_metrics + 0.6)
                fig, axes = plt.subplots(
                    n_metrics,
                    1,
                    figsize=(fig_width, fig_height),
                    squeeze=False,
                )
                axes_iter = axes.flatten()

            rendered_any_metric = False
            for idx, metric in enumerate(_METRIC_PRIORITIES):
                ax = axes_iter[idx]
                baseline_value = _mean(baseline_metrics[metric])
                granularity_means: Dict[str, float] = {}
                for gran, values_list in granularity_metrics[metric].items():
                    avg = _mean(values_list)
                    if avg is None or math.isnan(avg):
                        continue
                    granularity_means[gran] = avg
                if granularity_means:
                    prefer_high = metric != "lpips"
                    sorted_items = sorted(
                        granularity_means.items(),
                        key=lambda item: item[1],
                        reverse=prefer_high,
                    )
                    if preserve_three_panel:
                        sorted_items = sorted_items[:3]
                else:
                    sorted_items = []
                metric_rendered = _plot_average_metric(
                    ax,
                    metric,
                    baseline_value,
                    sorted_items,
                    show_ylabel=idx == 0,
                )
                rendered_any_metric = rendered_any_metric or metric_rendered

            if not rendered_any_metric:
                fig.text(
                    0.5,
                    0.48,
                    "No metric averages available",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="0.35",
                )

            title_suffix = _stage_label(stage)
            fig.suptitle(
                f"{csv_path.stem} — metric averages ({title_suffix})",
                fontsize=12,
            )
            if preserve_three_panel:
                fig.tight_layout(rect=(0, 0, 1, 0.92))
            else:
                fig.tight_layout(rect=(0, 0, 1, 0.96))

            out_png = plot_dir / f"{csv_path.stem}.png"
            _nice_save(fig, out_png, tight=False)
            saved.append(str(out_png))
            continue

        dataset_groups: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            raw_ds = (row.get("dataset") or "").strip()
            if raw_ds:
                # keep just the leaf name: "tandt_db/db/playroom" -> "playroom"
                dataset = PurePath(raw_ds.replace("\\", "/")).name
            else:
                # try to infer from model path; else leave empty
                model = (row.get("model") or "").strip().replace("\\", "/")
                parts = PurePath(model).parts
                dataset = parts[-2] if len(parts) >= 2 else "All scenes (average)"
            dataset_groups.setdefault(dataset, []).append(row)

        per_metric_limit = 3 if preserve_three_panel else None
        dataset_entries: Dict[str, Dict[str, Any]] = {}
        for dataset, ds_rows in dataset_groups.items():
            baseline_row = _baseline_row(ds_rows)
            metric_map = {
                metric: _topk_rows_by_metric(ds_rows, metric, max_items=per_metric_limit)
                for metric in _METRIC_PRIORITIES
            }
            if not baseline_row and not any(metric_map.values()):
                continue
            dataset_entries[dataset] = {
                "baseline": baseline_row,
                "metrics": metric_map,
            }

        if not dataset_entries:
            continue

        stem_lower = csv_path.stem.lower()
        if stem_lower.startswith("metrics_per_model_best"):
            granularity_counts: Dict[str, Counter[str]] = {metric: Counter() for metric in _METRIC_PRIORITIES}
            for entry in dataset_entries.values():
                metric_map = entry["metrics"]
                baseline_row = entry["baseline"]
                for metric in _METRIC_PRIORITIES:
                    candidates = list(metric_map.get(metric) or [])
                    if baseline_row is not None:
                        baseline_value = _metric_value(baseline_row, metric)
                        if baseline_value is not None and not math.isnan(baseline_value):
                            candidates.append((baseline_value, baseline_row))
                    if not candidates:
                        continue
                    prefer_high = metric != "lpips"
                    best_entry = max(
                        candidates,
                        key=lambda pair: pair[0],
                    ) if prefer_high else min(
                        candidates,
                        key=lambda pair: pair[0],
                    )
                    _, row = best_entry
                    granularity = _granularity_from_row(row)
                    granularity_counts[metric][granularity] += 1

            if not any(counter for counter in granularity_counts.values()):
                continue

            ncols = len(_METRIC_PRIORITIES)
            if preserve_three_panel:
                fig_width = max(9.0, 3.2 * ncols)
                fig_height = 3.6
                fig, axes = plt.subplots(
                    1,
                    ncols,
                    figsize=(fig_width, fig_height),
                    squeeze=False,
                )
                axes_iter = axes.flatten()
                count_limit = 3
            else:
                fig_width = max(8.0, 6.0)
                fig_height = max(9.0, 3.2 * ncols + 0.6)
                fig, axes = plt.subplots(
                    ncols,
                    1,
                    figsize=(fig_width, fig_height),
                    squeeze=False,
                )
                axes_iter = axes.flatten()
                count_limit = None
            for idx, metric in enumerate(_METRIC_PRIORITIES):
                ax = axes_iter[idx]
                counts = granularity_counts.get(metric, Counter())
                _plot_granularity_counts(
                    ax,
                    counts,
                    metric,
                    show_ylabel=idx == 0,
                    max_items=count_limit,
                )

            title_suffix = _stage_label(stage)
            fig.suptitle(
                f"{csv_path.stem} — granularity wins ({title_suffix})",
                fontsize=12,
            )
            if preserve_three_panel:
                fig.tight_layout(rect=(0, 0, 1, 0.92))
            else:
                fig.tight_layout(rect=(0, 0, 1, 0.96))

            out_png = plot_dir / f"{csv_path.stem}.png"
            _nice_save(fig, out_png, tight=False)
            saved.append(str(out_png))
            continue

        dataset_names = sorted(dataset_entries.keys(), key=str.lower)
        nrows = len(dataset_names)
        ncols = len(_METRIC_PRIORITIES)
        vertical_metrics_layout = stem_lower.startswith(
            "metrics_per_view_best_average_"
        ) or stem_lower.startswith("metrics_per_view_worst_average_")
        if vertical_metrics_layout:
            total_panels = max(1, nrows * ncols)
            fig_width = min(_MAX_CSV_FIG_WIDTH, max(8.0, 3.4))
            fig_height = max(8.0, 2.6 * total_panels)
            fig, axes = plt.subplots(
                total_panels,
                1,
                figsize=(fig_width, fig_height),
                squeeze=False,
            )
            axes_flat = axes.flatten()
        else:
            base_width = max(8.0, 3.0 * ncols)
            fig_width = min(_MAX_CSV_FIG_WIDTH, base_width)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width, 2.8 * nrows),
                squeeze=False,
            )
            column_title_set = [False] * ncols
        for row_idx, dataset in enumerate(dataset_names):
            entry = dataset_entries[dataset]
            metric_map = entry["metrics"]
            baseline_row = entry["baseline"]
            dataset_label_assigned = False
            for col_idx, metric in enumerate(_METRIC_PRIORITIES):
                if vertical_metrics_layout:
                    flat_idx = row_idx * ncols + col_idx
                    ax = axes_flat[flat_idx]
                else:
                    ax = axes[row_idx][col_idx]
                top_entries = list(metric_map.get(metric) or [])

                entries_with_baseline = top_entries
                if baseline_row is not None:
                    baseline_value = _metric_value(baseline_row, metric)
                    if baseline_value is not None and not math.isnan(baseline_value):
                        entries_with_baseline = [(baseline_value, baseline_row), *entries_with_baseline]

                if not entries_with_baseline:
                    ax.axis("off")
                    continue
                _plot_metric_topk(ax, entries_with_baseline, metric)
                if vertical_metrics_layout:
                    title = _METRIC_TITLES.get(metric, metric.upper())
                    ax.set_title(f"{dataset} — {title}", fontsize=10)
                    ax.set_ylabel("")
                else:
                    if not column_title_set[col_idx]:
                        title = _METRIC_TITLES.get(metric, metric.upper())
                        ax.set_title(title, fontsize=10)
                        column_title_set[col_idx] = True
                    if not dataset_label_assigned:
                        ax.set_ylabel(dataset, fontsize=10)
                        dataset_label_assigned = True
                    else:
                        ax.set_ylabel("")
                # if row_idx == nrows - 1:
                #     ax.set_xlabel("Granularity")

        title_suffix = _stage_label(stage)
        descriptor = "top 3 per dataset" if per_metric_limit else "all granularities per dataset"
        fig.suptitle(f"{csv_path.stem} — {descriptor} ({title_suffix})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_png = plot_dir / f"{csv_path.stem}.png"
        _nice_save(fig, out_png, tight=False)
        saved.append(str(out_png))

    for variance_path in variance_files:
        try:
            stage = _stage_from_filename(variance_path)
        except ValueError as exc:
            print(f"[visual_utils] Skipping variance CSV without stage suffix: {variance_path.name} ({exc})")
            continue

        with variance_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                for row in reader
            ]

        if not rows:
            continue

        stem = variance_path.stem
        stem_lower = stem.lower()
        if stem_lower.startswith("metrics_per_model_variance_"):
            try:
                variance_kind = _variance_kind_from_stem(variance_path.stem)
            except ValueError as exc:
                print(f"[visual_utils] Skipping variance CSV with unknown kind: {variance_path.name} ({exc})")
                continue

            preserve_three_panel = _preserve_three_panel_layout(stem)
            n_metrics = len(_METRIC_PRIORITIES)
            if preserve_three_panel:
                fig_width = max(9.0, 3.2 * n_metrics)
                fig_height = 3.6
                fig, axes = plt.subplots(
                    1,
                    n_metrics,
                    figsize=(fig_width, fig_height),
                    squeeze=False,
                )
                axes_iter = axes.flatten()
                entry_limit = 3
            else:
                fig_width = max(8.0, 6.0)
                fig_height = max(9.0, 3.2 * n_metrics + 0.6)
                fig, axes = plt.subplots(
                    n_metrics,
                    1,
                    figsize=(fig_width, fig_height),
                    squeeze=False,
                )
                axes_iter = axes.flatten()
                entry_limit = None
            rendered_any_metric = False
            for idx, metric in enumerate(_METRIC_PRIORITIES):
                ax = axes_iter[idx]
                baseline_entry, top_entries = _collect_metric_variance_entries(
                    rows,
                    metric,
                    variance_kind,
                    max_items=entry_limit,
                )
                metric_rendered = _plot_metric_variance_panel(
                    ax,
                    metric,
                    variance_kind,
                    baseline_entry,
                    top_entries,
                    show_ylabel=idx == 0,
                )
                rendered_any_metric = rendered_any_metric or metric_rendered

            if not rendered_any_metric:
                plt.close(fig)
                continue

            fig.suptitle(
                f"{variance_path.stem} — variance ({_stage_label(stage)})",
                fontsize=12,
            )
            if preserve_three_panel:
                fig.tight_layout(rect=(0, 0, 1, 0.92))
            else:
                fig.tight_layout(rect=(0, 0, 1, 0.96))

            out_png = plot_dir / f"{variance_path.stem}.png"
            _nice_save(fig, out_png, tight=False)
            saved.append(str(out_png))
            continue

        sorted_rows = sorted(rows, key=_variance_sort_key)
        max_items = min(8, len(sorted_rows))
        top_rows = sorted_rows[:max_items]

        numeric_columns = _collect_numeric_columns(top_rows)
        variance_series = [
            (name, values)
            for name, values in numeric_columns.items()
            if name.lower().startswith("var_")
        ]
        rank_series = [
            (name, values)
            for name, values in numeric_columns.items()
            if name.lower().startswith("rank_var_")
        ]

        if not variance_series and not rank_series:
            continue

        variance_series.sort(key=lambda item: _series_sort_key(item[0]))
        rank_series.sort(key=lambda item: _series_sort_key(item[0]))

        series_items = [
            *[(name, values, False) for name, values in variance_series],
            *[(name, values, False) for name, values in rank_series],
        ]

        labels = [_compose_label(row, idx) for idx, row in enumerate(top_rows)]
        tick_labels = [_compose_tick_label(row, idx) for idx, row in enumerate(top_rows)]

        n_panels = len(series_items)
        ncols = min(3, n_panels)
        nrows = math.ceil(n_panels / ncols)
        base_width = max(8.0, 0.6 * len(labels))
        fig_width = min(_MAX_CSV_FIG_WIDTH, base_width)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_width, 3.2 * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()
        for ax, (series_name, values, apply_ylim) in zip(axes_flat, series_items):
            _plot_single_series(
                ax,
                labels,
                tick_labels,
                values,
                series_name,
                apply_metric_ylim=apply_ylim,
            )

        for ax in axes_flat[len(series_items):]:
            ax.axis("off")

        fig.suptitle(
            f"{variance_path.stem} — variance overview ({_stage_label(stage)})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_png = plot_dir / f"{variance_path.stem}.png"
        _nice_save(fig, out_png, tight=False)
        saved.append(str(out_png))

    return saved


def _metric_ylim(metric: str):
    m = metric.upper()
    if m == "LPIPS":
        return (0.0, 0.4)
    if m == "SSIM":
        return (0.70, 1.00)
    if m == "PSNR":
        return (15.0, 40.0)
    return None  # leave auto for unknown metrics


def _as_view_index(name: str) -> int:
    """
    Extract a sortable numeric index from view/image names like '00012.png' or 'view_0012.png'.
    Falls back to lexicographic ordering if no digits are present.
    """
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else name


def _ensure_metrics_order(d: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    """Return (ordered_keys, values) sorted by embedded numeric index in keys."""
    keys = sorted(d.keys(), key=_as_view_index)
    vals = np.array([d[k] for k in keys], dtype=float)
    return keys, vals


def _nice_save(fig: plt.Figure, path: Path, tight: bool = True, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def create_metrics_visualizations(output_dir: str) -> List[str]:
    """
    Read `results.json` and `per_view.json` inside `output_dir`, generate visualizations, and save them as PNG.

    Generates:
      - results_bar_<metric>.png      : Bar charts per metric across all variants in results.json
      - per_view_<metric>_<variant>.png : Per-view line plot (sorted by view index), with mean/median bands
      - per_view_box_<metric>.png     : Boxplot comparing per-view distributions across variants

    Returns:
      List of file paths (as strings) to all saved PNGs.

    Notes:
      - Supports multiple variants (top-level keys) e.g., "ours_30000", "baseline", "fisher_80".
      - Expects metrics named PSNR, SSIM, LPIPS (extra metrics will be plotted if present).
    """
    out_dir = Path(output_dir)
    save_dir = out_dir / "plots"
    ensure_dir_for(save_dir)
    results_path = out_dir / "results.json"
    per_view_path = out_dir / "per_view.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json at {results_path}")
    if not per_view_path.exists():
        raise FileNotFoundError(f"Missing per_view.json at {per_view_path}")

    with results_path.open("r") as f:
        results = json.load(f)  # {variant: {metric: float}}

    with per_view_path.open("r") as f:
        per_view = json.load(f)  # {variant: {metric: {view_name: float}}}

    # Normalize keys to strings (in case)
    results = {str(k): v for k, v in results.items()}
    per_view = {str(k): v for k, v in per_view.items()}

    # Collect metric names
    metrics_set = set()
    for v in results.values():
        metrics_set.update(v.keys())
    for v in per_view.values():
        metrics_set.update(v.keys())
    metrics: List[str] = sorted(metrics_set, key=lambda m: {"PSNR": 0, "SSIM": 1, "LPIPS": 2}.get(m, 99))

    saved: List[str] = []

    # ============== Scene-level bars (results.json) ==============
    # For each metric, make a bar chart across variants
    variants = sorted(results.keys())
    for metric in metrics:
        # Only plot variants that have the metric
        xs = []
        ys = []
        for var in variants:
            val = results.get(var, {}).get(metric, None)
            if val is not None:
                xs.append(var)
                ys.append(val)

        if not xs:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(xs, ys)
        ax.set_title(f"{metric} (scene-level)")
        ax.set_ylabel(metric)
        ax.set_xlabel("Variant")
        for i, v in enumerate(ys):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)
        ax.grid(axis="y", alpha=0.3)
        ylim = _metric_ylim(metric)
        if ylim:
            ax.set_ylim(*ylim)

        out_png = save_dir / f"results_bar_{metric.lower()}.png"
        _nice_save(fig, out_png)
        saved.append(str(out_png))

    # ============== Per-view curves (per_view.json) ==============
    # Per metric, we also assemble a boxplot across variants for distribution comparison.
    per_metric_distributions: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in metrics}

    for var, m2d in per_view.items():
        for metric, view_map in m2d.items():
            if not isinstance(view_map, dict) or not view_map:
                continue
            view_names, values = _ensure_metrics_order(view_map)

            # Per-view line plot for this (variant, metric)
            fig, ax = plt.subplots(figsize=(9, 4.2))
            x = np.arange(len(values))
            ax.plot(x, values, marker="", linewidth=1.5)
            mean_v = float(np.mean(values))
            med_v = float(np.median(values))
            ax.axhline(mean_v, linestyle="--", linewidth=1.0, label=f"mean={mean_v:.3f}")
            ax.axhline(med_v, linestyle=":", linewidth=1.0, label=f"median={med_v:.3f}")

            # highlight best/worst points
            if metric.upper() in ("PSNR", "SSIM"):
                best_idx = int(np.argmax(values))
                worst_idx = int(np.argmin(values))
                ax.scatter([best_idx], [values[best_idx]], s=30, zorder=3, label=f"best={values[best_idx]:.3f}")
                ax.scatter([worst_idx], [values[worst_idx]], s=30, zorder=3, label=f"worst={values[worst_idx]:.3f}")
            else:  # lower is better (e.g., LPIPS)
                best_idx = int(np.argmin(values))
                worst_idx = int(np.argmax(values))
                ax.scatter([best_idx], [values[best_idx]], s=30, zorder=3, label=f"best={values[best_idx]:.3f}")
                ax.scatter([worst_idx], [values[worst_idx]], s=30, zorder=3, label=f"worst={values[worst_idx]:.3f}")

            ax.set_title(f"{metric} per-view — {var}")
            ax.set_xlabel("View index (sorted)")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, ncol=3, loc="best")
            ylim = _metric_ylim(metric)
            if ylim:
                ax.set_ylim(*ylim)

            out_png = save_dir / f"per_view_{metric.lower()}_{var}.png"
            _nice_save(fig, out_png)
            saved.append(str(out_png))

            # Record distribution for boxplot aggregation
            per_metric_distributions.setdefault(metric, {})[var] = values

    # ============== Per-view boxplots across variants ==============
    for metric, dist_by_var in per_metric_distributions.items():
        if not dist_by_var:
            continue
        tick_labels = sorted(dist_by_var.keys())
        data = [dist_by_var[v] for v in tick_labels]

        fig, ax = plt.subplots(figsize=(8.8, 4.2))
        b = ax.boxplot(data, tick_labels=tick_labels, showmeans=True, patch_artist=True)
        ax.set_title(f"{metric} per-view distribution across variants")
        ax.set_ylabel(metric)
        ax.set_xlabel("Variant")
        ax.grid(axis="y", alpha=0.3)
        ylim = _metric_ylim(metric)
        if ylim:
            ax.set_ylim(*ylim)

        # small annotation of means
        means = [np.mean(v) for v in data]
        for i, m in enumerate(means, start=1):
            ax.text(i, b["means"][0].get_ydata()[0] if b["means"] else m, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8)

        out_png = save_dir / f"per_view_box_{metric.lower()}.png"
        _nice_save(fig, out_png)
        saved.append(str(out_png))

    try:
        saved.extend(create_metrics_csv_visualizations(out_dir))
    except FileNotFoundError:
        pass

    return saved

if __name__ == "__main__":
    from gs_coresets.utils.default_paths import ROOT_DIR
    outputs_root = (ROOT_DIR.resolve() / "outputs" / "gs_coresets" / "prune90")
    model_dir = (outputs_root.resolve() / "tandt_db" / "db" / "drjohnson" / "model_baseline")

    print("Model visualisation:")
    print("Creating PNGs...")
    pngs = create_metrics_visualizations(str(model_dir))
    print("PNGs created.")
    print("\n".join(pngs))

    print("All datasets visualization:")
    print("Creating PNGs...")
    pngs = create_metrics_csv_visualizations(str(outputs_root))
    print("PNGs created.")
    print("\n".join(pngs))
