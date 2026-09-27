"""Machine-readable provenance for exact Gaussian selections."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gs_coresets_pytorch.utils.raw_ply_subset import sha256_file
from gs_coresets_pytorch.utils.output_safety import (
    preflight_outputs, publish_new_file, temporary_output,
)


SCHEMA = "3dgs_coresets.selection.v1"


def _indices_sha256(indices: Sequence[int]) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def probability_statistics(
    scores: torch.Tensor,
    probabilities: torch.Tensor,
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    """Summarize score support and the exact selection distribution."""
    score_values = scores.detach().to(device="cpu", dtype=torch.float64)
    probability_values = probabilities.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    selected = torch.as_tensor(selected_indices, dtype=torch.long)
    positive = probability_values > 0
    positive_values = probability_values[positive]
    entropy_terms = probability_values[positive]
    return {
        "score_count": int(score_values.numel()),
        "score_finite_count": int(torch.isfinite(score_values).sum().item()),
        "score_positive_count": int((score_values > 0).sum().item()),
        "probability_support_count": int(positive.sum().item()),
        "probability_sum": float(probability_values.sum().item()),
        "probability_min_positive": (
            float(positive_values.min().item())
            if positive_values.numel()
            else None
        ),
        "probability_max": float(probability_values.max().item()),
        "probability_entropy_nats": float(
            -(entropy_terms * torch.log(entropy_terms)).sum().item()
        ),
        "selected_probability_mass": float(
            probability_values[selected].sum().item()
        ),
        "selected_probability_min": float(
            probability_values[selected].min().item()
        ),
        "selected_probability_max": float(
            probability_values[selected].max().item()
        ),
    }


def write_selection_manifest(
    path: str | os.PathLike[str],
    *,
    source_ply: str | os.PathLike[str],
    output_ply: str | os.PathLike[str],
    score_path: str | os.PathLike[str] | None,
    selected_indices: Sequence[int],
    original_count: int,
    requested_count: int,
    prune_ratio: float,
    seed: int | None,
    sampling_mode: str,
    replacement: bool,
    temperature: float,
    unit_weights: bool,
    raw_subset_validation: Mapping[str, Any],
    probability_stats: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
    published_output_ply: str | os.PathLike[str] | None = None,
) -> Path:
    """Atomically create an immutable selection manifest."""
    manifest_path = Path(path)
    preflight_outputs(manifest_path, inputs=(source_ply, output_ply))
    indices = [int(index) for index in selected_indices]
    if len(indices) != requested_count:
        raise ValueError(
            f"selected {len(indices)} identities, expected {requested_count}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("selection manifest cannot contain duplicate identities")
    if indices != sorted(indices):
        raise ValueError("selection manifest identities must be sorted")
    if any(index < 0 or index >= original_count for index in indices):
        raise ValueError("selection manifest identity outside the source model")
    if not math.isclose(
        float(probability_stats["probability_sum"]),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("selection probabilities do not sum to one")

    source = Path(source_ply)
    output = Path(output_ply)
    published_output = Path(published_output_ply) if published_output_ply is not None else output
    score = Path(score_path) if score_path is not None else None
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method_name": "3DGS Coresets",
        "source_ply": str(source.resolve()),
        "source_ply_sha256": sha256_file(source),
        "output_ply": str(published_output.resolve()),
        "output_ply_sha256": sha256_file(output),
        "score_path": str(score.resolve()) if score is not None else None,
        "score_sha256": sha256_file(score) if score is not None else None,
        "original_count": int(original_count),
        "requested_count": int(requested_count),
        "retained_count": len(indices),
        "prune_ratio": float(prune_ratio),
        "seed": None if seed is None else int(seed),
        "sampling_mode": str(sampling_mode),
        "replacement": bool(replacement),
        "temperature": float(temperature),
        "unit_weights": bool(unit_weights),
        "raw_parameters_preserved": True,
        "original_indices_sorted": indices,
        "original_indices_int64le_sha256": _indices_sha256(indices),
        "probability_statistics": dict(probability_stats),
        "raw_subset_validation": dict(raw_subset_validation),
    }
    if extra:
        document["extra"] = dict(extra)

    with temporary_output(manifest_path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_new_file(temporary, manifest_path)
    return manifest_path


__all__ = [
    "SCHEMA",
    "probability_statistics",
    "write_selection_manifest",
]
