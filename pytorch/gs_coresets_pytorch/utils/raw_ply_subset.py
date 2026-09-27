"""Lossless row-subsetting helpers for GraphDECO Gaussian PLY files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from plyfile import PlyData, PlyElement, PlyListProperty

from gs_coresets_pytorch.utils.output_safety import (
    preflight_outputs, publish_new_file, temporary_output,
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_indices(indices: Iterable[int], row_count: int) -> np.ndarray:
    values = np.asarray(list(indices), dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("original Gaussian indices must be one-dimensional")
    if values.size == 0:
        raise ValueError("at least one original Gaussian index is required")
    if np.any(values < 0) or np.any(values >= row_count):
        raise IndexError(
            f"original Gaussian index outside [0, {row_count})"
        )
    if values.size > 1 and np.any(values[1:] <= values[:-1]):
        raise ValueError(
            "original Gaussian indices must be unique and strictly increasing"
        )
    return values


def _vertex_element(ply: PlyData) -> PlyElement:
    try:
        return ply["vertex"]
    except KeyError as exc:
        raise ValueError("source PLY has no vertex element") from exc


def _row_digest(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def validate_raw_ply_subset(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    original_indices: Iterable[int],
) -> dict[str, Any]:
    """Verify that output vertex records exactly match selected source records."""
    source = PlyData.read(str(source_path))
    output = PlyData.read(str(output_path))
    source_vertex = _vertex_element(source)
    output_vertex = _vertex_element(output)
    indices = _normalized_indices(original_indices, len(source_vertex.data))
    expected = np.array(source_vertex.data[indices], copy=True)
    observed = np.array(output_vertex.data, copy=False)

    if expected.dtype != observed.dtype:
        raise ValueError(
            "raw subset field layout changed: "
            f"{expected.dtype!r} != {observed.dtype!r}"
        )
    if expected.shape != observed.shape:
        raise ValueError(
            "raw subset row count changed: "
            f"{expected.shape!r} != {observed.shape!r}"
        )
    if expected.tobytes(order="C") != observed.tobytes(order="C"):
        raise ValueError("raw subset vertex-row bytes do not match the source")

    return {
        "row_count": int(expected.shape[0]),
        "property_names": list(expected.dtype.names or ()),
        "selected_row_sha256": _row_digest(expected),
        "output_row_sha256": _row_digest(observed),
        "source_format": (
            "ascii"
            if source.text
            else f"binary_{source.byte_order or '='}_endian"
        ),
    }


def save_raw_ply_subset(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    original_indices: Iterable[int],
) -> dict[str, Any]:
    """Write selected binary PLY rows without decoding Gaussian parameters.

    The complete structured vertex records are copied, so non-unit stored
    quaternions and any additional scalar properties remain bit-identical.
    List-valued vertex properties are rejected because they do not have a
    fixed raw row representation.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    preflight_outputs(output_path, inputs=(source_path,))
    source = PlyData.read(str(source_path))
    if source.text:
        raise ValueError(
            "raw-parameter preservation requires a binary source PLY"
        )
    source_vertex = _vertex_element(source)
    if any(
        isinstance(prop, PlyListProperty)
        for prop in source_vertex.properties
    ):
        raise ValueError(
            "raw-parameter preservation does not support list-valued "
            "vertex properties"
        )
    indices = _normalized_indices(original_indices, len(source_vertex.data))
    selected = np.array(source_vertex.data[indices], copy=True)
    replacement = PlyElement.describe(
        selected,
        source_vertex.name,
        comments=list(source_vertex.comments),
    )
    elements = [
        replacement if element.name == source_vertex.name else element
        for element in source.elements
    ]
    output = PlyData(
        elements,
        text=False,
        byte_order=source.byte_order,
        comments=list(source.comments),
        obj_info=list(source.obj_info),
    )

    with temporary_output(output_path) as temporary:
        output.write(str(temporary))
        metadata = validate_raw_ply_subset(
            source_path,
            temporary,
            indices.tolist(),
        )
        metadata.update(
            {
                "source_ply": str(source_path.resolve()),
                "output_ply": str(output_path.resolve()),
                "source_ply_sha256": sha256_file(source_path),
                "output_ply_sha256": sha256_file(temporary),
            }
        )
        publish_new_file(temporary, output_path)
    return metadata


__all__ = [
    "save_raw_ply_subset",
    "sha256_file",
    "validate_raw_ply_subset",
]
