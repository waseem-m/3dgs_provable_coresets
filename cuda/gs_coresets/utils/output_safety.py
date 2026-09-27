"""No-replace publication of new selection artifacts.

A PLY and its manifest cannot be atomically committed as a pair. Both are
prepared first, and the manifest is published last as the completion marker.
Handled failures roll back only links created by this invocation. A process
kill may leave an incomplete PLY; it is never silently overwritten on retry.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterable, Iterator


def preflight_outputs(
    *destinations: str | os.PathLike[str],
    inputs: Iterable[str | os.PathLike[str]] = (),
) -> None:
    """Reject existing, aliased, nested, or structurally invalid destinations."""
    paths = [Path(value).absolute() for value in destinations]
    resolved = [path.resolve() for path in paths]
    protected = {Path(value).resolve() for value in inputs}
    for position, path in enumerate(paths):
        if os.path.lexists(path):
            raise FileExistsError(f"output already exists: {path}")
        canonical = resolved[position]
        if canonical in protected:
            raise ValueError(f"output aliases an input: {path}")
        for other in resolved[:position]:
            if canonical == other or canonical in other.parents or other in canonical.parents:
                raise ValueError("output destinations must be distinct, non-nested files")
        parent = path.parent
        while not os.path.lexists(parent):
            parent = parent.parent
        if not parent.is_dir():
            raise NotADirectoryError(f"output parent is not a directory: {parent}")


@contextmanager
def temporary_output(destination: str | os.PathLike[str]) -> Iterator[Path]:
    """Reserve a unique staging file on the destination's filesystem."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def publish_new_file(staged: Path, destination: Path) -> None:
    """Atomically link a completed file; fail if anything already occupies dest.

    Unlike exists()+replace(), link() cannot overwrite a concurrent writer.
    Staging alongside the destination keeps this operation on one filesystem.
    """
    os.link(staged, destination)


@contextmanager
def selection_outputs(
    output_ply: Path,
    manifest: Path | None,
    *,
    inputs: Iterable[str | os.PathLike[str]] = (),
) -> Iterator[tuple[Path, Path | None]]:
    """Stage both selection files and publish only after the caller validates."""
    destinations = (output_ply, manifest) if manifest is not None else (output_ply,)
    preflight_outputs(*destinations, inputs=inputs)
    published: list[tuple[Path, os.stat_result]] = []
    with ExitStack() as stack:
        staged = []
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            directory = stack.enter_context(tempfile.TemporaryDirectory(
                prefix=f".{destination.name}.", dir=destination.parent,
            ))
            staged.append(Path(directory) / destination.name)
        yield staged[0], staged[1] if manifest is not None else None
        try:
            for source, destination in zip(staged, destinations):
                identity = source.stat()
                publish_new_file(source, destination)
                published.append((destination, identity))
        except BaseException:
            for destination, identity in reversed(published):
                try:
                    current = destination.lstat()
                except FileNotFoundError:
                    continue
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    destination.unlink()
            raise
