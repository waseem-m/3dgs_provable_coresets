#!/usr/bin/env python3
"""
Dual-output logging helpers shared across the CLI entrypoints.
"""

from __future__ import annotations

from __future__ import annotations
from contextlib import contextmanager
import codecs
import os, sys, io, subprocess, threading
from typing import Dict, Optional, Sequence

from gs_coresets.utils.io_utils import ensure_dir_for

__all__ = [
    "DualOutput",
    "log_enable",
    "log_disable",
    "log_to_file",
    "LoggingContext",
    "maybe_enable_logging",
    "run_tee"
]


_prev_streams = None
_log_file = None
_log_lock = threading.Lock()


class DualOutput(io.TextIOBase):
    """Mirror writes to both the console and a logfile."""

    def __init__(self, console, file, lock):
        self.console = console
        self.file = file
        self.lock = lock

    def writable(self) -> bool:  # pragma: no cover - required by TextIOBase
        return True

    def readable(self) -> bool:  # pragma: no cover - required by TextIOBase
        return False

    def seekable(self) -> bool:  # pragma: no cover - required by TextIOBase
        return False

    def write(self, s: str):  # type: ignore[override]
        if not isinstance(s, str):
            s = str(s)
        with self.lock:
            self.console.write(s)
            if self.file is not None:
                self.file.write(s)
            if s.endswith("\n") or s.endswith("\r"):
                self.console.flush()
                if self.file is not None:
                    self.file.flush()
        return len(s)

    def writelines(self, lines):  # pragma: no cover - rarely used
        for line in lines:
            self.write(line)

    def flush(self):  # pragma: no cover - passthrough
        with self.lock:
            self.console.flush()
            if self.file is not None:
                self.file.flush()

    def isatty(self):  # pragma: no cover - passthrough
        return getattr(self.console, "isatty", lambda: False)()

    def fileno(self):  # pragma: no cover - passthrough
        return self.console.fileno()

    @property
    def encoding(self):  # pragma: no cover - passthrough
        return getattr(self.console, "encoding", "utf-8")

    @property
    def errors(self):  # pragma: no cover - passthrough
        return getattr(self.console, "errors", "replace")


def log_enable(path: str) -> None:
    """Tee stdout/stderr to the console and the provided logfile."""
    global _prev_streams, _log_file
    if _log_file is not None:
        return  # already enabled

    ensure_dir_for(path, treat_as_file=True)
    _prev_streams = (sys.stdout, sys.stderr)
    _log_file = open(path, "a", encoding="utf-8")
    sys.stdout = DualOutput(sys.__stdout__, _log_file, _log_lock)
    sys.stderr = DualOutput(sys.__stderr__, _log_file, _log_lock)


def log_disable() -> None:
    """Restore the original stdout/stderr streams and close the logfile."""
    global _prev_streams, _log_file
    if _log_file is None:
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # pragma: no cover - defensive
        pass

    sys.stdout, sys.stderr = _prev_streams
    _prev_streams = None

    _log_file.flush()
    _log_file.close()
    _log_file = None


@contextmanager
def log_to_file(path: str):
    """Context manager that enables dual-output logging inside the block."""
    log_enable(path)
    try:
        yield
    finally:
        log_disable()


class LoggingContext:
    """Handle returned by maybe_enable_logging to manage teardown."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path

    def close(self) -> None:
        log_disable()

    def __enter__(self) -> "LoggingContext":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.close()


def maybe_enable_logging(log_path: Optional[str]) -> Optional[LoggingContext]:
    """
    Optionally enable dual-output logging.
    """
    if not log_path:
        return None

    log_enable(log_path)
    return LoggingContext(log_path=log_path)


def run_tee(
    cmd: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> int:
    """
    Run a subprocess and stream BOTH stdout+stderr (merged) to our stdout
    in real time. When --log is enabled, sys.stdout is DualOutput, so the same
    bytes mirrored here are also persisted to the logfile automatically.

    Returns the child's exit code.
    """
    env_full = dict(os.environ, **(env or {}))
    # Make Python children unbuffered; helps immediate prints
    env_full.setdefault("PYTHONUNBUFFERED", "1")

    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env_full,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr → stdout (preserve order)
        bufsize=0,                  # unbuffered binary pipe
        text=False,
    ) as p:
        assert p.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stream = p.stdout
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            tail = decoder.decode(b"", final=True)
            if tail:
                sys.stdout.write(tail)
                sys.stdout.flush()
        finally:
            stream.close()
        return int(p.wait() or 0)
