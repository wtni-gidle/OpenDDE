# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Plain/zstd text I/O for portable inference resources."""

from __future__ import annotations

from contextlib import contextmanager
import io
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, TextIO

try:
    import zstandard
except ImportError:  # Plain-text inputs remain usable in minimal environments.
    zstandard = None


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _require_zstandard() -> Any:
    if zstandard is None:
        raise ImportError(
            "Reading or writing zstd resources requires the 'zstandard' package."
        )
    return zstandard


@contextmanager
def open_text(path: str | os.PathLike[str]) -> Iterator[TextIO]:
    """Open UTF-8 plain or zstd text, selecting by content rather than suffix."""
    with open(path, "rb") as raw:
        magic = raw.read(len(_ZSTD_MAGIC))
        raw.seek(0)
        if magic == _ZSTD_MAGIC:
            with _require_zstandard().open(raw, "rt", encoding="utf-8") as handle:
                yield handle
        else:
            with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                yield handle


def read_text(path: str | os.PathLike[str]) -> str:
    """Read an entire plain or zstd text resource."""
    with open_text(path) as handle:
        return handle.read()


def write_zstd_text_atomic(path: str | os.PathLike[str], text: str) -> None:
    """Atomically publish UTF-8 text as a standard zstd frame."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(
                _require_zstandard().ZstdCompressor(level=3).compress(text.encode())
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
