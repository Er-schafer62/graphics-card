"""Display output: turns an RGBA8 framebuffer into a PNG file.

Only the standard library is used (zlib for the DEFLATE stream).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def write_png(path: str | Path, rgba: np.ndarray) -> Path:
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected an array of shape (height, width, 4)")
    height, width, _ = rgba.shape
    # Each scanline starts with filter type 0 (none).
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rgba.reshape(height, -1)], axis=1)
    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
           + _chunk(b"IEND", b""))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path
