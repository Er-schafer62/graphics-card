"""Ready-made workloads that exercise the replica end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .device import LaunchStats, RX590GME
from .kernels import builtin_kernel


def saxpy(gpu: RX590GME, a: float, x: np.ndarray, y: np.ndarray,
          block: int = 256) -> tuple[np.ndarray, LaunchStats]:
    """Compute ``a * x + y`` on the GPU and return (result, stats)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = x.size
    dx, dy = gpu.to_device(x), gpu.to_device(y)
    try:
        stats = gpu.launch(gpu.compile(builtin_kernel("saxpy")), grid=-(-n // block), block=block,
                           args=[n, float(a), dx, dy, block])
        return gpu.from_device(dy, np.float32, n), stats
    finally:
        gpu.free(dx)
        gpu.free(dy)


def mandelbrot(gpu: RX590GME, width: int = 320, height: int = 240, max_iter: int = 64,
               center: tuple[float, float] = (-0.6, 0.0), zoom: float = 1.0,
               block: tuple[int, int] = (16, 16)) -> tuple[np.ndarray, LaunchStats]:
    """Render the Mandelbrot set; returns (RGBA image of shape (h, w, 4), stats)."""
    scale = 3.0 / zoom / width
    x0 = center[0] - width / 2 * scale
    y0 = center[1] - height / 2 * scale
    fb = gpu.alloc(width * height * 4)
    try:
        stats = gpu.launch(gpu.compile(builtin_kernel("mandelbrot")),
                           grid=(-(-width // block[0]), -(-height // block[1])), block=block,
                           args=[width, height, fb, max_iter, x0, y0, scale, block[0], block[1]])
        return gpu.from_device(fb, np.uint8, (height, width, 4)), stats
    finally:
        gpu.free(fb)


DEFAULT_TRIANGLE = (
    (0.50, 0.08, (237, 28, 36)),    # AMD red
    (0.92, 0.90, (40, 200, 90)),
    (0.08, 0.90, (30, 110, 240)),
)


def triangle(gpu: RX590GME, width: int = 320, height: int = 240, vertices=DEFAULT_TRIANGLE,
             background: tuple[int, int, int] = (18, 18, 24),
             block: tuple[int, int] = (16, 16)) -> tuple[np.ndarray, LaunchStats]:
    """Rasterise a colour-interpolated triangle. Vertex positions are given
    as fractions of the screen size; returns (RGBA image, stats)."""
    data = np.zeros(16, dtype=np.float32)
    for i, (vx, vy, _) in enumerate(vertices):
        data[2 * i: 2 * i + 2] = (vx * width, vy * height)
    for i, (_, _, colour) in enumerate(vertices):
        data[6 + 3 * i: 9 + 3 * i] = colour
    bg = background[0] | background[1] << 8 | background[2] << 16 | 0xFF << 24
    tri, fb = gpu.to_device(data), gpu.alloc(width * height * 4)
    try:
        stats = gpu.launch(gpu.compile(builtin_kernel("triangle")),
                           grid=(-(-width // block[0]), -(-height // block[1])), block=block,
                           args=[width, height, fb, tri, block[0], block[1], bg])
        return gpu.from_device(fb, np.uint8, (height, width, 4)), stats
    finally:
        gpu.free(tri)
        gpu.free(fb)


def save_image(path: str | Path, rgba: np.ndarray) -> Path:
    from .display import write_png
    return write_png(path, rgba)
