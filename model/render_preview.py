"""Render shaded preview images of an STL file (NumPy z-buffer rasteriser).

    python render_preview.py stl/rx590gme_1to1.stl preview.png --view iso
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

VIEWS = {
    # (azimuth, elevation) in degrees of the camera around the model
    "iso": (215, 35),         # fans + top edge (label, heatpipes, power)
    "iso-pcie": (-35, 35),    # fans + PCIe edge
    "fans": (0, 89.9),
    "back": (0, -89.9),
    "io": (-90, 20),
    "top-edge": (180, 8),
    "installed": (180, 8),    # as seen through a case window: fans down, bracket left
}
FANS_DOWN = {"installed"}


def read_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    rec = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n, offset=84)
    return rec["v"].astype(np.float64)


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, -1)], axis=1).tobytes()

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def render(tris: np.ndarray, width: int = 960, height: int = 600, azimuth: float = -35,
           elevation: float = 35, color=(214, 40, 40), ss: int = 2) -> np.ndarray:
    """Render triangles (N, 3, 3). ``color`` is one RGB triple or one per triangle."""
    az, el = np.radians(azimuth), np.radians(elevation)
    # Camera basis: model z is "up" for the camera orbit.
    forward = -np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)])
    up_hint = np.array([0, 0, 1.0]) if abs(np.sin(el)) < 0.99 else np.array([0, 1.0, 0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pts = tris.reshape(-1, 3)
    centre = (pts.min(0) + pts.max(0)) / 2
    p = pts - centre
    sx, sy, depth = p @ right, p @ up, p @ forward
    W, H = width * ss, height * ss
    scale = 0.9 * min(W / (sx.max() - sx.min()), H / (sy.max() - sy.min()))
    X = (sx * scale + W / 2).reshape(-1, 3)
    Y = (H / 2 - sy * scale).reshape(-1, 3)
    Z = depth.reshape(-1, 3)

    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = -forward * 0.6 + up * 0.6 + right * 0.3
    light /= np.linalg.norm(light)
    shade = 0.28 + 0.72 * np.clip(normals @ light, 0, 1)
    colors = np.broadcast_to(np.asarray(color, dtype=float), (len(tris), 3))
    facing = (normals @ forward) < 0

    zbuf = np.full((H, W), np.inf)
    img = np.zeros((H, W, 3), dtype=np.float64)
    for i in np.nonzero(facing)[0]:
        x, y, z = X[i], Y[i], Z[i]
        x0, x1 = max(int(x.min()), 0), min(int(x.max()) + 1, W - 1)
        y0, y1 = max(int(y.min()), 0), min(int(y.max()) + 1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        region = zbuf[y0:y1 + 1, x0:x1 + 1]
        hit = inside & (zz < region)
        region[hit] = zz[hit]
        img[y0:y1 + 1, x0:x1 + 1][hit] = shade[i] * colors[i]

    covered = np.isfinite(zbuf)
    rgb = np.empty((H, W, 3))
    bg = np.array([246, 246, 248], dtype=float)
    rgb[:] = bg
    rgb[covered] = img[covered]
    # Darken depth discontinuities so edges read clearly.
    zf = np.where(covered, zbuf, zbuf[covered].max() + 50 if covered.any() else 0)
    edge = np.zeros((H, W), bool)
    edge[1:, :] |= np.abs(np.diff(zf, axis=0)) > 0.45
    edge[:, 1:] |= np.abs(np.diff(zf, axis=1)) > 0.45
    rgb[edge] *= 0.45
    rgb = rgb.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_parts(parts, width=960, height=600, azimuth=215, elevation=35) -> np.ndarray:
    """Render several meshes, each ``(triangles, rgb)``, in one image."""
    tris = np.concatenate([t for t, _ in parts])
    colors = np.concatenate([np.tile(np.asarray(c, float), (len(t), 1)) for t, c in parts])
    return render(tris, width, height, azimuth, elevation, colors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stl", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("--view", choices=sorted(VIEWS), default="iso")
    parser.add_argument("--size", default="960x600")
    args = parser.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    az, el = VIEWS[args.view]
    tris = read_stl(args.stl)
    if args.view in FANS_DOWN:
        tris[..., 0] *= -1          # turn the card over (180 degrees about y)
        tris[..., 2] *= -1
    write_png(args.png, render(tris, w, h, az, el))


if __name__ == "__main__":
    main()
