import numpy as np

from rx590gme import demos
from rx590gme.display import write_png


def test_triangle_covers_its_centroid_and_not_the_corners(gpu):
    image, stats = demos.triangle(gpu, 64, 48, background=(1, 2, 3))
    assert image.shape == (48, 64, 4)
    assert tuple(image[0, 0]) == (1, 2, 3, 255)
    assert tuple(image[0, 63]) == (1, 2, 3, 255)
    centre = image[30, 32]
    assert tuple(centre[:3]) != (1, 2, 3)
    # centroid is roughly an equal blend of the three vertex colours
    expected = np.mean([c for _, _, c in demos.DEFAULT_TRIANGLE], axis=0)
    assert np.all(np.abs(centre[:3].astype(int) - expected) < 40)
    assert stats.smem == stats.wavefronts  # vertex data comes in through the scalar cache


def test_mandelbrot_interior_is_black_and_exterior_is_not(gpu):
    image, stats = demos.mandelbrot(gpu, 48, 32, max_iter=32, center=(-0.6, 0.0))
    assert image.shape == (32, 48, 4)
    assert np.all(image[..., 3] == 255)
    scale = 3.0 / 48
    px, py = int((0.0 - (-0.6 - 24 * scale)) / scale), 16      # c = 0 is in the set
    assert tuple(image[py, px, :3]) == (0, 0, 0)
    assert image[0, 0, :3].sum() > 0 or image[0, 47, :3].sum() > 0
    assert stats.simd_utilization < 1.0  # escaping lanes are masked off


def test_png_writer(tmp_path):
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    path = write_png(tmp_path / "x.png", rgba)
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in data and data.endswith(b"IEND\xaeB`\x82")
