"""Checks on the 3D-printable card model (skipped when manifold3d is missing)."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("manifold3d")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import rx590gme_model as model  # noqa: E402


@pytest.fixture(scope="module")
def full():
    return model.build()


def size(solid):
    b = solid.bounding_box()
    return np.array([b[3] - b[0], b[4] - b[1], b[5] - b[2]]), np.array(b[:3])


def test_full_card_is_one_watertight_solid_on_the_bed(full):
    assert full.status().name == "NoError"
    assert len(full.decompose()) == 1
    dims, lo = size(full)
    assert lo[2] == pytest.approx(0, abs=1e-6)            # nothing below the build plate
    card = model.Card()
    assert dims[2] == pytest.approx(card.slot_pitch * card.slots, abs=0.05)   # bracket: dual slot
    assert 255 <= dims[0] <= 270                                               # 255 mm card + bracket


def test_card_body_is_dual_slot_thickness(full):
    # Everything except the I/O bracket stays within the 38 mm card thickness.
    body = full.trim_by_plane((1, 0, 0), 0.5)
    assert size(body)[0][2] == pytest.approx(38.0, abs=0.05)


def test_split_parts_fit_a_180mm_bed_and_keep_the_volume(full):
    front, rear, pins = model.split(full)
    for part in (front, rear):
        assert len(part.decompose()) == 1
        dims, lo = size(part)
        assert lo[2] == pytest.approx(0, abs=1e-6)
        assert sorted(dims[:2])[1] <= 180
    assert len(pins.decompose()) == 2
    holes = full.volume() - front.volume() - rear.volume()
    pin_holes = 2 * np.pi * 2.2 ** 2 * 20             # two 4.4 mm holes, 10 mm deep each side
    assert holes == pytest.approx(pin_holes, rel=0.05)


def test_half_scale_model(full):
    half = model.build(scale=0.5)
    assert len(half.decompose()) == 1
    assert size(half)[0][0] == pytest.approx(size(full)[0][0] / 2, rel=0.01)


def test_stl_writer_round_trips(tmp_path, full):
    tris, _ = model.write_stl(tmp_path / "card.stl", full, "test")
    data = (tmp_path / "card.stl").read_bytes()
    assert len(data) == 84 + 50 * tris
    assert int.from_bytes(data[80:84], "little") == tris == full.num_tri()
