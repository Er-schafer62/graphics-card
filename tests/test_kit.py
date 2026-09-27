"""Checks on the paint-and-assemble kit (skipped when manifold3d is missing)."""

import itertools
import sys
from pathlib import Path

import pytest

pytest.importorskip("manifold3d")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import kit  # noqa: E402


@pytest.fixture(scope="module")
def parts():
    return kit.build_kit()


def dims(solid):
    b = solid.bounding_box()
    return b[3] - b[0], b[4] - b[1], b[5] - b[2]


def test_kit_size_and_palette(parts):
    assert len(parts) <= 30
    assert len({p.number for p in parts}) == len(parts)
    assert {p.colour for p in parts} == {kit.RED, kit.BLACK, kit.WHITE, kit.YELLOW}


def test_every_part_prints_as_one_solid_on_the_bed(parts):
    for part in parts:
        solid = part.printable()
        assert solid.status().name == "NoError", part.name
        assert len(solid.decompose()) == 1, part.name
        assert solid.bounding_box()[2] == pytest.approx(0, abs=1e-6), part.name


def test_assembled_parts_never_overlap(parts):
    # Pegs sit in holes with clearance and inlays sit in recesses, so no two
    # parts may share any volume once assembled.
    for a, b in itertools.combinations(parts, 2):
        assert (a.assembled ^ b.assembled).volume() < 1e-3, (a.name, b.name)


def test_pegs_engage_their_sockets(parts):
    by_name = {p.name: p.assembled for p in parts}
    # The backplate pins pass right through the PCB into the heatsink.
    pins = by_name["backplate"].trim_by_plane((0, 0, 1), kit.BP_TOP + 0.1)
    assert (pins ^ by_name["heatsink"].hull()).volume() > 0
    # Hub caps sit on the axles that are part of the heatsink.
    for i in (1, 2):
        cap = by_name[f"hub_cap_{i}"]
        assert (cap.hull() ^ by_name["heatsink"]).volume() > 0


def test_split_pieces_fit_a_180mm_bed(parts):
    for part in parts:
        pieces = part.split or (part.printable(),)
        for piece in pieces:
            x, y, _ = dims(piece)
            assert max(x, y) <= 180, part.name
