"""Parametric, 3D-printable model of an AMD Radeon RX 590 GME graphics card.

The card modelled is a typical dual-fan, dual-slot RX 590 GME, such as
PowerColor's Red Dragon: 255 mm long and 38 mm thick, with DVI-D + HDMI +
DisplayPort outputs and a single 8-pin power connector. There is no vendor
branding; the top edge reads "RX 590 GME".

Coordinates are in millimetres, laid out in print orientation (backplate on
the bed):

    x  along the card: 0 is the inside face of the I/O bracket
    y  across the card: 0 is the PCB edge above the PCIe fingers
    z  up from the build plate: backplate at the bottom, fans on top

Run ``python rx590gme_model.py`` to write the STL files into ``stl/``.
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

SEGMENTS = 96


@dataclass(frozen=True)
class Card:
    length: float = 255.0          # shroud length, bracket not included
    body_y0: float = 3.0           # shroud bottom edge (PCIe side)
    body_y1: float = 121.0         # shroud top edge
    thickness: float = 38.0        # backplate to top of fan bezels (dual slot)
    backplate: float = 1.2
    pcb: float = 1.6
    pcb_length: float = 240.0
    pcb_height: float = 111.0
    bezel: float = 1.2             # raised fan rings and shroud accents
    fan_diameter: float = 88.0
    fan_x: tuple = (70.0, 186.0)
    fan_blades: int = 9
    pocket_depth: float = 12.0
    bracket_plate: float = 1.2
    slot_pitch: float = 20.32      # one expansion slot
    slots: int = 2

    @property
    def pcb_z0(self) -> float:
        return self.backplate

    @property
    def body_z0(self) -> float:
        return self.backplate + self.pcb

    @property
    def body_z1(self) -> float:
        return self.thickness - self.bezel

    @property
    def fan_y(self) -> float:
        return (self.body_y0 + self.body_y1) / 2

    @property
    def pocket_r(self) -> float:
        return self.fan_diameter / 2 + 2.0


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def box(x0, y0, z0, x1, y1, z1) -> Manifold:
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def cyl_z(cx, cy, z0, z1, r, segments=SEGMENTS) -> Manifold:
    return Manifold.cylinder(z1 - z0, r, circular_segments=segments).translate((cx, cy, z0))


def cyl_x(x0, x1, cy, cz, r, segments=48) -> Manifold:
    return Manifold.cylinder(x1 - x0, r, circular_segments=segments).rotate((0, 90, 0)).translate((x0, cy, cz))


def polygon(points) -> CrossSection:
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) < 0:
        pts = pts[::-1]                     # CrossSection wants counter-clockwise outlines
    return CrossSection([pts])


def union(parts) -> Manifold:
    parts = [p for p in parts if not p.is_empty()]
    return Manifold.batch_boolean(parts, OpType.Add) if parts else Manifold()


def rounded_rect(w, h, r) -> CrossSection:
    r = min(r, w / 2 - 1e-3, h / 2 - 1e-3)
    pts = []
    for cx, cy, a0 in ((w - r, h - r, 0), (r, h - r, 90), (r, r, 180), (w - r, r, 270)):
        for k in range(9):
            a = math.radians(a0 + k * 90 / 8)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return polygon(pts).translate((-w / 2, -h / 2))


# A tiny stroke font: each glyph is a list of polylines on a 4 x 6 grid.
GLYPHS = {
    "R": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(2, 3), (4, 0)]],
    "X": [[(0, 0), (4, 6)], [(0, 6), (4, 0)]],
    "5": [[(4, 6), (0, 6), (0, 3.4), (3, 3.4), (4, 2.4), (4, 1), (3, 0), (0, 0)]],
    "9": [[(4, 3.2), (1, 3.2), (0, 4.2), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (0, 0)]],
    "0": [[(1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0)]],
    "G": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2.2, 3)]],
    "M": [[(0, 0), (0, 6), (2, 3), (4, 6), (4, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    " ": [],
}


def text_2d(text: str, height: float, stroke: float) -> CrossSection:
    """Text as a 2D shape, baseline at y=0, starting at x=0."""
    unit = height / 6
    r = stroke / 2
    circle = [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    capsules = []
    for i, ch in enumerate(text):
        ox = i * 5.6 * unit
        for line in GLYPHS[ch]:
            for (ax, ay), (bx, by) in zip(line, line[1:]):
                pts = [(ox + ax * unit + cx, ay * unit + cy) for cx, cy in circle]
                pts += [(ox + bx * unit + cx, by * unit + cy) for cx, cy in circle]
                capsules.append(CrossSection.hull_points(np.array(pts)))
    return CrossSection.batch_boolean(capsules, OpType.Add)


def text_width(text: str, height: float) -> float:
    return (len(text) * 5.6 - 1.6) * height / 6


# ---------------------------------------------------------------------------
# Card parts
# ---------------------------------------------------------------------------

def fan(c: Card, cx: float, thin) -> Manifold:
    floor = c.body_z1 - c.pocket_depth
    top = c.body_z1 - 1.0
    r_hub, r_tip = 17.0, c.fan_diameter / 2
    hub = cyl_z(cx, c.fan_y, floor - 0.01, top, r_hub)
    hub -= cyl_z(cx, c.fan_y, top - 0.6, top + 1, r_hub - 3.0)          # sticker recess
    hub += cyl_z(cx, c.fan_y, top - 1.0, top - 0.3, 4.0, 32)             # centre cap

    # Swept blade outline: curved backwards, slightly wider at the root.
    radii = np.linspace(r_hub - 2, r_tip, 24)
    sweep = math.radians(38)
    left, right = [], []
    for r in radii:
        t = (r - radii[0]) / (radii[-1] - radii[0])
        a = sweep * t * t
        half = thin(2.4 - 0.7 * t) / 2 / r
        left.append((r * math.cos(a + half), r * math.sin(a + half)))
        right.append((r * math.cos(a - half), r * math.sin(a - half)))
    blade = polygon(left + right[::-1])
    blades = [blade.rotate(360 * k / c.fan_blades) for k in range(c.fan_blades)]
    blades = CrossSection.batch_boolean(blades, OpType.Add)
    blades = blades.extrude(top - 0.8 - floor, n_divisions=8, twist_degrees=-18)
    return hub + blades.translate((cx, c.fan_y, floor - 0.01))


def shroud_and_heatsink(c: Card, thin) -> Manifold:
    z0, z1 = c.body_z0, c.body_z1
    body = box(0, c.body_y0, z0 - 0.01, c.length, c.body_y1, z1)
    cuts = []

    # Fan pockets with a finned floor.
    floor = z1 - c.pocket_depth
    for cx in c.fan_x:
        cuts.append(cyl_z(cx, c.fan_y, floor, z1 + 1, c.pocket_r))
        grooves = union([box(x - thin(1.2) / 2, c.fan_y - c.pocket_r, floor - 1.0, x + thin(1.2) / 2,
                             c.fan_y + c.pocket_r, floor + 0.01)
                         for x in np.arange(cx - c.pocket_r, cx + c.pocket_r, 3.0)])
        cuts.append(grooves ^ cyl_z(cx, c.fan_y, floor - 2, floor + 1, c.pocket_r - 2.5))

    # Heatsink fins showing along the bottom (PCIe side) and top edges.
    gap = thin(1.4)
    for x in np.arange(20.0, 236.0, 3.0):
        cuts.append(box(x, c.body_y0 - 1, z0 + 2.5, x + gap, c.body_y0 + 7, z1 - 3.5))
    for x in np.arange(150.0, 204.0, 3.0):
        cuts.append(box(x, c.body_y1 - 7, z0 + 2.5, x + gap, c.body_y1 + 1, z1 - 3.5))

    body = body - union(cuts)

    # Three heatpipes crossing the exposed top fins.
    pipes = union([cyl_x(146, 208, c.body_y1 - 4.2, z, 3.0) for z in (z0 + 7, z0 + 15, z0 + 23)])

    # Raised fan bezels and angular accents on the shroud face.
    zt = c.thickness
    bezels = [cyl_z(cx, c.fan_y, z1 - 0.01, zt, c.pocket_r + 3.0) - cyl_z(cx, c.fan_y, z1 - 1, zt + 1, c.pocket_r)
              for cx in c.fan_x]
    mid = (c.fan_x[0] + c.fan_x[1]) / 2
    y0, y1, ym = c.body_y0 + 6, c.body_y1 - 6, c.fan_y
    accents = [
        polygon([(2, y0), (20, y0), (13, ym), (20, y1), (2, y1)]),
        polygon([(mid - 6, y0), (mid + 6, y0), (mid + 3, ym), (mid + 6, y1), (mid - 6, y1), (mid - 3, ym)]),
        polygon([(c.length - 2, y0), (c.length - 20, y0), (c.length - 13, ym),
                 (c.length - 20, y1), (c.length - 2, y1)]),
    ]
    accent = union([a.extrude(zt - z1 + 0.01).translate((0, 0, z1 - 0.01)) for a in accents])
    accent -= union([cyl_z(cx, c.fan_y, z1 - 1, zt + 1, c.pocket_r) for cx in c.fan_x])

    # "RX 590 GME" on the top edge, readable through a case window.
    # Sized to end before the split line between the fans.
    height = 12.0
    label = text_2d("RX 590 GME", height, thin(2.0))
    label = label.extrude(0.8).rotate((-90, 0, 0))       # text up = -z, faces +y
    label = label.translate((12, c.body_y1 - 0.01, z0 + 5.5 + height))

    fans = union([fan(c, cx, thin) for cx in c.fan_x])
    shroud = union([body, pipes, accent, label, fans] + bezels)
    # Full-height notch for the 8-pin power connector (no bridging when printed).
    return shroud - box(210, 104, z0 - 0.02, 234, c.body_y1 + 1, c.thickness + 1)


def pcb_and_backplate(c: Card, thin) -> Manifold:
    bp = box(1, 2, 0, c.length - 2, c.body_y1 - 1, c.backplate)
    # Vents and an engraved label on the underside of the backplate.
    vents = union([rounded_rect(18, thin(2.6), 1.3).extrude(0.7).translate((x, y, -0.01))
                   for x in (160, 182, 204, 226) for y in np.arange(20, 106, 6.0)])
    label = text_2d("RX 590 GME", 10, thin(1.6)).mirror((0, 1)).extrude(0.6)
    label = label.translate((40, 70, -0.01))            # text up = -y when seen from below
    bp -= vents + label

    pcb = box(0, 0, c.pcb_z0 - 0.01, c.pcb_length, c.pcb_height, c.body_z0)
    # PCIe x16 edge: 11.65 mm, key, 71.65 mm; 8.2 mm deep with chamfered corners.
    x0 = 37.0
    finger = CrossSection.batch_boolean([
        polygon([(x0, 0.1), (x0, -7.2), (x0 + 1, -8.2), (x0 + 11.65, -8.2), (x0 + 11.65, 0.1)]),
        polygon([(x0 + 13.55, 0.1), (x0 + 13.55, -8.2), (x0 + 84.2, -8.2), (x0 + 85.2, -7.2),
                 (x0 + 85.2, 0.1)]),
    ], OpType.Add)
    fingers = finger.extrude(c.pcb).translate((0, 0, c.pcb_z0))
    # Fill between backplate and shroud where the PCB does not reach, set back
    # 1.5 mm so the PCB still reads as a separate layer around the edge.
    filler = box(2.5, 3.5, c.pcb_z0 - 0.01, c.length - 3.5, c.body_y1 - 2.5, c.body_z0 + 0.01)

    # 8-pin PCIe power connector standing on the PCB, opening towards +y.
    z_base = c.body_z0
    housing = box(213.5, 103, z_base - 0.01, 233.5, 117, z_base + 9.6)
    holes = union([box(215.6 + 4.2 * i - 1.6, 108, z_base + 1.2 + 4.2 * j,
                       215.6 + 4.2 * i + 1.6, 118, z_base + 1.2 + 4.2 * j + 3.2)
                   for i in range(4) for j in range(2)])
    latch = box(221.5, 113, z_base + 9.59, 225.5, 116.5, z_base + 10.8)
    connector = housing - holes + latch
    return union([bp, pcb, fingers, filler, connector])


def io_bracket(c: Card, thin) -> tuple[Manifold, Manifold, Manifold]:
    t = thin(c.bracket_plate)
    width = c.slot_pitch * c.slots
    top = 108.0
    plate = box(-t, -2, 0, 0.01, top, width)
    tongue = box(-t, -12, 0, 0.01, -1.9, 10)
    tab = box(-11, top - t, 0, 0, top, width)
    for s in range(c.slots):            # screw notches in the fold-over tab
        zc = c.slot_pitch * (s + 0.5)
        notch = CrossSection.circle(2.2, 32).extrude(t + 2).rotate((-90, 0, 0)).translate((-7, top - t - 1, zc))
        notch += box(-12, top - t - 1, zc - 2.2, -7, top + 1, zc + 2.2)
        tab -= notch
    bracket = plate + tongue + tab

    zc = c.body_z0 + 7.5                # outputs sit in the first slot, next to the PCB
    depth = 8.0

    cuts, inserts = [], []
    # DVI-D: trapezoid shell, pin block, two thumbscrew holes.
    dvi_y = 80.0
    dvi = polygon([(-18.6, -5.25), (18.6, -5.25), (18.6, 3.0), (16.4, 5.25), (-16.4, 5.25), (-18.6, 3.0)])
    cuts.append(_port(dvi, dvi_y, zc, depth, t))
    inserts.append(box(2.5, dvi_y - 14, zc - 3, depth + 0.1, dvi_y + 14, zc + 3))
    for dy in (-22.3, 22.3):
        cuts.append(cyl_x(-t - 0.5, 6, dvi_y + dy, zc, 1.6, 24))
    # HDMI: 15.2 x 6.2 with the bottom corners cut.
    hdmi = polygon([(-7.6, 3.1), (7.6, 3.1), (7.6, -0.9), (5.4, -3.1), (-5.4, -3.1), (-7.6, -0.9)])
    cuts.append(_port(hdmi, 46.0, zc, depth, t))
    inserts.append(box(3.0, 46.0 - 5.8, zc - 0.6, depth + 0.1, 46.0 + 5.8, zc + 0.9))
    # DisplayPort: 16.3 x 6.5 with one corner cut.
    dp = polygon([(-8.15, 3.25), (8.15, 3.25), (8.15, -1.35), (6.25, -3.25), (-8.15, -3.25)])
    cuts.append(_port(dp, 22.0, zc, depth, t))
    inserts.append(box(3.0, 22.0 - 6.3, zc - 0.5, depth + 0.1, 22.0 + 6.3, zc + 1.0))
    # Exhaust vents across the second slot.
    vent = rounded_rect(thin(3.2), 12.5, 1.6)
    for y in np.arange(6.0, 104.0, 5.2):
        cuts.append(_port(vent, y, c.slot_pitch * 1.5 + 0.5, 3.0, t))
    return bracket, union(cuts), union(inserts)


def _port(cs: CrossSection, y: float, z: float, depth: float, t: float) -> Manifold:
    """Extrude an outline drawn in the (y, z) plane along +x, through the
    bracket and ``depth`` mm into the card."""
    solid = cs.extrude(depth + t + 0.5)               # along +z, outline in (x=y, y=z)
    solid = solid.transform(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))
    return solid.translate((-t - 0.5, y, z))


def build(c: Card = Card(), scale: float = 1.0, min_feature: float = 0.8) -> Manifold:
    """The whole card as one watertight solid, scaled by ``scale``."""
    def thin(v: float) -> float:          # keep thin walls printable after scaling
        return max(v, min_feature / scale)

    body = shroud_and_heatsink(c, thin) + pcb_and_backplate(c, thin)
    bracket, port_cuts, port_inserts = io_bracket(c, thin)
    card = (body + bracket) - port_cuts + port_inserts
    return card.scale((scale, scale, scale)) if scale != 1 else card


def split(card: Manifold, c: Card = Card(), at_x: float | None = None,
          pin_d: float = 4.0, pin_len: float = 18.0, clearance: float = 0.2):
    """Cut the full-size card in two between the fans, with holes for two
    alignment pins. Returns (bracket half, rear half, pins)."""
    at_x = (c.fan_x[0] + c.fan_x[1]) / 2 if at_x is None else at_x
    rear, front = card.split_by_plane((1, 0, 0), at_x)
    spots = [(c.fan_y - 22, c.body_z0 + 14), (c.fan_y + 22, c.body_z0 + 14)]
    holes = union([cyl_x(at_x - pin_len / 2 - 1, at_x + pin_len / 2 + 1, y, z, pin_d / 2 + clearance, 32)
                   for y, z in spots])
    front, rear = front - holes, rear - holes
    pins = union([Manifold.cylinder(pin_len, pin_d / 2, circular_segments=32).translate((i * 8.0, 0, 0))
                  for i in range(len(spots))])
    # Stand the rear half on the bed at its cut face's origin for convenience.
    return front, rear.translate((-at_x, 0, 0)), pins


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_stl(path: Path, solid: Manifold, name: str) -> tuple[int, float]:
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    tris = np.asarray(mesh.tri_verts, dtype=np.int64)
    a, b, cc = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(b - a, cc - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    records["n"] = normals
    records["v"] = np.stack([a, b, cc], axis=1)
    header = name.encode()[:80].ljust(80, b" ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + struct.pack("<I", len(tris)) + records.tobytes())
    return len(tris), solid.volume()


def describe(solid: Manifold) -> str:
    lo = solid.bounding_box()
    size = [lo[3] - lo[0], lo[4] - lo[1], lo[5] - lo[2]]
    return f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--out", default=str(Path(__file__).with_name("stl")))
    args = parser.parse_args()
    out = Path(args.out)
    c = Card()

    full = build(c)
    outputs = {"rx590gme_1to1.stl": full}
    front, rear, pins = split(full, c)
    outputs["rx590gme_1to1_part1_bracket.stl"] = front
    outputs["rx590gme_1to1_part2_rear.stl"] = rear
    outputs["rx590gme_1to1_pins_x2.stl"] = pins
    outputs["rx590gme_1to2.stl"] = build(c, scale=0.5)

    for filename, solid in outputs.items():
        if solid.status().name != "NoError":
            raise SystemExit(f"{filename}: manifold error {solid.status()}")
        pieces = len(solid.decompose())
        tris, volume = write_stl(out / filename, solid, f"RX 590 GME model - {filename}")
        print(f"{filename:36s} {describe(solid):26s} {tris:7,d} triangles  "
              f"{volume / 1000:6.1f} cm3  {pieces} piece(s)")


if __name__ == "__main__":
    main()
