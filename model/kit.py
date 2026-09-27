"""The RX 590 GME model as a paint-and-assemble kit.

Every part is printed and painted in a single colour (red, black, white or
yellow), then assembled with printed pegs and a little glue. The kit stacks
up like the real card:

    backplate  ->  PCB  ->  heatsink (yellow grille)  ->  shroud  ->  fans,
    bezels, accents and labels on top; I/O bracket, ports and 8-pin at the ends.

The backplate's pins pass through the PCB into the heatsink, and the shroud's
pegs drop into the heatsink, so the stack self-aligns. Each fan rotor turns on
an axle that is part of the heatsink; its hub cap is glued to the axle tip,
not to the rotor, so the fans still spin once the card is assembled.

Run ``python kit.py`` to write one STL per part (in print orientation) into
``kit/``. Add ``--split`` to also write the four long parts in two pieces
each, for beds smaller than 256 mm.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold, OpType

from rx590gme_model import (box, cyl_x, cyl_z, polygon, rounded_rect, text_2d, text_width, union,
                            write_stl)

RED, BLACK, WHITE, YELLOW = "red", "black", "white", "yellow"
JOIN_MITER = JoinType.Miter
PALETTE = {RED: (200, 32, 38), BLACK: (46, 46, 52), WHITE: (236, 236, 236), YELLOW: (236, 186, 36)}

FIT = 0.2          # radial / side clearance between mating parts
RECESS = 0.6       # depth of the pockets that locate the decals, bezels and accents
INLAY = 1.4        # thickness of those parts (they stand 0.8 mm proud)

# Stack heights (z, mm), backplate on the build plate.
BP_TOP = 2.0
PCB_TOP = BP_TOP + 1.6
HS_TOP = 24.8                  # heatsink top = fan pocket floor
SHROUD_TOP = 36.8
LENGTH = 255.0
Y0, Y1 = 3.0, 121.0
FAN_X = (70.0, 186.0)
FAN_Y = 62.0
FAN_HOLE_R = 46.0
NOTCH = (208.0, 101.0, 236.0)  # x0, y0, x1 of the 8-pin power connector cut-out
SKIRT_X1 = 140.0               # label skirt covers the top edge up to here

BACKPLATE_PINS = [(30, 20), (30, 100), (110, 62), (146, 62), (200, 20), (200, 95), (240, 62)]
SHROUD_PEGS = [(25, 40), (25, 85), (120, 30), (120, 94), (136, 30), (136, 94), (242, 40), (242, 85)]
PORT_PEGS = [(9, 34), (9, 60)]
POWER_PEGS = [(217, 105.5), (230, 105.5)]

# I/O layout
PORT_Z = PCB_TOP + 6.7
DVI_Y, HDMI_Y, DP_Y = 80.0, 46.0, 22.0
DVI_SCREW_DY = 22.3
BRACKET_T = 1.2
BRACKET_W = 2 * 20.32
BRACKET_TOP = 108.0


@dataclass
class Part:
    number: int
    name: str
    colour: str
    assembled: Manifold                    # where it sits in the finished card
    to_print: Manifold | None = None       # print orientation; defaults to assembled, dropped to z=0
    note: str = ""
    split: tuple = field(default=())       # optional (piece_a, piece_b) for small beds

    def printable(self) -> Manifold:
        solid = self.to_print if self.to_print is not None else self.assembled
        return on_bed(solid)


def on_bed(solid: Manifold) -> Manifold:
    b = solid.bounding_box()
    return solid.translate((-b[0], -b[1], -b[2]))


def flip_over(solid: Manifold) -> Manifold:
    """Turn a part upside down (180 degrees about x)."""
    return solid.rotate((180, 0, 0))


def peg(x, y, z0, z1, r=1.5) -> Manifold:
    return cyl_z(x, y, z0, z1, r, 24)


def hole(x, y, z0, z1, r=1.5 + FIT) -> Manifold:
    return cyl_z(x, y, z0, z1, r, 24)


def label_2d(text: str, height: float, stroke: float) -> CrossSection:
    """Text sitting on an underline bar, so the whole label is one piece."""
    width = text_width(text, height)
    bar = CrossSection.square((width + stroke, 1.6)).translate((-stroke / 2, -stroke / 2 - 1.3))
    return text_2d(text, height, stroke) + bar


def yz_outline(cs: CrossSection, x0: float, x1: float, y: float, z: float) -> Manifold:
    """Extrude an outline drawn in the (y, z) plane along x, from x0 to x1."""
    solid = cs.extrude(x1 - x0)
    solid = solid.transform(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))
    return solid.translate((x0, y, z))


DVI = polygon([(-18.6, -5.25), (18.6, -5.25), (18.6, 3.0), (16.4, 5.25), (-16.4, 5.25), (-18.6, 3.0)])
HDMI = polygon([(-7.6, 3.1), (7.6, 3.1), (7.6, -0.9), (5.4, -3.1), (-5.4, -3.1), (-7.6, -0.9)])
DP = polygon([(-8.15, 3.25), (8.15, 3.25), (8.15, -1.35), (6.25, -3.25), (-8.15, -3.25)])
PORTS = [(DVI, DVI_Y), (HDMI, HDMI_Y), (DP, DP_Y)]


def accent_shapes() -> list[CrossSection]:
    mid = (FAN_X[0] + FAN_X[1]) / 2
    y0, y1, ym = Y0 + 6, Y1 - 6, FAN_Y
    return [
        polygon([(2, y0), (20, y0), (13, ym), (20, y1), (2, y1)]),
        polygon([(mid - 6, y0), (mid + 6, y0), (mid + 3, ym), (mid + 6, y1), (mid - 6, y1), (mid - 3, ym)]),
        polygon([(LENGTH - 2, y0), (LENGTH - 20, y0), (LENGTH - 13, ym), (LENGTH - 20, y1), (LENGTH - 2, y1)]),
    ]


def annulus(r0, r1, segments=128) -> CrossSection:
    return CrossSection.circle(r1, segments) - CrossSection.circle(r0, segments)


# Top-edge label: reads correctly through a case window (fans down, bracket left).
TOP_LABEL = label_2d("RX 590 GME", 12.0, 2.0)
TOP_LABEL_X, TOP_LABEL_Z = 12.0, PCB_TOP + 4.4 + 12.0     # text top edge sits at PCB_TOP + 4.4
# Backplate label: reads correctly looking down on the installed card.
BP_LABEL = label_2d("RX 590 GME", 10.0, 1.6)
BP_LABEL_X, BP_LABEL_Y = 40.0, 70.0


def top_label_placed(cs: CrossSection, thickness: float, y_face: float) -> Manifold:
    # u -> +x, v -> -z, extrusion -> +y (outwards from the top edge)
    return cs.extrude(thickness).rotate((-90, 0, 0)).translate((TOP_LABEL_X, y_face, TOP_LABEL_Z))


def bp_label_placed(cs: CrossSection, thickness: float, z0: float) -> Manifold:
    # seen from below: u -> +x, v -> -y
    return cs.mirror((0, 1)).extrude(thickness).translate((BP_LABEL_X, BP_LABEL_Y, z0))


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def backplate() -> Manifold:
    plate = box(1, 2, 0, LENGTH - 2, Y1 - 1, BP_TOP)
    vents = union([rounded_rect(18, 2.6, 1.3).extrude(0.8).translate((x, y, -0.01))
                   for x in (160, 182, 204, 226) for y in np.arange(20, 106, 6.0)])
    recess = bp_label_placed(BP_LABEL.offset(FIT, circular_segments=16), RECESS + 0.01, -0.01)
    pins = union([peg(x, y, BP_TOP - 0.01, PCB_TOP + 2.4) for x, y in BACKPLATE_PINS])
    return plate - vents - recess + pins


def backplate_label() -> tuple[Manifold, Manifold]:
    assembled = bp_label_placed(BP_LABEL, INLAY, RECESS - INLAY)
    return assembled, BP_LABEL.extrude(INLAY)


def pcb() -> Manifold:
    board = box(0, 0, BP_TOP, 249, 118, PCB_TOP)
    x0 = 37.0     # PCIe x16 edge: 11.65 mm, key, 71.65 mm
    fingers = CrossSection.batch_boolean([
        polygon([(x0, 0.1), (x0, -7.2), (x0 + 1, -8.2), (x0 + 11.65, -8.2), (x0 + 11.65, 0.1)]),
        polygon([(x0 + 13.55, 0.1), (x0 + 13.55, -8.2), (x0 + 84.2, -8.2), (x0 + 85.2, -7.2), (x0 + 85.2, 0.1)]),
    ], OpType.Add).extrude(PCB_TOP - BP_TOP).translate((0, 0, BP_TOP))
    holes = union([hole(x, y, BP_TOP - 1, PCB_TOP + 1) for x, y in BACKPLATE_PINS])
    pegs = union([peg(x, y, PCB_TOP - 0.01, PCB_TOP + 2.4) for x, y in PORT_PEGS + POWER_PEGS])
    return board + fingers - holes + pegs


def heatsink() -> Manifold:
    x0, x1 = 13.2, 248.8
    block = box(x0, Y0, PCB_TOP, x1, Y1, HS_TOP)
    cuts = [
        box(x0 - 1, 117.8, PCB_TOP - 1, SKIRT_X1 + FIT, Y1 + 1, HS_TOP + 1),     # behind the label skirt
        box(NOTCH[0], NOTCH[1], PCB_TOP - 1, NOTCH[2], Y1 + 1, HS_TOP + 1),        # 8-pin connector
    ]
    # Fins: full-height slots along both long edges.
    cuts += [box(x, Y0 - 1, PCB_TOP - 1, x + 1.4, Y0 + 7, HS_TOP + 1) for x in np.arange(20.0, 236.0, 3.0)]
    cuts += [box(x, Y1 - 7, PCB_TOP - 1, x + 1.4, Y1 + 1, HS_TOP + 1) for x in np.arange(143.0, 206.0, 3.0)]
    # Grille under each fan.
    for cx in FAN_X:
        grooves = union([box(x - 0.6, FAN_Y - FAN_HOLE_R, HS_TOP - 1.0, x + 0.6, FAN_Y + FAN_HOLE_R, HS_TOP + 1)
                         for x in np.arange(cx - FAN_HOLE_R, cx + FAN_HOLE_R, 3.0)])
        cuts.append((grooves ^ cyl_z(cx, FAN_Y, HS_TOP - 2, HS_TOP + 2, FAN_HOLE_R - 2.5)) -
                    cyl_z(cx, FAN_Y, HS_TOP - 2, HS_TOP + 2, 6.5))
    cuts += [hole(x, y, PCB_TOP - 1, PCB_TOP + 2.6) for x, y in BACKPLATE_PINS]
    cuts += [hole(x, y, HS_TOP - 3.0, HS_TOP + 1) for x, y in SHROUD_PEGS]
    sink = block - union(cuts)

    pipes = union([cyl_x(SKIRT_X1 + 2, NOTCH[0] - 1, Y1 - 4.2, z, 2.8) for z in (7.5, 14.0, 20.5)])
    axles = union([cyl_z(cx, FAN_Y, HS_TOP - 0.01, HS_TOP + 1.0, 5.0, 48) +
                   cyl_z(cx, FAN_Y, HS_TOP + 0.99, 35.3, 2.0, 32) for cx in FAN_X])
    return sink + pipes + axles


def shroud() -> Manifold:
    plate = box(0, Y0, HS_TOP, LENGTH, Y1, SHROUD_TOP)
    skirt = box(6.0, 118.0, PCB_TOP, SKIRT_X1, Y1, HS_TOP + 0.01)
    front = box(0, Y0, 17.2, 6.0, Y1, HS_TOP + 0.01)
    rear = box(249.0, Y0, BP_TOP, LENGTH, Y1, HS_TOP + 0.01)
    body = union([plate, skirt, front, rear])
    cuts = [cyl_z(cx, FAN_Y, HS_TOP - 1, SHROUD_TOP + 1, FAN_HOLE_R, 128) for cx in FAN_X]
    cuts.append(box(NOTCH[0], NOTCH[1], BP_TOP - 1, NOTCH[2], Y1 + 1, SHROUD_TOP + 1))
    top = SHROUD_TOP - RECESS
    cuts += [annulus(FAN_HOLE_R - 1, 49.0 + FIT).extrude(RECESS + 1).translate((cx, FAN_Y, top)) for cx in FAN_X]
    cuts += [a.offset(FIT, JOIN_MITER).extrude(RECESS + 1).translate((0, 0, top)) for a in accent_shapes()]
    cuts.append(top_label_placed(TOP_LABEL.offset(FIT, circular_segments=16), RECESS + 0.01, Y1 - RECESS))
    body -= union(cuts)
    pegs = union([peg(x, y, HS_TOP - 2.6, HS_TOP + 0.01) for x, y in SHROUD_PEGS])
    return body + pegs


def fan_rotor(cx: float) -> Manifold:
    bottom, top = HS_TOP + 1.0, 35.8
    r_hub, r_tip = 17.0, 44.0
    hub = cyl_z(cx, FAN_Y, bottom, top, r_hub)
    hub -= cyl_z(cx, FAN_Y, top - 1.6, top + 1, 14.0)             # seat for the hub cap
    radii = np.linspace(r_hub - 2, r_tip, 24)
    sweep = math.radians(38)
    left, right = [], []
    for r in radii:
        t = (r - radii[0]) / (radii[-1] - radii[0])
        a = sweep * t * t
        half = (2.4 - 0.7 * t) / 2 / r
        left.append((r * math.cos(a + half), r * math.sin(a + half)))
        right.append((r * math.cos(a - half), r * math.sin(a - half)))
    blade = polygon(left + right[::-1])
    blades = CrossSection.batch_boolean([blade.rotate(360 * k / 9) for k in range(9)], OpType.Add)
    blades = blades.extrude(top - 1.0 - bottom, n_divisions=8, twist_degrees=-18).translate((cx, FAN_Y, bottom))
    rotor = hub + blades
    return rotor - cyl_z(cx, FAN_Y, bottom - 1, top + 1, 2.0 + 0.35, 32)   # spins on the axle


def hub_cap(cx: float) -> Manifold:
    cap = cyl_z(cx, FAN_Y, 34.4, 35.8, 14.0 - FIT - 0.2, 96)
    cap -= cyl_z(cx, FAN_Y, 34.39, 35.4, 2.05, 32)                  # glued onto the axle tip
    cap -= annulus(9.0, 10.0, 96).extrude(0.5).translate((cx, FAN_Y, 35.31))
    return cap


def bezel(cx: float) -> Manifold:
    return annulus(FAN_HOLE_R - 0.8, 49.0, 128).extrude(INLAY).translate((cx, FAN_Y, SHROUD_TOP - RECESS))


def accent(shape: CrossSection) -> Manifold:
    return shape.extrude(INLAY).translate((0, 0, SHROUD_TOP - RECESS))


def top_label() -> tuple[Manifold, Manifold]:
    return top_label_placed(TOP_LABEL, INLAY, Y1 - RECESS), TOP_LABEL.extrude(INLAY)


def io_bracket() -> Manifold:
    t = BRACKET_T
    plate = box(-t, -2, 0, 0, BRACKET_TOP, BRACKET_W)
    tongue = box(-t, -12, 0, 0, -1.9, 10)
    tab = box(-11, BRACKET_TOP - t, 0, -t + 0.01, BRACKET_TOP, BRACKET_W)
    for s in range(2):
        zc = 20.32 * (s + 0.5)
        notch = CrossSection.circle(2.2, 32).extrude(t + 2).rotate((-90, 0, 0)).translate((-7, BRACKET_TOP - t - 1, zc))
        notch += box(-12, BRACKET_TOP - t - 1, zc - 2.2, -7, BRACKET_TOP + 1, zc + 2.2)
        tab -= notch
    bracket = plate + tongue + tab
    cuts = [yz_outline(cs.offset(FIT, JOIN_MITER), -t - 1, 1, y, PORT_Z) for cs, y in PORTS]
    cuts += [cyl_x(-t - 1, 1, DVI_Y + dy, PORT_Z, 2.2, 24) for dy in (-DVI_SCREW_DY, DVI_SCREW_DY)]
    vent = rounded_rect(3.2, 12.5, 1.6)
    cuts += [yz_outline(vent, -t - 1, 1, y, 20.32 * 1.5 + 0.5) for y in np.arange(6.0, 104.0, 5.2)]
    return bracket - union(cuts)


def port_block() -> Manifold:
    body = box(0, 6, PCB_TOP, 12, 106, 17.0)
    shells = union([yz_outline(cs, -BRACKET_T, 0.01, y, PORT_Z) for cs, y in PORTS])
    screws = union([cyl_x(-BRACKET_T, 0.01, DVI_Y + dy, PORT_Z, 2.0, 24) for dy in (-DVI_SCREW_DY, DVI_SCREW_DY)])
    block = body + shells + screws
    cuts = [yz_outline(DVI.offset(-0.8, JOIN_MITER), -BRACKET_T - 1, 6.0, DVI_Y, PORT_Z)]
    for cs, y in PORTS[1:]:
        cuts.append(yz_outline(cs.offset(-0.7, JOIN_MITER), -BRACKET_T - 1, 6.0, y, PORT_Z))
    cuts += [cyl_x(-BRACKET_T - 1, 6.0, DVI_Y + dy, PORT_Z, 1.1, 16) for dy in (-DVI_SCREW_DY, DVI_SCREW_DY)]
    cuts += [hole(x, y, PCB_TOP - 1, PCB_TOP + 2.6) for x, y in PORT_PEGS]
    block -= union(cuts)
    # Connector tongues stand on the cavity floor so they print without support.
    tongues = union([box(-0.6, HDMI_Y - 5.2, PORT_Z - 2.5, 6.01, HDMI_Y + 5.2, PORT_Z + 0.9),
                     box(-0.6, DP_Y - 5.8, PORT_Z - 2.65, 6.01, DP_Y + 5.8, PORT_Z + 1.0)])
    return block + tongues


def dvi_insert() -> Manifold:
    flange = yz_outline(DVI.offset(-0.8 - FIT, JOIN_MITER), 5.0, 6.0, DVI_Y, PORT_Z)
    pins = box(-0.9, DVI_Y - 13, PORT_Z - 3, 5.01, DVI_Y + 13, PORT_Z + 3)
    holes = union([box(-1.5, DVI_Y - 9.1 + 2.6 * i - 0.6, PORT_Z - 2.2 + 2.2 * j - 0.6,
                       2.0, DVI_Y - 9.1 + 2.6 * i + 0.6, PORT_Z - 2.2 + 2.2 * j + 0.6)
                   for i in range(8) for j in range(3)])
    return flange + pins - holes


def power_connector() -> Manifold:
    housing = box(213.5, 103, PCB_TOP, 233.5, 117, PCB_TOP + 9.6)
    holes = union([box(215.6 + 4.2 * i - 1.6, 108, PCB_TOP + 1.2 + 4.2 * j,
                       215.6 + 4.2 * i + 1.6, 118, PCB_TOP + 1.2 + 4.2 * j + 3.2)
                   for i in range(4) for j in range(2)])
    latch = box(221.5, 113, PCB_TOP + 9.59, 225.5, 116.5, PCB_TOP + 10.8)
    sockets = union([hole(x, y, PCB_TOP - 1, PCB_TOP + 2.6) for x, y in POWER_PEGS])
    return housing - holes + latch - sockets


def lay_on_inner_face(solid: Manifold) -> Manifold:
    """Rotate so the face at the card's front (+x) lies on the bed."""
    return solid.rotate((0, 90, 0))


def split_x(solid: Manifold, at: float) -> tuple[Manifold, Manifold]:
    right, left = solid.split_by_plane((1, 0, 0), at)
    return left, right


def build_kit() -> list[Part]:
    bp_label, bp_label_print = backplate_label()
    edge_label, edge_label_print = top_label()
    bp, board, sink, cover = backplate(), pcb(), heatsink(), shroud()
    bracket = io_bracket()
    insert = dvi_insert()

    parts = [
        Part(1, "backplate", BLACK, bp, note="pins face up", split=split_x(bp, 90.0)),
        Part(2, "backplate_label", WHITE, bp_label, bp_label_print, "glue into the recess under the backplate"),
        Part(3, "pcb", BLACK, board, note="optionally paint the PCIe fingers yellow/gold",
             split=split_x(board, 150.0)),
        Part(4, "heatsink", YELLOW, sink, note="the grille, fins, heatpipes and fan axles",
             split=split_x(sink, 128.0)),
        Part(5, "shroud", BLACK, cover, flip_over(cover), "printed upside down, top face on the bed",
             split=tuple(flip_over(p) for p in split_x(cover, 128.0))),
        Part(6, "fan_rotor_1", BLACK, fan_rotor(FAN_X[0])),
        Part(7, "fan_rotor_2", BLACK, fan_rotor(FAN_X[1])),
        Part(8, "hub_cap_1", RED, hub_cap(FAN_X[0]), flip_over(hub_cap(FAN_X[0])), "glue to the axle tip only"),
        Part(9, "hub_cap_2", RED, hub_cap(FAN_X[1]), flip_over(hub_cap(FAN_X[1])), "glue to the axle tip only"),
        Part(10, "fan_bezel_1", RED, bezel(FAN_X[0])),
        Part(11, "fan_bezel_2", RED, bezel(FAN_X[1])),
    ]
    for i, (shape, name) in enumerate(zip(accent_shapes(), ("front", "middle", "rear"))):
        parts.append(Part(12 + i, f"accent_{name}", RED, accent(shape)))
    parts += [
        Part(15, "edge_label", WHITE, edge_label, edge_label_print, "glue into the recess on the top edge"),
        Part(16, "io_bracket", BLACK, bracket, lay_on_inner_face(bracket), "inner face on the bed"),
        Part(17, "port_block", BLACK, port_block(), note="DVI, HDMI and DisplayPort connectors"),
        Part(18, "dvi_insert", WHITE, insert, lay_on_inner_face(insert), "push into the DVI shell"),
        Part(19, "power_connector", BLACK, power_connector(), note="8-pin PCIe power"),
    ]
    return parts


ASSEMBLY = """\
1. Backplate (1) on the table, pins up. Glue the white label (2) into the recess on its underside.
2. Drop the PCB (3) over the backplate pins.
3. Press the heatsink (4) onto the same pins; glue.
4. Push the port block (17) and 8-pin connector (19) onto the PCB pegs. Slide the
   white DVI insert (18) into the DVI shell.
5. Lower the shroud (5) over the heatsink; its pegs locate it. Glue.
6. Slide the fan rotors (6, 7) onto the axles. Put a drop of glue on each axle
   tip - not on the rotor - and press a hub cap (8, 9) on. The fans still spin.
7. Glue the red bezels (10, 11) and accents (12-14) into their recesses on the
   shroud and the white edge label (15) into the recess on the top edge.
8. Fit the I/O bracket (16) over the port shells and glue it to the port block
   and the front of the shroud.
"""


# How far each part moves in the exploded view (mm).
EXPLODE = {
    "backplate_label": (0, 0, -30), "pcb": (0, 0, 14), "heatsink": (0, 0, 34), "shroud": (0, 0, 62),
    "fan_rotor": (0, 0, 88), "hub_cap": (0, 0, 108), "fan_bezel": (0, 0, 76), "accent": (0, 0, 76),
    "edge_label": (0, 40, 62), "io_bracket": (-55, 0, 14), "port_block": (-22, 0, 14),
    "dvi_insert": (-40, 0, 14), "power_connector": (0, 30, 14),
}


def triangles(solid: Manifold) -> np.ndarray:
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    return verts[np.asarray(mesh.tri_verts, dtype=np.int64)]


def render_previews(parts: list[Part], folder: Path) -> None:
    from render_preview import render_parts, write_png

    def scene(exploded: bool, flip: bool = False):
        out = []
        for part in parts:
            solid = part.assembled
            if exploded:
                key = next((k for k in EXPLODE if part.name.startswith(k)), None)
                if key:
                    solid = solid.translate(EXPLODE[key])
            t = triangles(solid)
            if flip:                          # fans down, as installed in a case
                t = t.copy()
                t[..., 0] *= -1
                t[..., 2] *= -1
            out.append((t, PALETTE[part.colour]))
        return out

    folder.mkdir(parents=True, exist_ok=True)
    write_png(folder / "kit_assembled.png", render_parts(scene(False), 960, 560, 215, 35))
    write_png(folder / "kit_exploded.png", render_parts(scene(True), 960, 760, 215, 28))
    write_png(folder / "kit_installed.png", render_parts(scene(False, True), 960, 420, 180, 8))
    write_png(folder / "kit_io.png", render_parts(scene(False), 720, 560, -90, 20))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the paint-and-assemble RX 590 GME kit as STL files.")
    parser.add_argument("-o", "--out", default=str(Path(__file__).with_name("kit")))
    parser.add_argument("--split", action="store_true", help="also write split versions of the long parts")
    parser.add_argument("--previews", metavar="DIR", help="also render colour preview images into DIR")
    args = parser.parse_args()
    out = Path(args.out)
    parts = build_kit()
    for part in parts:
        solid = part.printable()
        pieces = [("", solid)]
        if args.split and part.split:
            pieces += [(f"_{s}", on_bed(p)) for s, p in zip("ab", part.split)]
        for suffix, piece in pieces:
            name = f"{part.number:02d}_{part.name}{suffix}_{part.colour}.stl"
            folder = out / ("split" if suffix else "")
            write_stl(folder / name, piece, f"RX 590 GME kit - {name}")
            b = piece.bounding_box()
            print(f"{name:40s} {b[3] - b[0]:6.1f} x {b[4] - b[1]:6.1f} x {b[5] - b[2]:5.1f} mm  "
                  f"{len(piece.decompose())} piece(s)")
    if args.previews:
        render_previews(parts, Path(args.previews))


if __name__ == "__main__":
    main()
