"""Lay the kit out on print plates and write them as 3MF files.

3MF files carry their units (millimetres), so a slicer cannot mistake them
for inches the way it can with STL. Each plate holds parts of one colour,
arranged for a square bed (220 x 220 mm by default: FlashForge Adventurer 5M,
Bambu A1, Ender 3 V3 and similar). Parts are rotated by 90 degrees where that
packs better.

    python plates.py                 # 220 mm bed, writes kit/plates/*.3mf
    python plates.py --bed 256       # e.g. Bambu P1S / X1
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from manifold3d import Manifold

import kit

GAP = 6.0       # space between parts
MARGIN = 8.0    # keep clear of the bed edge

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


@dataclass
class Placed:
    name: str
    solid: Manifold


def print_pieces(split: bool) -> list[tuple[str, str, Manifold]]:
    """(name, colour, solid in print orientation) for every piece to print."""
    pieces = []
    for part in kit.build_kit():
        base = f"{part.number:02d}_{part.name}"
        if split and part.split:
            pieces += [(f"{base}_{s}", part.colour, kit.on_bed(p)) for s, p in zip("ab", part.split)]
        else:
            pieces.append((base, part.colour, part.printable()))
    return pieces


def size_xy(solid: Manifold) -> tuple[float, float]:
    b = solid.bounding_box()
    return b[3] - b[0], b[4] - b[1]


def pack(pieces: list[tuple[str, Manifold]], bed: float) -> list[list[Placed]]:
    """Shelf packing: tallest pieces first, rows filled left to right."""
    usable = bed - 2 * MARGIN
    items = []
    for name, solid in pieces:
        w, h = size_xy(solid)
        if min(w, h) > usable:
            raise ValueError(f"{name} ({w:.0f} x {h:.0f} mm) does not fit a {bed:.0f} mm bed")
        # Long side along x makes flatter shelves, unless it only fits the other way.
        if (h > w and h <= usable) or w > usable:
            solid, (w, h) = solid.rotate((0, 0, 90)), (h, w)
        items.append((name, kit.on_bed(solid), w, h))
    items.sort(key=lambda it: -it[3])

    plates: list[dict] = []
    for name, solid, w, h in items:
        for plate in plates:
            shelf = plate["shelves"][-1]
            if shelf["x"] + w <= usable and h <= shelf["h"]:
                break
            if plate["y"] + h <= usable:
                plate["shelves"].append({"y": plate["y"], "x": 0.0, "h": h})
                plate["y"] += h + GAP
                break
        else:
            plate = {"shelves": [{"y": 0.0, "x": 0.0, "h": h}], "y": h + GAP, "items": []}
            plates.append(plate)
        shelf = plate["shelves"][-1]
        x, y = MARGIN + shelf["x"], MARGIN + shelf["y"]
        plate["items"].append(Placed(name, solid.translate((x, y, 0))))
        shelf["x"] += w + GAP
    return [p["items"] for p in plates]


def write_3mf(path: Path, items: list[Placed]) -> None:
    objects, build = [], []
    for i, item in enumerate(items, start=1):
        mesh = item.solid.to_mesh()
        verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
        tris = np.asarray(mesh.tri_verts, dtype=np.int64)
        v = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in verts)
        t = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in tris)
        objects.append(f'<object id="{i}" name="{item.name}" type="model"><mesh>'
                       f'<vertices>{v}</vertices><triangles>{t}</triangles></mesh></object>')
        build.append(f'<item objectid="{i}"/>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources>{"".join(objects)}</resources><build>{"".join(build)}</build></model>')
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in (("[Content_Types].xml", CONTENT_TYPES), ("_rels/.rels", RELS),
                           ("3D/3dmodel.model", model)):
            # Fixed timestamp so regenerating identical plates gives identical files.
            info = zipfile.ZipInfo(name, date_time=(2020, 3, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)


def render_sheet(plates: list[tuple[str, str, list[Placed]]], bed: float, path: Path, tile: int = 360) -> None:
    """Top-down picture of every plate, in its filament colour, on a grey bed."""
    from render_preview import render_parts, write_png
    cols = 4
    rows = -(-len(plates) // cols)
    sheet = np.full((rows * tile, cols * tile, 3), 246, dtype=np.uint8)
    bed_plate = kit.box(0, 0, -1.0, bed, bed, -0.5)
    for i, (_, colour, items) in enumerate(plates):
        scene = [(kit_triangles(bed_plate), (205, 205, 210))]
        scene += [(kit_triangles(item.solid), kit.PALETTE[colour]) for item in items]
        img = render_parts(scene, tile, tile, 0, 89.9)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = img
    write_png(path, sheet)


def kit_triangles(solid: Manifold) -> np.ndarray:
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    return verts[np.asarray(mesh.tri_verts, dtype=np.int64)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bed", type=float, default=220.0, help="square bed size in mm (default 220)")
    parser.add_argument("-o", "--out", default=str(Path(__file__).with_name("kit") / "plates"))
    parser.add_argument("--preview", metavar="PNG", help="also render all plates into one image")
    args = parser.parse_args()

    split = args.bed - 2 * MARGIN < 255      # the long parts need splitting on small beds
    pieces = print_pieces(split)
    out = Path(args.out)
    written = []
    for colour in (kit.BLACK, kit.RED, kit.WHITE, kit.YELLOW):
        group = [(n, s) for n, c, s in pieces if c == colour]
        for plate in pack(group, args.bed):
            path = out / f"plate_{len(written) + 1:02d}_{colour}.3mf"
            write_3mf(path, plate)
            written.append((path.name, colour, plate))
            print(f"{path.name:26s} " + ", ".join(p.name for p in plate))
    if args.preview:
        render_sheet(written, args.bed, Path(args.preview))


if __name__ == "__main__":
    main()
