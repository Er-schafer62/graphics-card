"""Command line front end: ``rx590gme <command>`` or ``python -m rx590gme <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import demos
from .device import RX590GME, occupancy
from .isa import AssemblerError, assemble
from .kernels import builtin_kernel, builtin_kernel_names
from .specs import spec_sheet


def _gpu(args: argparse.Namespace) -> RX590GME:
    return RX590GME(clock_mhz=args.clock)


def cmd_info(args: argparse.Namespace) -> int:
    print(spec_sheet(clock_mhz=args.clock))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    gpu = _gpu(args)
    rng = np.random.default_rng(590)
    x = rng.standard_normal(args.n, dtype=np.float32)
    y = rng.standard_normal(args.n, dtype=np.float32)
    result, stats = demos.saxpy(gpu, 2.5, x, y)
    expected = np.float32(2.5) * x + y
    ok = np.allclose(result, expected, rtol=1e-6, atol=1e-6)
    print(f"SAXPY on {args.n:,} elements: {'PASS' if ok else 'FAIL'}\n")
    print(stats)
    return 0 if ok else 1


def cmd_mandelbrot(args: argparse.Namespace) -> int:
    gpu = _gpu(args)
    image, stats = demos.mandelbrot(gpu, args.width, args.height, args.iterations,
                                    center=(args.cx, args.cy), zoom=args.zoom)
    path = demos.save_image(args.output, image)
    print(stats)
    print(f"\nwrote {path}")
    return 0


def cmd_triangle(args: argparse.Namespace) -> int:
    gpu = _gpu(args)
    image, stats = demos.triangle(gpu, args.width, args.height)
    path = demos.save_image(args.output, image)
    print(stats)
    print(f"\nwrote {path}")
    return 0


def cmd_asm(args: argparse.Namespace) -> int:
    if args.source in builtin_kernel_names():
        source, label = builtin_kernel(args.source), f"built-in kernel '{args.source}'"
    else:
        path = Path(args.source)
        if not path.is_file():
            print(f"error: {args.source} is neither a file nor a built-in kernel "
                  f"({', '.join(builtin_kernel_names())})", file=sys.stderr)
            return 2
        source, label = path.read_text(), str(path)
    try:
        program = assemble(source)
    except AssemblerError as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return 1
    waves = -(-args.block // 64)
    occ = occupancy(program, waves)
    print(f"{label}: kernel '{program.name}', {len(program.instructions)} instructions")
    print(f"  VGPRs {program.vgpr_count}, SGPRs {program.sgpr_count}, LDS {program.lds_bytes} bytes")
    print(f"  occupancy with {args.block}-thread workgroups: {occ.waves_per_simd}/10 waves per SIMD "
          f"(limited by {occ.limited_by})")
    if args.disassemble:
        print()
        print(program.disassemble())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rx590gme",
                                     description="Software replica of the AMD Radeon RX 590 GME")
    parser.add_argument("--clock", type=float, default=None,
                        help="shader clock in MHz (default: 1380 boost; base is 1257)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print the card's specification").set_defaults(fn=cmd_info)

    p = sub.add_parser("bench", help="run and verify a SAXPY benchmark")
    p.add_argument("-n", type=int, default=1 << 18, help="number of elements")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("mandelbrot", help="render the Mandelbrot set to a PNG")
    p.add_argument("-o", "--output", default="mandelbrot.png")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--iterations", type=int, default=64)
    p.add_argument("--cx", type=float, default=-0.6)
    p.add_argument("--cy", type=float, default=0.0)
    p.add_argument("--zoom", type=float, default=1.0)
    p.set_defaults(fn=cmd_mandelbrot)

    p = sub.add_parser("triangle", help="rasterise a shaded triangle to a PNG")
    p.add_argument("-o", "--output", default="triangle.png")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.set_defaults(fn=cmd_triangle)

    p = sub.add_parser("asm", help="assemble a kernel and report its resource usage")
    p.add_argument("source", help="path to a .s file or the name of a built-in kernel")
    p.add_argument("--block", type=int, default=256, help="threads per workgroup")
    p.add_argument("-d", "--disassemble", action="store_true")
    p.set_defaults(fn=cmd_asm)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
