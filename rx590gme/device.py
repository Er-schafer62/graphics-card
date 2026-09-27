"""The RX 590 GME device: VRAM, kernel dispatch and performance model.

Dispatch works like the hardware's command processor and workload manager.
The grid is split into workgroups, workgroups are handed to the 36 compute
units in round-robin order, and each workgroup is split into 64-lane
wavefronts that are spread over the CU's four SIMDs.

Functionally every wavefront is executed exactly. The timing model is a
throughput (roofline) estimate, not a cycle-accurate one:

* each SIMD issues one wave64 vector instruction every 4 clocks,
* the CU's scalar unit and LDS are shared by all of its waves,
* the busiest CU sets the compute time,
* DRAM traffic (whole 64-byte lines) moves at 256 GB/s,
* the kernel takes max(compute, memory) plus a fixed launch overhead.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .isa import Program, assemble, float_bits
from .memory import VRAM, Buffer
from .specs import RX590_GME, GPUSpec
from .wavefront import (BARRIER, FULL_MASK, LANES, RUNNING, CompiledKernel,
                        Wavefront, compile_program)

USER_SGPRS = 16          # kernel arguments are preloaded into s0-s15
WORKGROUP_ID_SGPR = 16   # s16, s17, s18 = workgroup id x, y, z
LAUNCH_OVERHEAD_US = 4.0


def _dim3(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (int, np.integer)):
        value = (int(value),)
    dims = tuple(int(v) for v in value) + (1, 1, 1)
    dims = dims[:3]
    if any(d < 1 for d in dims):
        raise ValueError(f"dimensions must be positive, got {value}")
    return dims  # type: ignore[return-value]


def pack_args(args: Iterable) -> list[int]:
    """Lay out kernel arguments as dwords: buffers become 64-bit addresses
    (two dwords), floats become IEEE-754 bits, ints are stored as-is."""
    dwords: list[int] = []
    for arg in args:
        if isinstance(arg, Buffer):
            dwords += [arg.address & 0xFFFFFFFF, arg.address >> 32]
        elif isinstance(arg, (bool, np.bool_)):
            dwords.append(int(arg))
        elif isinstance(arg, (int, np.integer)):
            dwords.append(int(arg) & 0xFFFFFFFF)
        elif isinstance(arg, (float, np.floating)):
            dwords.append(float_bits(float(arg)))
        else:
            raise TypeError(f"unsupported kernel argument {arg!r}")
    if len(dwords) > USER_SGPRS:
        raise ValueError(f"kernel arguments need {len(dwords)} dwords; at most {USER_SGPRS} fit in SGPRs")
    return dwords


@dataclass
class Occupancy:
    waves_per_simd: int
    limited_by: str
    vgprs: int
    sgprs: int
    lds_bytes: int


def occupancy(program: Program, waves_per_workgroup: int, spec: GPUSpec = RX590_GME) -> Occupancy:
    """How many waves each SIMD can hold at once, and which resource limits it."""
    vgprs = max(4, -(-max(program.vgpr_count, 1) // 4) * 4)  # allocated in blocks of 4
    # User + system SGPRs, plus VCC, allocated in blocks of 16.
    sgprs = -(-(max(program.sgpr_count, WORKGROUP_ID_SGPR + 3) + 2) // 16) * 16
    lds = -(-program.lds_bytes // 512) * 512
    simds = spec.simds_per_cu
    wpw = waves_per_workgroup

    limits = {
        "wave slots": spec.max_waves_per_simd * simds // wpw,
        "VGPRs": (spec.vgprs_per_simd // vgprs) * simds // wpw,
        "SGPRs": (spec.sgprs_per_simd // sgprs) * simds // wpw,
    }
    if lds:
        limits["LDS"] = spec.lds_per_cu_bytes // lds
    limited_by = min(limits, key=lambda k: limits[k])
    workgroups_per_cu = limits[limited_by]
    waves = min(spec.max_waves_per_simd, -(-workgroups_per_cu * wpw // simds))
    return Occupancy(waves, limited_by, vgprs, sgprs, program.lds_bytes)


@dataclass
class LaunchStats:
    kernel: str
    grid: tuple
    block: tuple
    workgroups: int
    wavefronts: int
    instructions: int
    valu: int
    salu: int
    branches: int
    smem: int
    vmem: int
    lds: int
    lane_ops: int
    flops: int
    dram_bytes: int
    compute_cycles: int
    clock_mhz: float
    occupancy: Occupancy
    cus_used: int
    sim_seconds: float
    spec: GPUSpec = field(repr=False, default=RX590_GME)

    @property
    def compute_time_s(self) -> float:
        return self.compute_cycles / (self.clock_mhz * 1e6)

    @property
    def memory_time_s(self) -> float:
        return self.dram_bytes / (self.spec.memory_bandwidth_gbs * 1e9)

    @property
    def bound(self) -> str:
        return "memory" if self.memory_time_s > self.compute_time_s else "compute"

    @property
    def time_s(self) -> float:
        return max(self.compute_time_s, self.memory_time_s) + LAUNCH_OVERHEAD_US * 1e-6

    @property
    def gflops(self) -> float:
        return self.flops / self.time_s / 1e9

    @property
    def bandwidth_gbs(self) -> float:
        return self.dram_bytes / self.time_s / 1e9

    @property
    def simd_utilization(self) -> float:
        """Fraction of VALU lanes that did useful work (divergence costs this)."""
        return self.lane_ops / (self.valu * LANES) if self.valu else 0.0

    def __str__(self) -> str:
        occ = self.occupancy
        peak_gflops = self.spec.fp32_gflops(self.clock_mhz)
        rows = [
            ("Kernel", self.kernel),
            ("Grid x block", f"{self.grid} x {self.block}"),
            ("Workgroups / waves", f"{self.workgroups:,} / {self.wavefronts:,} on {self.cus_used} CUs"),
            ("Occupancy", f"{occ.waves_per_simd}/{self.spec.max_waves_per_simd} waves per SIMD "
                          f"(limited by {occ.limited_by}; {occ.vgprs} VGPRs, {occ.sgprs} SGPRs, "
                          f"{occ.lds_bytes} B LDS)"),
            ("Instructions", f"{self.instructions:,} (VALU {self.valu:,}, SALU {self.salu:,}, "
                             f"branch {self.branches:,}, VMEM {self.vmem:,}, SMEM {self.smem:,}, "
                             f"LDS {self.lds:,})"),
            ("Lane utilization", f"{self.simd_utilization * 100:.1f}%"),
            ("DRAM traffic", f"{self.dram_bytes / 1e6:.3f} MB"),
            ("Estimated time", f"{self.time_s * 1e6:,.1f} µs ({self.bound} bound @ {self.clock_mhz:g} MHz)"),
            ("Throughput", f"{self.gflops:,.1f} GFLOPS ({self.gflops / peak_gflops * 100:.1f}% of peak), "
                           f"{self.bandwidth_gbs:,.1f} GB/s "
                           f"({self.bandwidth_gbs / self.spec.memory_bandwidth_gbs * 100:.1f}% of peak)"),
            ("Simulation wall time", f"{self.sim_seconds:.2f} s"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k.ljust(width)} : {v}" for k, v in rows)


class RX590GME:
    """A software replica of the AMD Radeon RX 590 GME."""

    def __init__(self, clock_mhz: float | None = None, watchdog_instructions: int = 20_000_000,
                 spec: GPUSpec = RX590_GME):
        self.spec = spec
        self.clock_mhz = float(clock_mhz if clock_mhz is not None else spec.boost_clock_mhz)
        if not 0 < self.clock_mhz <= 3000:
            raise ValueError("clock must be between 0 and 3000 MHz")
        self.watchdog = watchdog_instructions
        self.vram = VRAM(spec.memory_size_bytes)
        self.history: list[LaunchStats] = []

    def __repr__(self) -> str:
        return (f"<{self.spec.name}: {self.spec.compute_units} CUs, "
                f"{self.spec.memory_size_bytes >> 30} GB {self.spec.memory_type}, {self.clock_mhz:g} MHz>")

    # ---- memory ----------------------------------------------------------

    def alloc(self, nbytes: int) -> Buffer:
        return self.vram.alloc(nbytes)

    def free(self, buf: Buffer) -> None:
        self.vram.free(buf)

    def to_device(self, array: np.ndarray) -> Buffer:
        data = np.ascontiguousarray(array)
        buf = self.alloc(max(data.nbytes, 4))
        self.vram.write_bytes(buf.address, data.tobytes())
        return buf

    def upload(self, buf: Buffer, array: np.ndarray) -> None:
        data = np.ascontiguousarray(array)
        if data.nbytes > buf.size:
            raise ValueError(f"{data.nbytes} bytes do not fit in {buf!r}")
        self.vram.write_bytes(buf.address, data.tobytes())

    def from_device(self, buf: Buffer, dtype=np.uint32, shape: int | tuple | None = None) -> np.ndarray:
        dtype = np.dtype(dtype)
        count = int(np.prod(shape)) if shape is not None else buf.size // dtype.itemsize
        raw = self.vram.read_bytes(buf.address, count * dtype.itemsize)
        out = np.frombuffer(raw, dtype=dtype).copy()
        return out.reshape(shape) if shape is not None else out

    # ---- kernels ---------------------------------------------------------

    def compile(self, source: str | Program, name: str | None = None) -> CompiledKernel:
        program = source if isinstance(source, Program) else assemble(source, name)
        return compile_program(program)

    def load_kernel(self, path: str | Path) -> CompiledKernel:
        path = Path(path)
        return self.compile(path.read_text(), name=None)

    def launch(self, kernel: CompiledKernel | str, grid: int | Sequence[int],
               block: int | Sequence[int], args: Iterable = ()) -> LaunchStats:
        """Run ``kernel`` over ``grid`` workgroups of ``block`` threads each.

        ABI: arguments are preloaded into s0-s15 (see :func:`pack_args`), the
        workgroup id is in s16/s17/s18 and the thread id within the workgroup
        is in v0/v1/v2 (x/y/z).
        """
        if isinstance(kernel, str):
            kernel = self.compile(kernel)
        spec = self.spec
        grid3, block3 = _dim3(grid), _dim3(block)
        threads = block3[0] * block3[1] * block3[2]
        if threads > spec.max_workgroup_size:
            raise ValueError(f"workgroup of {threads} threads exceeds the {spec.max_workgroup_size} limit")
        waves_per_wg = -(-threads // LANES)
        user = {i: v for i, v in enumerate(pack_args(args))}
        lds_words = kernel.program.lds_bytes // 4

        # Thread ids and EXEC masks for each wave of a workgroup never change.
        wave_ids = []
        for w in range(waves_per_wg):
            flat = np.arange(w * LANES, (w + 1) * LANES, dtype=np.uint32)
            ids = np.stack([flat % block3[0], (flat // block3[0]) % block3[1],
                            flat // (block3[0] * block3[1])]).astype(np.uint32)
            valid = min(LANES, threads - w * LANES)
            wave_ids.append((ids, FULL_MASK if valid == LANES else (1 << valid) - 1))

        n_cu = spec.compute_units
        simd_cycles = np.zeros((n_cu, spec.simds_per_cu), dtype=np.int64)
        scalar_cycles = np.zeros(n_cu, dtype=np.int64)
        lds_cycles = np.zeros(n_cu, dtype=np.int64)
        totals = dict.fromkeys(("n_instr", "n_valu", "n_salu", "n_branch", "n_smem", "n_vmem",
                                "n_lds", "lane_ops", "flops", "dram_bytes"), 0)

        started = time.perf_counter()
        workgroups = itertools.product(range(grid3[2]), range(grid3[1]), range(grid3[0]))
        n_wg = 0
        for n_wg, (gz, gy, gx) in enumerate(workgroups, start=1):
            index = n_wg - 1
            cu = index % n_cu
            lds = np.zeros(lds_words, dtype=np.uint32)
            sgprs = dict(user)
            sgprs.update({WORKGROUP_ID_SGPR: gx, WORKGROUP_ID_SGPR + 1: gy, WORKGROUP_ID_SGPR + 2: gz})
            waves = [Wavefront(kernel, self.vram, lds, sgprs, ids, mask, self.watchdog)
                     for ids, mask in wave_ids]
            self._run_workgroup(waves)

            first_simd = (index // n_cu) * waves_per_wg
            for i, wave in enumerate(waves):
                simd_cycles[cu, (first_simd + i) % spec.simds_per_cu] += wave.simd_cycles
                scalar_cycles[cu] += wave.scalar_cycles
                lds_cycles[cu] += wave.lds_cycles
                for key in totals:
                    totals[key] += getattr(wave, key)

        per_cu = np.maximum(np.maximum(simd_cycles.max(axis=1), scalar_cycles), lds_cycles)
        stats = LaunchStats(
            kernel=kernel.name, grid=grid3, block=block3, workgroups=n_wg,
            wavefronts=n_wg * waves_per_wg, instructions=totals["n_instr"],
            valu=totals["n_valu"], salu=totals["n_salu"], branches=totals["n_branch"],
            smem=totals["n_smem"], vmem=totals["n_vmem"], lds=totals["n_lds"],
            lane_ops=totals["lane_ops"], flops=totals["flops"], dram_bytes=totals["dram_bytes"],
            compute_cycles=int(per_cu.max()), clock_mhz=self.clock_mhz,
            occupancy=occupancy(kernel.program, waves_per_wg, spec),
            cus_used=min(n_wg, n_cu), sim_seconds=time.perf_counter() - started, spec=spec,
        )
        self.history.append(stats)
        return stats

    @staticmethod
    def _run_workgroup(waves: list[Wavefront]) -> None:
        """Run the waves of one workgroup, releasing s_barrier once every
        wave that has not finished is waiting at it."""
        while True:
            for wave in waves:
                if wave.state == RUNNING:
                    wave.run()
            waiting = [w for w in waves if w.state == BARRIER]
            if not waiting:
                return
            for wave in waiting:
                wave.state = RUNNING

    # ---- display ---------------------------------------------------------

    def scanout(self, framebuffer: Buffer, width: int, height: int, path: str | Path) -> Path:
        """Send an RGBA8 framebuffer to the display output (a PNG file)."""
        from .display import write_png

        pixels = self.from_device(framebuffer, np.uint8, (height, width, 4))
        return write_png(path, pixels)
