"""Hardware specification of the AMD Radeon RX 590 GME.

Every number the simulator uses for the chip's shape (compute units, SIMDs,
register files, LDS, caches) and for its timing model (clocks, memory
bandwidth) comes from this one table.
"""

from __future__ import annotations

from dataclasses import dataclass

KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB


@dataclass(frozen=True)
class GPUSpec:
    # Identity
    name: str
    gpu: str
    architecture: str
    family: str
    process_nm: int
    transistors_millions: int
    die_size_mm2: int
    release: str
    bus_interface: str

    # Shader array
    shader_engines: int
    cus_per_shader_engine: int
    simds_per_cu: int
    simd_width: int
    wavefront_size: int
    tmus_per_cu: int
    rops: int

    # Per-SIMD / per-CU resources
    vgprs_per_simd: int
    sgprs_per_simd: int
    max_sgprs_per_wave: int
    max_waves_per_simd: int
    max_waves_per_cu: int
    max_workgroup_size: int
    lds_per_cu_bytes: int
    l1_per_cu_bytes: int
    l2_bytes: int

    # Clocks
    base_clock_mhz: int
    boost_clock_mhz: int

    # Memory
    memory_size_bytes: int
    memory_type: str
    memory_bus_width_bits: int
    memory_data_rate_gbps: float

    # Power
    tdp_watts: int
    fp64_ratio: int

    # ---- derived figures -------------------------------------------------

    @property
    def compute_units(self) -> int:
        return self.shader_engines * self.cus_per_shader_engine

    @property
    def stream_processors(self) -> int:
        return self.compute_units * self.simds_per_cu * self.simd_width

    @property
    def tmus(self) -> int:
        return self.compute_units * self.tmus_per_cu

    @property
    def memory_bandwidth_gbs(self) -> float:
        """Peak DRAM bandwidth in GB/s (10^9 bytes)."""
        return self.memory_bus_width_bits / 8 * self.memory_data_rate_gbps

    def fp32_gflops(self, clock_mhz: float | None = None) -> float:
        """Peak single precision throughput (one FMA = 2 FLOPs per lane per clock)."""
        clock = self.boost_clock_mhz if clock_mhz is None else clock_mhz
        return self.stream_processors * 2 * clock / 1000

    def fp64_gflops(self, clock_mhz: float | None = None) -> float:
        return self.fp32_gflops(clock_mhz) / self.fp64_ratio

    def pixel_fillrate_gpix(self, clock_mhz: float | None = None) -> float:
        clock = self.boost_clock_mhz if clock_mhz is None else clock_mhz
        return self.rops * clock / 1000

    def texture_rate_gtex(self, clock_mhz: float | None = None) -> float:
        clock = self.boost_clock_mhz if clock_mhz is None else clock_mhz
        return self.tmus * clock / 1000


RX590_GME = GPUSpec(
    name="AMD Radeon RX 590 GME",
    gpu="Polaris 20 XTX",
    architecture="Graphics Core Next 4.0 (GCN 4)",
    family="Polaris",
    process_nm=14,
    transistors_millions=5700,
    die_size_mm2=232,
    release="March 2020",
    bus_interface="PCIe 3.0 x16",
    shader_engines=4,
    cus_per_shader_engine=9,
    simds_per_cu=4,
    simd_width=16,
    wavefront_size=64,
    tmus_per_cu=4,
    rops=32,
    vgprs_per_simd=256,
    sgprs_per_simd=800,
    max_sgprs_per_wave=102,
    max_waves_per_simd=10,
    max_waves_per_cu=40,
    max_workgroup_size=1024,
    lds_per_cu_bytes=64 * KiB,
    l1_per_cu_bytes=16 * KiB,
    l2_bytes=2 * MiB,
    base_clock_mhz=1257,
    boost_clock_mhz=1380,
    memory_size_bytes=8 * GiB,
    memory_type="GDDR5",
    memory_bus_width_bits=256,
    memory_data_rate_gbps=8.0,
    tdp_watts=175,
    fp64_ratio=16,
)


def spec_sheet(spec: GPUSpec = RX590_GME, clock_mhz: float | None = None) -> str:
    """A GPU-Z style summary of the card."""
    clock = spec.boost_clock_mhz if clock_mhz is None else clock_mhz
    rows = [
        ("Name", spec.name),
        ("GPU", f"{spec.gpu} ({spec.family})"),
        ("Architecture", spec.architecture),
        ("Process", f"{spec.process_nm} nm"),
        ("Transistors", f"{spec.transistors_millions / 1000:.1f} billion"),
        ("Die size", f"{spec.die_size_mm2} mm²"),
        ("Release", spec.release),
        ("Bus interface", spec.bus_interface),
        ("Shader engines", str(spec.shader_engines)),
        ("Compute units", f"{spec.compute_units} ({spec.cus_per_shader_engine} per SE)"),
        ("Stream processors", str(spec.stream_processors)),
        ("TMUs / ROPs", f"{spec.tmus} / {spec.rops}"),
        ("Wavefront size", str(spec.wavefront_size)),
        ("LDS per CU", f"{spec.lds_per_cu_bytes // KiB} KiB"),
        ("L1 per CU / L2", f"{spec.l1_per_cu_bytes // KiB} KiB / {spec.l2_bytes // MiB} MiB"),
        ("Base / boost clock", f"{spec.base_clock_mhz} / {spec.boost_clock_mhz} MHz"),
        ("Simulated clock", f"{clock:g} MHz"),
        ("Memory", f"{spec.memory_size_bytes // GiB} GB {spec.memory_type}"),
        ("Memory bus", f"{spec.memory_bus_width_bits}-bit @ {spec.memory_data_rate_gbps:g} Gbps"),
        ("Bandwidth", f"{spec.memory_bandwidth_gbs:.0f} GB/s"),
        ("FP32", f"{spec.fp32_gflops(clock) / 1000:.2f} TFLOPS"),
        ("FP64", f"{spec.fp64_gflops(clock):.0f} GFLOPS (1:{spec.fp64_ratio})"),
        ("Pixel fillrate", f"{spec.pixel_fillrate_gpix(clock):.1f} GPixel/s"),
        ("Texture rate", f"{spec.texture_rate_gtex(clock):.1f} GTexel/s"),
        ("Board power", f"{spec.tdp_watts} W"),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k.ljust(width)} : {v}" for k, v in rows)
