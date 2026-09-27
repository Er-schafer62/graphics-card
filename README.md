# RX 590 GME — a replica

This repo contains two replicas of the **AMD Radeon RX 590 GME**:

1. **[A 3D-printable model of the card](model/)** at 1:1 scale. It comes as a 19-part paint-and-assemble kit in red, black, white and yellow, or as a one-piece print. There's also a half-scale desk model.
2. **A software replica of the GPU**: a simulator built to the chip's specs that runs real shader programs (described below).

![3D-printable RX 590 GME kit](docs/kit_assembled.png)

## 3D-printable model

**Paint-and-assemble kit** ([`model/kit/`](model/kit/)): 19 single-colour parts, all printable without supports. The fans still spin after assembly. [`model/kit/split/`](model/kit/split/) has the four long parts in two pieces each, for beds of 180 × 180 mm or more. See [the kit guide](model/README.md#paint-and-assemble-kit) for the parts list, colours and assembly steps.

![Exploded kit](docs/kit_exploded.png)

**One-piece model:**

| File | For |
|---|---|
| [`model/stl/rx590gme_1to1.stl`](model/stl/rx590gme_1to1.stl) | Full size in one piece (266 mm long; needs a large bed) |
| [`model/stl/rx590gme_1to1_part1_bracket.stl`](model/stl/rx590gme_1to1_part1_bracket.stl) + [`part2_rear`](model/stl/rx590gme_1to1_part2_rear.stl) + [`pins_x2`](model/stl/rx590gme_1to1_pins_x2.stl) | Full size on any bed of at least 180 × 180 mm |
| [`model/stl/rx590gme_1to2.stl`](model/stl/rx590gme_1to2.stl) | Half-scale desk model |

Print with the backplate on the bed. Supports are only needed under the PCIe fingers. See [`model/README.md`](model/README.md) for print settings, assembly and how to customise the parametric model.

## Software replica

A software replica of the **AMD Radeon RX 590 GME** graphics card (Polaris 20 XTX, GCN 4). It is built to the card's specs and runs real shader programs:

- **36 compute units** in 4 shader engines. Each CU has 4 × SIMD16 units, 64 KiB of LDS and a scalar unit.
- **64-lane wavefronts** with a 64-bit EXEC mask, VCC and SCC, so divergent code works the way it does on GCN hardware.
- **VGPR, SGPR and LDS limits** that decide occupancy (waves per SIMD), as on the real chip.
- **8 GB of GDDR5 VRAM** on a 256-bit bus at 8 Gbps. The full address space is modelled, and unmapped accesses page-fault.
- **A GCN-style assembler** for kernels written in AMD-style assembly (`v_mac_f32`, `s_and_saveexec_b64`, `buffer_load_dword`, `ds_read_b32`, `s_barrier`, …).
- **A performance model** that uses the card's clocks and memory bandwidth to estimate kernel time, GFLOPS, GB/s, lane utilization and whether a kernel is compute or memory bound.
- **A display output** that scans a framebuffer out to PNG.

| Rasteriser (`triangle.s`) | Divergent compute (`mandelbrot.s`) |
|---|---|
| ![triangle](docs/triangle.png) | ![mandelbrot](docs/mandelbrot.png) |

Both images were rendered by kernels running on the replica.

> This is a functional simulator written in Python with NumPy. It is not hardware, and it does not run AMD drivers or real game binaries. See [Accuracy](#accuracy) for what is modelled exactly and what is estimated.

### Specifications

```
$ python -m rx590gme info
Name               : AMD Radeon RX 590 GME
GPU                : Polaris 20 XTX (Polaris)
Architecture       : Graphics Core Next 4.0 (GCN 4)
Process            : 14 nm
Transistors        : 5.7 billion
Die size           : 232 mm²
Release            : March 2020
Bus interface      : PCIe 3.0 x16
Shader engines     : 4
Compute units      : 36 (9 per SE)
Stream processors  : 2304
TMUs / ROPs        : 144 / 32
Wavefront size     : 64
LDS per CU         : 64 KiB
L1 per CU / L2     : 16 KiB / 2 MiB
Base / boost clock : 1257 / 1380 MHz
Memory             : 8 GB GDDR5
Memory bus         : 256-bit @ 8 Gbps
Bandwidth          : 256 GB/s
FP32               : 6.36 TFLOPS
FP64               : 397 GFLOPS (1:16)
Pixel fillrate     : 44.2 GPixel/s
Texture rate       : 198.7 GTexel/s
Board power        : 175 W
```

All of these live in [`rx590gme/specs.py`](rx590gme/specs.py). The clocks are AMD's reference clocks; AIB partner cards ship with boost clocks of up to about 1460 MHz, and `--clock` lets you simulate any of them.

### Quick start

```bash
pip install -e .            # needs Python 3.9+ and NumPy
rx590gme info               # spec sheet (or: python -m rx590gme info)
rx590gme bench              # SAXPY benchmark, verified against NumPy
rx590gme mandelbrot -o mandelbrot.png --width 480 --height 320 --iterations 96
rx590gme triangle -o triangle.png
rx590gme asm mandelbrot -d  # assemble a kernel, show VGPR/SGPR use, occupancy, disassembly
rx590gme --clock 1257 bench # run at the base clock instead of boost
```

Sample benchmark output:

```
SAXPY on 262,144 elements: PASS

Kernel               : saxpy
Grid x block         : (1024, 1, 1) x (256, 1, 1)
Workgroups / waves   : 1,024 / 4,096 on 36 CUs
Occupancy            : 10/10 waves per SIMD (limited by wave slots; 8 VGPRs, 32 SGPRs, 0 B LDS)
Instructions         : 49,152 (VALU 16,384, SALU 8,192, branch 4,096, VMEM 12,288, SMEM 0, LDS 0)
Lane utilization     : 100.0%
DRAM traffic         : 3.146 MB
Estimated time       : 16.3 µs (memory bound @ 1380 MHz)
Throughput           : 32.2 GFLOPS (0.5% of peak), 193.1 GB/s (75.4% of peak)
```

### Writing your own kernel

```python
import numpy as np
from rx590gme import RX590GME

gpu = RX590GME()                        # or RX590GME(clock_mhz=1257)

kernel = gpu.compile("""
    .kernel square
    ; s[0:1] = &data, s2 = n, s16 = workgroup id, v0 = thread id
    s_lshl_b32          s3, s16, 6          ; 64 threads per workgroup
    v_add_u32           v1, s3, v0          ; global id
    v_cmp_lt_u32        vcc, v1, s2
    s_and_saveexec_b64  s[4:5], vcc         ; if (i < n)
    v_lshlrev_b32       v2, 2, v1
    buffer_load_dword   v3, v2, s[0:1]
    v_mul_f32           v3, v3, v3
    buffer_store_dword  v3, v2, s[0:1]
    s_endpgm
""")

data = gpu.to_device(np.arange(100, dtype=np.float32))
stats = gpu.launch(kernel, grid=2, block=64, args=[data, 100])
print(gpu.from_device(data, np.float32, 100)[:5])   # [ 0.  1.  4.  9. 16.]
print(stats)
```

#### Kernel ABI

| Where | What |
|---|---|
| `s0`–`s15` | Kernel arguments, packed as dwords. A buffer takes two dwords (a 64-bit address), a float becomes its IEEE-754 bits, and an int is stored as-is. |
| `s16`, `s17`, `s18` | Workgroup ID x, y, z |
| `v0`, `v1`, `v2` | Thread ID within the workgroup (x, y, z) |
| `exec` | Starts with a lane set for each thread that exists. The last wave of a workgroup can be partial. |

#### Instruction set

| Class | Instructions |
|---|---|
| Vector ALU | `v_mov_b32` `v_add/sub/subrev_u32` `v_mul_lo/hi_u32` `v_mul_u32_u24` `v_mad_u32_u24` `v_and/or/xor/not_b32` `v_lshlrev/lshrrev_b32` `v_ashrrev_i32` `v_min/max_{u32,i32,f32}` `v_add/sub/subrev/mul_f32` `v_mac_f32` `v_mad_f32` `v_fma_f32` `v_cndmask_b32` `v_readfirstlane_b32` `v_cvt_{f32_u32,f32_i32,u32_f32,i32_f32}` `v_floor/fract/trunc_f32` |
| Transcendental (¼ rate) | `v_rcp_f32` `v_sqrt_f32` `v_rsq_f32` `v_exp_f32` `v_log_f32` `v_sin_f32` `v_cos_f32` |
| Vector compare | `v_cmp_{lt,le,gt,ge,eq,ne}_{f32,u32,i32}` → `vcc` or `s[n:n+1]` |
| Scalar ALU | `s_mov/not_b32/b64` `s_add/sub_u32` `s_mul_i32` `s_and/or/xor_b32/b64` `s_andn2_b64` `s_lshl/lshr_b32` `s_min/max_u32` `s_and_saveexec_b64` `s_or_saveexec_b64` `s_bcnt1_i32_b64` `s_cmp_{eq,lg,lt,le,gt,ge}_{u32,i32}` |
| Flow | `s_branch` `s_cbranch_{scc0,scc1,execz,execnz,vccz,vccnz}` `s_barrier` `s_waitcnt` `s_nop` `s_endpgm` |
| Memory | `s_load_dword` `s_load_dwordx{2,4,8,16}` `buffer_load_dword` `buffer_store_dword` `ds_read_b32` `ds_write_b32` |

Directives: `.kernel <name>` and `.lds <bytes>`. Comments start with `;` or `//`.

### Accuracy

**Modelled exactly**
- Every lane of every wavefront executes. Results are bit-exact IEEE-754 single precision, and integers wrap at 32 bits.
- EXEC masking, divergence, uniform branches and barriers within a workgroup.
- VGPR, SGPR, LDS and wave-slot limits for occupancy, using the GCN allocation granularities.
- Memory coalescing. DRAM traffic is counted in 64-byte cache lines, so strided access costs what it should.
- LDS bank conflicts: 32 banks, serviced one half-wave at a time, with broadcast for lanes reading the same address.
- Quarter-rate instructions: transcendentals and 32-bit integer multiply.
- Page-granular memory protection: null or wild pointers fault, and runaway kernels trip a watchdog (like Windows TDR).

**Estimated or simplified**
- **Timing is a roofline estimate, not cycle-accurate.** Each SIMD issues one wave instruction every 4 clocks. The busiest CU sets the compute time. Memory runs at the full 256 GB/s. Kernel time is max(compute, memory) plus 4 µs of launch overhead. Caches, memory latency and clock boost behaviour are not modelled.
- Memory operations complete immediately, so `s_waitcnt` is accepted but does nothing.
- Buffer instructions take a 64-bit base address in `s[n:n+1]` instead of a 128-bit resource descriptor.
- There are no carry-outs from `v_add_u32`, and the one-SGPR-per-VALU constant bus limit is not enforced.
- `v_fma_f32` is computed in double precision and then rounded. This is exact for the product but can double-round the sum in rare cases.
- There is no fixed-function graphics pipeline (TMUs, ROPs, rasteriser hardware). Graphics is done in compute shaders, as `triangle.s` shows.
- Instruction encodings are not binary compatible with real GCN machine code. Programs are assembled from text.

## Layout

```
model/          3D-printable model: kit + one-piece generators, preview renderer, STL files
rx590gme/
  specs.py      card specification and spec sheet
  isa.py        instruction set and two-pass assembler
  wavefront.py  wavefront state and instruction execution
  memory.py     8 GB paged VRAM
  device.py     RX590GME: allocation, dispatch, occupancy, performance model
  display.py    framebuffer -> PNG
  demos.py      saxpy / mandelbrot / triangle workloads
  cli.py        command line interface
  kernels/      saxpy.s, mandelbrot.s, triangle.s
tests/          pytest suite (pip install -e ".[test]" && pytest; model tests need manifold3d)
```

## Spec sources

- [Tom's Hardware — AMD Polaris Revival: Radeon RX 590 GME](https://www.tomshardware.com/news/amd-radeon-rx-590-gme-graphics-cards)
- [VideoCardz — AMD Radeon RX 590 GME features Polaris 20 XTX](https://videocardz.com/newz/amd-radeon-rx-590-gme-features-polaris-20-xtx)
- [AMD3D — RX 590 GME, a 14 nm Polaris 20 with lower clock rate](https://www.amd3d.com/radeon/amd-radeon-rx-590-gme-a-14nm-polaris-20-with-lower-clock-rate/)
- AMD, *Graphics Core Next Architecture, Generation 3/4 ISA reference* (instruction semantics)
