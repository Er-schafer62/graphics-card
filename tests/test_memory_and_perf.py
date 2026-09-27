import numpy as np
import pytest

from rx590gme import GPUOutOfMemory, RX590GME, assemble, occupancy, pack_args
from rx590gme.memory import PAGE_SIZE


def test_alloc_free_and_page_backing(gpu):
    a = gpu.alloc(100)
    b = gpu.alloc(PAGE_SIZE * 3)
    assert a.address % 256 == 0 and b.address >= a.address + 256
    data = np.arange(PAGE_SIZE * 3 // 4, dtype=np.uint32)
    gpu.upload(b, data)
    np.testing.assert_array_equal(gpu.from_device(b, np.uint32), data)
    gpu.free(a)
    gpu.free(b)
    assert gpu.vram.used_bytes == 0
    assert gpu.vram.resident_bytes == 0


def test_eight_gigabytes_can_be_addressed_but_not_exceeded(gpu):
    big = gpu.alloc(7 * 2**30)             # only touched pages use host memory
    gpu.upload(big, np.ones(16, dtype=np.uint32))
    assert gpu.vram.resident_bytes == PAGE_SIZE
    with pytest.raises(GPUOutOfMemory):
        gpu.alloc(2 * 2**30)
    gpu.free(big)
    assert gpu.vram.resident_bytes == 0


def test_pack_args(gpu):
    buf = gpu.alloc(4)
    assert pack_args([1, -1, 1.0, buf]) == [1, 0xFFFFFFFF, 0x3F800000, buf.address & 0xFFFFFFFF, buf.address >> 32]
    with pytest.raises(ValueError):
        pack_args(list(range(17)))


def test_occupancy_is_limited_by_vgprs():
    heavy = assemble("v_mov_b32 v99, 0\ns_endpgm")    # 100 VGPRs -> 2 waves per SIMD
    occ = occupancy(heavy, waves_per_workgroup=1)
    assert (occ.waves_per_simd, occ.limited_by) == (2, "VGPRs")


def test_occupancy_is_limited_by_lds():
    lds = assemble(".lds 32768\ns_endpgm")            # two workgroups per CU
    occ = occupancy(lds, waves_per_workgroup=1)
    assert (occ.waves_per_simd, occ.limited_by) == (1, "LDS")


STRIDED = """
    v_mul_lo_u32        v1, v0, s2
    buffer_load_dword   v2, v1, s[0:1]
    s_endpgm
"""


def test_uncoalesced_access_costs_more_dram_traffic(gpu):
    buf = gpu.alloc(64 * 64 * 4)
    kernel = gpu.compile(STRIDED)
    coalesced = gpu.launch(kernel, grid=1, block=64, args=[buf, 4])
    strided = gpu.launch(kernel, grid=1, block=64, args=[buf, 64])
    assert coalesced.dram_bytes == 256
    assert strided.dram_bytes == 64 * 64


def test_lds_bank_conflicts_serialise():
    from rx590gme.wavefront import _lds_bank_cycles
    active = np.ones(64, dtype=bool)
    lanes = np.arange(64, dtype=np.uint32)
    assert _lds_bank_cycles(lanes * 4, active) == 2            # conflict free
    assert _lds_bank_cycles(lanes * 128, active) == 64         # every lane in bank 0
    assert _lds_bank_cycles(np.zeros(64, np.uint32), active) == 2  # broadcast


def test_lower_clock_means_longer_compute_bound_kernels():
    from rx590gme import demos
    fast, slow = RX590GME(), RX590GME(clock_mhz=1257)
    _, a = demos.mandelbrot(fast, 32, 32, 16)
    _, b = demos.mandelbrot(slow, 32, 32, 16)
    assert a.compute_cycles == b.compute_cycles
    assert b.compute_time_s == pytest.approx(a.compute_time_s * 1380 / 1257)
