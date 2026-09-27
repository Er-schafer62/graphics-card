import numpy as np
import pytest

from rx590gme import GPUExecutionError, GPUHangError, GPUMemoryFault, builtin_kernel, demos


def test_saxpy_matches_numpy_for_a_partial_last_wave(gpu):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000, dtype=np.float32)
    y = rng.standard_normal(1000, dtype=np.float32)
    result, stats = demos.saxpy(gpu, 3.0, x, y, block=128)
    np.testing.assert_array_equal(result, np.float32(3.0) * x + y)
    assert stats.workgroups == 8
    assert stats.wavefronts == 16
    assert stats.flops == 2 * 1000


def test_divergent_if_else_with_exec_mask(gpu):
    # out[i] = i even ? i * 10 : i + 1000
    kernel = gpu.compile("""
        v_lshlrev_b32       v1, 2, v0
        v_and_b32           v2, v0, 1
        v_cmp_eq_u32        vcc, v2, 0
        s_and_saveexec_b64  s[4:5], vcc        ; if (even)
        v_mul_lo_u32        v3, v0, 10
        s_andn2_b64         exec, s[4:5], exec ; else
        v_add_u32           v3, v0, 1000
        s_mov_b64           exec, s[4:5]       ; endif
        buffer_store_dword  v3, v1, s[0:1]
        s_endpgm
    """)
    out = gpu.alloc(64 * 4)
    stats = gpu.launch(kernel, grid=1, block=64, args=[out])
    i = np.arange(64)
    np.testing.assert_array_equal(gpu.from_device(out, np.uint32, 64), np.where(i % 2 == 0, i * 10, i + 1000))
    assert stats.simd_utilization < 1.0


def test_lds_and_barrier_reverse_a_workgroup(gpu):
    # Each 256-thread workgroup (4 waves) reverses its slice through LDS; the
    # barrier is what makes the other waves' writes visible.
    kernel = gpu.compile("""
        .lds 1024
        v_lshlrev_b32       v1, 2, v0
        s_lshl_b32          s2, s16, 10
        v_add_u32           v2, s2, v1
        buffer_load_dword   v3, v2, s[0:1]
        ds_write_b32        v1, v3
        s_barrier
        v_sub_u32           v4, 1020, v1
        ds_read_b32         v5, v4
        buffer_store_dword  v5, v2, s[0:1]
        s_endpgm
    """)
    data = np.arange(512, dtype=np.uint32)
    buf = gpu.to_device(data)
    stats = gpu.launch(kernel, grid=2, block=256, args=[buf])
    expected = np.concatenate([data[:256][::-1], data[256:][::-1]])
    np.testing.assert_array_equal(gpu.from_device(buf, np.uint32, 512), expected)
    assert stats.lds == 2 * 4 * 2
    assert stats.occupancy.limited_by in ("wave slots", "LDS")


def test_scalar_loop_and_readfirstlane(gpu):
    # sum 1..10 in a scalar loop, then broadcast through a vector register
    kernel = gpu.compile("""
        s_mov_b32           s4, 0
        s_mov_b32           s5, 1
    loop:
        s_add_u32           s4, s4, s5
        s_add_u32           s5, s5, 1
        s_cmp_le_u32        s5, 10
        s_cbranch_scc1      loop
        v_mov_b32           v1, s4
        v_add_u32           v1, v1, v0
        v_readfirstlane_b32 s6, v1
        v_mov_b32           v2, s6
        v_lshlrev_b32       v3, 2, v0
        buffer_store_dword  v2, v3, s[0:1]
        s_endpgm
    """)
    out = gpu.alloc(64 * 4)
    gpu.launch(kernel, grid=1, block=64, args=[out])
    assert set(gpu.from_device(out, np.uint32, 64)) == {55}


def test_float_ops_and_conversions(gpu):
    kernel = gpu.compile("""
        v_cvt_f32_u32       v1, v0
        v_mul_f32           v2, v1, 0.5
        v_sqrt_f32          v3, v1
        v_cvt_u32_f32       v4, v3
        v_fract_f32         v5, v2
        v_lshlrev_b32       v6, 2, v0
        buffer_store_dword  v2, v6, s[0:1]
        buffer_store_dword  v4, v6, s[2:3]
        buffer_store_dword  v5, v6, s[4:5]
        s_endpgm
    """)
    a, b, c = gpu.alloc(256), gpu.alloc(256), gpu.alloc(256)
    gpu.launch(kernel, grid=1, block=64, args=[a, b, c])
    i = np.arange(64, dtype=np.float32)
    np.testing.assert_array_equal(gpu.from_device(a, np.float32, 64), i * 0.5)
    np.testing.assert_array_equal(gpu.from_device(b, np.uint32, 64), np.sqrt(i).astype(np.uint32))
    np.testing.assert_array_equal(gpu.from_device(c, np.float32, 64), np.where(np.arange(64) % 2, 0.5, 0))


def test_null_pointer_faults(gpu):
    kernel = gpu.compile("v_lshlrev_b32 v1, 2, v0\nbuffer_store_dword v0, v1, s[0:1]\ns_endpgm")
    with pytest.raises(GPUMemoryFault, match="page fault"):
        gpu.launch(kernel, grid=1, block=64, args=[0, 0])


def test_lds_out_of_bounds_faults(gpu):
    kernel = gpu.compile(".lds 64\nv_lshlrev_b32 v1, 2, v0\nds_write_b32 v1, v0\ns_endpgm")
    with pytest.raises(GPUMemoryFault, match="LDS"):
        gpu.launch(kernel, grid=1, block=64)


def test_infinite_loop_trips_the_watchdog(gpu):
    kernel = gpu.compile("spin: s_branch spin\ns_endpgm")
    with pytest.raises(GPUHangError, match="hung"):
        gpu.launch(kernel, grid=1, block=64)


def test_running_off_the_end_is_an_error(gpu):
    kernel = gpu.compile("s_branch end\ns_endpgm\nend:")
    with pytest.raises(GPUExecutionError):
        gpu.launch(kernel, grid=1, block=64)


def test_workgroup_size_limit(gpu):
    with pytest.raises(ValueError, match="1024"):
        gpu.launch(gpu.compile("s_endpgm"), grid=1, block=(32, 33))


def test_builtin_kernels_all_assemble(gpu):
    for name in ("saxpy", "mandelbrot", "triangle"):
        assert gpu.compile(builtin_kernel(name)).name == name
