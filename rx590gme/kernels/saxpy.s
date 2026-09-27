; y[i] = a * x[i] + y[i]
;
; Arguments (s0-s15):  s0 = n, s1 = a (f32), s[2:3] = &x, s[4:5] = &y, s6 = blockDim.x
; System:              s16 = blockIdx.x, v0 = threadIdx.x
.kernel saxpy

    s_mul_i32           s7, s16, s6         ; first element of this workgroup
    v_add_u32           v1, s7, v0          ; i = blockIdx.x * blockDim.x + threadIdx.x
    v_cmp_lt_u32        vcc, v1, s0         ; lanes with i < n ...
    s_and_saveexec_b64  s[8:9], vcc         ; ... stay active
    s_cbranch_execz     done

    v_lshlrev_b32       v2, 2, v1           ; byte offset = i * 4
    buffer_load_dword   v3, v2, s[2:3]      ; x[i]
    buffer_load_dword   v4, v2, s[4:5]      ; y[i]
    s_waitcnt           vmcnt(0)
    v_mac_f32           v4, s1, v3          ; y[i] += a * x[i]
    buffer_store_dword  v4, v2, s[4:5]
done:
    s_endpgm
