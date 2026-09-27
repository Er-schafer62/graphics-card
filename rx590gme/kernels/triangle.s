; Rasterises one Gouraud-shaded triangle, the way a pixel shader would see it.
;
; For each pixel centre the three edge functions give barycentric weights;
; the pixel is covered when all three are >= 0, and its colour is the
; weighted blend of the vertex colours.
;
; Arguments:  s0 = width, s1 = height, s[2:3] = &framebuffer, s[4:5] = &triangle,
;             s6 = blockDim.x, s7 = blockDim.y, s8 = background colour (RGBA8)
; Triangle:   16 floats: x0 y0 x1 y1 x2 y2, r0 g0 b0, r1 g1 b1, r2 g2 b2 (0-255), pad
; System:     s16/s17 = workgroup id x/y, v0/v1 = thread id x/y
.kernel triangle

    s_load_dwordx16     s[20:35], s[4:5], 0 ; vertex data -> s20..s35
    s_mul_i32           s9, s16, s6
    v_add_u32           v2, s9, v0          ; px
    s_mul_i32           s9, s17, s7
    v_add_u32           v3, s9, v1          ; py
    v_cmp_lt_u32        vcc, v2, s0
    s_and_saveexec_b64  s[10:11], vcc
    v_cmp_lt_u32        vcc, v3, s1
    s_and_saveexec_b64  s[10:11], vcc
    s_cbranch_execz     done
    s_waitcnt           lgkmcnt(0)

    v_cvt_f32_u32       v4, v2
    v_add_f32           v4, 0.5, v4         ; x = px + 0.5
    v_cvt_f32_u32       v5, v3
    v_add_f32           v5, 0.5, v5         ; y = py + 0.5

    ; w0 = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    v_sub_f32           v6, s24, s22
    v_sub_f32           v7, s25, s23
    v_sub_f32           v8, v5, s23
    v_sub_f32           v9, v4, s22
    v_mul_f32           v10, v6, v8
    v_mul_f32           v11, v7, v9
    v_sub_f32           v10, v10, v11
    ; w1 = (x0 - x2) * (y - y2) - (y0 - y2) * (x - x2)
    v_sub_f32           v6, s20, s24
    v_sub_f32           v7, s21, s25
    v_sub_f32           v8, v5, s25
    v_sub_f32           v9, v4, s24
    v_mul_f32           v11, v6, v8
    v_mul_f32           v12, v7, v9
    v_sub_f32           v11, v11, v12
    ; w2 = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
    v_sub_f32           v6, s22, s20
    v_sub_f32           v7, s23, s21
    v_sub_f32           v8, v5, s21
    v_sub_f32           v9, v4, s20
    v_mul_f32           v12, v6, v8
    v_mul_f32           v13, v7, v9
    v_sub_f32           v12, v12, v13

    ; Barycentrics b_i = w_i / (w0 + w1 + w2); works for either winding.
    v_add_f32           v13, v10, v11
    v_add_f32           v13, v13, v12
    v_rcp_f32           v13, v13
    v_mul_f32           v10, v10, v13
    v_mul_f32           v11, v11, v13
    v_mul_f32           v12, v12, v13

    v_mov_b32           v20, s8             ; background
    v_cmp_ge_f32        vcc, v10, 0
    s_and_saveexec_b64  s[12:13], vcc       ; coverage test, saving on-screen lanes
    v_cmp_ge_f32        vcc, v11, 0
    s_and_b64           exec, exec, vcc
    v_cmp_ge_f32        vcc, v12, 0
    s_and_b64           exec, exec, vcc
    s_cbranch_execz     shaded

    v_mul_f32           v14, v10, s26       ; red
    v_mac_f32           v14, v11, s29
    v_mac_f32           v14, v12, s32
    v_mul_f32           v15, v10, s27       ; green
    v_mac_f32           v15, v11, s30
    v_mac_f32           v15, v12, s33
    v_mul_f32           v16, v10, s28       ; blue
    v_mac_f32           v16, v11, s31
    v_mac_f32           v16, v12, s34
    v_cvt_u32_f32       v14, v14
    v_cvt_u32_f32       v15, v15
    v_cvt_u32_f32       v16, v16
    v_min_u32           v14, v14, 255
    v_min_u32           v15, v15, 255
    v_min_u32           v16, v16, 255
    v_lshlrev_b32       v15, 8, v15
    v_lshlrev_b32       v16, 16, v16
    v_or_b32            v20, v14, v15
    v_or_b32            v20, v20, v16
    v_or_b32            v20, v20, 0xff000000
shaded:
    s_mov_b64           exec, s[12:13]

    v_mul_lo_u32        v21, v3, s0
    v_add_u32           v21, v21, v2
    v_lshlrev_b32       v21, 2, v21         ; (py * width + px) * 4
    buffer_store_dword  v20, v21, s[2:3]
done:
    s_endpgm
