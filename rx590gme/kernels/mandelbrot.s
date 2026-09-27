; Renders the Mandelbrot set into an RGBA8 framebuffer.
;
; Every pixel iterates z = z^2 + c on its own lane. Lanes whose orbit escapes
; are switched off in EXEC while the rest of the wavefront keeps iterating:
; this is how divergent loops run on GCN.
;
; Arguments:  s0 = width, s1 = height, s[2:3] = &framebuffer, s4 = max iterations,
;             s5 = x of left edge (f32), s6 = y of top edge (f32), s7 = units per pixel (f32),
;             s8 = blockDim.x, s9 = blockDim.y
; System:     s16/s17 = workgroup id x/y, v0/v1 = thread id x/y
.kernel mandelbrot

    s_mul_i32           s10, s16, s8
    v_add_u32           v2, s10, v0         ; px
    s_mul_i32           s10, s17, s9
    v_add_u32           v3, s10, v1         ; py
    v_cmp_lt_u32        vcc, v2, s0
    s_and_saveexec_b64  s[18:19], vcc       ; px < width
    v_cmp_lt_u32        vcc, v3, s1
    s_and_saveexec_b64  s[18:19], vcc       ; py < height
    s_cbranch_execz     exit
    s_mov_b64           s[22:23], exec      ; remember the on-screen lanes

    ; c = (x0 + px * scale, y0 + py * scale)
    v_cvt_f32_u32       v4, v2
    v_cvt_f32_u32       v5, v3
    v_mov_b32           v6, s5
    v_mac_f32           v6, v4, s7          ; cr
    v_mov_b32           v7, s6
    v_mac_f32           v7, v5, s7          ; ci
    v_mov_b32           v8, 0               ; zr
    v_mov_b32           v9, 0               ; zi
    v_mov_b32           v10, 0              ; iteration count
    s_mov_b32           s20, 0              ; wave-wide loop counter

loop:
    v_mul_f32           v11, v8, v8         ; zr^2
    v_mul_f32           v12, v9, v9         ; zi^2
    v_add_f32           v13, v11, v12       ; |z|^2
    v_cmp_lt_f32        vcc, v13, 4.0
    s_and_b64           exec, exec, vcc     ; retire lanes that escaped
    s_cbranch_execz     loop_end
    v_mul_f32           v14, v8, v9         ; zr * zi
    v_sub_f32           v8, v11, v12
    v_add_f32           v8, v8, v6          ; zr' = zr^2 - zi^2 + cr
    v_fma_f32           v9, v14, 2.0, v7    ; zi' = 2 * zr * zi + ci
    v_add_u32           v10, v10, 1
    s_add_u32           s20, s20, 1
    s_cmp_lt_u32        s20, s4
    s_cbranch_scc1      loop
loop_end:
    s_mov_b64           exec, s[22:23]      ; all on-screen lanes again

    ; Colour: t = iterations / max, classic polynomial palette.
    v_cvt_f32_u32       v4, v10
    v_cvt_f32_u32       v5, s4
    v_rcp_f32           v5, v5
    v_mul_f32           v4, v4, v5          ; t
    v_sub_f32           v5, 1.0, v4         ; u = 1 - t
    v_mul_f32           v11, v4, v4         ; t^2
    v_mul_f32           v12, v5, v5         ; u^2
    v_mul_f32           v13, v11, v4
    v_mul_f32           v13, v13, v5
    v_mul_f32           v13, v13, 2295.0    ; r = 9 * u * t^3 * 255
    v_mul_f32           v14, v11, v12
    v_mul_f32           v14, v14, 3825.0    ; g = 15 * u^2 * t^2 * 255
    v_mul_f32           v15, v12, v5
    v_mul_f32           v15, v15, v4
    v_mul_f32           v15, v15, 2167.5    ; b = 8.5 * u^3 * t * 255
    v_cvt_u32_f32       v13, v13
    v_cvt_u32_f32       v14, v14
    v_cvt_u32_f32       v15, v15
    v_min_u32           v13, v13, 255
    v_min_u32           v14, v14, 255
    v_min_u32           v15, v15, 255

    ; Pack RGBA8 and store at framebuffer[(py * width + px) * 4]
    v_lshlrev_b32       v14, 8, v14
    v_lshlrev_b32       v15, 16, v15
    v_or_b32            v13, v13, v14
    v_or_b32            v13, v13, v15
    v_or_b32            v13, v13, 0xff000000
    v_mul_lo_u32        v16, v3, s0
    v_add_u32           v16, v16, v2
    v_lshlrev_b32       v16, 2, v16
    buffer_store_dword  v13, v16, s[2:3]
exit:
    s_endpgm
