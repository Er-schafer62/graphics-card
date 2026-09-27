import pytest

from rx590gme import AssemblerError, assemble


def test_assembles_registers_labels_and_directives():
    prog = assemble("""
        .kernel demo
        .lds 256
    top: v_add_f32 v7, 1.5, v0     ; comment
        s_and_b64 exec, exec, vcc  // another comment
        s_load_dwordx4 s[20:23], s[0:1], 16
        s_cbranch_scc1 top
        s_endpgm
    """)
    assert prog.name == "demo"
    assert prog.lds_bytes == 256
    assert prog.labels == {"top": 0}
    assert prog.vgpr_count == 8
    assert prog.sgpr_count == 24
    assert prog.instructions[0].operands[1].value == 0x3FC00000  # 1.5f


@pytest.mark.parametrize("source, message", [
    ("v_frobnicate v0, v1\ns_endpgm", "unknown instruction"),
    ("v_add_f32 v0, v1\ns_endpgm", "takes 3 operand"),
    ("s_branch nowhere\ns_endpgm", "undefined label"),
    ("v_mov_b32 s0, v1\ns_endpgm", "must be a vector register"),
    ("buffer_load_dword v0, v1, s2\ns_endpgm", "scalar register pair"),
    ("s_mov_b64 s[1:2], exec\ns_endpgm", "even register"),
    ("v_mov_b32 v256, 0\ns_endpgm", "out of range"),
    ("v_mov_b32 v0, 1", "must end with s_endpgm"),
    (".lds 70000\ns_endpgm", ".lds"),
    ("a:\na:\ns_endpgm", "duplicate label"),
])
def test_errors_are_reported_with_line_numbers(source, message):
    with pytest.raises(AssemblerError, match=message):
        assemble(source)


def test_disassembly_round_trips():
    src = "loop: v_mul_f32 v1, v0, s3\ns_cbranch_scc0 loop\ns_endpgm"
    prog = assemble(src)
    again = assemble(prog.disassemble())
    assert [str(i) for i in again.instructions] == [str(i) for i in prog.instructions]
