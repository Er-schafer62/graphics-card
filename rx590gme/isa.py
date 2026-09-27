"""A GCN 4 flavoured instruction set and a two-pass assembler for it.

The syntax follows AMD's GCN assembly conventions::

    .kernel saxpy          ; kernel name
    .lds    1024           ; bytes of LDS (local data share) per workgroup

    loop:
        v_mul_f32     v2, v0, s4        ; vector ALU, SGPR broadcast source
        v_cmp_lt_f32  vcc, v2, 4.0      ; per-lane compare into VCC
        s_and_b64     exec, exec, vcc   ; scalar ops on the 64-bit EXEC mask
        s_cbranch_execz done
        buffer_store_dword v2, v3, s[2:3]
    done:
        s_endpgm

Registers: ``v0``-``v255`` (vector, one 32-bit value per lane), ``s0``-``s101``
(scalar, 32-bit), ``s[a:b]`` for consecutive scalar registers (64-bit values
live in pairs, low dword first), and the special 64-bit registers ``exec``
and ``vcc``. Literals are integers (``42``, ``-1``, ``0xff000000``) or floats
(``2.0``, ``0.5``, ``1e-3``); a float literal is encoded as IEEE-754 single
precision bits, an integer literal is used as raw bits.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

MAX_VGPRS = 256
MAX_SGPRS = 102


class AssemblerError(ValueError):
    def __init__(self, message: str, line_no: int | None = None, line: str | None = None):
        self.line_no = line_no
        self.line = line
        where = f"line {line_no}: " if line_no is not None else ""
        suffix = f"\n    {line.strip()}" if line else ""
        super().__init__(f"{where}{message}{suffix}")


@dataclass(frozen=True)
class Operand:
    kind: str  # 'v', 's', 'exec', 'vcc', 'imm', 'label', 'raw'
    value: int | str = 0
    count: int = 1  # number of consecutive registers for s[a:b]

    def __str__(self) -> str:
        if self.kind == "v":
            return f"v{self.value}"
        if self.kind == "s":
            if self.count == 1:
                return f"s{self.value}"
            return f"s[{self.value}:{self.value + self.count - 1}]"
        if self.kind in ("exec", "vcc"):
            return self.kind
        if self.kind == "imm":
            return hex(self.value) if isinstance(self.value, int) and abs(self.value) > 0xFFFF else str(self.value)
        return str(self.value)


@dataclass(frozen=True)
class Instruction:
    opcode: str
    operands: tuple
    line_no: int
    text: str

    def __str__(self) -> str:
        return f"{self.opcode} " + ", ".join(str(o) for o in self.operands)


@dataclass
class Program:
    name: str
    instructions: list
    labels: dict
    vgpr_count: int
    sgpr_count: int
    lds_bytes: int
    source: str = ""

    def disassemble(self) -> str:
        by_index: dict[int, list[str]] = {}
        for label, idx in self.labels.items():
            by_index.setdefault(idx, []).append(label)
        out = [f".kernel {self.name}"]
        if self.lds_bytes:
            out.append(f".lds {self.lds_bytes}")
        for i, ins in enumerate(self.instructions):
            for label in by_index.get(i, []):
                out.append(f"{label}:")
            out.append(f"    {ins}")
        for label in by_index.get(len(self.instructions), []):
            out.append(f"{label}:")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Opcode table
#
# Operand kinds used in signatures:
#   VD    vector register destination        VS    vector register source
#   SRC   vector ALU source: vN, sN, literal
#   SD    scalar register destination        SS    scalar 32-bit source (sN, literal)
#   SD64  64-bit scalar dest (s[a:a+1], exec, vcc)
#   SS64  64-bit scalar source (s[a:a+1], exec, vcc, literal)
#   MASK  lane mask: vcc or s[a:a+1]
#   SBASE 64-bit base address in s[a:a+1]
#   SDN   N consecutive scalar registers (for s_load_dwordxN)
#   IMM   literal                             LABEL branch target
# ---------------------------------------------------------------------------

OPCODES: dict[str, tuple[str, tuple[str, ...]]] = {}


def _op(category: str, signature: tuple[str, ...], *names: str) -> None:
    for name in names:
        OPCODES[name] = (category, signature)


# Vector ALU
_op("valu", ("VD", "SRC"), "v_mov_b32", "v_not_b32",
    "v_cvt_f32_u32", "v_cvt_f32_i32", "v_cvt_u32_f32", "v_cvt_i32_f32",
    "v_floor_f32", "v_fract_f32", "v_trunc_f32")
_op("valu_trans", ("VD", "SRC"), "v_rcp_f32", "v_sqrt_f32", "v_rsq_f32", "v_exp_f32", "v_log_f32",
    "v_sin_f32", "v_cos_f32")
_op("valu", ("VD", "SRC", "SRC"),
    "v_add_u32", "v_sub_u32", "v_subrev_u32", "v_mul_u32_u24",
    "v_and_b32", "v_or_b32", "v_xor_b32",
    "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32",
    "v_min_u32", "v_max_u32", "v_min_i32", "v_max_i32",
    "v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32", "v_min_f32", "v_max_f32",
    "v_mac_f32")
_op("valu_quarter", ("VD", "SRC", "SRC"), "v_mul_lo_u32", "v_mul_hi_u32")
_op("valu", ("VD", "SRC", "SRC", "SRC"), "v_mad_f32", "v_fma_f32", "v_mad_u32_u24")
_op("valu", ("VD", "SRC", "SRC", "MASK"), "v_cndmask_b32")
_op("valu", ("SD", "VS"), "v_readfirstlane_b32")

VCMP_OPS = ("lt", "le", "gt", "ge", "eq", "ne")
VCMP_TYPES = ("f32", "u32", "i32")
for _c in VCMP_OPS:
    for _t in VCMP_TYPES:
        _op("vcmp", ("MASK", "SRC", "SRC"), f"v_cmp_{_c}_{_t}")

# Scalar ALU
_op("salu", ("SD", "SS"), "s_mov_b32", "s_not_b32")
_op("salu", ("SD", "SS", "SS"),
    "s_add_u32", "s_sub_u32", "s_mul_i32", "s_and_b32", "s_or_b32", "s_xor_b32",
    "s_lshl_b32", "s_lshr_b32", "s_min_u32", "s_max_u32")
_op("salu", ("SD64", "SS64"), "s_mov_b64", "s_not_b64",
    "s_and_saveexec_b64", "s_or_saveexec_b64")
_op("salu", ("SD", "SS64"), "s_bcnt1_i32_b64")
_op("salu", ("SD64", "SS64", "SS64"), "s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64")
SCMP_OPS = ("eq", "lg", "lt", "le", "gt", "ge")
for _c in SCMP_OPS:
    for _t in ("u32", "i32"):
        _op("scmp", ("SS", "SS"), f"s_cmp_{_c}_{_t}")

# Program flow
_op("branch", ("LABEL",), "s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
    "s_cbranch_execz", "s_cbranch_execnz", "s_cbranch_vccz", "s_cbranch_vccnz")
_op("special", (), "s_endpgm", "s_barrier")
_op("special", ("IMM",), "s_nop")
_op("special", ("RAW",), "s_waitcnt")

# Scalar memory
_op("smem", ("SD", "SBASE", "IMM"), "s_load_dword")
for _n in (2, 4, 8, 16):
    _op("smem", (f"SD{_n}", "SBASE", "IMM"), f"s_load_dwordx{_n}")

# Vector memory (buffer) and LDS
_op("vmem_load", ("VD", "VS", "SBASE"), "buffer_load_dword")
_op("vmem_store", ("VS", "VS", "SBASE"), "buffer_store_dword")
_op("lds_read", ("VD", "VS"), "ds_read_b32")
_op("lds_write", ("VS", "VS"), "ds_write_b32")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^([A-Za-z_.$][\w.$]*):\s*(.*)$")
_IDENT_RE = re.compile(r"^[A-Za-z_.$][\w.$]*$")
_FLOAT_RE = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+(\.\d*)?[eE][-+]?\d+)$")


def float_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def parse_literal(tok: str) -> int | None:
    tok = tok.strip()
    if _FLOAT_RE.match(tok):
        return float_bits(float(tok))
    try:
        return int(tok, 0)
    except ValueError:
        return None


def parse_operand(tok: str) -> Operand:
    tok = tok.strip()
    low = tok.lower()
    if low in ("exec", "vcc"):
        return Operand(low)
    m = re.fullmatch(r"v(\d+)", low)
    if m:
        return Operand("v", int(m.group(1)))
    m = re.fullmatch(r"s(\d+)", low)
    if m:
        return Operand("s", int(m.group(1)))
    m = re.fullmatch(r"s\[(\d+)\s*:\s*(\d+)\]", low)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            raise ValueError(f"bad register range {tok}")
        return Operand("s", a, b - a + 1)
    lit = parse_literal(tok)
    if lit is not None:
        return Operand("imm", lit)
    if _IDENT_RE.match(tok):
        return Operand("label", tok)
    raise ValueError(f"cannot parse operand '{tok}'")


def _matches(kind: str, op: Operand) -> bool:
    if kind == "VD" or kind == "VS":
        return op.kind == "v"
    if kind == "SRC":
        return op.kind in ("v", "imm") or (op.kind == "s" and op.count == 1)
    if kind == "SD":
        return op.kind == "s" and op.count == 1
    if kind == "SS":
        return op.kind == "imm" or (op.kind == "s" and op.count == 1)
    if kind == "SD64":
        return op.kind in ("exec", "vcc") or (op.kind == "s" and op.count == 2)
    if kind == "SS64":
        return op.kind in ("exec", "vcc", "imm") or (op.kind == "s" and op.count == 2)
    if kind == "MASK":
        return op.kind == "vcc" or (op.kind == "s" and op.count == 2)
    if kind == "SBASE":
        return op.kind == "s" and op.count == 2
    if kind.startswith("SD") and kind[2:].isdigit():
        return op.kind == "s" and op.count == int(kind[2:])
    if kind == "IMM":
        return op.kind == "imm"
    if kind == "LABEL":
        return op.kind == "label"
    raise AssertionError(kind)


_KIND_NAMES = {
    "VD": "a vector register", "VS": "a vector register",
    "SRC": "a vector register, scalar register or literal",
    "SD": "a scalar register", "SS": "a scalar register or literal",
    "SD64": "a 64-bit scalar register pair, exec or vcc",
    "SS64": "a 64-bit scalar register pair, exec, vcc or literal",
    "MASK": "vcc or a scalar register pair", "SBASE": "a scalar register pair s[n:n+1]",
    "IMM": "a literal", "LABEL": "a label",
}


def assemble(source: str, name: str | None = None) -> Program:
    """Assemble GCN-style source text into a :class:`Program`."""
    kernel_name = name
    lds_bytes = 0
    pending: list[tuple[int, str, str, str]] = []  # (line_no, opcode, rest, raw line)
    labels: dict[str, int] = {}

    for line_no, raw in enumerate(source.splitlines(), start=1):
        line = re.split(r";|//", raw, maxsplit=1)[0].strip()
        while line:
            m = _LABEL_RE.match(line)
            if not m:
                break
            label = m.group(1)
            if label in labels:
                raise AssemblerError(f"duplicate label '{label}'", line_no, raw)
            labels[label] = len(pending)
            line = m.group(2).strip()
        if not line:
            continue
        if line.startswith("."):
            parts = line.split(None, 1)
            directive = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if directive == ".kernel":
                if not _IDENT_RE.match(arg):
                    raise AssemblerError(".kernel needs a name", line_no, raw)
                kernel_name = kernel_name or arg
            elif directive == ".lds":
                value = parse_literal(arg)
                if value is None or value < 0 or value > 64 * 1024 or value % 4:
                    raise AssemblerError(".lds needs a size in bytes (multiple of 4, at most 65536)",
                                         line_no, raw)
                lds_bytes = value
            else:
                raise AssemblerError(f"unknown directive '{directive}'", line_no, raw)
            continue
        parts = line.split(None, 1)
        pending.append((line_no, parts[0].lower(), parts[1].strip() if len(parts) > 1 else "", raw))

    instructions: list[Instruction] = []
    max_v, max_s = -1, -1
    for line_no, opcode, rest, raw in pending:
        if opcode not in OPCODES:
            raise AssemblerError(f"unknown instruction '{opcode}'", line_no, raw)
        _, signature = OPCODES[opcode]
        if signature == ("RAW",):
            ops: tuple = (Operand("raw", rest),) if rest else ()
            instructions.append(Instruction(opcode, ops, line_no, raw.strip()))
            continue
        tokens = [t for t in (x.strip() for x in rest.split(",")) if t] if rest else []
        if len(tokens) != len(signature):
            raise AssemblerError(
                f"'{opcode}' takes {len(signature)} operand(s), got {len(tokens)}", line_no, raw)
        ops_list = []
        for i, (kind, tok) in enumerate(zip(signature, tokens)):
            try:
                op = parse_operand(tok)
            except ValueError as exc:
                raise AssemblerError(str(exc), line_no, raw) from None
            if kind == "LABEL" and op.kind == "label" and op.value not in labels:
                raise AssemblerError(f"undefined label '{op.value}'", line_no, raw)
            if not _matches(kind, op):
                expected = _KIND_NAMES.get(kind, f"{kind[2:]} consecutive scalar registers")
                raise AssemblerError(f"operand {i + 1} of '{opcode}' must be {expected}, got '{tok}'",
                                     line_no, raw)
            if op.kind == "v":
                if op.value >= MAX_VGPRS:
                    raise AssemblerError(f"v{op.value} out of range (v0-v{MAX_VGPRS - 1})", line_no, raw)
                max_v = max(max_v, op.value)
            elif op.kind == "s":
                top = op.value + op.count - 1
                if top >= MAX_SGPRS:
                    raise AssemblerError(f"s{top} out of range (s0-s{MAX_SGPRS - 1})", line_no, raw)
                if op.count == 2 and op.value % 2:
                    raise AssemblerError("64-bit scalar register pairs must start on an even register",
                                         line_no, raw)
                max_s = max(max_s, top)
            ops_list.append(op)
        instructions.append(Instruction(opcode, tuple(ops_list), line_no, raw.strip()))

    if not instructions or instructions[-1].opcode not in ("s_endpgm", "s_branch"):
        # Falling off the end of a program would run into garbage on hardware.
        raise AssemblerError("program must end with s_endpgm")

    return Program(
        name=kernel_name or "kernel",
        instructions=instructions,
        labels=labels,
        vgpr_count=max_v + 1,
        sgpr_count=max_s + 1,
        lds_bytes=lds_bytes,
        source=source,
    )
