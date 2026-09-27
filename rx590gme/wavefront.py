"""Execution of 64-lane wavefronts, the unit of work a GCN SIMD runs.

A wavefront has 64 lanes. Vector instructions operate on all lanes at once, and
the 64-bit EXEC mask decides which lanes may write results. Scalar instructions
run once per wavefront on the CU's scalar unit and manage control flow, EXEC,
VCC and SCC. Branching is uniform: to run divergent ``if``/``else`` code, a
kernel narrows EXEC, runs both sides, and restores EXEC, the same way real
GCN code does.

Programs are translated once into a list of Python closures ("microcode"),
so the interpreter loop only has to call ``code[pc](wave)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .isa import MAX_SGPRS, OPCODES, Instruction, Operand, Program
from .memory import VRAM, GPUMemoryFault

LANES = 64
FULL_MASK = (1 << 64) - 1
M32 = 0xFFFFFFFF
_BITS = np.left_shift(np.uint64(1), np.arange(LANES, dtype=np.uint64))

RUNNING, BARRIER, DONE = 0, 1, 2

# Issue cost in SIMD clocks: a wave64 instruction occupies a SIMD16 for 4 clocks;
# transcendentals and 32-bit integer multiplies run at quarter rate.
CYCLES = {"valu": 4, "valu_trans": 16, "valu_quarter": 16, "vcmp": 4,
          "vmem_load": 4, "vmem_store": 4}
CACHE_LINE = 64


class GPUHangError(RuntimeError):
    """The watchdog tripped: a wavefront ran too many instructions (think TDR)."""


class GPUExecutionError(RuntimeError):
    """A kernel did something illegal, e.g. ran past its last instruction."""


def mask_to_bool(mask: int) -> np.ndarray:
    return (np.uint64(mask) & _BITS) != 0


def bool_to_mask(lanes: np.ndarray) -> int:
    return int(_BITS[lanes].sum(dtype=np.uint64))


def popcount(x: int) -> int:
    return bin(x).count("1")


def _f(x: np.ndarray) -> np.ndarray:
    return x.view(np.float32)


def _i(x: np.ndarray) -> np.ndarray:
    return x.view(np.int32)


def _u(x: np.ndarray) -> np.ndarray:
    return x.view(np.uint32)


def _signed32(x: int) -> int:
    return x - (1 << 32) if x & 0x80000000 else x


def _cvt_u32_f32(a: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(_f(a).astype(np.float64), nan=0.0)
    return np.clip(x, 0, M32).astype(np.uint32)


def _cvt_i32_f32(a: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(_f(a).astype(np.float64), nan=0.0)
    return np.clip(x, -(2**31), 2**31 - 1).astype(np.int32).view(np.uint32)


_TWO_PI = np.float32(2 * np.pi)

UNARY: dict[str, Callable] = {
    "v_mov_b32": lambda a: a,
    "v_not_b32": lambda a: ~a,
    "v_cvt_f32_u32": lambda a: a.astype(np.float32).view(np.uint32),
    "v_cvt_f32_i32": lambda a: _i(a).astype(np.float32).view(np.uint32),
    "v_cvt_u32_f32": _cvt_u32_f32,
    "v_cvt_i32_f32": _cvt_i32_f32,
    "v_floor_f32": lambda a: _u(np.floor(_f(a))),
    "v_fract_f32": lambda a: _u(_f(a) - np.floor(_f(a))),
    "v_trunc_f32": lambda a: _u(np.trunc(_f(a))),
    "v_rcp_f32": lambda a: _u(np.float32(1) / _f(a)),
    "v_sqrt_f32": lambda a: _u(np.sqrt(_f(a))),
    "v_rsq_f32": lambda a: _u(np.float32(1) / np.sqrt(_f(a))),
    "v_exp_f32": lambda a: _u(np.exp2(_f(a))),
    "v_log_f32": lambda a: _u(np.log2(_f(a))),
    # Like the hardware, sin/cos take their argument in revolutions: sin(2*pi*x).
    "v_sin_f32": lambda a: _u(np.sin(_TWO_PI * _f(a))),
    "v_cos_f32": lambda a: _u(np.cos(_TWO_PI * _f(a))),
}

BINARY: dict[str, Callable] = {
    "v_add_u32": lambda a, b: a + b,
    "v_sub_u32": lambda a, b: a - b,
    "v_subrev_u32": lambda a, b: b - a,
    "v_mul_u32_u24": lambda a, b: (a & 0xFFFFFF) * (b & 0xFFFFFF),
    "v_mul_lo_u32": lambda a, b: a * b,
    "v_mul_hi_u32": lambda a, b: ((a.astype(np.uint64) * b.astype(np.uint64)) >> np.uint64(32)).astype(np.uint32),
    "v_and_b32": lambda a, b: a & b,
    "v_or_b32": lambda a, b: a | b,
    "v_xor_b32": lambda a, b: a ^ b,
    "v_lshlrev_b32": lambda a, b: b << (a & 31),
    "v_lshrrev_b32": lambda a, b: b >> (a & 31),
    "v_ashrrev_i32": lambda a, b: _u(_i(b) >> _i(a & 31)),
    "v_min_u32": np.minimum,
    "v_max_u32": np.maximum,
    "v_min_i32": lambda a, b: _u(np.minimum(_i(a), _i(b))),
    "v_max_i32": lambda a, b: _u(np.maximum(_i(a), _i(b))),
    "v_add_f32": lambda a, b: _u(_f(a) + _f(b)),
    "v_sub_f32": lambda a, b: _u(_f(a) - _f(b)),
    "v_subrev_f32": lambda a, b: _u(_f(b) - _f(a)),
    "v_mul_f32": lambda a, b: _u(_f(a) * _f(b)),
    "v_min_f32": lambda a, b: _u(np.fmin(_f(a), _f(b))),
    "v_max_f32": lambda a, b: _u(np.fmax(_f(a), _f(b))),
}

TERNARY: dict[str, Callable] = {
    # v_mad_f32 rounds after the multiply; v_fma_f32 is fused (computed in
    # double precision, which is exact for the product of two floats).
    "v_mad_f32": lambda a, b, c: _u(_f(a) * _f(b) + _f(c)),
    "v_fma_f32": lambda a, b, c: _u((_f(a).astype(np.float64) * _f(b) + _f(c)).astype(np.float32)),
    "v_mad_u32_u24": lambda a, b, c: (a & 0xFFFFFF) * (b & 0xFFFFFF) + c,
}

FLOPS = {"v_add_f32": 1, "v_sub_f32": 1, "v_subrev_f32": 1, "v_mul_f32": 1,
         "v_mad_f32": 2, "v_fma_f32": 2, "v_mac_f32": 2}

_VCMP = {"lt": np.less, "le": np.less_equal, "gt": np.greater,
         "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal}
_VCONV = {"f32": _f, "u32": lambda x: x, "i32": _i}
_SCMP = {"eq": lambda a, b: a == b, "lg": lambda a, b: a != b, "lt": lambda a, b: a < b,
         "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}


# ---------------------------------------------------------------------------
# Wavefront state
# ---------------------------------------------------------------------------

class Wavefront:
    def __init__(self, kernel: "CompiledKernel", vram: VRAM, lds: np.ndarray,
                 sgprs: dict[int, int], local_ids: np.ndarray, exec_mask: int,
                 watchdog: int):
        self.kernel = kernel
        self.vram = vram
        self.lds = lds
        self.watchdog = watchdog
        self.vgpr = np.zeros((max(kernel.program.vgpr_count, 3), LANES), dtype=np.uint32)
        self.vgpr[0:3] = local_ids
        self.sgpr = [0] * MAX_SGPRS
        for index, value in sgprs.items():
            self.sgpr[index] = value & M32
        self.vcc = 0
        self.scc = 0
        self.pc = 0
        self.state = RUNNING
        self.set_exec(exec_mask)

        # Performance counters
        self.n_instr = 0
        self.n_valu = 0
        self.n_salu = 0
        self.n_branch = 0
        self.n_smem = 0
        self.n_vmem = 0
        self.n_lds = 0
        self.lane_ops = 0          # sum of active lanes over VALU instructions
        self.flops = 0
        self.simd_cycles = 0       # vector issue time on this wave's SIMD
        self.scalar_cycles = 0     # time on the CU's shared scalar unit
        self.lds_cycles = 0        # time on the CU's LDS, including bank conflicts
        self.dram_bytes = 0        # traffic in 64-byte cache lines

    def set_exec(self, mask: int) -> None:
        mask &= FULL_MASK
        self.exec = mask
        self.exec_full = mask == FULL_MASK
        self.exec_bool = mask_to_bool(mask)
        self.active = popcount(mask)

    def write_v(self, reg: int, value: np.ndarray) -> None:
        if self.exec_full:
            self.vgpr[reg] = value
        elif self.active:
            np.copyto(self.vgpr[reg], value, where=self.exec_bool)

    def account_valu(self, cycles: int, flops_per_lane: int) -> None:
        self.n_valu += 1
        self.simd_cycles += cycles
        self.lane_ops += self.active
        self.flops += flops_per_lane * self.active

    def run(self) -> None:
        """Execute until the wave ends (s_endpgm) or waits at s_barrier."""
        code = self.kernel.code
        n = len(code)
        with np.errstate(all="ignore"):
            while self.state == RUNNING:
                pc = self.pc
                if pc >= n:
                    raise GPUExecutionError(
                        f"kernel '{self.kernel.program.name}' ran past its last instruction")
                self.pc = pc + 1
                self.n_instr += 1
                if self.n_instr > self.watchdog:
                    ins = self.kernel.program.instructions[pc]
                    raise GPUHangError(
                        f"kernel '{self.kernel.program.name}' hung: a wavefront executed more than "
                        f"{self.watchdog:,} instructions (last at line {ins.line_no}: {ins.text})")
                try:
                    code[pc](self)
                except GPUMemoryFault as exc:
                    ins = self.kernel.program.instructions[pc]
                    raise GPUMemoryFault(
                        f"{exc}\n  in kernel '{self.kernel.program.name}', line {ins.line_no}: "
                        f"{ins.text}") from None


# ---------------------------------------------------------------------------
# Operand accessors
# ---------------------------------------------------------------------------

def _vsrc(op: Operand) -> Callable[[Wavefront], np.ndarray]:
    if op.kind == "v":
        n = op.value
        return lambda w: w.vgpr[n]
    if op.kind == "s":
        n = op.value
        return lambda w: np.full(LANES, w.sgpr[n], dtype=np.uint32)
    if op.kind == "imm":
        const = np.full(LANES, op.value & M32, dtype=np.uint32)
        const.flags.writeable = False
        return lambda w: const
    raise AssertionError(op)


def _s32(op: Operand) -> Callable[[Wavefront], int]:
    if op.kind == "s":
        n = op.value
        return lambda w: w.sgpr[n]
    const = op.value & M32
    return lambda w: const


def _s64(op: Operand) -> Callable[[Wavefront], int]:
    if op.kind == "s":
        n = op.value
        return lambda w: w.sgpr[n] | (w.sgpr[n + 1] << 32)
    if op.kind == "exec":
        return lambda w: w.exec
    if op.kind == "vcc":
        return lambda w: w.vcc
    const = op.value & FULL_MASK
    return lambda w: const


def _w32(op: Operand) -> Callable[[Wavefront, int], None]:
    n = op.value

    def write(w: Wavefront, value: int) -> None:
        w.sgpr[n] = value & M32
    return write


def _w64(op: Operand) -> Callable[[Wavefront, int], None]:
    if op.kind == "exec":
        return lambda w, value: w.set_exec(value)
    if op.kind == "vcc":
        def write_vcc(w: Wavefront, value: int) -> None:
            w.vcc = value & FULL_MASK
        return write_vcc
    n = op.value

    def write(w: Wavefront, value: int) -> None:
        w.sgpr[n] = value & M32
        w.sgpr[n + 1] = (value >> 32) & M32
    return write


# ---------------------------------------------------------------------------
# Instruction translation
# ---------------------------------------------------------------------------

def _compile_valu(ins: Instruction, category: str) -> Callable[[Wavefront], None]:
    op = ins.opcode
    ops = ins.operands
    cycles = CYCLES[category]
    flops = FLOPS.get(op, 0)

    if op == "v_readfirstlane_b32":
        write = _w32(ops[0])
        src = ops[1].value

        def run(w: Wavefront) -> None:
            m = w.exec
            lane = (m & -m).bit_length() - 1 if m else 0
            write(w, int(w.vgpr[src][lane]))
            w.account_valu(cycles, 0)
        return run

    dst = ops[0].value
    if op == "v_cndmask_b32":
        ra, rb, rmask = _vsrc(ops[1]), _vsrc(ops[2]), _s64(ops[3])

        def run(w: Wavefront) -> None:
            w.write_v(dst, np.where(mask_to_bool(rmask(w)), rb(w), ra(w)))
            w.account_valu(cycles, flops)
    elif op == "v_mac_f32":
        ra, rb = _vsrc(ops[1]), _vsrc(ops[2])

        def run(w: Wavefront) -> None:
            w.write_v(dst, _u(_f(ra(w)) * _f(rb(w)) + _f(w.vgpr[dst])))
            w.account_valu(cycles, flops)
    elif op in UNARY:
        fn, ra = UNARY[op], _vsrc(ops[1])

        def run(w: Wavefront) -> None:
            w.write_v(dst, fn(ra(w)))
            w.account_valu(cycles, flops)
    elif op in BINARY:
        fn, ra, rb = BINARY[op], _vsrc(ops[1]), _vsrc(ops[2])

        def run(w: Wavefront) -> None:
            w.write_v(dst, fn(ra(w), rb(w)))
            w.account_valu(cycles, flops)
    elif op in TERNARY:
        fn, ra, rb, rc = TERNARY[op], _vsrc(ops[1]), _vsrc(ops[2]), _vsrc(ops[3])

        def run(w: Wavefront) -> None:
            w.write_v(dst, fn(ra(w), rb(w), rc(w)))
            w.account_valu(cycles, flops)
    else:
        raise AssertionError(op)
    return run


def _compile_vcmp(ins: Instruction) -> Callable[[Wavefront], None]:
    _, _, cond, typ = ins.opcode.split("_")
    cmp, conv = _VCMP[cond], _VCONV[typ]
    write = _w64(ins.operands[0])
    ra, rb = _vsrc(ins.operands[1]), _vsrc(ins.operands[2])

    def run(w: Wavefront) -> None:
        result = cmp(conv(ra(w)), conv(rb(w)))
        if not w.exec_full:
            result &= w.exec_bool  # inactive lanes write 0
        write(w, bool_to_mask(result))
        w.account_valu(4, 0)
    return run


def _salu_result(w: Wavefront, write: Callable, result: int, scc: bool | None) -> None:
    write(w, result)
    if scc is not None:
        w.scc = int(scc)
    w.n_salu += 1
    w.scalar_cycles += 1


def _compile_salu(ins: Instruction) -> Callable[[Wavefront], None]:
    op, ops = ins.opcode, ins.operands

    if op in ("s_and_saveexec_b64", "s_or_saveexec_b64"):
        write, rs = _w64(ops[0]), _s64(ops[1])
        use_and = op == "s_and_saveexec_b64"

        def run(w: Wavefront) -> None:
            src, old = rs(w), w.exec
            w.set_exec(src & old if use_and else src | old)
            _salu_result(w, write, old, w.exec != 0)
        return run

    if op == "s_bcnt1_i32_b64":
        write, rs = _w32(ops[0]), _s64(ops[1])

        def run(w: Wavefront) -> None:
            r = popcount(rs(w))
            _salu_result(w, write, r, r != 0)
        return run

    if op.endswith("_b64"):
        write = _w64(ops[0])
        srcs = [_s64(o) for o in ops[1:]]
        mask = FULL_MASK
    else:
        write = _w32(ops[0])
        srcs = [_s32(o) for o in ops[1:]]
        mask = M32

    table: dict[str, Callable] = {
        "s_mov_b32": lambda a: (a, None),
        "s_mov_b64": lambda a: (a, None),
        "s_not_b32": lambda a: (~a & mask, (~a & mask) != 0),
        "s_not_b64": lambda a: (~a & mask, (~a & mask) != 0),
        "s_add_u32": lambda a, b: ((a + b) & mask, a + b > mask),
        "s_sub_u32": lambda a, b: ((a - b) & mask, a < b),
        "s_mul_i32": lambda a, b: ((a * b) & mask, None),
        "s_lshl_b32": lambda a, b: ((a << (b & 31)) & mask, ((a << (b & 31)) & mask) != 0),
        "s_lshr_b32": lambda a, b: (a >> (b & 31), (a >> (b & 31)) != 0),
        "s_min_u32": lambda a, b: (min(a, b), a < b),
        "s_max_u32": lambda a, b: (max(a, b), a > b),
    }
    for name, fn in (("and", lambda a, b: a & b), ("or", lambda a, b: a | b),
                     ("xor", lambda a, b: a ^ b), ("andn2", lambda a, b: a & ~b & mask)):
        for width in ("b32", "b64"):
            table[f"s_{name}_{width}"] = (lambda f: lambda a, b: (f(a, b), f(a, b) != 0))(fn)
    fn = table[op]

    if len(srcs) == 1:
        (ra,) = srcs

        def run(w: Wavefront) -> None:
            r, scc = fn(ra(w))
            _salu_result(w, write, r, scc)
    else:
        ra, rb = srcs

        def run(w: Wavefront) -> None:
            r, scc = fn(ra(w), rb(w))
            _salu_result(w, write, r, scc)
    return run


def _compile_scmp(ins: Instruction) -> Callable[[Wavefront], None]:
    _, _, cond, typ = ins.opcode.split("_")
    cmp = _SCMP[cond]
    ra, rb = _s32(ins.operands[0]), _s32(ins.operands[1])
    signed = typ == "i32"

    def run(w: Wavefront) -> None:
        a, b = ra(w), rb(w)
        if signed:
            a, b = _signed32(a), _signed32(b)
        w.scc = int(cmp(a, b))
        w.n_salu += 1
        w.scalar_cycles += 1
    return run


def _compile_branch(ins: Instruction, program: Program) -> Callable[[Wavefront], None]:
    target = program.labels[ins.operands[0].value]
    cond: Callable[[Wavefront], bool] = {
        "s_branch": lambda w: True,
        "s_cbranch_scc0": lambda w: w.scc == 0,
        "s_cbranch_scc1": lambda w: w.scc == 1,
        "s_cbranch_execz": lambda w: w.exec == 0,
        "s_cbranch_execnz": lambda w: w.exec != 0,
        "s_cbranch_vccz": lambda w: w.vcc == 0,
        "s_cbranch_vccnz": lambda w: w.vcc != 0,
    }[ins.opcode]

    def run(w: Wavefront) -> None:
        if cond(w):
            w.pc = target
        w.n_branch += 1
        w.scalar_cycles += 1
    return run


def _compile_special(ins: Instruction) -> Callable[[Wavefront], None]:
    op = ins.opcode
    if op == "s_endpgm":
        def run(w: Wavefront) -> None:
            w.state = DONE
    elif op == "s_barrier":
        def run(w: Wavefront) -> None:
            w.state = BARRIER
            w.scalar_cycles += 1
    elif op == "s_nop":
        stall = (ins.operands[0].value & 0xF) + 1

        def run(w: Wavefront) -> None:
            w.simd_cycles += stall
    else:  # s_waitcnt: memory is synchronous in this model
        def run(w: Wavefront) -> None:
            w.scalar_cycles += 1
    return run


def _compile_smem(ins: Instruction) -> Callable[[Wavefront], None]:
    dst = ins.operands[0].value
    count = ins.operands[0].count
    rbase = _s64(ins.operands[1])
    offset = ins.operands[2].value
    steps = np.arange(count, dtype=np.uint64) * np.uint64(4)

    def run(w: Wavefront) -> None:
        addr = (rbase(w) + offset) & FULL_MASK
        values = w.vram.read_u32(steps + np.uint64(addr))
        w.sgpr[dst:dst + count] = [int(v) for v in values]
        w.n_smem += 1
        w.scalar_cycles += 1
        w.dram_bytes += 4 * count
    return run


def _lane_addresses(w: Wavefront, voff: int, base: int) -> np.ndarray:
    off = w.vgpr[voff] if w.exec_full else w.vgpr[voff][w.exec_bool]
    return off.astype(np.uint64) + np.uint64(base & FULL_MASK)


def _compile_vmem(ins: Instruction, category: str) -> Callable[[Wavefront], None]:
    reg, voff = ins.operands[0].value, ins.operands[1].value
    rbase = _s64(ins.operands[2])
    load = category == "vmem_load"

    def run(w: Wavefront) -> None:
        w.n_vmem += 1
        w.simd_cycles += 4
        if not w.active:
            return
        addrs = _lane_addresses(w, voff, rbase(w))
        if load:
            values = w.vram.read_u32(addrs)
            if w.exec_full:
                w.vgpr[reg] = values
            else:
                w.vgpr[reg][w.exec_bool] = values
        else:
            values = w.vgpr[reg] if w.exec_full else w.vgpr[reg][w.exec_bool]
            w.vram.write_u32(addrs, values)
        # Coalescing: DRAM traffic is whole cache lines, not requested bytes.
        w.dram_bytes += CACHE_LINE * np.unique(addrs >> np.uint64(6)).size
    return run


def _lds_bank_cycles(addrs: np.ndarray, active: np.ndarray) -> int:
    """GCN LDS: 32 banks of dwords, serviced one half-wave at a time.
    Lanes reading the same dword are broadcast; distinct dwords in the same
    bank serialise."""
    dwords = addrs >> 2
    cycles = 0
    for half in (slice(0, 32), slice(32, 64)):
        sel = active[half]
        if sel.any():
            unique = np.unique(dwords[half][sel])
            cycles += int(np.bincount(unique & 31).max())
    return cycles


def _compile_lds(ins: Instruction, category: str) -> Callable[[Wavefront], None]:
    read = category == "lds_read"
    if read:
        dst, vaddr = ins.operands[0].value, ins.operands[1].value
    else:
        vaddr, src = ins.operands[0].value, ins.operands[1].value

    def run(w: Wavefront) -> None:
        w.n_lds += 1
        w.simd_cycles += 4
        if not w.active:
            return
        all_addrs = w.vgpr[vaddr]
        addrs = all_addrs if w.exec_full else all_addrs[w.exec_bool]
        size = w.lds.size * 4
        if np.any(addrs & 3) or np.any(addrs >= size):
            bad = int(addrs[((addrs & 3) != 0) | (addrs >= size)][0])
            raise GPUMemoryFault(
                f"LDS access at 0x{bad:x} is misaligned or outside the {size}-byte allocation")
        index = (addrs >> 2).astype(np.intp)
        if read:
            if w.exec_full:
                w.vgpr[dst] = w.lds[index]
            else:
                w.vgpr[dst][w.exec_bool] = w.lds[index]
        else:
            w.lds[index] = w.vgpr[src] if w.exec_full else w.vgpr[src][w.exec_bool]
        w.lds_cycles += _lds_bank_cycles(all_addrs, w.exec_bool)
    return run


@dataclass
class CompiledKernel:
    program: Program
    code: list

    @property
    def name(self) -> str:
        return self.program.name


def compile_program(program: Program) -> CompiledKernel:
    code = []
    for ins in program.instructions:
        category, _ = OPCODES[ins.opcode]
        if category.startswith("valu"):
            fn = _compile_valu(ins, category)
        elif category == "vcmp":
            fn = _compile_vcmp(ins)
        elif category == "salu":
            fn = _compile_salu(ins)
        elif category == "scmp":
            fn = _compile_scmp(ins)
        elif category == "branch":
            fn = _compile_branch(ins, program)
        elif category == "special":
            fn = _compile_special(ins)
        elif category == "smem":
            fn = _compile_smem(ins)
        elif category.startswith("vmem"):
            fn = _compile_vmem(ins, category)
        elif category.startswith("lds"):
            fn = _compile_lds(ins, category)
        else:
            raise AssertionError(category)
        code.append(fn)
    return CompiledKernel(program, code)
