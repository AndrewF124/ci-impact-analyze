"""Track eligible basic-block entries in the pinned interpreter."""

from __future__ import annotations

import dis
import hashlib
import inspect
import tracemalloc
from functools import cache
from types import CodeType


JUMPS = frozenset(dis.hasjump if hasattr(dis, "hasjump") else dis.hasjrel + dis.hasjabs)
EXITS = {"RETURN_VALUE", "RETURN_CONST", "RAISE_VARARGS", "RERAISE"}
UNSUPPORTED_FLAGS = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
LITERALS = (type(None), bool, int, float, complex, str, bytes)


def identity(code):
    return f"{code.co_qualname}:{code.co_firstlineno}"


def fingerprint(code):
    data = (code.co_code, code.co_consts, code.co_names, code.co_varnames, code.co_flags,
            code.co_posonlyargcount, code.co_argcount, code.co_kwonlyargcount,
            code.co_cellvars, code.co_freevars, code.co_exceptiontable)
    return hashlib.sha256(repr(data).encode()).hexdigest()


@cache
def layout(code):
    """Map instruction offsets to their blocks for small, ordinary functions."""
    if (not code.co_flags & inspect.CO_NEWLOCALS or code.co_flags & UNSUPPORTED_FLAGS
            or code.co_exceptiontable or any(type(c) not in LITERALS for c in code.co_consts)):
        return None
    instructions = tuple(dis.get_instructions(code))
    if not instructions or len(instructions) > 180:
        return None
    starts = {instructions[0].offset}
    for index, op in enumerate(instructions):
        if op.opcode in JUMPS:
            starts.add(op.argval)
        elif op.opname not in EXITS:
            continue
        if index + 1 < len(instructions):
            starts.add(instructions[index + 1].offset)
    if not 2 < len(starts) <= 32:
        return None
    # Coverage of separate source lines already handles most functions. Track
    # only ones with a live source line shared across possible block entries.
    current = instructions[0].offset
    by_offset = {}
    lines = {}
    for op in instructions:
        if op.offset in starts:
            current = op.offset
        by_offset[op.offset] = current
        if op.positions.lineno is not None:
            lines.setdefault(op.positions.lineno, set()).add(current)
    if not any(len(blocks) > 1 for blocks in lines.values()):
        return None
    return instructions, by_offset, starts


def by_identity(code):
    for part in code.co_consts:
        if isinstance(part, CodeType):
            yield identity(part), part
            yield from by_identity(part)


class BlockTrace:
    def __init__(self, coverage):
        self.coverage = coverage
        self.code_rows = {}
        self.files = {}

    def start(self, path, code):
        if not path.startswith("sqlglot/"):
            return
        mask = self.coverage.mask | self.coverage.shared_mask
        if mask & self.coverage.line_only_mask:
            return
        trace = layout(code)
        if trace is None:
            return
        instructions, _, starts = trace
        if code not in self.code_rows:
            key = identity(code)
            digest = fingerprint(code)
            row = self.files.setdefault(path, {}).setdefault(key, [digest, 0, 0, False, {}])
            if row[0] != digest:
                row[0] = ""
            self.code_rows[code] = row, starts

        row, _ = self.code_rows[code]
        row[3] |= self.coverage.capture_startup
        if not mask or tracemalloc.is_tracing():
            row[2] |= mask
            return
        row[1] |= mask
        entry = instructions[0].offset
        row[4][entry] = row[4].get(entry, 0) | mask
        monitor = self.monitor
        local = monitor.get_local_events(monitor.COVERAGE_ID, code)
        if not local & monitor.events.INSTRUCTION:
            monitor.set_local_events(monitor.COVERAGE_ID, code, local | monitor.events.INSTRUCTION)

    def instruction(self, code, offset):
        row, starts = self.code_rows[code]
        if offset in starts:
            mask = self.coverage.mask | self.coverage.shared_mask
            if mask & self.coverage.line_only_mask or tracemalloc.is_tracing():
                row[2] |= mask
            elif mask:
                row[4][offset] = row[4].get(offset, 0) | mask
        return self.monitor.DISABLE

    def begin(self, monitor):
        self.monitor = monitor
        monitor.register_callback(monitor.COVERAGE_ID, monitor.events.INSTRUCTION, self.instruction)

    def suspend(self):
        tool = self.monitor.COVERAGE_ID
        flag = self.monitor.events.INSTRUCTION
        for code in self.code_rows:
            local = self.monitor.get_local_events(tool, code)
            if local & flag:
                self.monitor.set_local_events(tool, code, local & ~flag)

    def encoded(self):
        return {path: {key: [digest, format(seen, "x"), format(unknown, "x"), startup,
                            {str(offset): format(bits, "x") for offset, bits in blocks.items()}]
                       for key, (digest, seen, unknown, startup, blocks) in entries.items()}
                for path, entries in self.files.items()}

def unchanged_blocks(profile, old_module, new_module, old_function, new_function):
    """Prove the changed instructions were outside recorded execution."""
    old_functions = [(key, code) for key, code in by_identity(old_module)
                     if code.co_name == old_function.name and code.co_firstlineno == old_function.lineno]
    new_functions = [code for _, code in by_identity(new_module)
                     if code.co_name == new_function.name and code.co_firstlineno == new_function.lineno]
    if len(old_functions) != 1 or len(new_functions) != 1:
        return 0
    (key, old), new = old_functions[0], new_functions[0]
    if key not in profile:
        return 0
    digest, seen, unknown, startup, covered = profile[key]
    if startup or fingerprint(old) != digest:
        return 0
    fields = ("co_qualname", "co_flags", "co_names", "co_varnames", "co_cellvars", "co_freevars",
              "co_exceptiontable", "co_argcount", "co_posonlyargcount", "co_kwonlyargcount")
    if any(getattr(old, field) != getattr(new, field) for field in fields):
        return 0
    info = layout(old)
    if info is None:
        return 0
    before, blocks, _ = info
    after = tuple(dis.get_instructions(new))
    if [(op.offset, op.opname, op.positions) for op in before] != [(op.offset, op.opname, op.positions) for op in after]:
        return 0
    affected = set()
    for a, b in zip(before, after):
        if a.opname == "LOAD_CONST" and type(a.argval) is type(b.argval) and type(a.argval) in LITERALS:
            if repr(a.argval) != repr(b.argval):
                affected.add(blocks[a.offset])
        elif a.arg != b.arg:
            if a.opname not in {"COMPARE_OP", "IS_OP"}:
                return 0
            affected.add(blocks[a.offset])
    if not affected:
        return 0
    hit = int(unknown, 16)
    for offset in affected:
        hit |= int(covered.get(str(offset), "0"), 16)
    return int(seen, 16) & ~hit
