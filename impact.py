"""Select tests using pinned source and per-test coverage measured at build time."""

from __future__ import annotations

import ast
import gzip
import json
from functools import cache
from pathlib import Path

from patch_parser import FilePatch, parse_patch

SOURCE_COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
CALLABLES = (*FUNCTIONS, ast.Lambda)
YIELDS = (ast.Yield, ast.YieldFrom)
OPERATOR_KINDS = (ast.operator, ast.boolop, ast.cmpop, ast.unaryop)
EXPRESSION_FIELDS = {ast.If: "test", ast.Assign: "value", ast.AnnAssign: "value", ast.Return: "value"}


class UnknownImpact(ValueError):
    """There is insufficient evidence to narrow the test set."""


def changed_callables(before: ast.AST, after: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
    """Pair callables with body edits, or reject a structural edit."""
    callables = []

    def compare(old, new):
        if type(old) is not type(new):
            raise UnknownImpact("changed AST structure")
        if isinstance(old, CALLABLES):
            for field, value in ast.iter_fields(old):
                if field != "body":
                    compare(value, getattr(new, field))
            old_body = old.body if isinstance(old.body, list) else [old.body]
            new_body = new.body if isinstance(new.body, list) else [new.body]
            if [ast.dump(node) for node in old_body] != [ast.dump(node) for node in new_body]:
                if any(isinstance(node, YIELDS) for function in (old, new) for node in ast.walk(function)):
                    # A generator can be called without running any of its body.
                    raise UnknownImpact("changed generator function")
                callables.append((old, new))
        elif isinstance(old, ast.AST):
            for field, value in ast.iter_fields(old):
                compare(value, getattr(new, field))
        elif isinstance(old, list):
            if len(old) != len(new):
                raise UnknownImpact("changed statement or argument count")
            for old_item, new_item in zip(old, new):
                compare(old_item, new_item)
        elif old != new:
            raise UnknownImpact("changed value outside a function body")

    compare(before, after)
    return callables


def same_expression_shape(old, new) -> bool:
    """Allow value/operator edits with the same binding and expression shape."""
    if any(isinstance(old, kind) and isinstance(new, kind) for kind in OPERATOR_KINDS):
        return True
    if type(old) is not type(new):
        return False
    if isinstance(old, (ast.Lambda, ast.GeneratorExp)):
        return False
    if isinstance(old, ast.Constant):
        return type(old.value) is type(new.value)
    if isinstance(old, ast.AST):
        return all(same_expression_shape(value, getattr(new, field)) for field, value in ast.iter_fields(old))
    if isinstance(old, list):
        return len(old) == len(new) and all(same_expression_shape(a, b) for a, b in zip(old, new))
    return old == new


def changed_statements(before: ast.AST, after: ast.AST) -> list[tuple[int, int]]:
    """Find old statement spans for edits confined to supported expressions."""
    spans = []

    def compare(old, new):
        if type(old) is not type(new):
            return False
        if isinstance(old, ast.AST):
            if isinstance(old, (*CALLABLES, ast.ClassDef)):
                return ast.dump(old) == ast.dump(new)
            expression_field = EXPRESSION_FIELDS.get(type(old))
            for field, value in ast.iter_fields(old):
                replacement = getattr(new, field)
                if field == expression_field and isinstance(value, ast.expr):
                    if ast.dump(value) != ast.dump(replacement):
                        if not same_expression_shape(value, replacement):
                            return False
                        spans.append((old.lineno, value.end_lineno))
                elif not compare(value, replacement):
                    return False
            return True
        if isinstance(old, list):
            return len(old) == len(new) and all(compare(a, b) for a, b in zip(old, new))
        return old == new

    return spans if isinstance(before, FUNCTIONS) and compare(before.body, after.body) else []


class ImpactIndex:
    def __init__(self, pinned: Path, coverage_path: Path, tests: list[dict]):
        self.pinned = pinned
        with gzip.open(coverage_path, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        if raw["source_commit"] != SOURCE_COMMIT or not raw["tests"] or not raw["lines"]:
            raise UnknownImpact("coverage does not match pinned sqlglot")

        test_bits = {node_id: 1 << index for index, node_id in enumerate(raw["tests"])}
        self.lines = raw["lines"]
        self.shared = raw["shared"]
        self.startup = raw["startup_lines"]
        self.always = set()
        self.numbers = {}
        unreliable = set(raw["unreliable"])
        for test in tests:
            node_id, number = test["id"], test["number"]
            if node_id in unreliable or node_id not in test_bits:
                self.always.add(number)
            else:
                self.numbers[test_bits[node_id]] = number

    @cache
    def line_masks(self, path: str) -> dict[int, int]:
        masks = {int(line): int(bits, 16) for line, bits in self.lines.get(path, {}).items()}
        for line, bits in self.shared.get(path, {}).items():
            number = int(line)
            masks[number] = masks.get(number, 0) | int(bits, 16)
        return masks

    @cache
    def original(self, path: str) -> tuple[str, ast.AST]:
        contents = (self.pinned / path).read_bytes()
        return contents.decode("utf-8"), ast.parse(contents, filename=path, type_comments=True)

    def python_impact(self, file: FilePatch, path: str) -> int:
        original, before = self.original(path)
        patched = file.apply(original).encode("utf-8")
        compile(patched, path, "exec", dont_inherit=True)
        after = ast.parse(patched, filename=path, type_comments=True)
        callables = changed_callables(before, after)
        lines = self.line_masks(path)
        mask = lines.get(0, 0)
        startup = self.startup.get(path, ())
        if 0 in startup:
            raise UnknownImpact("source read directly during collection")
        for function, replacement in callables:
            decorators = function.decorator_list if isinstance(function, FUNCTIONS) else ()
            start = min((node.lineno for node in decorators), default=function.lineno)
            if any(start <= line <= function.end_lineno for line in startup):
                raise UnknownImpact("changed function ran during collection")
            hits = 0
            for line in range(start, function.end_lineno + 1):
                hits |= lines.get(line, 0)
            if not hits:
                raise UnknownImpact("uncovered function")
            spans = changed_statements(function, replacement)
            statement_hits = 0
            for first, last in spans:
                span_hits = 0
                for line in range(first, last + 1):
                    span_hits |= lines.get(line, 0)
                if not span_hits:
                    # An absent span does not establish that it was reachable.
                    statement_hits = 0
                    break
                statement_hits |= span_hits
            mask |= statement_hits or hits
        return mask

    def file_impact(self, file: FilePatch) -> int:
        path = file.old_path
        if path is None or file.new_path is None:
            raise UnknownImpact("added or deleted file")
        if path.startswith("sqlglot/") and path.endswith(".py"):
            return self.python_impact(file, path)
        if path.startswith("tests/fixtures/"):
            if 0 in self.startup.get(path, ()):
                raise UnknownImpact("fixture read during collection")
            readers = self.line_masks(path).get(0, 0)
            if readers:
                return readers
        raise UnknownImpact("unmapped file")

    def select(self, patch_text: str) -> frozenset[int] | None:
        """Return selected test numbers, or None when the full suite is needed."""
        try:
            mask = 0
            for file in parse_patch(patch_text):
                mask |= self.file_impact(file)
        except (ValueError, OSError, SyntaxError, RecursionError):
            return None

        selected = set(self.always)
        while mask:
            bit = mask & -mask
            if bit in self.numbers:
                selected.add(self.numbers[bit])
            mask ^= bit
        return frozenset(selected)
