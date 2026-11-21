"""Select tests using pinned source and per-test coverage measured at build time."""

from __future__ import annotations

import ast
import gzip
import json
from functools import cache
from pathlib import Path

from patch_parser import FilePatch, parse_patch
from semantic import comparison_edits, derived_field_edits, literal_field_edits, site_key

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
    def __init__(self, pinned: Path, coverage_path: Path, tests: list[dict], isolation_path=None):
        self.pinned = pinned
        with gzip.open(coverage_path, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        if raw["source_commit"] != SOURCE_COMMIT or not raw["tests"] or not raw["lines"]:
            raise UnknownImpact("coverage does not match pinned sqlglot")

        test_bits = {node_id: 1 << index for index, node_id in enumerate(raw["tests"])}
        self.test_bits = test_bits
        self.lines = raw["lines"]
        self.shared = raw["shared"]
        self.startup = raw["startup_lines"]
        self.semantic = raw.get("semantic", {})
        self.isolated = {}
        if isolation_path is not None:
            try:
                with gzip.open(isolation_path, "rt", encoding="utf-8") as stream:
                    profiles = json.load(stream)
            except (OSError, ValueError, EOFError):
                profiles = {}
            for node_id, paths in profiles.items():
                bit = test_bits.get(node_id, 0)
                for path, lines in paths.items():
                    hits = self.isolated.setdefault(path, {})
                    for line in lines:
                        hits[line] = hits.get(line, 0) | bit
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

    def observations(self, kind: str, key: str) -> tuple[int, bool]:
        bits, startup = self.semantic.get(kind, {}).get(key, ("0", False))
        return int(bits, 16), startup

    def equivalent_comparison(self, original: str, patched: bytes, path: str) -> int:
        if not self.semantic.get("comparison_seen"):
            return 0
        site = comparison_edits(original.encode("utf-8"), patched)
        if site is None:
            return 0
        key = f"{path}:{site_key(site)}"
        seen, startup = self.observations("comparison_seen", key)
        uncertain, _ = self.observations("comparison_uncertain", key)
        incomplete, _ = self.observations("incomplete", "*")
        uncertain |= incomplete
        return 0 if startup else seen & ~uncertain

    def python_impact(self, file: FilePatch, path: str) -> int:
        original, before = self.original(path)
        patched = file.apply(original).encode("utf-8")
        compile(patched, path, "exec", dont_inherit=True)
        after = ast.parse(patched, filename=path, type_comments=True)
        callables = changed_callables(before, after)
        lines = self.line_masks(path)
        isolated = self.isolated.get(path, {})
        mask = lines.get(0, 0) | isolated.get(0, 0)
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
            isolated_hits = 0
            # A cold import can bind a definition without running its body.
            for line in range(function.lineno + 1, function.end_lineno + 1):
                isolated_hits |= isolated.get(line, 0)
            if not hits:
                raise UnknownImpact("uncovered function")
            spans = changed_statements(function, replacement)
            statement_hits = 0
            isolated_statements = 0
            for first, last in spans:
                span_hits = 0
                for line in range(first, last + 1):
                    span_hits |= lines.get(line, 0)
                    isolated_statements |= isolated.get(line, 0)
                if not span_hits:
                    # An absent span does not establish that it was reachable.
                    statement_hits = 0
                    break
                statement_hits |= span_hits
            function_mask = (statement_hits | isolated_statements) if statement_hits else (hits | isolated_hits)
            if statement_hits and len(callables) == 1:
                function_mask &= ~self.equivalent_comparison(original, patched, path)
            mask |= function_mask
        return mask

    def patched_files(self, files: list[FilePatch]) -> dict[str, bytes]:
        edits = {}
        for file in files:
            path = file.old_path
            if path != file.new_path or not path or not path.startswith("sqlglot/") or not path.endswith(".py"):
                raise UnknownImpact("verification requires existing Python source")
            original, _ = self.original(path)
            edits[path] = file.apply(original).encode("utf-8")
        return edits

    def state_impact(self, files, verification):
        """Use measured consumers for restricted edits with a checked import result."""
        if not self.semantic.get("field"):
            return None
        edits = self.patched_files(files)
        derived = self.derived_impact(edits, verification)
        if derived is not None:
            return derived
        changes = {}
        mask = 0
        for path, patched in edits.items():
            original, _ = self.original(path)
            changed = literal_field_edits(original.encode("utf-8"), patched, path)
            if not changed or 0 in self.startup.get(path, ()):
                return None
            changes.update(changed)
            mask |= self.line_masks(path).get(0, 0)

        # Plain declarations can still feed another binding during class creation.
        # Compare a fresh import and require evidence for each state change.
        state_changes = verification.changed_state(edits)
        if state_changes is None or not changes.keys() <= state_changes:
            return None
        for field in state_changes:
            readers, startup = self.observations("field", field)
            unresolved, unresolved_startup = self.observations("unknown_owner", field.rsplit(":", 1)[1])
            if not readers or unresolved_startup or (startup and field not in changes):
                return None
            keys = changes.get(field)
            if keys and not startup:
                readers, _ = self.observations("whole_value", field)
                for key in keys:
                    bits, _ = self.observations("dict_key", f"{field}:{key}")
                    readers |= bits
            mask |= readers | unresolved
        incomplete, startup_incomplete = self.observations("incomplete", "*")
        if startup_incomplete:
            return None
        mask |= incomplete
        return self.test_numbers(mask) | self.always

    def derived_impact(self, edits, verification):
        changes = {}
        mask = 0
        for path, patched in edits.items():
            original, _ = self.original(path)
            allowed = derived_field_edits(original.encode("utf-8"), patched, path)
            if not allowed or 0 in self.startup.get(path, ()):
                return None
            changes.update(allowed)
            mask |= self.line_masks(path).get(0, 0)
        names = {field.rsplit(":", 1)[1] for field in changes}
        if names == {"arg_types"}:
            kind = "required"
            affected_names = {"arg_types", "required_args"}
            observed_field, source = "required_args", "sqlglot/expressions/core.py"
        elif names == {"SHOW_PARSERS"}:
            kind = "show"
            affected_names = names
            observed_field, source = "SHOW_PARSERS", "sqlglot/parser.py"
        elif names == {"NUMERIC_ESCAPES"}:
            kind = "numeric"
            affected_names = names
            observed_field, source = "NUMERIC_ESCAPES", "sqlglot/tokenizer_core.py"
        else:
            return None
        if not self.semantic.get(kind):
            return None
        state = verification.changed_state(edits, entries=True)
        expected = {f"{field}#{key}" for field, keys in changes.items() for key in keys}
        if state is None or not expected <= state:
            return None
        allowed_keys = set().union(*changes.values())
        for change in state:
            field, separator, key = change.partition("#")
            name = field.rsplit(":", 1)[1]
            if name not in affected_names or (separator and key not in allowed_keys):
                return None
            if separator and name == observed_field:
                readers, startup = self.observations(kind, key if kind == "numeric" else change)
                if startup:
                    return None
                mask |= readers
        uncertain, startup_uncertain = self.observations("derived_uncertain", kind)
        incomplete, startup_incomplete = self.observations("incomplete", "*")
        if startup_uncertain or startup_incomplete:
            return None
        # Memory-sensitive tests retain their normal coverage if they can reach
        # the validator or dispatch point that was unavailable to semantic tracing.
        functions = {
            "required": {"error_messages"},
            "show": {"_find_parser"},
            "numeric": {"_read_byte_escape", "_scan_numeric_escape"},
        }
        _, tree = self.original(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in functions[kind]:
                mask |= incomplete & self.line_masks(source).get(node.lineno, 0)
        return self.test_numbers(mask | uncertain) | self.always

    def tpc_replay_scope(self, edits: dict[str, bytes], node_id: str) -> bool:
        bit = self.test_bits.get(node_id, 0)
        if not bit or not self.semantic.get("tpc_inside"):
            return False
        for path, patched in edits.items():
            inside, _ = self.observations("tpc_inside", path)
            outside, _ = self.observations("tpc_outside", path)
            if not inside & bit or outside & bit or self.line_masks(path).get(0, 0) & bit:
                return False
            _, original = self.original(path)
            changes = changed_callables(original, ast.parse(patched, type_comments=True))
            if not changes or not all(changed_statements(a, b) for a, b in changes):
                return False
        return True

    def query_cases(self, edits, node_id, profile):
        """Map supported body edits to isolated cases, refusing setup changes."""
        bit = self.test_bits.get(node_id, 0)
        if not bit or not profile:
            return None
        cases = 0
        for path, patched in edits.items():
            if self.line_masks(path).get(0, 0) & bit:
                return None
            _, original = self.original(path)
            functions = changed_callables(original, ast.parse(patched, type_comments=True))
            lines = profile["lines"].get(path, {})
            for before, after in functions:
                spans = changed_statements(before, after)
                if not spans or any(before.lineno <= line <= before.end_lineno for line in self.startup.get(path, ())):
                    return None
                for first, last in spans:
                    for line in range(first, last + 1):
                        covered = int(lines.get(str(line), "0"), 16)
                        if not covered and self.isolated.get(path, {}).get(line, 0) & bit:
                            return None
                        cases |= covered
        if cases & 1:
            return None
        return [i for i in range(len(profile["times"])) if cases & (2 << i)]

    def warm_plan(self, edits, node_ids):
        """Restrict in-memory replacement to plain, post-import function bodies."""
        test_mask = 0
        for node_id in node_ids:
            test_mask |= self.test_bits.get(node_id, 0)
        plan = {}
        for path, patched in edits.items():
            if self.line_masks(path).get(0, 0) & test_mask or 0 in self.startup.get(path, ()):
                return None
            _, before = self.original(path)
            locations = []
            for old, new in changed_callables(before, ast.parse(patched, type_comments=True)):
                if (not isinstance(old, FUNCTIONS) or old.decorator_list
                        or any(old.lineno <= line <= old.end_lineno for line in self.startup.get(path, ()))):
                    return None
                locations.append((old.name, old.lineno, new.lineno))
            if locations:
                plan[path] = locations
        return plan or None

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
            return self.select_files(parse_patch(patch_text))
        except (ValueError, OSError, SyntaxError, RecursionError):
            return None

    def select_files(self, files: list[FilePatch]) -> frozenset[int] | None:
        try:
            mask = 0
            for file in files:
                mask |= self.file_impact(file)
        except (ValueError, OSError, SyntaxError, RecursionError):
            return None
        return self.test_numbers(mask) | self.always

    def nearby(self, files: list[FilePatch]) -> frozenset[int]:
        """Return tests that reach the old source around otherwise scoped edits."""
        mask = 0
        for file in files:
            if file.old_path is None:
                continue
            lines = self.line_masks(file.old_path)
            for hunk in file.hunks:
                for line in range(max(1, hunk.old_offset - 8), hunk.old_offset + hunk.old_count + 9):
                    mask |= lines.get(line, 0)
        return self.test_numbers(mask)

    def test_numbers(self, mask: int) -> frozenset[int]:
        selected = set()
        while mask:
            bit = mask & -mask
            if bit in self.numbers:
                selected.add(self.numbers[bit])
            mask ^= bit
        return frozenset(selected)
