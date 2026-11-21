"""Select tests using pinned source and per-test coverage measured at build time."""

from __future__ import annotations

import ast
import copy
import gzip
import json
from functools import cache
from pathlib import Path
from types import CodeType

from patch_parser import FilePatch, parse_patch
from semantic import MISSING, Shape, comparison_edits, derived_field_edits, literal, literal_field_edits, site_key

SOURCE_COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
CALLABLES = (*FUNCTIONS, ast.Lambda)
YIELDS = (ast.Yield, ast.YieldFrom)
OPERATOR_KINDS = (ast.operator, ast.boolop, ast.cmpop, ast.unaryop)
EXPRESSION_FIELDS = {
    ast.If: "test",
    ast.Assign: "value",
    ast.AnnAssign: "value",
    ast.Return: "value",
    ast.Expr: "value",
    ast.For: "iter",
    ast.AugAssign: "value",
}


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
                if isinstance(old, FUNCTIONS):
                    start = len(callables)
                    try:
                        compare(old_body, new_body)
                    except UnknownImpact:
                        del callables[start:]
                    # Keep nested closures separate when their bindings and
                    # expression shape are unchanged by the patch.
                    if len(callables) > start and all(changed_statements(a, b) for a, b in callables[start:]):
                        return
                    del callables[start:]
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
                        # A bare string can supply a function's __doc__ without running it.
                        if isinstance(old, ast.Expr) and not isinstance(value, ast.Call):
                            return False
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


def only_expected_values(before, after):
    """Recognize literal expectation edits while leaving the test program intact."""
    assertions = {"assertEqual", "assertDictEqual", "assertListEqual", "assertTupleEqual", "assertSequenceEqual"}

    def normalized(function):
        function = copy.deepcopy(function)
        expected = []
        for node in ast.walk(function):
            if (isinstance(node, ast.Call) and len(node.args) >= 2
                    and isinstance(node.func, ast.Attribute) and node.func.attr in assertions
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                    and literal(node.args[1]) is not MISSING):
                expected.append(node.args[1])
        return ast.dump(Shape(expected).visit(function))

    return normalized(before) == normalized(after)


def same_bytecode(before, after):
    """Compare compiled behavior and binding metadata, ignoring source positions."""
    fields = (
        "co_code", "co_flags", "co_names", "co_varnames", "co_freevars", "co_cellvars",
        "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_exceptiontable",
        "co_name", "co_qualname",
    )
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        return False
    if len(before.co_consts) != len(after.co_consts):
        return False
    for old, new in zip(before.co_consts, after.co_consts):
        if type(old) is not type(new):
            return False
        if isinstance(old, CodeType):
            if not same_bytecode(old, new):
                return False
        elif repr(old) != repr(new):
            return False
    return True


class ImpactIndex:
    def __init__(self, pinned: Path, coverage_path: Path, tests: list[dict], isolation_path=None, subprocess_path=None):
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
        self.fixture_directories = raw.get("fixture_directories", False)
        self.module_discovery = raw.get("module_discovery", False)
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
        self.subprocess_imports = {}
        child_profiles = {}
        if subprocess_path is not None:
            try:
                child_profiles = json.loads(subprocess_path.read_bytes())
            except (OSError, ValueError):
                pass
        for node_id, profile in child_profiles.items():
            bit = test_bits.get(node_id, 0)
            for path in profile["imported"]:
                self.subprocess_imports[path] = self.subprocess_imports.get(path, 0) | bit
            for path, lines in profile["lines"].items():
                hits = self.isolated.setdefault(path, {})
                for line in lines:
                    hits[line] = hits.get(line, 0) | bit
        self.always = set()
        self.numbers = {}
        unreliable = set(raw["unreliable"]) - child_profiles.keys()
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

    @cache
    def compiled(self, path):
        return compile(self.original(path)[0], path, "exec", dont_inherit=True)

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

    def python_impact(self, file: FilePatch, path: str, *, compare_operands=True) -> int:
        original, before = self.original(path)
        patched = file.apply(original).encode("utf-8")
        compiled = compile(patched, path, "exec", dont_inherit=True)
        after = ast.parse(patched, filename=path, type_comments=True)
        lines = self.line_masks(path)
        isolated = self.isolated.get(path, {})
        mask = lines.get(0, 0) | isolated.get(0, 0)
        startup = self.startup.get(path, ())
        if 0 in startup:
            raise UnknownImpact("source read directly during collection")
        try:
            callables = changed_callables(before, after)
        except UnknownImpact:
            if same_bytecode(self.compiled(path), compiled):
                return mask
            raise
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
            if not (hits or isolated_hits):
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
            if compare_operands and statement_hits and len(callables) == 1:
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

        @cache
        def changed_state():
            return verification.changed_state(edits, entries=True)

        derived = self.derived_impact(edits, changed_state)
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
            mask |= self.declaration_readers(path)

        # Plain declarations can still feed another binding during class creation.
        # Compare a fresh import and require evidence for each state change.
        state_changes = changed_state()
        if state_changes is not None:
            state_changes = {name for name in state_changes if "#" not in name}
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

    def derived_impact(self, edits, changed_state):
        changes = {}
        mask = 0
        for path, patched in edits.items():
            original, _ = self.original(path)
            allowed = derived_field_edits(original.encode("utf-8"), patched, path)
            if not allowed or 0 in self.startup.get(path, ()):
                return None
            changes.update(allowed)
            mask |= self.declaration_readers(path)
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
        state = changed_state()
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

    def declaration_readers(self, path):
        return (self.line_masks(path).get(0, 0) | self.isolated.get(path, {}).get(0, 0)
                | self.subprocess_imports.get(path, 0))

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

    def file_impact(self, file: FilePatch, *, compare_operands=True) -> int:
        path = file.old_path
        fixture = file.new_path or path
        if fixture and fixture.startswith("tests/fixtures/"):
            return self.fixture_impact(file, fixture)
        if path is None or file.new_path is None:
            raise UnknownImpact("added or deleted file")
        if path in {"README.md", "CHANGELOG.md"} or (path.startswith("posts/") and path.endswith(".md")):
            # Restrict this to existing Markdown so discovery and imports stay unchanged.
            if 0 in self.startup.get(path, ()):
                raise UnknownImpact("documentation read during collection")
            file.apply((self.pinned / path).read_bytes().decode("utf-8"))
            return self.declaration_readers(path)
        if path.startswith("sqlglot/") and path.endswith(".py"):
            return self.python_impact(file, path, compare_operands=compare_operands)
        if path.startswith("tests/") and Path(path).name.startswith("test_") and path.endswith(".py"):
            return self.test_source_impact(file, path)
        raise UnknownImpact("unmapped file")

    def fixture_impact(self, file, path):
        if 0 in self.startup.get(path, ()):
            raise UnknownImpact("fixture read during collection")
        readers = self.line_masks(path).get(0, 0) | self.isolated.get(path, {}).get(0, 0)
        if file.old_path and file.new_path:
            if readers:
                return readers
        elif self.fixture_directories and path.endswith(".sql"):
            source = self.pinned / path
            if file.old_path is None and source.exists():
                raise UnknownImpact("fixture already exists")
            if file.old_path and not readers:
                raise UnknownImpact("deleted fixture had no tracked readers")
            file.apply(source.read_bytes().decode("utf-8") if file.old_path else "")
            for parent in Path(path).parents:
                directory = parent.as_posix()
                if directory == "tests":
                    break
                if 0 in self.startup.get(directory, ()):
                    raise UnknownImpact("fixture directory listed during collection")
                readers |= self.line_masks(directory).get(0, 0) | self.isolated.get(directory, {}).get(0, 0)
            return readers
        raise UnknownImpact("unmapped fixture")

    def test_source_impact(self, file, path):
        original, tree = self.original(path)
        patched = file.apply(original)
        compile(patched, path, "exec", dont_inherit=True)
        changes = changed_callables(tree, ast.parse(patched, filename=path, type_comments=True))
        direct_tests = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                direct_tests[node] = f"{path}::{node.name}"
            elif isinstance(node, ast.ClassDef):
                direct_tests.update((method, f"{path}::{node.name}::{method.name}")
                                    for method in node.body if isinstance(method, ast.FunctionDef))
        mask = self.line_masks(path).get(0, 0)
        startup = self.startup.get(path, ())
        if 0 in startup:
            raise UnknownImpact("test source read during collection")
        for before, after in changes:
            node = direct_tests.get(before)
            if (not node or not before.name.startswith("test_") or before.decorator_list
                    or any(before.lineno <= line <= before.end_lineno for line in startup)
                    or not only_expected_values(before, after)):
                raise UnknownImpact("test edit may affect collection or other tests")
            members = self.test_bits.get(node, 0)
            if not members:
                raise UnknownImpact("edited test not in the baseline catalog")
            mask |= members
            for line in range(before.lineno, before.end_lineno + 1):
                mask |= self.line_masks(path).get(line, 0) | self.isolated.get(path, {}).get(line, 0)
        return mask

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
                # Another changed file can change the operands reaching this site.
                mask |= self.file_impact(file, compare_operands=len(files) == 1)
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
