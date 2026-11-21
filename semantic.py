"""Recognize guarded source changes and record their baseline dependencies."""

from __future__ import annotations

import ast
import inspect
from enum import Enum
from pathlib import Path
from sys import _getframe

MISSING = object()
SWAPS = {ast.Is: ast.Eq, ast.Eq: ast.Is, ast.IsNot: ast.NotEq, ast.NotEq: ast.IsNot}
SAFE_TYPES = (type(None), bool, int, float, str, bytes)
TPC_TESTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpch": "tpc-h",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": "tpc-ds",
}


def module_name(path: str) -> str:
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def field_key(owner: type, name: str) -> str:
    return f"{owner.__module__}:{owner.__qualname__}:{name}"


def literal(node):
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return MISSING


def class_fields(tree, module):
    """Map plain class assignments to the nodes supplying their values."""
    fields = {}

    def visit(statements, prefix):
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                visit(statement.body, prefix + statement.name + ".")
            elif prefix and isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name) and statement.value is not None:
                    name = targets[0].id
                    if not name.startswith("__"):
                        fields[f"{module}:{prefix[:-1]}:{name}"] = statement.value

    visit(tree.body, "")
    return fields


class Shape(ast.NodeTransformer):
    def __init__(self, values):
        self.values = frozenset(values)

    def generic_visit(self, node):
        if node in self.values:
            return ast.Constant(value=None)
        return super().generic_visit(node)


def literal_field_edits(original: bytes, patched: bytes, path: str):
    """Return changed bindings and literal keys if the rest of the AST is equal."""
    left, right = (ast.parse(data, type_comments=True) for data in (original, patched))
    old = class_fields(left, module_name(path))
    new = class_fields(right, module_name(path))
    if old.keys() != new.keys():
        return {}
    changes = {}
    for key, before in old.items():
        after = new[key]
        if ast.dump(before) == ast.dump(after):
            continue
        a, b = literal(before), literal(after)
        if a is MISSING or b is MISSING or type(a) is not type(b):
            return {}
        if type(a) is dict:
            if list(a) != list(b) or not all(type(k) is str for k in a):
                return {}
            keys = [k for k in a if repr(a[k]) != repr(b[k])]
        else:
            keys = None
        changes[key] = keys
    if not changes or ast.dump(Shape(old[k] for k in changes).visit(left)) != ast.dump(
        Shape(new[k] for k in changes).visit(right)
    ):
        return {}
    return changes


def simple_comparisons(tree):
    """Comparisons that start a condition, with operands readable without calls."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp, ast.While)):
            continue
        comparison = node.test
        if isinstance(comparison, ast.Compare) and len(comparison.ops) == 1 and type(comparison.ops[0]) in SWAPS:
            terms = [comparison.left, comparison.comparators[0]]
            if all(isinstance(t, (ast.Name, ast.Attribute, ast.Constant)) for t in terms):
                yield comparison


def site_key(node):
    return f"{node.lineno}:{node.col_offset}"


def comparison_edits(original: bytes, patched: bytes):
    """Identify patches changing only one supported comparison operator."""
    a, b = (ast.parse(data, type_comments=True) for data in (original, patched))
    left, right = list(simple_comparisons(a)), list(simple_comparisons(b))
    if len(left) != len(right):
        return None
    changes = [(x, y) for x, y in zip(left, right) if type(x.ops[0]) is not type(y.ops[0])]
    if len(changes) != 1:
        return None
    old, new = changes[0]
    if SWAPS[type(old.ops[0])] is not type(new.ops[0]):
        return None
    new.ops = old.ops
    return old if ast.dump(a) == ast.dump(b) else None


def resolve(node, frame):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return frame.f_locals.get(node.id, frame.f_globals.get(node.id, MISSING))
    if isinstance(node, ast.Attribute):
        owner = resolve(node.value, frame)
        if owner is not MISSING:
            expected = type.__getattribute__ if isinstance(owner, type) else object.__getattribute__
            if type(owner).__getattribute__ is expected:
                value = inspect.getattr_static(owner, node.attr, MISSING)
                if not hasattr(type(value), "__get__"):
                    return value
    return MISSING


def equal_operators(a, b):
    """Certify agreement only when no user equality implementation can run."""
    if a is MISSING or b is MISSING:
        return False
    if type(a) in SAFE_TYPES and type(b) in SAFE_TYPES:
        return (a is b) == (a == b)
    if a is None or b is None:
        other = b if a is None else a
        return type(other).__eq__ is object.__eq__ and type(other).__ne__ is object.__ne__
    # Enum members may have custom equality. Exact object operators are safe.
    return all(isinstance(x, Enum) and type(x).__eq__ is object.__eq__ and type(x).__ne__ is object.__ne__ for x in (a, b))


class SemanticTrace:
    """Attach per-test bitsets to attribute reads and comparison operands."""

    def __init__(self, sites=None):
        self.masks = {}
        self.context = None
        self.counts = {}
        self.memory_sensitive = set()
        if sites is not None:
            self.sites = sites
            return
        trees = {p.as_posix(): ast.parse(p.read_bytes()) for p in Path("sqlglot").rglob("*.py")}
        names = {key.rsplit(":", 1)[-1] for path, tree in trees.items()
                 for key, value in class_fields(tree, module_name(path)).items()
                 if literal(value) is not MISSING}
        trees.update((p.as_posix(), ast.parse(p.read_bytes())) for p in Path("tests").rglob("*.py"))
        self.sites = {}
        for path, tree in trees.items():
            if path.startswith("tests/") and any(
                isinstance(node, ast.Import) and any(alias.name == "tracemalloc" for alias in node.names)
                or isinstance(node, ast.ImportFrom) and node.module == "tracemalloc"
                for node in ast.walk(tree)
            ):
                self.memory_sensitive.add(path)
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in names:
                    parent = parents.get(node)
                    lookup = None
                    if isinstance(parent, ast.Subscript) and parent.value is node:
                        lookup = parent.slice
                    elif isinstance(parent, ast.Attribute) and parent.attr == "get":
                        call = parents.get(parent)
                        if isinstance(call, ast.Call) and call.func is parent and call.args:
                            lookup = call.args[0]
                    self.sites.setdefault((path, node.lineno), []).append((node, lookup))
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"getattr", "hasattr"}:
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in names:
                        read = ast.Attribute(value=node.args[0], attr=node.args[1].value, ctx=ast.Load())
                        self.sites.setdefault((path, node.lineno), []).append((read, None))
            if path.startswith("tests/"):
                continue
            for node in simple_comparisons(tree):
                self.sites.setdefault((path, node.lineno), []).append((node, None))

    def put(self, kind, key, mask, startup):
        row = self.masks.setdefault(kind, {}).setdefault(key, [0, False])
        row[0] |= mask
        row[1] |= startup

    def observe(self, sites, path, line, mask, startup):
        context = (mask, startup)
        if self.context != context:
            self.context = context
            self.counts.clear()
        where = (path, line)
        count = self.counts.get(where, 0) + 1
        self.counts[where] = count
        if count > 1024:
            # Retain the test for this site when the bounded trace is incomplete.
            for node, _ in sites:
                if isinstance(node, ast.Compare):
                    self.put("comparison_uncertain", f"{path}:{site_key(node)}", mask, startup)
                else:
                    self.put("unknown_owner", node.attr, mask, startup)
            return False
        frame = _getframe(2)
        for node, lookup in sites:
            if isinstance(node, ast.Compare):
                key = f"{path}:{site_key(node)}"
                a, b = resolve(node.left, frame), resolve(node.comparators[0], frame)
                self.put("comparison_seen", key, mask, startup)
                if not equal_operators(a, b):
                    self.put("comparison_uncertain", key, mask, startup)
                continue
            owner = resolve(node.value, frame)
            cls = owner if isinstance(owner, type) else type(owner)
            expected = type.__getattribute__ if isinstance(owner, type) else object.__getattribute__
            if type(owner).__getattribute__ is not expected:
                self.put("unknown_owner", node.attr, mask, startup)
                continue
            defining = next((base for base in cls.__mro__ if node.attr in vars(base)), None)
            if defining is None:
                self.put("unknown_owner", node.attr, mask, startup)
                continue
            key = field_key(defining, node.attr)
            self.put("field", key, mask, startup)
            field = vars(defining)[node.attr]
            selector = resolve(lookup, frame)
            if type(field) is dict and type(selector) is str:
                self.put("dict_key", f"{key}:{selector}", mask, startup)
            else:
                self.put("whole_value", key, mask, startup)
        return True

    def encoded(self):
        return {kind: {key: [format(bits, "x"), startup] for key, (bits, startup) in rows.items()}
                for kind, rows in self.masks.items()}

    def merge(self, source):
        for kind, rows in source.items():
            for key, (bits, startup) in rows.items():
                self.put(kind, key, int(bits, 16), startup)
