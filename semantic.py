"""Recognize guarded source changes and record their baseline dependencies."""

from __future__ import annotations

import ast
import inspect
import operator
from enum import Enum
from pathlib import Path
from sys import _getframe
from types import FunctionType

MISSING = object()
SWAPS = {
    ast.Is: ast.Eq, ast.Eq: ast.Is, ast.IsNot: ast.NotEq, ast.NotEq: ast.IsNot,
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
}
ORDER_OPERATORS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
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


def derived_field_edits(original: bytes, patched: bytes, path: str):
    """Accept existing flag or dispatch entries with unchanged declaration shape."""
    changes = {}
    trees = [ast.parse(contents, type_comments=True) for contents in (original, patched)]
    left, right = [class_fields(tree, module_name(path)) for tree in trees]
    for field, a in left.items():
        name = field.rsplit(":", 1)[1]
        b = right.get(field)
        if not isinstance(a, ast.Dict) or not isinstance(b, ast.Dict) or ast.dump(a) == ast.dump(b):
            continue
        if name == "arg_types" and path.startswith("sqlglot/expressions/"):
            compatible = lambda x, y: type(literal(x)) is bool and type(literal(y)) is bool
        elif name == "SHOW_PARSERS" and path.startswith("sqlglot/parsers/"):
            compatible = matching_show_factory
        elif name == "NUMERIC_ESCAPES" and path.startswith("sqlglot/dialects/"):
            compatible = matching_numeric_escape
        else:
            continue
        if len(a.keys) != len(b.keys):
            return {}
        keys = set()
        for ak, bk, av, bv in zip(a.keys, b.keys, a.values, b.values):
            akey = ast.dump(ak) if ak is not None else None
            bkey = ast.dump(bk) if bk is not None else None
            if akey != bkey:
                return {}
            if ast.dump(av) != ast.dump(bv):
                key = literal(ak)
                if not isinstance(key, str) or not compatible(av, bv):
                    return {}
                keys.add(key)
        if keys:
            changes[field] = keys
    if changes and ast.dump(Shape(left[k] for k in changes).visit(trees[0])) == ast.dump(
        Shape(right[k] for k in changes).visit(trees[1])
    ):
        return changes
    return {}


def matching_show_factory(a, b):
    if not all(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
               and call.func.id == "_show_parser" for call in (a, b)):
        return False
    if [kw.arg for kw in a.keywords] != [kw.arg for kw in b.keywords] or len(a.args) != len(b.args):
        return False
    values = (node for call in (a, b) for node in (*call.args, *(kw.value for kw in call.keywords)))
    return all(type(literal(node)) in SAFE_TYPES for node in values) and all(
        kw.arg is not None for call in (a, b) for kw in call.keywords
    )


def matching_numeric_escape(a, b):
    for node in (a, b):
        value = literal(node)
        if type(value) is not tuple or len(value) != 4 or any(type(part) is not int for part in value):
            return False
    return True


def show_closure(value):
    """Identify closures returned by the pinned SHOW parser factories."""
    if (type(value) is not FunctionType or not value.__module__.startswith("sqlglot.parsers.")
            or not value.__qualname__.startswith("_show_parser.<locals>.")):
        return None
    contents = []
    for cell in value.__closure__ or ():
        item = cell.cell_contents
        if type(item) is tuple and all(type(element) in SAFE_TYPES for element in item):
            contents.append(item)
        elif type(item) is dict and all(type(k) is str and type(v) in SAFE_TYPES for k, v in item.items()):
            contents.append(item)
        elif type(item) in SAFE_TYPES:
            contents.append(item)
        else:
            return None
    return value.__module__, value.__qualname__, contents


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


def cached_property_comparison(node, parents):
    if {type(node.ops[0]), SWAPS[type(node.ops[0])]} not in (
        {ast.Is, ast.Eq}, {ast.IsNot, ast.NotEq}
    ):
        return False
    condition = parents.get(node)
    function = parents.get(condition)
    if not (isinstance(condition, ast.If) and isinstance(function, ast.FunctionDef)
            and function.body[0] is condition and any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in function.decorator_list)):
        return False
    a, b = node.left, node.comparators[0]
    if isinstance(a, ast.Constant):
        a, b = b, a
    return (isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)
            and isinstance(b, ast.Constant) and b.value is None)


def resolve_cached_operand(node, frame):
    if isinstance(node, ast.Attribute):
        owner = resolve(node.value, frame)
        cls = type(owner)
        if cls.__getattribute__ is object.__getattribute__ and all(
            node.attr not in vars(base) for base in cls.__mro__
        ):
            try:
                return object.__getattribute__(owner, node.attr)
            except AttributeError:
                return MISSING
    return resolve(node, frame)


def equal_operators(a, b, op=ast.Is):
    """Compare both operators only on operand types whose behavior is known."""
    if a is MISSING or b is MISSING:
        return False
    if op in ORDER_OPERATORS:
        if type(a) not in SAFE_TYPES or type(b) not in SAFE_TYPES:
            return False
        try:
            return ORDER_OPERATORS[op](a, b) == ORDER_OPERATORS[SWAPS[op]](a, b)
        except TypeError:
            return False
    if type(a) in SAFE_TYPES and type(b) in SAFE_TYPES:
        return (a is b) == (a == b)
    if a is None or b is None:
        other = b if a is None else a
        return type(other).__eq__ is object.__eq__ and type(other).__ne__ is object.__ne__
    if type(a) is type(b) and isinstance(a, Enum):
        cls = type(a)
        if cls.__eq__ is object.__eq__ and cls.__ne__ is object.__ne__:
            return True
        for base in (str, int):
            if cls.__eq__ is base.__eq__ and cls.__ne__ is base.__ne__:
                return (a is b) == base.__eq__(a, b)
    return False


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
                if (path == "sqlglot/tokenizer_core.py" and isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                        and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "numeric_escapes"):
                    self.sites.setdefault((path, node.lineno), []).append((node, "numeric"))
                if (path == "sqlglot/parser.py" and isinstance(node, ast.FunctionDef)
                        and node.name == "_find_parser"):
                    for statement in ast.walk(node):
                        if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Subscript)
                                and isinstance(statement.value.value, ast.Name)
                                and statement.value.value.id == "parsers"):
                            self.sites.setdefault((path, statement.lineno), []).append((statement, "show"))
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in names:
                    parent = parents.get(node)
                    lookup = None
                    if (path == "sqlglot/expressions/core.py" and node.attr == "required_args"
                            and isinstance(parent, ast.For) and parent.iter is node):
                        self.sites.setdefault((path, node.lineno), []).append((node, "required"))
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
                fast = cached_property_comparison(node, parents)
                self.sites.setdefault((path, node.lineno), []).append((node, fast))

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
        frame = _getframe(2)
        active = False
        for node, lookup in sites:
            if isinstance(lookup, str):
                self.observe_derived(lookup, frame, mask, startup)
                active = True
                continue
            if isinstance(node, ast.Compare):
                key = f"{path}:{site_key(node)}"
                uncertain, at_startup = self.masks.get("comparison_uncertain", {}).get(key, (0, False))
                if (mask and uncertain & mask == mask) or (startup and at_startup):
                    continue
                if count > (2_000_000 if lookup else 16_384):
                    self.put("comparison_uncertain", key, mask, startup)
                    continue
                operand = resolve_cached_operand if lookup else resolve
                a, b = operand(node.left, frame), operand(node.comparators[0], frame)
                self.put("comparison_seen", key, mask, startup)
                if not equal_operators(a, b, type(node.ops[0])):
                    self.put("comparison_uncertain", key, mask, startup)
                else:
                    active = True
                continue
            if count > 1024:
                if count == 1025:
                    self.put("unknown_owner", node.attr, mask, startup)
                continue
            active = True
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
        return active

    def observe_derived(self, kind, frame, mask, startup):
        locals_ = frame.f_locals
        if kind == "numeric":
            key = "0" if locals_["is_octal"] else locals_["peek"]
            self.put(kind, key, mask, startup)
            return
        cls = type(locals_["self"])
        if kind == "required":
            values = locals_["self"].args
            for key in cls.arg_types:
                value = values.get(key)
                if value is None or (type(value) is list and not value):
                    self.put("required", f"{field_key(cls, 'required_args')}#{key}", mask, startup)
        else:
            defining = next((base for base in cls.__mro__ if "SHOW_PARSERS" in vars(base)), None)
            if defining and locals_.get("parsers") is vars(defining)["SHOW_PARSERS"]:
                key = " ".join(locals_["this"])
                self.put("show", f"{field_key(defining, 'SHOW_PARSERS')}#{key}", mask, startup)

    def encoded(self):
        return {kind: {key: [format(bits, "x"), startup] for key, (bits, startup) in rows.items()}
                for kind, rows in self.masks.items()}

    def merge(self, source):
        for kind, rows in source.items():
            for key, (bits, startup) in rows.items():
                self.put(kind, key, int(bits, 16), startup)
