"""Bound structural fallbacks using only the pinned checkout and test traces."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from copy import copy
from functools import cache
from importlib.util import resolve_name
from pathlib import PurePosixPath

from semantic import MISSING, literal, module_name

DIALECTS = "sqlglot/dialects/"
REGISTRY_NAMES = {"classes", "_classes", "DIALECTS", "Dialects", "DIALECT_MODULE_NAMES"}
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
DEFAULT_ATTRIBUTES = {"__defaults__", "__kwdefaults__", "__signature__", "__wrapped__"}


def pure_literal(node):
    return literal(node) is not MISSING and not any(isinstance(child, ast.Call) for child in ast.walk(node))


def inert_module(tree):
    body = tree.body
    if ast.get_docstring(tree, clean=False) is not None:
        body = body[1:]
    return all(
        isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and not node.targets[0].id.startswith("__")
        and pure_literal(node.value)
        for node in body
    )


def default_edit(before, after):
    """Find one plain top-level function whose only changes are literal defaults."""
    if len(before.body) != len(after.body):
        return None
    changed = None
    for old, new in zip(before.body, after.body):
        if ast.dump(old) == ast.dump(new):
            continue
        if changed is not None or not all(
            isinstance(node, ast.FunctionDef) and not node.decorator_list for node in (old, new)
        ):
            return None
        left = old.args.defaults + old.args.kw_defaults
        right = new.args.defaults + new.args.kw_defaults
        if (len(old.args.defaults) != len(new.args.defaults) or len(left) != len(right)
                or any((a is None) != (b is None) for a, b in zip(left, right))):
            return None
        for a, b in zip(left, right):
            if a is not None and ast.dump(a) != ast.dump(b) and not (pure_literal(a) and pure_literal(b)):
                return None
        normalized = copy(new)
        normalized.args = copy(new.args)
        normalized.args.defaults = old.args.defaults
        normalized.args.kw_defaults = old.args.kw_defaults
        if ast.dump(old) != ast.dump(normalized):
            return None
        changed = old
    return changed


def appended_import_error(before, after):
    if len(after.body) != len(before.body) + 1:
        return False
    statement = after.body[-1]
    if not isinstance(statement, ast.Raise) or statement.cause is not None:
        return False
    error = statement.exc
    if isinstance(error, ast.Call):
        if error.keywords or len(error.args) > 1 or any(
            not isinstance(arg, ast.Constant) or type(arg.value) is not str for arg in error.args
        ):
            return False
        error = error.func
    return (isinstance(error, ast.Name) and error.id in {"ImportError", "ModuleNotFoundError"}
            and [ast.dump(node) for node in before.body] == [ast.dump(node) for node in after.body[:-1]])


def import_targets(tree, package):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parent = resolve_name("." * node.level + (node.module or ""), package)
            yield parent
            for alias in node.names:
                yield f"{parent}.{alias.name}"
                if parent == "sqlglot.dialects":
                    yield f"{parent}.{alias.name.lower()}"
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            func = node.func
            if (isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}
                    or isinstance(func, ast.Attribute) and func.attr == "import_module"):
                yield node.args[0].value


class FallbackIndex:
    def __init__(self, impact, catalog):
        self.impact = impact
        self.pinned = impact.pinned
        self.by_file = defaultdict(list)
        for test in catalog:
            self.by_file[test["file"]].append((test["id"], test["number"]))

    @cache
    def source_files(self):
        return tuple(
            file.relative_to(self.pinned).as_posix()
            for directory in ("sqlglot", "tests")
            for file in (self.pinned / directory).rglob("*.py")
        )

    @cache
    def dialect_paths(self):
        _, tree = self.impact.original(DIALECTS + "__init__.py")
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "DIALECTS"):
                names = literal(node.value)
                if isinstance(names, list) and all(isinstance(name, str) for name in names):
                    return {DIALECTS + name.lower() + ".py" for name in names}
        return set()

    @cache
    def importers(self):
        files = {module_name(path): path for path in self.source_files()}
        edges = defaultdict(set)
        for name, path in files.items():
            _, tree = self.impact.original(path)
            package = name if path.endswith("/__init__.py") else name.rpartition(".")[0]
            for target in set(import_targets(tree, package)) & files.keys():
                if target != name:
                    edges[files[target]].add(path)
        return edges

    def reverse_dependencies(self, path):
        edges = self.importers()
        affected, pending = {path}, [path]
        while pending:
            for importer in edges.get(pending.pop(), frozenset()) - affected:
                pending.append(importer)
                affected.add(importer)
        return affected

    @cache
    def file_mask(self, path):
        mask = self.impact.declaration_readers(path)
        for traces in (self.impact.line_masks(path), self.impact.isolated.get(path, {})):
            for readers in traces.values():
                mask |= readers
        return mask

    def span_mask(self, path, first, last):
        lines = self.impact.line_masks(path)
        isolated = self.impact.isolated.get(path, {})
        mask = 0
        for number in range(first, last + 1):
            mask |= lines.get(number, 0) | isolated.get(number, 0)
        return mask

    def test_scope(self, path, matches):
        _, tree = self.impact.original(path)
        entries = self.by_file[path]
        extra = set()
        mask = 0

        def visit(node, prefix, owner):
            nonlocal mask
            if isinstance(node, DEFINITIONS):
                prefix += "::" + node.name
                owner = node
            if matches(node):
                if owner is None:
                    extra.update(number for _, number in entries)
                else:
                    qualified = path + prefix
                    members = {number for name, number in entries
                               if name == qualified or name.startswith((qualified + "::", qualified + "["))}
                    extra.update(members)
                    if not members:
                        mask |= self.span_mask(path, owner.lineno, owner.end_lineno)
            for child in ast.iter_child_nodes(node):
                visit(child, prefix, owner)

        visit(tree, "", None)
        return extra | self.impact.test_numbers(mask)

    @cache
    def dialect_consumers(self, stems):
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(stems))) + r")\b", re.I)

        def matches(node):
            if isinstance(node, ast.Constant):
                value = node.value
            elif isinstance(node, ast.Name):
                value = node.id
            elif isinstance(node, ast.Attribute):
                value = node.attr
            elif isinstance(node, ast.alias):
                value = node.name
            else:
                return False
            return isinstance(value, str) and (value in REGISTRY_NAMES or pattern.search(value) is not None)

        return frozenset().union(*(self.test_scope(path, matches) for path in self.by_file if path.endswith(".py")))

    def broken_import_scope(self, path):
        affected = self.reverse_dependencies(path)
        stems = frozenset(PurePosixPath(name).stem for name in affected if name in self.dialect_paths())
        for stem in stems:
            affected.update(f"sqlglot/{family}/{stem}.py" for family in ("parsers", "generators"))
        # Startup coverage lacks test attribution. Cold collection supplies it.
        mask = self.file_mask("dialect:registry")
        selected = set(self.dialect_consumers(stems))
        for stem in stems:
            mask |= self.file_mask("dialect:" + stem)
        for name in affected:
            if name in self.by_file:
                selected.update(number for _, number in self.by_file[name])
            else:
                mask |= self.file_mask(name)
                if name.startswith("sqlglot/"):
                    mask |= self.file_mask("import:" + module_name(name))
        return selected | self.impact.test_numbers(mask) | self.impact.always

    def default_scope(self, path, function):
        if any(function.lineno <= n <= function.end_lineno for n in self.impact.startup.get(path, ())):
            return None
        if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(function)):
            return None
        mask = self.file_mask(path)
        if not mask or 0 in self.impact.startup.get(path, ()):
            return None
        name = function.name
        word = re.compile(r"\b" + re.escape(name) + r"\b")
        selected = set()
        for source in self.source_files():
            text = (self.pinned / source).read_text(encoding="utf-8")
            if not word.search(text):
                continue
            _, tree = self.impact.original(source)
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            readers = set()
            for node, parent in parents.items():
                if isinstance(node, ast.alias) and node.name == name and node.asname:
                    return None
                if (isinstance(node, ast.Constant) and node.value == name
                        and not isinstance(parent, ast.Expr)):
                    # An attribute fetched by name can escape the original binding.
                    return None
                is_name = isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
                is_attribute = isinstance(node, ast.Attribute) and node.attr == name
                if not (is_name or is_attribute):
                    continue
                if isinstance(parent, ast.Call) and parent.func is node:
                    continue
                if isinstance(parent, ast.Attribute) and parent.attr in DEFAULT_ATTRIBUTES:
                    readers.add(parent)
                elif (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute)
                      and isinstance(parent.func.value, ast.Name) and parent.func.value.id == "inspect"
                      and parent.func.attr in {"signature", "getfullargspec"}):
                    readers.add(parent)
                else:
                    return None
            if readers:
                # A collection-time signature read cannot be safely attributed.
                for reader in readers:
                    child, ancestor = reader, parents[reader]
                    while not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if ancestor not in parents:
                            return None
                        child, ancestor = ancestor, parents[ancestor]
                    if child not in ancestor.body:
                        return None
                    if any(ancestor.lineno <= n <= ancestor.end_lineno
                           for n in self.impact.startup.get(source, ())):
                        return None
                    mask |= self.span_mask(source, ancestor.lineno, ancestor.end_lineno)
                if source in self.by_file:
                    selected.update(self.test_scope(source, readers.__contains__))
        return selected | self.impact.test_numbers(mask) | self.impact.always

    def added_module(self, file, path):
        parent = PurePosixPath(path).parent.as_posix()
        stem = PurePosixPath(path).stem
        if (not self.impact.module_discovery or stem.startswith("__")
                or not (self.pinned / parent / "__init__.py").is_file()
                or (self.pinned / path).exists() or (self.pinned / path).with_suffix("").exists()):
            return None
        contents = file.apply("")
        compile(contents, path, "exec", dont_inherit=True)
        if not inert_module(ast.parse(contents, filename=path)):
            return None
        key = "import:" + module_name(path)
        if any(0 in self.impact.startup.get(name, ()) for name in (path, parent, key)):
            return None
        basename = re.compile(r"\b" + re.escape(stem) + r"\b", re.I)
        if any(basename.search((self.pinned / source).read_text(encoding="utf-8"))
               for source in self.source_files()):
            return None
        mask = self.file_mask(path) | self.file_mask(parent) | self.file_mask(key)
        return self.impact.test_numbers(mask) | self.impact.always

    def select(self, files):
        """Return candidate tests and, for a broken import, its collection overlay."""
        if len(files) != 1:
            return None
        file = files[0]
        path = file.new_path
        if path is None or not re.fullmatch(r"sqlglot/(?:[a-zA-Z_]\w*/)*[a-zA-Z_]\w*\.py", path):
            return None
        if file.old_path is None:
            selected = self.added_module(file, path)
            return None if selected is None else (selected, None)
        original, before = self.impact.original(path)
        patched = file.apply(original).encode("utf-8")
        try:
            compile(patched, path, "exec", dont_inherit=True)
        except SyntaxError:
            broken = True
        else:
            after = ast.parse(patched, filename=path, type_comments=True)
            if (function := default_edit(before, after)) is not None:
                selected = self.default_scope(path, function)
                return None if selected is None else (selected, None)
            broken = appended_import_error(before, after)
        if (broken and self.impact.module_discovery and path in self.dialect_paths()
                and 0 not in self.impact.startup.get(path, ())):
            return self.broken_import_scope(path), {path: patched}
        return None
