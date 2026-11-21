"""Fork verification children from an interpreter with pristine modules loaded."""

from __future__ import annotations

import importlib
import json
import os
import re
import select
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from functools import lru_cache
from importlib.abc import MetaPathFinder
from importlib.util import source_from_cache, spec_from_file_location
from pathlib import Path
from types import CodeType, FunctionType

from semantic import module_name


FAST_PATH = re.compile(r"tests/(?:dialects/)?test_[a-zA-Z0-9_]+\.py")
FAST_EXCLUDED = frozenset({
    "tests/test_build.py",
    "tests/test_dialect_entry_points.py",
    "tests/test_dialect_imports.py",
    "tests/test_executor.py",
    "tests/test_integration_loader.py",
    "tests/test_optimizer.py",
    "tests/test_sqlglotc.py",
    "tests/test_typing.py",
})
PYTEST_HOOKS = frozenset({
    "pytestmark", "pytest_plugins", "setup_module", "teardown_module",
    "setup_function", "teardown_function", "setup_class", "teardown_class",
    "setup_method", "teardown_method",
})
PROCESS_EVENTS = frozenset({
    "subprocess.Popen", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn",
    "os.spawn", "os.startfile", "os.system", "_posixsubprocess.fork_exec",
})


def fast_test_id(node_id: str) -> bool:
    """Cheap index-side check for ordinary pinned unittest IDs."""
    parts = node_id.split("::")
    if len(parts) != 3:
        return False
    path, cls, method = parts
    return (FAST_PATH.fullmatch(path) is not None and path not in FAST_EXCLUDED
            and cls.startswith("Test") and cls.isidentifier()
            and method.startswith("test_") and method.isidentifier())


def _has_pytest_behavior(namespace):
    return not PYTEST_HOOKS.isdisjoint(namespace) or any(
        getattr(value, "_fixture_function_marker", None) is not None
        or getattr(value, "_pytestfixturefunction", None) is not None
        for value in namespace.values() if type(value).__module__.startswith("_pytest.")
        or type(value) is FunctionType
    )


def _test_cases(pinned, ids):
    cases = []
    for node_id in ids:
        path, owner, method = node_id.split("::")
        expected = pinned / path
        # A conftest can alter the behaviour of otherwise ordinary TestCases.
        for directory in (expected.parent, *expected.parent.parents):
            if (directory / "conftest.py").exists():
                return None
            if directory == pinned:
                break
        module = importlib.import_module(module_name(path))
        cls = vars(module).get(owner)
        if (Path(module.__file__).resolve() != expected.resolve() or _has_pytest_behavior(vars(module))
                or type(cls) is not type or not issubclass(cls, unittest.TestCase)
                or cls.__module__ != module.__name__):
            return None
        test = vars(cls).get(method)
        if (type(test) is not FunctionType or hasattr(test, "pytestmark")
                or any(_has_pytest_behavior(vars(base)) for base in cls.__mro__
                       if base is not unittest.TestCase and base is not object)
                or any(getattr(cls, name) is not getattr(unittest.TestCase, name)
                       for name in ("__init__", "__call__", "run", "_callTestMethod"))):
            return None
        cases.append(cls(method))
    return cases


def unittest_passes(pinned, ids, *, fixtures=()):
    """Return passes only after unittest has finished all fixtures and cleanups."""
    if any(not fast_test_id(node_id) for node_id in ids):
        return None

    spawned = False
    active = True
    pending = {os.path.normcase(os.path.abspath(path)) for path in fixtures}

    def audit(event, args):
        nonlocal spawned
        if not active:
            return
        if event in PROCESS_EVENTS:
            spawned = True
            raise RuntimeError("subprocess cannot use a warm verification body")
        if (pending and event == "open" and isinstance(args[0], (str, bytes))
                and (args[2] & os.O_ACCMODE) != os.O_WRONLY):
            pending.discard(os.path.normcase(os.path.abspath(os.fsdecode(args[0]))))

    sys.addaudithook(audit)
    try:
        cases = _test_cases(pinned, ids)
        if cases is None:
            return None

        class Results(unittest.TestResult):
            def __init__(self):
                super().__init__()
                self.known = {id(case) for case in cases}
                self.passed = set()
                self.inconclusive = False

            def addSuccess(self, test):
                self.passed.add(id(test))

            def addError(self, test, err):
                if id(test) not in self.known:
                    self.inconclusive = True
                self.passed.discard(id(test))

            def addSkip(self, test, reason):
                if id(test) not in self.known:
                    self.inconclusive = True
                self.passed.discard(id(test))

        results = Results()
        results.failfast = True
        unittest.TestSuite(cases).run(results)
        if spawned or pending or results.inconclusive:
            return None
        return [node_id for node_id, case in zip(ids, cases) if id(case) in results.passed]
    finally:
        active = False


def code_by_line(code):
    result = {}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            result[value.co_name, value.co_firstlineno] = value
            result.update(code_by_line(value))
    return result


@lru_cache(maxsize=32)
def _pristine_bodies(pinned, path):
    source = pinned / path
    return code_by_line(compile(source.read_bytes(), str(source), "exec", dont_inherit=True))


def install_bodies(edits, plan, *, pinned=None):
    """Replace only plain functions whose loaded code matches the old binding."""
    replacements = []
    for path, locations in plan.items():
        name = module_name(path)
        module = sys.modules[name]
        code = compile(edits[path], str(Path.cwd() / path), "exec", dont_inherit=True)
        compiled = code_by_line(code)
        for function_name, old_line, new_line in locations:
            body = compiled.get((function_name, new_line))
            if body is None or "<locals>" in body.co_qualname:
                return False
            current = module
            for part in body.co_qualname.split("."):
                if not isinstance(current, type) and current is not module:
                    return False
                current = vars(current).get(part)
            if type(current) is not FunctionType or current.__module__ != name:
                return False
            old = current.__code__
            if pinned is not None and old != _pristine_bodies(pinned, path).get((function_name, old_line)):
                return False
            if (old.co_firstlineno != old_line or old.co_qualname != body.co_qualname
                    or old.co_flags != body.co_flags or old.co_freevars != body.co_freevars
                    or old.co_cellvars != body.co_cellvars):
                return False
            replacements.append((current, body))
    for function, body in replacements:
        function.__code__ = body
    return bool(replacements)


class WarmProcess:
    def __init__(self, pinned):
        self.pinned = pinned
        self.process = None

    def _request(self, request, timeout):
        if not hasattr(os, "fork"):
            return None
        try:
            if self.process is None:
                self.process = subprocess.Popen(
                    [sys.executable, __file__, str(self.pinned)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            self.process.stdin.write(json.dumps(request).encode() + b"\n")
            self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], timeout + 1.3)
            if ready:
                return json.loads(self.process.stdout.readline())
        except (OSError, ValueError):
            pass
        self.close()
        return None

    def request(self, edits, plan, ids, timeout, *, fast=False):
        payload = {path: contents.decode("utf-8") for path, contents in edits.items()}
        reply = self._request(["tests", payload, timeout, plan, ids, fast], timeout)
        if isinstance(reply, list) and all(isinstance(node_id, str) and node_id in ids for node_id in reply):
            return reply
        return None

    def request_state(self, edits, modules, unordered, expected, timeout):
        payload = {path: contents.decode("utf-8") for path, contents in edits.items()}
        reply = self._request(["state", payload, timeout, modules, unordered, expected], timeout)
        if isinstance(reply, dict) and reply and all(type(value) is str for value in reply.values()):
            return reply
        return None

    def close(self):
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.process = None


def bootstrap_import(pinned, unordered, snapshot):
    """Record the exact first import and its reads before reusing it after fork."""
    reads = set()
    active = True

    def audit(event, args):
        if not active or event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = os.path.abspath(os.fsdecode(args[0]))
        if path.endswith(".pyc"):
            try:
                path = source_from_cache(path)
            except ValueError:
                return
        try:
            reads.add(Path(path).relative_to(pinned).as_posix())
        except ValueError:
            pass

    sys.addaudithook(audit)
    try:
        prefix = snapshot(["sqlglot"], unordered.copy())
    finally:
        active = False
    return prefix, reads


class PatchedModule(MetaPathFinder):
    def __init__(self, path):
        self.path = path
        self.name = module_name(path)
        self.used = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            self.used = True
            return spec_from_file_location(fullname, Path.cwd() / self.path)
        return None


def reuse_state(boot, edits, params, verifier, snapshot, timeout):
    modules, unordered, expected = params
    if len(edits) != 1 or not boot or not modules or modules[0] != "sqlglot":
        return None
    path = next(iter(edits))
    prefix, reads = boot
    if (prefix != expected or path in reads or not path.startswith("sqlglot/")
            or path.endswith("/__init__.py") or module_name(path) in sys.modules):
        return None

    def probe():
        # Already-loaded packages keep their pristine __path__. Route the one
        # edited module to the overlay without evicting any package state.
        loader = PatchedModule(path)
        sys.meta_path.insert(0, loader)
        source = os.path.normcase(os.fspath(verifier.pinned / path))
        read_pristine = False

        def audit(event, args):
            nonlocal read_pristine
            if event != "open" or not isinstance(args[0], (str, bytes)):
                return
            filename = os.path.abspath(os.fsdecode(args[0]))
            if filename.endswith(".pyc"):
                try:
                    filename = source_from_cache(filename)
                except ValueError:
                    return
            read_pristine |= os.path.normcase(filename) == source

        sys.addaudithook(audit)
        state = snapshot(modules[1:], unordered)
        return prefix | state if loader.used and not read_pristine else None

    return verifier.run(edits, probe, timeout)


def serve(pinned):
    from verify import Verification, pytest_passes, snapshot

    os.chdir(pinned)
    sys.path.insert(0, str(pinned))
    verifier = Verification(pinned, {})
    boot = None
    for line in sys.stdin.buffer:
        answer = None
        try:
            kind, edits, timeout, *params = json.loads(line)
            edits = {path: contents.encode("utf-8") for path, contents in edits.items()}
            with redirect_stdout(sys.stderr):
                if kind == "state":
                    if "sqlglot" not in sys.modules:
                        boot = bootstrap_import(pinned, params[1], snapshot)
                    answer = reuse_state(boot, edits, params, verifier, snapshot, timeout)
                elif kind == "tests":
                    boot = None  # Test requests can advance past the captured import prefix.
                    plan, ids, fast = params
                    importlib.import_module("sqlglot")
                    for path in plan:
                        importlib.import_module(module_name(path))

                    def probe():
                        if plan:
                            if not install_bodies(edits, plan, pinned=pinned):
                                return None
                            return unittest_passes(pinned, ids) if fast else pytest_passes(ids)
                        if not fast or not edits or any(not path.startswith("tests/fixtures/") or not path.endswith(".sql")
                                                       for path in edits):
                            return None
                        return unittest_passes(pinned, ids, fixtures=edits)

                    answer = verifier.run(edits, probe, timeout)
        except (OSError, ValueError, ImportError, TypeError):
            pass
        print(json.dumps(answer), flush=True)


if __name__ == "__main__":
    serve(Path(sys.argv[1]))
