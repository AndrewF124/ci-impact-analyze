"""Bounded verification of patched source from the pinned checkout."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

from query_replay import load_profiles, run_cases
from semantic import TPC_TESTS, module_name, show_closure
from warm_verify import WarmProcess


def digest(value):
    return hashlib.sha256(json.dumps(value, default=str, ensure_ascii=True).encode()).hexdigest()


def config_value(value, depth=0):
    """Fingerprint data without invoking application-defined serialization."""
    if depth > 6:
        return None
    kind = type(value)
    if kind in (type(None), bool, int, float, str, bytes):
        return (kind.__name__, repr(value))
    if kind is dict:
        pairs = [(config_value(k, depth + 1), config_value(v, depth + 1)) for k, v in value.items()]
        return ("dict", sorted(pairs, key=repr))
    if kind in (list, tuple, set, frozenset):
        members = [config_value(v, depth + 1) for v in value]
        if kind in (set, frozenset):
            members.sort(key=repr)
        return (kind.__name__, members)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    return None


def unordered_tuple_bindings(source):
    """Find the top-level tuple bindings whose order comes from a set."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Name) and call.func.id == "tuple"
                    and len(call.args) == 1 and isinstance(call.args[0], ast.Set) and not call.keywords):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return sorted(names)


def snapshot(modules, unordered_by_module=None):
    fields = {}
    if unordered_by_module is None:
        unordered_by_module = {}

    def record(prefix, obj, module, unordered=()):
        for name, value in vars(obj).items():
            if name.startswith("__"):
                continue
            if isinstance(value, type) and value.__module__ == module:
                qualifier = f"{module}:{value.__qualname__}:"
                if qualifier == prefix[:-1] + "." + name + ":" or prefix == module + "::":
                    if value.__qualname__.split(".")[-1] == name and qualifier != prefix:
                        record(qualifier, value, module)
                continue
            signature = config_value(value)
            if isinstance(obj, type):
                field = prefix + name + "#"
                if name == "arg_types" and type(value) is dict:
                    fields.update((field + k, digest(v)) for k, v in value.items()
                                  if type(k) is str and type(v) is bool)
                elif name == "required_args" and type(value) is set:
                    fields.update((field + k, "required") for k in value if type(k) is str)
                elif name == "NUMERIC_ESCAPES" and type(value) is dict:
                    fields.update((field + k, digest(config_value(v))) for k, v in value.items()
                                  if type(k) is str)
                elif name == "SHOW_PARSERS" and type(value) is dict:
                    callbacks = [(key, show_closure(v)) for key, v in value.items() if type(key) is str]
                    fields.update((field + k, digest(v)) for k, v in callbacks if v is not None)
            if signature is not None:
                if name in unordered and signature[0] == "tuple":
                    signature = ("tuple-from-set", sorted(signature[1], key=repr))
                fields[prefix + name] = digest(signature)

    for name in modules:
        module = importlib.import_module(name)
        # tuple({...}) can produce different orders across independently imported
        # interpreters. Explicit tuples still retain their order in the signature.
        if name not in unordered_by_module:
            unordered_by_module[name] = unordered_tuple_bindings(Path(module.__file__).read_bytes())
        record(name + "::", module, name, unordered_by_module[name])
    return fields


def tpc_signatures(dataset):
    from sqlglot import parse_one, transpile
    from sqlglot.optimizer import optimize
    from sqlglot.schema import ensure_schema
    from sqlglot.serde import dump
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA, load_sql_fixture_pairs

    schema = TPCH_SCHEMA if dataset == "tpc-h" else TPCDS_SCHEMA
    signatures = []
    for options, sql, _ in load_sql_fixture_pairs(f"optimizer/{dataset}/{dataset}.sql"):
        parsed = parse_one(sql)
        if options.get("execute"):
            optimized = optimize(parsed, ensure_schema(schema), leave_tables_isolated=True)
            signatures.append(digest((sql, dump(optimized), transpile(sql, write="duckdb"))))
        else:
            # The real test parses the query before a worker checks its execute flag.
            signatures.append(digest((sql, dump(parsed))))
    return signatures


def overlay(source, target, edits):
    """Point unmodified paths at the read-only pinned checkout."""
    for name in ("sqlglot", "tests", "pyproject.toml"):
        relevant = {path: contents for path, contents in edits.items() if path.startswith(name + "/")}
        if not relevant:
            (target / name).symlink_to(source / name)
            continue
        for path, contents in relevant.items():
            (target / path).parent.mkdir(parents=True, exist_ok=True)
            (target / path).write_bytes(contents)
        changed_parents = {parent for path in relevant for parent in Path(path).parents if parent != Path(".")}
        for parent in changed_parents:
            for child in (source / parent).iterdir():
                output = target / parent / child.name
                if not output.exists():
                    output.symlink_to(child)


def pytest_passes(ids):
    import pytest

    class Results:
        def __init__(self):
            self.calls = set()
            self.finished = set()
            self.failed = set()
            self.collection_failed = False

        def pytest_collectreport(self, report):
            self.collection_failed |= report.failed

        def pytest_runtest_logreport(self, report):
            if report.failed or report.skipped:
                self.failed.add(report.nodeid)
            elif report.when == "call" and report.passed:
                self.calls.add(report.nodeid)
            elif report.when == "teardown" and report.passed:
                self.finished.add(report.nodeid)

    results = Results()
    status = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", *ids], plugins=[results])
    if results.collection_failed or status not in (0, 1):
        return []
    return sorted(set(ids) & results.calls & results.finished - results.failed)


def pytest_collection():
    import pytest

    class Collection:
        def __init__(self):
            self.errors = set()
            self.tests = []

        def pytest_collectreport(self, report):
            if report.failed:
                self.errors.add(report.nodeid)

        def pytest_collection_finish(self, session):
            self.tests = [item.nodeid for item in session.items]

    report = Collection()
    status = pytest.main(
        ["-q", "--collect-only", "--continue-on-collection-errors", "-p", "no:cacheprovider", "--tb=no", "tests"],
        plugins=[report],
    )
    if status == 0 or (status in (1, 2, 5) and report.errors):
        return report.tests, sorted(report.errors)
    return None


class Verification:
    def __init__(self, pinned, baseline, query_path=None):
        self.pinned = pinned
        self.baseline = baseline
        self.started = time.monotonic()
        self.warmed = False
        self.pristine = None
        self.queries = load_profiles(query_path) if query_path is not None else {}
        self.warm = WarmProcess(pinned) if query_path is not None else None

    def run(self, edits, operation, timeout):
        if not hasattr(os, "fork"):
            return None
        import resource

        if time.monotonic() - self.started > 55:
            return None
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        if children.ru_utime + children.ru_stime > 24:
            return None
        with tempfile.TemporaryDirectory(prefix="impact-") as tmp:
            root = Path(tmp)
            overlay(self.pinned, root, edits)
            result = root / ".result"
            pid = os.fork()
            if pid == 0:
                try:
                    os.setsid()
                    with open(os.devnull, "w") as sink:
                        os.dup2(sink.fileno(), 1)
                        os.dup2(sink.fileno(), 2)
                    os.chdir(root)
                    sys.path.insert(0, str(root))
                    os.environ["PYTHONPATH"] = str(root)
                    answer = operation()
                    result.write_text(json.dumps(answer), encoding="utf-8")
                except BaseException:
                    os._exit(1)
                os._exit(0)

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                finished, status = os.waitpid(pid, os.WNOHANG)
                if finished:
                    if status == 0 and result.is_file():
                        return json.loads(result.read_bytes())
                    return None
                time.sleep(0.01)
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os.waitpid(pid, 0)
        return None

    def changed_state(self, edits, *, entries=False):
        modules = self.baseline["modules"]
        unordered = self.baseline.get("unordered_tuple_bindings", {})
        if self.pristine is None:
            self.pristine = self.run({}, lambda: snapshot(modules, unordered), 2.5)
        if self.pristine is None or self.pristine.keys() != self.baseline["state"].keys():
            return None
        # Only edited modules can change the syntax used to construct a tuple.
        unchanged = unordered.copy()
        for path in edits:
            unchanged.pop(module_name(path), None)
        current = self.run(edits, lambda: snapshot(modules, unchanged), 2.5)
        if current is None:
            return None
        old = self.pristine
        return {key for key in old.keys() | current.keys()
                if (entries or "#" not in key) and old.get(key) != current.get(key)}

    def collection_impact(self, edits, tests):
        result = self.run(edits, pytest_collection, 6.0)
        if result is None:
            return None
        collected, errors = map(set, result)
        files = {test["file"] for test in tests}
        # An error without a file or directory cannot be scoped reliably.
        if any(not error or not any(path == error or path.startswith(error.rstrip("/") + "/")
                                    for path in files) for error in errors):
            return None
        return {
            test["number"] for test in tests
            if test["id"] not in collected or any(test["file"] == error
                or test["file"].startswith(error.rstrip("/") + "/") for error in errors)
        }

    def exact_passes(self, edits, ids, timeout, plan=None, cold=()):
        passed = []
        if plan and self.warm:
            warm_ids = [node_id for node_id in ids if node_id not in cold]
            result = self.warm.request(edits, plan, warm_ids, timeout)
            if result is not None:
                passed, ids = result, cold
        if not ids:
            return passed
        if not self.warmed:
            # Full-overlay verification imports sqlglot only in its children.
            import pytest  # noqa: F401

            self.warmed = True
        return passed + (self.run(edits, lambda: pytest_passes(ids), timeout) or [])

    def close(self):
        if self.warm:
            self.warm.close()

    def tpc_passes(self, edits, node_id):
        expected = self.baseline["tpc"].get(node_id)
        if expected:
            return self.run(edits, lambda: tpc_signatures(TPC_TESTS[node_id]) == expected, 4.5) is True
        return False

    def queries_pass(self, edits, node_id, indices, available_ms):
        profile = self.queries[node_id]
        # The build records instrumented CPU, an upper estimate for plain replay.
        estimate = 900 + sum(profile["times"][index] for index in indices)
        if estimate * 1.5 > available_ms or estimate > 6_000:
            return False
        if not indices:
            return True
        timeout = min(8.0, max(2.5, 1.5 * estimate / 1000))
        return self.run(edits, lambda: bool(run_cases(node_id, profile, indices)), timeout) is True


if __name__ == "__main__":
    modules = sorted({module_name(path.as_posix()) for path in Path("sqlglot").rglob("*.py")
                      if path.name != "__main__.py"})
    unordered = {}
    state = snapshot(modules, unordered)
    tpc = {name: tpc_signatures(dataset) for name, dataset in TPC_TESTS.items()}
    data = {"modules": modules, "state": state, "tpc": tpc, "unordered_tuple_bindings": unordered}
    Path(sys.argv[1]).write_text(json.dumps(data), encoding="utf-8")
