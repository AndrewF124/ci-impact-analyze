"""Profile and rerun pinned SQL fixture cases with their original assertions."""

from __future__ import annotations

import gzip
import ast
import json
import os
import pickle
import sys
import time
from functools import cache
from pathlib import Path
from concurrent.futures import Future
from contextlib import nullcontext
from unittest.mock import patch

from semantic import TPC_TESTS


class QueryTrace:
    """Attribute executed lines to setup or an individual fixture case."""

    def __init__(self):
        self.prefix = str(Path.cwd()) + os.sep
        self.monitor = sys.monitoring
        self.tool = self.monitor.COVERAGE_ID
        self.lines = {}
        self.case = 1

    @cache
    def source(self, filename):
        return filename[len(self.prefix):] if filename.startswith(self.prefix + "sqlglot/") else None

    def start(self, code, offset):
        if path := self.source(code.co_filename):
            if not self.monitor.get_local_events(self.tool, code):
                self.monitor.set_local_events(self.tool, code, self.monitor.events.LINE)
            self.record(path, code.co_firstlineno)
        return self.monitor.DISABLE

    def line(self, code, line):
        self.record(self.source(code.co_filename), line)
        return self.monitor.DISABLE

    def record(self, path, line):
        hits = self.lines.setdefault(path, {})
        hits[line] = hits.get(line, 0) | self.case

    def switch(self, case):
        self.case = 1 << (case + 1)
        self.monitor.restart_events()

    def __enter__(self):
        self.monitor.use_tool_id(self.tool, "query replay profile")
        self.monitor.register_callback(self.tool, self.monitor.events.PY_START, self.start)
        self.monitor.register_callback(self.tool, self.monitor.events.LINE, self.line)
        self.monitor.set_events(self.tool, self.monitor.events.PY_START)
        return self

    def __exit__(self, *_):
        self.monitor.set_events(self.tool, 0)
        self.monitor.register_callback(self.tool, self.monitor.events.PY_START, None)
        self.monitor.register_callback(self.tool, self.monitor.events.LINE, None)
        self.monitor.free_tool_id(self.tool)


def run_executor_cases(dataset, indices=None, trace=None, before_case=None, *, target_dataset_only=False):
    """Use the suite's setup, exact worker functions and result assertions."""
    from sqlglot import parse_one
    from tests import test_executor as suite

    tpch = dataset == "tpc-h"
    other_schema = "TPCDS_SCHEMA" if tpch else "TPCH_SCHEMA"
    # Optimizer-only checkpoints never consult the other dataset. Use the pinned
    # setup for the relevant rows, fixture SQL and DuckDB views.
    context = patch.object(suite, other_schema, {}) if target_dataset_only else nullcontext()
    with context:
        suite.TestExecutor.setUpClass()
    try:
        instance = suite.TestExecutor()
        cases = instance.tpch_sqls if tpch else instance.tpcds_sqls
        schema, tables = ((suite.TPCH_SCHEMA, instance.tpch_tables) if tpch
                          else (suite.TPCDS_SCHEMA, instance.tpcds_tables))
        suite.initializer(schema, tables)
        timings = [0] * len(cases)
        for index in range(len(cases)) if indices is None else indices:
            if trace is not None:
                trace.switch(index)
            start = time.process_time()
            options, sql, _ = cases[index]
            if before_case is not None:
                before_case(index, options)
            # The pinned test serializes the input/result across a process pool.
            inputs = pickle.loads(pickle.dumps((parse_one(sql), options)))
            result = pickle.loads(pickle.dumps(suite.mp_execute(*inputs)))
            if result is not None:
                instance.subtestHelper(index, result, tpch=tpch)
            timings[index] = round((time.process_time() - start) * 1000)
        return timings
    finally:
        if trace is not None:
            trace.switch(-1)
        suite.TestExecutor.tearDownClass()


class InlinePool:
    """Preserve the pinned pool's argument and result serialization."""

    def __init__(self, **_):
        pass

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            function, args, kwargs = pickle.loads(pickle.dumps((function, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(function(*args, **kwargs))))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self):
        pass


def optimizer_fixtures(suite):
    """Find tests whose entire body is one check_file call without a case filter."""
    tree = ast.parse(Path(suite.__file__).read_bytes())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TestOptimizer")
    for method in owner.body:
        if not isinstance(method, ast.FunctionDef) or method.decorator_list or len(method.body) != 1:
            continue
        expression = method.body[0]
        if not isinstance(expression, ast.Expr) or not isinstance(expression.value, ast.Call):
            continue
        call = expression.value
        if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self" and call.func.attr == "check_file"
                and 0 < len(call.args) <= 2 and isinstance(call.args[0], ast.Constant)
                and type(call.args[0].value) is str and all(kw.arg not in (None, "only") for kw in call.keywords)):
            yield method.name, f"optimizer/{call.args[0].value}.sql"


def run_optimizer_cases(method, fixture, indices=None, trace=None):
    from tests import test_optimizer as suite

    cases = list(suite.load_sql_fixture_pairs(fixture))
    owner = suite.TestOptimizer
    instance = owner(method)
    with patch.object(suite, "ProcessPoolExecutor", InlinePool):
        owner.setUpClass()
        try:
            instance.setUp()
            timings = [0] * len(cases)
            for index in range(len(cases)) if indices is None else indices:
                if trace is not None:
                    trace.switch(index)
                start = time.process_time()
                with patch.object(suite, "load_sql_fixture_pairs", return_value=iter([cases[index]])):
                    getattr(instance, method)()
                timings[index] = round((time.process_time() - start) * 1000)
            return timings
        finally:
            if trace is not None:
                trace.switch(-1)
            owner.conn.close()
            owner.doClassCleanups()


def run_cases(node_id, profile, indices=None, trace=None):
    if node_id in TPC_TESTS:
        return run_executor_cases(TPC_TESTS[node_id], indices, trace)
    return run_optimizer_cases(node_id.rsplit("::", 1)[1], profile["fixture"], indices, trace)


def load_profiles(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return {}


def build_profiles(output):
    from tests import test_optimizer

    profiles = {node_id: {} for node_id in TPC_TESTS}
    fixtures = [(sum(1 for _ in test_optimizer.load_sql_fixture_pairs(fixture)), method, fixture)
                for method, fixture in optimizer_fixtures(test_optimizer)]
    # Profile a bounded set of larger methods within the Docker build budget.
    for size, method, fixture in sorted(fixtures, reverse=True)[:3]:
        if size:
            profiles[f"tests/test_optimizer.py::TestOptimizer::{method}"] = {"fixture": fixture}
    for node_id, profile in profiles.items():
        with QueryTrace() as trace:
            times = run_cases(node_id, profile, trace=trace)
        profile["times"] = times
        profile["lines"] = {path: {line: format(mask, "x") for line, mask in hits.items()}
                            for path, hits in trace.lines.items()}
        print(f"Profiled {node_id}: {len(times)} cases", flush=True)
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(profiles, stream, separators=(",", ":"))


if __name__ == "__main__":
    build_profiles(Path(sys.argv[1]))
