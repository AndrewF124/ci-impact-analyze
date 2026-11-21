"""Measure per-test coverage in a checkout of the pinned sqlglot source.

Run from the checkout during the Docker build. The output uses pytest node IDs
and a bit mask per source line. Line zero records files opened for reading.
Collection hits and shared fixture hits are recorded alongside per-test hits.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from multiprocessing.util import Finalize
from pathlib import Path
from unittest.mock import patch

SOURCE_COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"


class Coverage:
    def __init__(self, *, capture_startup=False):
        self.prefix = os.path.normcase(os.getcwd() + os.sep)
        self.lines: dict[str, dict[int, int]] = {}
        self.shared: dict[str, dict[int, int]] = {}
        self.startup_lines: dict[str, set[int]] = {}
        self.capture_startup = capture_startup
        self.incomplete = 0
        self.current = None
        self.mask = 0
        self.shared_mask = 0

    @cache
    def source_path(self, filename: str) -> str | None:
        if not os.path.isabs(filename):
            return None
        normalized = os.path.normcase(filename)
        if normalized.startswith(self.prefix) and not normalized.endswith(".pyc"):
            return filename[len(self.prefix):].replace(os.sep, "/")
        return None

    def record(self, path: str, line: int, *, startup: bool = True) -> None:
        if self.mask:
            hits = self.lines.setdefault(path, {})
            hits[line] = hits.get(line, 0) | self.mask
        if self.shared_mask:
            hits = self.shared.setdefault(path, {})
            hits[line] = hits.get(line, 0) | self.shared_mask
        if self.capture_startup and startup:
            self.startup_lines.setdefault(path, set()).add(line)

    def start(self, code, _offset):
        if path := self.source_path(code.co_filename):
            if not sys.monitoring.get_local_events(sys.monitoring.COVERAGE_ID, code):
                sys.monitoring.set_local_events(
                    sys.monitoring.COVERAGE_ID, code, sys.monitoring.events.LINE
                )
            if code.co_name != "<module>":
                self.record(path, code.co_firstlineno)
        return sys.monitoring.DISABLE

    def line(self, code, line):
        # A definition's line runs when it is bound at import time. Only an
        # actual code entry or file read is a startup dependency for body edits.
        self.record(self.source_path(code.co_filename), line, startup=False)
        return sys.monitoring.DISABLE

    def audit(self, event, args):
        if not self.mask and not self.capture_startup:
            return
        if event in {"subprocess.Popen", "os.exec", "os.posix_spawn"}:
            self.incomplete |= self.mask | self.shared_mask
        if event == "open":
            filename, _mode, flags = args
            if isinstance(filename, (str, bytes)) and (flags & os.O_ACCMODE) != os.O_WRONLY:
                name = os.path.abspath(os.fsdecode(filename))
                if path := self.source_path(name):
                    # Importing Python code is tracked by execution. Keep raw
                    # source reads by tests and introspection as dependencies.
                    caller = sys._getframe(1)
                    if (
                        path.endswith(".py")
                        and caller.f_code.co_filename == "<frozen importlib._bootstrap_external>"
                        and caller.f_code.co_name == "get_data"
                        and caller.f_back is not None
                        and caller.f_back.f_code.co_name == "get_code"
                    ):
                        return
                    self.record(path, 0)

    def begin(self) -> None:
        monitor = sys.monitoring
        tool = monitor.COVERAGE_ID
        monitor.use_tool_id(tool, "blast-radius coverage")
        monitor.register_callback(tool, monitor.events.PY_START, self.start)
        monitor.register_callback(tool, monitor.events.LINE, self.line)
        monitor.set_events(tool, monitor.events.PY_START)
        sys.addaudithook(self.audit)

    def switch(self, test_index: int | None, shared_mask: int = 0) -> None:
        if test_index == self.current and shared_mask == self.shared_mask:
            return
        self.current = test_index
        self.mask = 0 if test_index is None else 1 << test_index
        self.shared_mask = shared_mask
        if self.mask:
            self.capture_startup = False
            sys.monitoring.restart_events()

    def encoded(self) -> dict:
        def encode(table):
            return {
                path: {line: format(mask, "x") for line, mask in hits.items()}
                for path, hits in table.items()
            }

        return {"lines": encode(self.lines), "shared": encode(self.shared)}

    def merge(self, result: dict) -> None:
        self.incomplete |= int(result["incomplete"], 16)
        for table, encoded in ((self.lines, result["lines"]), (self.shared, result["shared"])):
            for path, incoming in encoded.items():
                hits = table.setdefault(path, {})
                for line, mask in incoming.items():
                    number = int(line)
                    hits[number] = hits.get(number, 0) | int(mask, 16)


_worker_coverage = None


def _save_worker(directory: Path, coverage: Coverage) -> None:
    output = directory / f"{os.getpid()}.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump({**coverage.encoded(), "incomplete": format(coverage.incomplete, "x")}, stream)


def _start_worker(directory: Path, test_index: int | None, shared_mask: int, initializer, initargs) -> None:
    global _worker_coverage
    _worker_coverage = Coverage()
    _worker_coverage.begin()
    Finalize(None, _save_worker, args=(directory, _worker_coverage), exitpriority=10)
    _worker_coverage.switch(test_index, shared_mask)
    if initializer is not None:
        initializer(*initargs)


def _run_covered(test_index: int | None, shared_mask: int, function, args, kwargs):
    _worker_coverage.switch(test_index, shared_mask)
    return function(*args, **kwargs)


def measure(output: Path) -> None:
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    import pytest

    coverage = Coverage(capture_startup=True)

    class TestContexts:
        def __init__(self):
            self.tests = {}
            self.fixture_masks = {}
            self.passed = set()
            self.failed = set()

        def pytest_collection_finish(self, session):
            self.tests = {item.nodeid: index for index, item in enumerate(session.items)}
            for index, item in enumerate(session.items):
                for scope in item.listchain():
                    if scope is not item:
                        self.fixture_masks[scope] = self.fixture_masks.get(scope, 0) | (1 << index)

        @pytest.hookimpl(wrapper=True)
        def pytest_fixture_setup(self, fixturedef, request):
            previous = coverage.shared_mask
            test_index = coverage.current
            if fixturedef.scope != "function":
                coverage.switch(test_index, self.fixture_masks.get(request.node, 0))
            try:
                return (yield)
            finally:
                coverage.switch(test_index, previous)

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_protocol(self, item, nextitem):
            coverage.switch(self.tests[item.nodeid])
            try:
                return (yield)
            finally:
                coverage.switch(None)

        def pytest_runtest_logreport(self, report):
            if report.failed or report.skipped:
                self.failed.add(report.nodeid)
            elif report.when == "call" and report.passed:
                self.passed.add(report.nodeid)

    contexts = TestContexts()
    original_init = ProcessPoolExecutor.__init__
    original_submit = ProcessPoolExecutor.submit

    with tempfile.TemporaryDirectory(prefix="sqlglot-coverage-") as temporary:
        directory = Path(temporary)

        def init_pool(pool, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
            # Pinned optimizer and executor tests use spawn. Bound their pools
            # to keep memory and CPU use predictable on the build host.
            if max_workers is None:
                max_workers = os.process_cpu_count() or 1
            original_init(
                pool,
                max_workers=min(max_workers, 3),
                mp_context=mp_context,
                initializer=_start_worker,
                initargs=(directory, coverage.current, coverage.shared_mask, initializer, initargs),
                **kwargs,
            )

        def submit(pool, function, /, *args, **kwargs):
            return original_submit(
                pool, _run_covered, coverage.current, coverage.shared_mask, function, args, kwargs
            )

        coverage.begin()
        with patch.object(ProcessPoolExecutor, "__init__", init_pool), patch.object(
            ProcessPoolExecutor, "submit", submit
        ):
            code = pytest.main(
                ["-q", "--tb=short", "--continue-on-collection-errors", "tests"],
                plugins=[contexts],
            )

        for file in directory.glob("*.json"):
            coverage.merge(json.loads(file.read_text(encoding="utf-8")))

    if not contexts.tests:
        raise RuntimeError(f"pytest did not collect tests (exit {code})")
    unmeasured = contexts.tests.keys() - contexts.passed
    unmeasured.update(contexts.failed)
    unmeasured.update(
        test for test, index in contexts.tests.items() if coverage.incomplete & (1 << index)
    )
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(
            {
                "source_commit": SOURCE_COMMIT,
                "tests": list(contexts.tests),
                "unreliable": sorted(unmeasured),
                "startup_lines": {path: sorted(lines) for path, lines in coverage.startup_lines.items()},
                **coverage.encoded(),
            },
            stream,
            separators=(",", ":"),
        )
    print(f"Measured {len(contexts.tests)} tests, {len(unmeasured)} incomplete (pytest exit {code})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: build_coverage.py OUTPUT.json.gz")
    os.environ["SKIP_INTEGRATION"] = "0"
    measure(Path(sys.argv[1]).resolve())
