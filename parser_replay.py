"""Replay the parser limit regression when its large inputs are unaffected."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import subprocess
import sys
import unittest
from functools import cache
from pathlib import Path
from unittest.mock import patch

from impact import changed_callables, changed_statements
from query_replay import QueryTrace, load_profiles


PARSER_TEST = "tests/test_parser.py::TestParser::test_max_nodes"
SUITE_PATH = "tests/test_parser.py"
LARGE_QUERY_LENGTH = 500_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ParserTrace(QueryTrace):
    """Use the same line and entry events as the fixture query profiler."""

    @cache
    def source(self, filename):
        relative = os.path.normcase(filename).removeprefix(os.path.normcase(self.prefix))
        relative = relative.replace(os.sep, "/")
        return relative if relative.startswith("sqlglot/") else None


def exercise(rows=None, trace=None):
    """Run the original unittest, recording or replaying its large parse errors."""
    from sqlglot.errors import ParseError
    from tests import test_parser as suite

    original = suite.parse_one
    recording = rows is None
    observed = []
    large_calls = 0

    def parse_one(sql, *args, **kwargs):
        nonlocal large_calls
        if len(sql) < LARGE_QUERY_LENGTH:
            return original(sql, *args, **kwargs)
        if args or kwargs.keys() != {"max_nodes"} or type(kwargs["max_nodes"]) is not int:
            raise ValueError("unexpected large parser call")
        signature = [digest(sql.encode()), kwargs["max_nodes"]]
        index = large_calls
        large_calls += 1
        try:
            if recording:
                try:
                    return original(sql, **kwargs)
                except ParseError as error:
                    if type(error) is not ParseError or len(error.args) != 1 or type(error.args[0]) is not str:
                        raise ValueError("unexpected large parser error") from error
                    observed.append([signature, error.args[0], error.errors])
                    raise
            if index >= len(rows) or signature != rows[index][0]:
                raise ValueError("large parser input changed")
            _, message, errors = rows[index]
            raise ParseError(message, errors)
        finally:
            if trace is not None and large_calls == 2:
                trace.switch(0)

    result = unittest.TestResult()
    with patch.object(suite, "parse_one", parse_one):
        suite.TestParser("test_max_nodes").run(result)
    if (result.testsRun != 1 or not result.wasSuccessful() or result.skipped
            or result.expectedFailures or large_calls != 2 or recording and len(observed) != 2):
        raise ValueError("parser limit test did not complete as expected")
    return observed


def build_profile(output):
    trace = ParserTrace()
    with trace:
        large_errors = exercise(trace=trace)
    # Bit zero covers every library line reached before both large calls finish.
    protected = {path: sorted(line for line, hits in lines.items() if hits & 1)
                 for path, lines in trace.lines.items()}
    protected = {path: lines for path, lines in protected.items() if lines}
    if "sqlglot/parser.py" not in protected:
        raise ValueError("large parser coverage unavailable")
    profile = {"suite": digest(Path(SUITE_PATH).read_bytes()), "protected": protected,
               "large_errors": large_errors}
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        json.dump(profile, stream, separators=(",", ":"))
    # The small execution must succeed in a separate, freshly imported interpreter.
    subprocess.run([sys.executable, __file__, "--check", str(output)], check=True, timeout=12)


def eligible(profile, pinned, edits):
    """Require an expression-only edit outside the protected execution prefix."""
    if not profile.get("protected") or len(profile.get("large_errors", ())) != 2 or len(edits) != 1:
        return False
    path, patched = next(iter(edits.items()))
    if not path.startswith("sqlglot/") or not path.endswith(".py"):
        return False
    if digest((pinned / SUITE_PATH).read_bytes()) != profile.get("suite"):
        return False
    before = ast.parse((pinned / path).read_bytes(), type_comments=True)
    after = ast.parse(patched, type_comments=True)
    functions = changed_callables(before, after)
    if len(functions) != 1:
        return False
    old, new = functions[0]
    # Definition-time effects and changes to the body's control flow are unknown.
    if not isinstance(old, ast.FunctionDef) or old.decorator_list:
        return False
    spans = changed_statements(old, new)
    if not spans:
        return False
    protected = profile["protected"].get(path, ())
    return not any(first <= line <= last for first, last in spans for line in protected)


def run_cheap(profile):
    if digest(Path(SUITE_PATH).read_bytes()) != profile.get("suite"):
        return False
    try:
        exercise(profile["large_errors"])
    except (ValueError, KeyError, TypeError, IndexError):
        return False
    return True


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(0 if run_cheap(load_profiles(Path(sys.argv[2]))) else 1)
    output = Path(sys.argv[1]).resolve()
    try:
        build_profile(output)
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        print(f"Parser phase profile unavailable: {error}", file=sys.stderr)
        with gzip.open(output, "wt", encoding="utf-8") as stream:
            json.dump({}, stream)
