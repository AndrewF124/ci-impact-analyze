"""Measure how much CPU the pinned warm verifier spends on actual tests."""

from __future__ import annotations

import gzip
import importlib
import itertools
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from semantic import module_name
from warm_verify import PROCESS_EVENTS, _test_cases, fast_test_id


PROFILE_VERSION = 1
BUILD_SECONDS = 55
OVERLAY_AND_FORK_MS = 80


class ReplayCosts:
    def __init__(self, data):
        def measured(section):
            values = data.get(section)
            if not isinstance(values, dict):
                return {}
            return {key: value for key, value in values.items()
                    if isinstance(key, str) and type(value) is int and 0 <= value <= 60_000}

        self.tests = measured("tests")
        self.modules = measured("modules")
        self.classes = measured("classes")
        self.startup_ms = measured("runtime").get("startup", 0)

    def test_ms(self, node_id):
        return self.tests.get(node_id)

    def estimate_ms(self, ids):
        """Include conservative overhead and charge module/class costs once per group."""
        tests = list(ids)
        if not tests or any(node_id not in self.tests for node_id in tests):
            return None
        modules = {node_id.split("::", 1)[0] for node_id in tests}
        classes = [owner for owner, _ in itertools.groupby(node_id.rsplit("::", 1)[0] for node_id in tests)]
        if any(module not in self.modules for module in modules) or any(owner not in self.classes for owner in classes):
            return None
        measured = (
            self.startup_ms
            + sum(self.tests[node_id] for node_id in tests)
            + sum(self.modules[path] for path in modules)
            + sum(self.classes[owner] for owner in classes)
        )
        return OVERLAY_AND_FORK_MS + (3 * measured + 1) // 2


def load_replay_costs(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == PROFILE_VERSION and data.get("python") == list(sys.version_info[:2]):
            return ReplayCosts(data)
    except (OSError, TypeError, ValueError):
        pass
    return ReplayCosts({})


def empty_profile():
    return {"version": PROFILE_VERSION, "python": list(sys.version_info[:2]), "runtime": {},
            "tests": {}, "modules": {}, "classes": {}}


def write_profile(path, profile):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(profile, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def milliseconds(nanos):
    return (max(0, nanos) + 999_999) // 1_000_000


class TimedResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.started = {}
        self.elapsed = {}
        self.passed = set()

    def startTest(self, test):
        super().startTest(test)
        self.started[id(test)] = time.process_time_ns()

    def stopTest(self, test):
        self.elapsed[id(test)] = time.process_time_ns() - self.started[id(test)]
        super().stopTest(test)

    def addSuccess(self, test):
        self.passed.add(id(test))


def measure_class(pinned, ids):
    cases = []
    assembly = {}
    for node_id in ids:
        start = time.process_time_ns()
        probe = _test_cases(pinned, [node_id])
        if probe is not None:
            case = probe[0]
            cases.append((node_id, case))
            assembly[id(case)] = time.process_time_ns() - start
    if not cases:
        return None

    result = TimedResult()
    start = time.process_time_ns()
    unittest.TestSuite([case for _, case in cases]).run(result)
    elapsed = time.process_time_ns() - start
    if not result.wasSuccessful() or result.skipped or result.expectedFailures:
        return None
    tests = {node_id: max(1, milliseconds(assembly[id(case)] + result.elapsed[id(case)]))
             for node_id, case in cases if id(case) in result.passed and id(case) in result.elapsed}
    return milliseconds(elapsed - sum(result.elapsed.values())), tests


def measure(coverage_path, output):
    with gzip.open(coverage_path, "rt", encoding="utf-8") as stream:
        coverage = json.load(stream)
    unreliable = set(coverage["unreliable"])
    durations = coverage["durations"]
    ids = (node_id for node_id in coverage["tests"] if node_id not in unreliable and fast_test_id(node_id)
           and isinstance(durations.get(node_id), (int, float)) and 0 <= durations[node_id] <= 5)
    pinned = Path.cwd()
    sys.path.insert(0, str(pinned))
    profile = empty_profile()
    deadline = time.monotonic() + BUILD_SECONDS - 4

    start = time.process_time_ns()
    importlib.import_module("sqlglot")
    profile["runtime"]["startup"] = milliseconds(time.process_time_ns() - start)

    spawned = False

    def prevent_process(event, _):
        nonlocal spawned
        if event in PROCESS_EVENTS:
            spawned = True
            raise RuntimeError("warm replay calibration cannot isolate a new process")

    sys.addaudithook(prevent_process)
    for path, candidates in itertools.groupby(ids, lambda node_id: node_id.split("::", 1)[0]):
        if time.monotonic() >= deadline:
            break
        spawned = False
        try:
            start = time.process_time_ns()
            importlib.import_module(module_name(path))
        except Exception:
            continue
        if spawned:
            continue
        profile["modules"][path] = milliseconds(time.process_time_ns() - start)
        for owner, group in itertools.groupby(candidates, lambda node_id: node_id.rsplit("::", 1)[0]):
            if time.monotonic() >= deadline:
                return
            spawned = False
            try:
                measured = measure_class(pinned, list(group))
            except Exception:
                continue
            if measured is not None and not spawned:
                setup, tests = measured
                profile["classes"][owner] = max(setup, profile["classes"].get(owner, 0))
                profile["tests"].update(tests)
                write_profile(output, profile)


def build(coverage_path, output):
    write_profile(output, empty_profile())
    try:
        subprocess.run([sys.executable, "-B", __file__, "--measure", str(coverage_path), str(output)],
                       timeout=BUILD_SECONDS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass
    calibrated = load_replay_costs(output)
    print(f"Calibrated warm replay CPU for {len(calibrated.tests)} tests")


if __name__ == "__main__":
    if sys.argv[1] == "--measure":
        measure(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
