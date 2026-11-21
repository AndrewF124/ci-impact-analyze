"""Check expensive tests in fresh interpreters alongside the shared-suite map."""

import gzip
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def cold_dependencies(node_id, result, ordered):
    paths = {}
    for scope in (result["lines"], result["shared"], result["startup_lines"]):
        for path, lines in scope.items():
            paths.setdefault(path, set()).update(map(int, lines))

    if (not ordered or node_id not in ordered.get("ordered_tests", ())
            or "worker_tests" not in result or node_id in result["worker_tests"]
            or "noninitializers" not in result):
        return paths

    bit = 1 << ordered["tests"].index(node_id)
    earlier = bit - 1
    # Only remove a declaration that is known to have executed before the target
    # in the normal process. A cold read, ordinary execution, or attribution to
    # target setup/teardown still preserves the dependency.
    for path, initialized in result.get("initializers", {}).items():
        if path not in paths or not path.startswith("sqlglot/"):
            continue
        ordinary = set(result["noninitializers"].get(path, ()))
        cold_startup = set(result["startup_lines"].get(path, ()))
        original_startup = set(ordered.get("startup_initializers", {}).get(path, ()))
        original_init = ordered.get("initializers", {}).get(path, {})
        original_lines = ordered["lines"].get(path, {})
        shared_lines = ordered["shared"].get(path, {})
        for key in initialized:
            line = int(key)
            if line <= 0 or line in ordinary or line in cold_startup:
                continue
            if (int(original_lines.get(key, "0"), 16) | int(shared_lines.get(key, "0"), 16)) & bit:
                continue
            if line in original_startup or int(original_init.get(key, "0"), 16) & earlier:
                paths[path].discard(line)
    return paths


def isolated_run(task, ordered=None):
    node_id, output = task
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("build_coverage.py")), str(output), node_id],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=95, check=True,
        )
        result = read(output)
    except (OSError, ValueError, subprocess.SubprocessError):
        return node_id, None
    if result["tests"] != [node_id] or result["unreliable"]:
        return node_id, None
    return node_id, {path: sorted(lines) for path, lines in cold_dependencies(node_id, result, ordered).items()
                     if lines}


def build(shared_path, output):
    shared = read(shared_path)
    ranked = sorted(shared["durations"], key=shared["durations"].get, reverse=True)
    reliable = [node_id for node_id in ranked if node_id not in shared["unreliable"]][:12]
    with tempfile.TemporaryDirectory(prefix="isolated-profile-") as temp:
        tasks = [(node_id, Path(temp) / f"{i}.json.gz") for i, node_id in enumerate(reliable)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            run = partial(isolated_run, ordered=shared)
            profiles = {node_id: lines for node_id, lines in pool.map(run, tasks) if lines}
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(profiles, stream, separators=(",", ":"))
    print(f"Isolated coverage completed for {len(profiles)} of {len(reliable)} slow tests")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
