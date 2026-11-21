"""Check expensive tests in fresh interpreters alongside the shared-suite map."""

import gzip
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def isolated_run(task):
    node_id, output = task
    try:
        subprocess.run(
            [sys.executable, "/build_coverage.py", str(output), node_id],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=95, check=True,
        )
        result = read(output)
    except (OSError, ValueError, subprocess.SubprocessError):
        return node_id, None
    if result["tests"] != [node_id] or result["unreliable"]:
        return node_id, None
    paths = {}
    for scope in (result["lines"], result["shared"], result["startup_lines"]):
        for path, lines in scope.items():
            paths.setdefault(path, set()).update(map(int, lines))
    return node_id, {path: sorted(lines) for path, lines in paths.items()}


def build(shared_path, output):
    shared = read(shared_path)
    ranked = sorted(shared["durations"], key=shared["durations"].get, reverse=True)
    reliable = [node_id for node_id in ranked if node_id not in shared["unreliable"]][:12]
    with tempfile.TemporaryDirectory(prefix="isolated-profile-") as temp:
        tasks = [(node_id, Path(temp) / f"{i}.json.gz") for i, node_id in enumerate(reliable)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            profiles = {node_id: lines for node_id, lines in pool.map(isolated_run, tasks) if lines}
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(profiles, stream, separators=(",", ":"))
    print(f"Isolated coverage completed for {len(profiles)} of {len(reliable)} slow tests")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
