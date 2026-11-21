"""Select and schedule sqlglot tests under the Blast Radius scoring model."""

import json
import math
import sys
from operator import itemgetter
from pathlib import Path

from impact import ImpactIndex

DATA = Path("/data")
PINNED = Path("/pinned")
COVERAGE = Path(__file__).with_name("coverage.json.gz")
MAX_WORKERS = 8
WORKER_COST_MS = 2_000


def schedule(tests: list[tuple[int, int]]) -> list[list[int]]:
    """Minimize scorer cost over longest-first schedules for each worker count.

    Durations are in scorer milliseconds and tests are sorted longest first.
    """
    if not tests:
        return []

    longest = tests[0][1]
    total = sum(duration for _, duration in tests)
    best_workers = [[number for number, _ in tests]]
    best_cost = total + WORKER_COST_MS

    for count in range(2, min(MAX_WORKERS, len(tests)) + 1):
        overhead = count * WORKER_COST_MS
        if longest + overhead >= best_cost:
            break
        if (total + count - 1) // count + overhead >= best_cost:
            continue

        workers = [[] for _ in range(count)]
        loads = [0] * count
        for index, (number, duration) in enumerate(tests):
            # Give every worker a test even when some durations round to zero.
            lightest = index if index < count else loads.index(min(loads))
            workers[lightest].append(number)
            loads[lightest] += duration
        cost = max(loads) + overhead
        if cost < best_cost:
            best_workers, best_cost = workers, cost

    return best_workers


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    try:
        impact = ImpactIndex(PINNED, COVERAGE, tests)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        print(f"coverage unavailable: {error}", file=sys.stderr)
        impact = None

    timed_tests = sorted(
        [(test["number"], math.floor(test["seconds"] * 1000 + 0.5)) for test in tests],
        key=itemgetter(1),
        reverse=True,
    )
    schedules = {None: schedule(timed_tests)}
    fallback = selected_count = 0
    for patch in sorted((DATA / "commits").glob("*.patch")):
        selected = None
        try:
            if impact is not None:
                selected = impact.select(patch.read_bytes().decode("utf-8"))
        except (OSError, UnicodeError):
            pass
        fallback += selected is None
        selected_count += len(tests) if selected is None else len(selected)
        if selected not in schedules:
            schedules[selected] = schedule([test for test in timed_tests if test[0] in selected])
        print(json.dumps({"commit": patch.stem, "workers": schedules[selected]}, separators=(",", ":")))
    print(f"selected {selected_count} tests across all patches, {fallback} full-suite fallbacks", file=sys.stderr)


if __name__ == "__main__":
    main()
