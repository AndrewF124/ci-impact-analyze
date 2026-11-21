"""Select and schedule sqlglot tests under the Blast Radius scoring model."""

import json
import math
import sys
from operator import itemgetter
from pathlib import Path

from impact import ImpactIndex
from patch_parser import parse_patch
from verify import TPC_TESTS, Verification

DATA = Path("/data")
PINNED = Path("/pinned")
COVERAGE = Path(__file__).with_name("coverage.json.gz")
ISOLATION = Path(__file__).with_name("isolated.json.gz")
VERIFICATION = Path(__file__).with_name("verification.json")
QUERY_PROFILES = Path(__file__).with_name("query-profiles.json.gz")
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


def add_to_slack(workers: list[list[int]], durations: dict[int, int], nearby: frozenset[int]) -> list[list[int]]:
    """Add nearby coverage using the original worker count and makespan."""
    if len(workers) < 2 or not nearby:
        return workers
    result = [worker.copy() for worker in workers]
    loads = [sum(durations[test] for test in worker) for worker in workers]
    limit = max(loads)
    present = {test for worker in workers for test in worker}
    for test in sorted(nearby - present, key=lambda number: (-durations[number], number)):
        duration = durations[test]
        lightest = loads.index(min(loads))
        if loads[lightest] + duration <= limit:
            result[lightest].append(test)
            loads[lightest] += duration
    return result


def verify_selected(impact, files, selected, verification, durations, node_ids):
    if not selected:
        return frozenset()
    # Exact cold runs can also establish an outcome for an untraced test.
    cheap = sorted(n for n in selected if durations[n] <= 500)
    query_tests = [n for n in selected - impact.always
                   if node_ids[n] in TPC_TESTS or node_ids[n] in verification.queries]
    probe = len(cheap) <= 12 and 0 < sum(durations[n] for n in cheap) <= 900
    if not probe and not query_tests:
        return frozenset()
    edits = impact.patched_files(files)
    passed = set()
    for n in sorted(query_tests, key=durations.__getitem__, reverse=True):
        node = node_ids[n]
        candidates = selected - passed
        jobs = sorted(((test, durations[test]) for test in candidates), key=itemgetter(1), reverse=True)
        without = schedule([job for job in jobs if job[0] != n])
        savings = schedule_cost(schedule(jobs), durations) - schedule_cost(without, durations)
        if savings <= 0:
            continue
        cases = impact.query_cases(edits, node, verification.queries.get(node))
        if cases is not None and verification.queries_pass(edits, node, cases, savings):
            passed.add(n)
        elif (node in TPC_TESTS and savings >= 2_000
              and impact.tpc_replay_scope(edits, node) and verification.tpc_passes(edits, node)):
            passed.add(n)
    if probe:
        jobs = sorted(((n, durations[n]) for n in selected - passed), key=itemgetter(1), reverse=True)
        without = set(cheap)
        current = schedule_cost(schedule(jobs), durations)
        optimistic = schedule_cost(schedule([job for job in jobs if job[0] not in without]), durations)
        # Pay the import/collection cost only when even a clean probe can repay it.
        if current - optimistic > 400 + sum(durations[n] for n in cheap):
            ids = [node_ids[n] for n in cheap]
            cold = [node_ids[n] for n in cheap if n in impact.always]
            warm = [node_ids[n] for n in cheap if n not in impact.always]
            try:
                plan = impact.warm_plan(edits, warm) if warm else None
            except ValueError:
                plan = None
            verified = set(verification.exact_passes(edits, ids, 1.8, plan, cold))
            passed.update(n for n in cheap if node_ids[n] in verified)
    return frozenset(passed)


def schedule_cost(workers, durations):
    return max((sum(durations[n] for n in group) for group in workers), default=0) + WORKER_COST_MS * len(workers)


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    try:
        impact = ImpactIndex(PINNED, COVERAGE, tests, ISOLATION)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        print(f"coverage unavailable: {error}", file=sys.stderr)
        impact = None
    try:
        verification = Verification(PINNED, json.loads(VERIFICATION.read_bytes()), QUERY_PROFILES)
    except (OSError, ValueError):
        verification = None

    timed_tests = sorted(
        [(test["number"], math.floor(test["seconds"] * 1000 + 0.5)) for test in tests],
        key=itemgetter(1),
        reverse=True,
    )
    durations = dict(timed_tests)
    node_ids = {test["number"]: test["id"] for test in tests}
    schedules = {None: schedule(timed_tests)}
    fallback = selected_count = 0
    for patch in sorted((DATA / "commits").glob("*.patch")):
        selected = None
        files = []
        verified = frozenset()
        try:
            if impact is not None:
                files = parse_patch(patch.read_bytes().decode("utf-8"))
                selected = impact.select_files(files)
            if verification is not None and files:
                if selected is None:
                    selected = impact.state_impact(files, verification)
                if selected:
                    verified = verify_selected(impact, files, selected, verification, durations, node_ids)
                    selected -= verified
        except (OSError, ValueError, SyntaxError, RecursionError):
            pass
        fallback += selected is None
        selected_count += len(tests) if selected is None else len(selected)
        if selected not in schedules:
            schedules[selected] = schedule([test for test in timed_tests if test[0] in selected])
        workers = schedules[selected]
        if selected is not None and len(workers) > 1:
            workers = add_to_slack(workers, durations, impact.nearby(files) - verified)
        print(json.dumps({"commit": patch.stem, "workers": workers}, separators=(",", ":")))
    print(f"selected {selected_count} tests across all patches, {fallback} full-suite fallbacks", file=sys.stderr)
    if verification is not None:
        verification.close()


if __name__ == "__main__":
    main()
