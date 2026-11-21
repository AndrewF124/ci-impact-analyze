"""Select and schedule sqlglot tests under the Blast Radius scoring model."""

import json
import math
import sys
import time
from operator import itemgetter
from pathlib import Path

from case_replay import PREFIX as OPTIMIZER_FIXTURE_PREFIX, fixture_cases, optimizer_cases
from fallback import FallbackIndex
from impact import ImpactIndex
from optimizer_replay import SOURCE as OPTIMIZER_SOURCE, TEST as OPTIMIZER_TEST
from parser_replay import PARSER_TEST
from patch_parser import parse_patch
from replay_costs import load_replay_costs
from semantic import TPC_TESTS
from verify import Verification
from warm_verify import fast_test_id

DATA = Path("/data")
PINNED = Path("/pinned")
COVERAGE = Path(__file__).with_name("coverage.json.gz")
ISOLATION = Path(__file__).with_name("isolated.json.gz")
SUBPROCESS_PROFILE = Path(__file__).with_name("subprocess.json")
VERIFICATION = Path(__file__).with_name("verification.json")
QUERY_PROFILES = Path(__file__).with_name("query-profiles.json.gz")
CHECKPOINTS = Path(__file__).with_name("optimizer-checkpoints.json.gz")
PARSER_PROFILE = Path(__file__).with_name("parser-profile.json.gz")
REPLAY_COSTS = Path(__file__).with_name("replay-costs.json")
STRUCTURAL_REFERENCES = Path(__file__).with_name("structural-references.json")
MAX_WORKERS = 8
WORKER_COST_MS = 2_000
COLLECTION_PROBE_COST_MS = 3_000
WARM_BATCH_SECONDS = 12


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
    query_tests = [n for n in selected - impact.always if node_ids[n] in verification.queries]
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


def warm_prefix(selected, always, node_ids, durations, saving_for, cost_model=None):
    """Choose an affordable roster in suite order, keeping unresolved later tests."""
    roster = []
    cost = 400
    for number in sorted(selected):
        node_id = node_ids[number]
        if number in always or not fast_test_id(node_id):
            break
        measured = cost_model.test_ms(node_id) if cost_model is not None else None
        estimate = durations[number] if measured is None else measured
        if estimate > (750 if measured is None else 1_250):
            break
        cost += estimate + 8
        if cost > 2_000 or len(roster) >= 32:
            break
        roster.append(number)

    best = (0, 0, frozenset())
    for length in sorted({1, 2, 4, 8, 16, len(roster)}):
        if not 0 < length <= len(roster):
            continue
        removal = frozenset(roster[:length])
        savings = saving_for(selected, removal)
        cost = cost_model.estimate_ms(node_ids[n] for n in roster[:length]) if cost_model is not None else None
        if cost is None:
            # Mixed measured and unmeasured rosters must pay the original conservative estimate.
            if any(durations[n] > 750 for n in removal):
                continue
            cost = 400 + sum(durations[n] + 8 for n in removal)
            if cost > 1_750:
                continue
        benefit = savings - cost
        if benefit > best[0]:
            best = (benefit, savings, removal)
    return best


def main(*, replay=False, checkpoints=True, warm_probes=True, diagnostics=False) -> None:
    tests = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    try:
        impact = ImpactIndex(PINNED, COVERAGE, tests, ISOLATION, SUBPROCESS_PROFILE)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        print(f"coverage unavailable: {error}", file=sys.stderr)
        impact = None
    try:
        query_path = QUERY_PROFILES if replay or checkpoints else None
        verification = Verification(
            PINNED, json.loads(VERIFICATION.read_bytes()), query_path,
            CHECKPOINTS if checkpoints else None,
            fast=warm_probes, parser_path=PARSER_PROFILE if checkpoints else None,
        )
    except (OSError, ValueError):
        verification = None

    timed_tests = sorted(
        [(test["number"], math.floor(test["seconds"] * 1000 + 0.5)) for test in tests],
        key=itemgetter(1),
        reverse=True,
    )
    durations = dict(timed_tests)
    by_duration = {n: i for i, (n, _) in enumerate(timed_tests)}
    node_ids = {test["number"]: test["id"] for test in tests}
    numbers_by_id = {node: n for n, node in node_ids.items()}
    optimizer_test = numbers_by_id.get(OPTIMIZER_TEST)
    parser_test = numbers_by_id.get(PARSER_TEST)
    tpc_tests = {numbers_by_id[node] for node in TPC_TESTS if node in numbers_by_id}
    case_tests = {n for n, node in node_ids.items() if node.startswith(OPTIMIZER_FIXTURE_PREFIX)}
    schedules = {None: schedule(timed_tests)}
    full_cost = schedule_cost(schedules[None], durations)
    plans = []

    def workers_for(selected):
        if selected not in schedules:
            jobs = [timed_tests[by_duration[n]] for n in sorted(selected, key=by_duration.__getitem__)]
            schedules[selected] = schedule(jobs)
        return schedules[selected]

    def saving_for(selected, removal):
        before = schedule_cost(workers_for(selected), durations)
        after = schedule_cost(workers_for(selected - removal), durations)
        return before - after

    # Resolve structural fallbacks throughout the batch before spending the
    # remaining verification budget on candidates that already have a schedule.
    for patch in sorted((DATA / "commits").glob("*.patch")):
        selected = None
        files = []
        try:
            if impact is not None:
                files = parse_patch(patch.read_bytes().decode("utf-8"))
                selected = impact.select_files(files)
            if selected is None and verification is not None and files:
                selected = impact.state_impact(files, verification)
        except (OSError, ValueError, SyntaxError, RecursionError):
            pass
        plans.append((patch.stem, files, selected, False))

    fallback_index = None
    for index, (commit_id, files, selected, _) in enumerate(plans):
        if selected is not None or impact is None or not files:
            continue
        try:
            if fallback_index is None:
                fallback_index = FallbackIndex(impact, tests, STRUCTURAL_REFERENCES)
            scope = fallback_index.select(files)
            if scope is None:
                continue
            candidates, edits = scope
            candidates = frozenset(candidates)
            if edits is not None:
                if (verification is None
                        or schedule_cost(workers_for(candidates), durations) + COLLECTION_PROBE_COST_MS >= full_cost):
                    continue
                affected = verification.collection_impact(edits, tests)
                if affected is None:
                    continue
                candidates |= affected
            if schedule_cost(workers_for(candidates), durations) < full_cost:
                plans[index] = (commit_id, files, candidates, True)
        except (OSError, ValueError, SyntaxError, RecursionError, ImportError):
            pass

    verified_by_commit = {}
    if checkpoints and verification is not None:
        checks = []
        for commit_id, files, selected, scoped in plans:
            if not selected or scoped or len(files) != 1:
                continue
            path = files[0].old_path
            if not path or path != files[0].new_path:
                continue
            if path.startswith("sqlglot/") and path.endswith(".py"):
                targets = {parser_test}
                if path.startswith("sqlglot/optimizer/"):
                    targets.update(tpc_tests | case_tests)
            elif path.startswith("tests/fixtures/optimizer/") and path.endswith(".sql"):
                targets = case_tests
            else:
                continue
            for number in sorted((targets & selected) - impact.always):
                savings = saving_for(selected, {number})
                minimum = 1_800 if number in case_tests else 3_000
                if savings >= minimum:
                    checks.append((savings, commit_id, files, selected, number))

        # Verification has a shared CPU budget. Spend it where a pass can save the most.
        for _, commit_id, files, selected, number in sorted(checks, key=itemgetter(0), reverse=True):
            try:
                savings = saving_for(selected - verified_by_commit.get(commit_id, set()), {number})
                if number in case_tests:
                    if savings < 1_800:
                        continue
                    node = node_ids[number]
                    profile = verification.queries.get(node)
                    if not profile:
                        continue
                    path = files[0].old_path
                    edits = impact.patched_files(files, fixtures=True)
                    if path.startswith("tests/fixtures/"):
                        cases = fixture_cases(PINNED, edits, node) if impact.fixture_case_scope(path, node) else None
                    else:
                        cases = optimizer_cases(PINNED, edits, node, impact, profile)
                    if cases and verification.fixture_cases_pass(edits, node, cases, savings):
                        verified_by_commit.setdefault(commit_id, set()).add(number)
                    continue
                if savings < 3_000:
                    continue
                edits = impact.patched_files(files)
                if number == parser_test:
                    path = files[0].old_path
                    if (0 not in impact.startup.get(path, ())
                            and not impact.declaration_readers(path) & impact.test_bits.get(PARSER_TEST, 0)
                            and impact.warm_plan(edits, [PARSER_TEST])
                            and verification.parser_passes(edits, savings)):
                        verified_by_commit.setdefault(commit_id, set()).add(number)
                    continue
                if number == optimizer_test and OPTIMIZER_SOURCE in edits and impact.tpc_replay_scope(edits, OPTIMIZER_TEST):
                    before, _ = impact.original(OPTIMIZER_SOURCE)
                    if verification.optimizer_passes(before.encode("utf-8"), edits, savings):
                        verified_by_commit.setdefault(commit_id, set()).add(number)
                        continue
                node = node_ids[number]
                if impact.optimizer_query_scope(edits, node):
                    cases = impact.query_cases(edits, node, verification.queries.get(node))
                    if cases and verification.queries_pass(edits, node, cases, savings, optimizer_only=True):
                        verified_by_commit.setdefault(commit_id, set()).add(number)
            except (OSError, ValueError, SyntaxError, RecursionError, KeyError, TypeError):
                pass

    if warm_probes and verification is not None and verification.warm is not None:
        costs = load_replay_costs(REPLAY_COSTS)
        jobs = []
        for commit_id, files, selected, scoped in plans:
            if not selected or scoped or len(files) != 1:
                continue
            # Keep verified earlier candidates in order when checking later candidates.
            # Their state changes can still matter to a later test.
            benefit, savings, roster = warm_prefix(selected, impact.always, node_ids, durations, saving_for, costs)
            if benefit:
                jobs.append((benefit, savings, commit_id, files, roster))
        deadline = time.monotonic() + WARM_BATCH_SECONDS
        for _, savings, commit_id, files, roster in sorted(jobs, key=itemgetter(0), reverse=True):
            remaining = deadline - time.monotonic()
            if remaining <= 1.45:
                break
            try:
                edits = impact.patched_files(files, fixtures=True)
                ids = [node_ids[n] for n in sorted(roster)]
                plan = impact.warm_plan(edits, ids, strict=True)
                timeout = min(2.0, remaining - 1.3, savings / 2_000)
                if plan is not None and verification.warm_passes(edits, plan, ids, timeout):
                    verified_by_commit.setdefault(commit_id, set()).update(roster)
            except (OSError, ValueError, SyntaxError, RecursionError, KeyError, TypeError):
                pass

    fallback = selected_count = 0
    for commit_id, files, selected, scoped in plans:
        verified = frozenset(verified_by_commit.get(commit_id, ()))
        if verified:
            selected -= verified
        if replay and verification is not None and selected and not scoped:
            try:
                verified |= verify_selected(impact, files, selected, verification, durations, node_ids)
                selected -= verified
            except (OSError, ValueError, SyntaxError, RecursionError):
                pass
        fallback += selected is None
        selected_count += len(tests) if selected is None else len(selected)
        workers = workers_for(selected)
        if selected is not None and len(workers) > 1:
            workers = add_to_slack(workers, durations, impact.nearby(files) - verified)
        if diagnostics:
            busiest = max(workers, key=lambda group: sum(durations[n] for n in group), default=[])
            drivers = sorted(busiest, key=durations.__getitem__, reverse=True)[:3]
            print(json.dumps({"commit": commit_id, "schedule_ms": schedule_cost(workers, durations),
                              "fallback": selected is None, "verified": len(verified),
                              "busiest_tests": [[node_ids[n], durations[n]] for n in drivers]}), file=sys.stderr)
        print(json.dumps({"commit": commit_id, "workers": workers}, separators=(",", ":")))
    print(f"selected {selected_count} tests across all patches, {fallback} full-suite fallbacks", file=sys.stderr)
    if verification is not None:
        verification.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="run optional test and query probes")
    parser.add_argument("--no-checkpoints", action="store_true", help="skip targeted slow-test verification")
    parser.add_argument("--no-warm-probes", action="store_true", help="skip default cost-gated verification")
    parser.add_argument("--diagnostics", action="store_true", help="report per-patch schedule costs on stderr")
    args = parser.parse_args()
    main(replay=args.replay, checkpoints=not args.no_checkpoints,
         warm_probes=not args.no_warm_probes, diagnostics=args.diagnostics)
