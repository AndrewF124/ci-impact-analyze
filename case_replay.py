"""Select independent cases of a pinned optimizer fixture for verification."""

from __future__ import annotations

import ast
import hashlib
import io
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath

from query_replay import run_optimizer_cases
from semantic import integer_guard, integer_guard_edit


TEST_FILE = "tests/test_optimizer.py"
HELPERS = "tests/helpers.py"
FIXTURES = "tests/fixtures/"
PREFIX = f"{TEST_FILE}::TestOptimizer::"
MAX_CASES = 32
# Only use the replay properties of these exact, pinned test and loader sources.
SUITE_HASHES = {
    TEST_FILE: "35c02d71cacfa5d8e24319ef245963da3c1151b891281bac1864a00fa183cbf7",
    HELPERS: "c0dc285fccd1b8bee4a451f5962118fe771de005843b56e3af4c7136f74c3ed3",
}


@dataclass(frozen=True)
class CaseReplay:
    fixture: str
    indices: tuple[int, ...]
    fixture_digest: str
    case_count: int


def digest(data):
    return hashlib.sha256(data).hexdigest()


@cache
def pinned_contract(root):
    try:
        sources = {path: (root / path).read_bytes() for path in SUITE_HASHES}
    except OSError:
        return {}, None
    if any(digest(source) != SUITE_HASHES[path] for path, source in sources.items()):
        return {}, None

    tree = ast.parse(sources[TEST_FILE])
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TestOptimizer")
    methods = {}
    for method in owner.body:
        if not isinstance(method, ast.FunctionDef) or method.decorator_list or len(method.body) != 1:
            continue
        statement = method.body[0]
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
                or call.func.attr != "check_file" or not isinstance(call.func.value, ast.Name)
                or call.func.value.id != "self" or len(call.args) != 2):
            continue
        fixture = call.args[0]
        keywords = {kw.arg: kw.value for kw in call.keywords}
        if (not isinstance(fixture, ast.Constant) or type(fixture.value) is not str
                or None in keywords or "only" in keywords):
            continue
        execute = keywords.get("execute", ast.Constant(None))
        if not isinstance(execute, ast.Constant) or execute.value not in (False, None):
            continue
        name = PurePosixPath(fixture.value)
        if name.is_absolute() or ".." in name.parts or "\\" in fixture.value:
            continue
        methods[PREFIX + method.name] = (f"optimizer/{name}.sql", execute.value is False)

    tree = ast.parse(sources[HELPERS])
    selected = {"_filter_comments", "_extract_meta", "load_sql_fixture_pairs"}
    loader = ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in selected], type_ignores=[])
    return methods, compile(loader, HELPERS, "exec", dont_inherit=True)


def loaded_cases(loader, data, *, disables_execution):
    # Use the actual pinned loader and metadata parser on in-memory bytes.
    environment = {"os": os, "FIXTURES_DIR": "",
                   "open": lambda *_args, **_kwargs: io.StringIO(data.decode("utf-8"), newline=None)}
    exec(loader, environment)
    cases = list(environment["load_sql_fixture_pairs"](""))
    if not disables_execution and any(meta.get("execute", "").lower() in {"1", "true"} for meta, _, _ in cases):
        return None
    return cases


def fixture_cases(pinned, edits, node_id):
    """Map a fixture-only patch to changed pairs without moving positions."""
    if len(edits) != 1 or not node_id.startswith(PREFIX):
        return None
    methods, loader = pinned_contract(pinned)
    contract = methods.get(node_id)
    if contract is None:
        return None
    fixture, disables_execution = contract
    path = FIXTURES + fixture
    if path not in edits:
        return None
    try:
        old = loaded_cases(loader, (pinned / path).read_bytes(), disables_execution=disables_execution)
        new = loaded_cases(loader, edits[path], disables_execution=disables_execution)
    except (OSError, ValueError, IndexError):
        return None
    if old is None or new is None or len(old) != len(new):
        return None
    indices = tuple(i for i, (before, after) in enumerate(zip(old, new)) if before != after)
    if not 0 < len(indices) <= MAX_CASES:
        return None
    return CaseReplay(fixture, indices, digest(edits[path]), len(new))


def pure_guard(before, after, impact, path, node_id):
    """Require the edited expression to select between effect-free local returns."""
    from impact import changed_callables

    guard = integer_guard_edit(before, after)
    if guard is None:
        return False
    functions = changed_callables(before, after)
    if len(functions) != 1:
        return False
    function, _ = functions[0]
    if not isinstance(function, ast.FunctionDef):
        return False
    body = function.body[1:] if ast.get_docstring(function, clean=False) is not None else function.body
    if len(body) != 2 or not isinstance(body[0], ast.If) or len(body[0].body) != 1 or body[0].orelse:
        return False
    guard_node = body[0].test
    if (parts := integer_guard(guard_node)) is None:
        return False
    local = {arg.arg for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
    if parts[0] not in local or any(
        not isinstance(stmt, ast.Return)
        or not (isinstance(stmt.value, ast.Constant)
                or isinstance(stmt.value, ast.Name) and stmt.value.id in local)
        for stmt in (body[0].body[0], body[1])
    ):
        return False
    bit = impact.test_bits.get(node_id, 0)
    site = guard[0]
    uncertain, setup = impact.observations("integer_guard_unknown", f"{path}:{site}")
    incomplete, _ = impact.observations("incomplete", "*")
    observed = impact.integer_guards.get(path, {}).get(site, {})
    return not setup and not (uncertain | incomplete) & bit and any(int(mask, 16) & bit for mask in observed)


def optimizer_cases(pinned, edits, node_id, impact, profile):
    """Map a pure optimizer guard edit to positively traced fixture cases."""
    if len(edits) != 1 or not profile or not node_id.startswith(PREFIX):
        return None
    methods, loader = pinned_contract(pinned)
    contract = methods.get(node_id)
    if contract is None:
        return None
    fixture, disables_execution = contract
    if profile.get("fixture") != fixture:
        return None
    path, patched = next(iter(edits.items()))
    if not path.startswith("sqlglot/optimizer/") or not impact.warm_plan(edits, [node_id]):
        return None
    _, before = impact.original(path)
    after = ast.parse(patched, type_comments=True)
    if not pure_guard(before, after, impact, path, node_id):
        return None
    indices = impact.query_cases(edits, node_id, profile)
    if not indices or len(indices) > MAX_CASES:
        return None
    data = (pinned / (FIXTURES + fixture)).read_bytes()
    cases = loaded_cases(loader, data, disables_execution=disables_execution)
    if cases is None or len(cases) != len(profile["times"]):
        return None
    return CaseReplay(fixture, tuple(indices), digest(data), len(cases))


def run_verified(node_id, replay):
    """Run the original fixture assertions against the current overlay."""
    root = Path.cwd()
    methods, _ = pinned_contract(root)
    contract = methods.get(node_id)
    if contract is None or contract[0] != replay.fixture:
        return False
    try:
        if digest((root / (FIXTURES + replay.fixture)).read_bytes()) != replay.fixture_digest:
            return False
        times = run_optimizer_cases(node_id.removeprefix(PREFIX), replay.fixture, replay.indices)
        return len(times) == replay.case_count
    except (AssertionError, OSError, ValueError, IndexError):
        return False
