"""Measure and verify predicate-pushdown edits in the pinned TPC-DS test."""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import io
import json
import pickle
import time
from enum import Enum
from functools import cache
from pathlib import Path
from unittest.mock import patch

from semantic import ORDER_OPERATORS, SWAPS, Shape, literal, site_key

SOURCE = "sqlglot/optimizer/pushdown_predicates.py"
RULE = "pushdown_predicates"
TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
OPERATORS = {kind.__name__: function for kind, function in ORDER_OPERATORS.items()}
ALTERNATIVES = {kind.__name__: SWAPS[kind].__name__ for kind in ORDER_OPERATORS}
REDUCERS = {"min": min, "max": max}
PROFILE_VERSION = 1


def rule_sites(tree):
    rule = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == RULE), None)
    if rule is None:
        return []
    sites = []
    for node in ast.walk(rule):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in ORDER_OPERATORS
                and all(isinstance(term, ast.Name) for term in (node.left, node.comparators[0]))):
            sites.append((node, node.ops[0], type(node.ops[0])))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in REDUCERS
              and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp)
              and len(node.keywords) == 1 and node.keywords[0].arg == "default"):
            default = node.keywords[0].value
            value = literal(default)
            if type(value) is int:
                sites.append((node, default, value))
    return sites


def changed_site(before, after):
    """Recognize a single supported value edit in an otherwise equal module."""
    old, new = (ast.parse(data, type_comments=True) for data in (before, after))
    left, right = rule_sites(old), rule_sites(new)
    if len(left) != len(right):
        return None
    differences = [(a, b) for a, b in zip(left, right) if a[2] != b[2]]
    if len(differences) != 1:
        return None
    (node, old_value, value), (replacement, new_value, changed) = differences[0]
    if type(node) is not type(replacement):
        return None
    if isinstance(node, ast.Compare) and SWAPS[value] is not changed:
        return None
    if ast.dump(Shape([old_value]).visit(old)) != ast.dump(Shape([new_value]).visit(new)):
        return None
    return site_key(node)


@cache
def expression_slots(cls):
    names = []
    for base in cls.__mro__:
        slots = vars(base).get("__slots__", ())
        names.extend((slots,) if isinstance(slots, str) else slots)
    return tuple(dict.fromkeys(name for name in names if name not in {"__dict__", "__weakref__"}))


class ExpressionPickler(pickle.Pickler):
    """Keep the expression graph, including parent links, unset args and aliases."""

    def __init__(self, stream, expression_type):
        super().__init__(stream, protocol=5)
        self.expression_type = expression_type
        self.hashes = []

    def reducer_override(self, value):
        if not isinstance(value, self.expression_type):
            return NotImplemented
        slots = {name: getattr(value, name) for name in expression_slots(type(value)) if hasattr(value, name)}
        # Cached hashes depend on the process's randomized hash seed.
        if "_hash" in slots:
            self.hashes.append((value, slots["_hash"]))
            slots["_hash"] = None
        return type(value), (), (getattr(value, "__dict__", None), slots)


def check_hashes(hashes):
    if any(expected is not None for _, expected in hashes):
        # Reject stale cached hashes instead of treating them as derived state.
        for node, _ in hashes:
            node._hash = None
        try:
            if any(expected is not None and hash(node) != expected for node, expected in hashes):
                raise ValueError("optimizer checkpoint contains a stale expression hash")
        finally:
            for node, expected in hashes:
                node._hash = expected


def snapshot(expression, schema):
    from sqlglot import exp

    stream = io.BytesIO()
    writer = ExpressionPickler(stream, exp.Expr)
    writer.dump((expression, schema))
    check_hashes(writer.hashes)
    return stream.getvalue()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(expression, schema):
    """Compare graph state independently of pickle memoization of strings."""
    from sqlglot import Dialect, exp
    from sqlglot.schema import MappingSchema

    seen = {}
    hashes = []

    def encode(value):
        kind = type(value)
        if kind in (str, int, bool, type(None), float, bytes):
            return (kind.__name__, repr(value)) if kind in (float, bytes) else value
        label = f"{kind.__module__}.{kind.__qualname__}"
        if isinstance(value, Enum):
            return label, value.name
        key = id(value)
        if key in seen:
            return "@", seen[key]
        seen[key] = len(seen)
        if kind is dict:
            return "dict", [(encode(k), encode(v)) for k, v in value.items()]
        if kind in (list, tuple):
            return kind.__name__, [encode(item) for item in value]
        if isinstance(value, exp.Expr):
            hashes.append((value, value._hash))
            slots = [(name, encode(getattr(value, name))) for name in expression_slots(kind)
                     if name != "_hash" and hasattr(value, name)]
            return label, slots, encode(getattr(value, "__dict__", None))
        if isinstance(value, (MappingSchema, Dialect)):
            return label, encode(vars(value))
        raise ValueError(f"unsupported optimizer checkpoint value {label}")

    graph = encode((expression, schema))
    check_hashes(hashes)
    return digest(json.dumps(graph, ensure_ascii=True, separators=(",", ":")).encode())


class Recorder:
    def __init__(self):
        from sqlglot.optimizer.optimizer import RULES, optimize
        from sqlglot.optimizer.pushdown_predicates import pushdown_predicates

        self.rule = pushdown_predicates
        self.original_optimize = optimize
        self.rules = RULES
        self.index = RULES.index(pushdown_predicates)
        self.cases = {}
        self.expected = set()
        self.recorded = {}
        self.overhead = {}
        self.affected = {}
        self.instrumented_rule = self.instrument(Path(SOURCE).read_bytes())

    def instrument(self, source):
        tree = ast.parse(source)
        sites = {node: value for node, _, value in rule_sites(tree)}
        self.affected = {site_key(node): 0 for node in sites}

        class Instrument(ast.NodeTransformer):
            def visit_Compare(self, node):
                if node not in sites:
                    return self.generic_visit(node)
                observed = ast.Call(ast.Name("_blast_comparison", ast.Load()),
                                    [ast.Constant(site_key(node)), ast.Constant(sites[node].__name__),
                                     node.left, node.comparators[0]], [])
                return ast.copy_location(observed, node)

            def visit_Call(self, node):
                if node not in sites:
                    return self.generic_visit(node)
                observed = ast.Call(ast.Name("_blast_default", ast.Load()),
                                    [ast.Constant(site_key(node)), ast.Constant(node.func.id),
                                     node.func, self.visit(node.args[0]), node.keywords[0].value], [])
                return ast.copy_location(observed, node)

        namespace = {"_blast_comparison": self.comparison, "_blast_default": self.default,
                     "__name__": self.rule.__module__}
        exec(compile(ast.fix_missing_locations(Instrument().visit(tree)), SOURCE, "exec", dont_inherit=True),
             namespace)
        return namespace[RULE]

    def comparison(self, key, name, a, b):
        result = OPERATORS[name](a, b)
        if type(a) is not int or type(b) is not int or result != OPERATORS[ALTERNATIVES[name]](a, b):
            self.affected[key] |= 1 << self.case
        return result

    def default(self, key, kind, reducer, values, default):
        if reducer is not REDUCERS[kind]:
            self.affected[key] |= 1 << self.case
            return reducer(values, default=default)
        missing = object()
        result = reducer(values, default=missing)
        if result is missing:
            self.affected[key] |= 1 << self.case
            return default
        return result

    def before_case(self, index, options):
        self.case = index
        if options.get("execute"):
            self.expected.add(index)

    def checkpoint(self, expression, dialect=None):
        if dialect is not None or self.case in self.recorded:
            raise ValueError("unsupported TPC-DS optimizer invocation")
        start = time.process_time()
        packed = snapshot(expression, self.schema)
        recovered, recovered_schema = pickle.loads(packed)
        self.overhead[self.case] = time.process_time() - start

        started_rule = time.process_time()
        result = self.rule(expression, dialect=dialect)
        rule_time = time.process_time() - started_rule

        start = time.process_time()
        after = fingerprint(result, self.schema)
        checked = self.instrumented_rule(recovered, dialect=dialect)
        if fingerprint(checked, recovered_schema) != after:
            raise ValueError(f"optimizer checkpoint tracing changed query {self.case}")
        self.overhead[self.case] += time.process_time() - start
        self.recorded[self.case] = [base64.b64encode(packed).decode(), after, None, round(rule_time * 1000)]
        return result

    def optimize(self, expression, schema, **options):
        if options != {"leave_tables_isolated": True, "dialect": None}:
            raise ValueError("TPC-DS optimizer flags differ from the pinned test")
        self.schema = schema
        rules = (*self.rules[:self.index], self.checkpoint, *self.rules[self.index + 1:])
        value = self.original_optimize(expression, schema, rules=rules, **options)
        start = time.process_time()
        self.recorded[self.case][2] = fingerprint(value, schema)
        self.overhead[self.case] += time.process_time() - start
        return value


def build_profile(output):
    from sqlglot import executor
    from query_replay import run_executor_cases

    recorder = Recorder()
    with patch.object(executor, "optimize", recorder.optimize):
        times = run_executor_cases("tpc-ds", before_case=recorder.before_case)
    if recorder.recorded.keys() != recorder.expected:
        raise ValueError("TPC-DS checkpoint coverage is incomplete")
    profile = {
        "version": PROFILE_VERSION,
        "source": digest(Path(SOURCE).read_bytes()),
        "cases": {index: [*state, max(1, times[index] - round(1000 * recorder.overhead[index]))]
                  for index, state in recorder.recorded.items()},
        "affected": {name: format(bits, "x") for name, bits in recorder.affected.items()},
    }
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(profile, stream, separators=(",", ":"))
    print(f"Recorded {len(profile['cases'])} optimizer checkpoints and {len(profile['affected'])} value sites",
          flush=True)


def cases_for(profile, site):
    if profile.get("version") != PROFILE_VERSION or site not in profile.get("affected", {}):
        return None
    mask = int(profile["affected"][site], 16)
    indices = [index for index in map(int, profile["cases"]) if mask & (1 << index)]
    return indices if sum(1 << index for index in indices) == mask else None


def replay(profile, indices, budget_ms):
    start = time.process_time()
    from sqlglot.optimizer.optimizer import RULES, optimize
    from sqlglot.optimizer.pushdown_predicates import pushdown_predicates
    from query_replay import run_executor_cases

    pivot = RULES.index(pushdown_predicates)
    remaining = []
    for index in indices:
        if (time.process_time() - start) * 1000 >= budget_ms:
            return False
        payload, expected_rule, expected_final, _, _ = profile["cases"][str(index)]
        expression, schema = pickle.loads(base64.b64decode(payload))
        result = pushdown_predicates(expression)
        if fingerprint(result, schema) == expected_rule:
            continue
        final = optimize(result, schema, copy=False, rules=RULES[pivot + 1:], leave_tables_isolated=True)
        if fingerprint(final, schema) != expected_final:
            remaining.append(index)
    if not remaining:
        return True
    # Replay the original test assertions only for queries that still differ.
    estimate = 1_200 + sum(profile["cases"][str(index)][4] for index in remaining)
    if (time.process_time() - start) * 1000 + estimate > budget_ms:
        return False
    return bool(run_executor_cases("tpc-ds", indices=remaining))


if __name__ == "__main__":
    import sys

    output = Path(sys.argv[1])
    try:
        build_profile(output)
    except (OSError, ValueError, ImportError, RecursionError, pickle.PickleError, AssertionError) as error:
        print(f"Optimizer checkpoints unavailable: {error}", file=sys.stderr)
        with gzip.open(output, "wt", encoding="utf-8") as stream:
            json.dump({}, stream)
