"""Measure import dependencies for small, reproducible Python child tests."""

import ast
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path


def environment_check(node):
    """Extract a literal Python command from a simple subprocess smoke test."""
    if not isinstance(node, ast.FunctionDef) or node.decorator_list or len(node.body) != 4:
        return None
    statement, code, result, assertion = node.body
    if not isinstance(statement, ast.Import) or [(n.name, n.asname) for n in statement.names] != [("subprocess", None)]:
        return None
    if not all(isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
               and isinstance(stmt.targets[0], ast.Name) for stmt in (code, result)):
        return None
    if not isinstance(code.value, ast.Constant) or type(code.value.value) is not str:
        return None
    variable = code.targets[0].id
    call = result.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and ast.dump(call.func) == ast.dump(ast.parse("subprocess.run", mode="eval").body)
            and len(call.args) == 1 and isinstance(call.args[0], ast.List)
            and ast.dump(call.args[0]) == ast.dump(ast.parse(f'["python", "-c", {variable}]', mode="eval").body)
            and len(call.keywords) == 2 and {kw.arg for kw in call.keywords} == {"text", "capture_output"}
            and all(isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in call.keywords)):
        return None
    if not isinstance(assertion, ast.Assert) or assertion.msg is not None:
        return None
    test = assertion.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.In)
            and isinstance(test.left, ast.Constant) and type(test.left.value) is str
            and ast.dump(test.comparators[0]) == ast.dump(ast.parse(f"{result.targets[0].id}.stdout", mode="eval").body)):
        return None
    program = ast.parse(code.value.value)
    if not (len(program.body) == 3 and isinstance(program.body[0], ast.Import)
            and [(n.name, n.asname) for n in program.body[0].names] == [("sqlglot", None)]
            and isinstance(program.body[1], ast.Assert) and isinstance(program.body[2], ast.Expr)):
        return None
    for node in ast.walk(program):
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in {"len", "print"}):
            return None
    return code.value.value, test.left.value


def observe(program, output):
    root = str(Path.cwd()) + os.sep
    lines = {}
    imported = set()

    def trace(frame, event, arg):
        name = frame.f_code.co_filename
        if not name.startswith(root + "sqlglot/"):
            return None
        path = name[len(root):]
        if frame.f_code.co_name == "<module>":
            imported.add(path)
        elif event == "call":
            lines.setdefault(path, set()).add(frame.f_code.co_firstlineno)
        elif event == "line":
            lines.setdefault(path, set()).add(frame.f_lineno)
        return trace

    def audit(event, args):
        if event != "open" or not isinstance(args[0], str):
            return
        path = os.path.abspath(args[0])
        if path.startswith(root) and not path.endswith(".pyc"):
            caller = sys._getframe(1).f_code
            if caller.co_filename == "<frozen importlib._bootstrap_external>" and caller.co_name == "get_data":
                return
            lines.setdefault(path[len(root):], set()).add(0)

    sys.addaudithook(audit)
    sys.settrace(trace)
    try:
        exec(compile(program, "<string>", "exec"), {"__name__": "__main__"})
    finally:
        sys.settrace(None)
    Path(output).write_text(json.dumps({
        "imported": sorted(imported),
        "lines": {path: sorted(entries) for path, entries in lines.items()},
    }), encoding="utf-8")


def build(coverage, output):
    with gzip.open(coverage, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    result = {}
    scratch = Path(output).with_suffix(".trace")
    for node_id in raw["unreliable"]:
        parts = node_id.split("::")
        if len(parts) != 3 or not parts[0].endswith(".py"):
            continue
        file, owner, method = parts
        tree = ast.parse(Path(file).read_bytes())
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner), None)
        if cls is None:
            continue
        entry = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method), None)
        extracted = environment_check(entry)
        if extracted is None:
            continue
        program, expected = extracted
        try:
            baseline = subprocess.run([sys.executable, "-c", program], capture_output=True, timeout=10, check=True)
            traced = subprocess.run(
                [sys.executable, __file__, "--trace", program, str(scratch)],
                capture_output=True, timeout=10, check=True,
            )
            actual = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node_id],
                capture_output=True, timeout=20, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if baseline.stdout == traced.stdout and expected.encode() in baseline.stdout and b"1 passed" in actual.stdout:
            profile = json.loads(scratch.read_bytes())
            if profile["imported"]:
                result[node_id] = profile
        scratch.unlink(missing_ok=True)
    Path(output).write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"Profiled {len(result)} separate Python test processes", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--trace":
        observe(sys.argv[2], sys.argv[3])
    else:
        build(sys.argv[1], sys.argv[2])
