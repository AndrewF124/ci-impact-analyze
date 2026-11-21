"""Fork verification children from an interpreter with pristine modules loaded."""

from __future__ import annotations

import importlib
import json
import select
import subprocess
import sys
from pathlib import Path
from types import CodeType, FunctionType

from semantic import module_name


def code_by_line(code):
    result = {}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            result[value.co_name, value.co_firstlineno] = value
            result.update(code_by_line(value))
    return result


def install_bodies(edits, plan):
    """Replace only plain functions whose loaded code matches the old binding."""
    replacements = []
    for path, locations in plan.items():
        name = module_name(path)
        module = sys.modules[name]
        code = compile(edits[path], str(Path.cwd() / path), "exec", dont_inherit=True)
        compiled = code_by_line(code)
        for function_name, old_line, new_line in locations:
            body = compiled.get((function_name, new_line))
            if body is None or "<locals>" in body.co_qualname:
                return False
            current = module
            for part in body.co_qualname.split("."):
                if not isinstance(current, type) and current is not module:
                    return False
                current = vars(current).get(part)
            if type(current) is not FunctionType or current.__module__ != name:
                return False
            old = current.__code__
            if (old.co_firstlineno != old_line or old.co_qualname != body.co_qualname
                    or old.co_flags != body.co_flags or old.co_freevars != body.co_freevars
                    or old.co_cellvars != body.co_cellvars):
                return False
            replacements.append((current, body))
    for function, body in replacements:
        function.__code__ = body
    return bool(replacements)


class WarmProcess:
    def __init__(self, pinned):
        self.pinned = pinned
        self.process = None

    def request(self, edits, plan, ids, timeout):
        if self.process is None:
            self.process = subprocess.Popen(
                [sys.executable, __file__, str(self.pinned)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        try:
            request = ({path: contents.decode("utf-8") for path, contents in edits.items()}, plan, ids, timeout)
            self.process.stdin.write(json.dumps(request).encode() + b"\n")
            self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], timeout + 1.3)
            if ready:
                return json.loads(self.process.stdout.readline())
        except (OSError, ValueError):
            pass
        self.close()
        return None

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        try:
            self.process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.process = None


def serve(pinned):
    from verify import Verification, pytest_passes

    sys.path.insert(0, str(pinned))
    import pytest  # noqa: F401
    import sqlglot  # noqa: F401

    verifier = Verification(pinned, {})
    for line in sys.stdin.buffer:
        try:
            edits, plan, ids, timeout = json.loads(line)
            for path in plan:
                importlib.import_module(module_name(path))
            edits = {path: contents.encode("utf-8") for path, contents in edits.items()}
            answer = verifier.run(edits, lambda: pytest_passes(ids) if install_bodies(edits, plan) else None, timeout)
        except (OSError, ValueError, ImportError):
            answer = None
        print(json.dumps(answer), flush=True)


if __name__ == "__main__":
    serve(Path(sys.argv[1]))
