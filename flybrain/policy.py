"""Fail-closed dependency checks and runtime network/process guard for fly-only.

This is a research-path guard against accidental fallback, not a hostile-code sandbox.
"""

import ast
import builtins
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import sys


LEGACY = frozenset(
    {
        "fly",
        "reward",
        "algorithm_model",
        "algorithm_phrases",
        "goal_model",
        "program_model",
        "pseudocode",
        "code_generator",
        "learning",
        "lab",
    }
)
FORBIDDEN = LEGACY | {
    "re",
    "regex",
    "urllib",
    "http",
    "socket",
    "requests",
    "httpx",
    "openai",
    "anthropic",
    "transformers",
    "torch",
    "importlib",
    "subprocess",
    "ctypes",
}
ACTIVE = ContextVar("codingfly_fly_only", default=False)
TEACHER_MODULES = frozenset(
    {
        "evaluation.code_data",
        "evaluation.curriculum_v2",
        "evaluation.nl_data",
        "evaluation.nl_data_v3",
        "models.train_code",
        "models.train_sequence",
        "models.train_nl",
        "models.train_nl_v3",
    }
)


class ForbiddenDependency(RuntimeError):
    pass


def verify_generation_tree(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    checked = []
    for directory in ("flybrain", "models"):
        for path in sorted((root / directory).glob("*.py")):
            # Guard implementation and CLI orchestration may inspect imports/process provenance.
            if path.name == "policy.py" or path.name.startswith("train_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = (
                    [item.name for item in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                if any(
                    name.split(".")[0] in FORBIDDEN or name in TEACHER_MODULES
                    for name in names
                ):
                    raise ForbiddenDependency(
                        f"Forbidden dependency in {path.name}:{node.lineno}"
                    )
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"eval", "exec", "__import__"}
                ):
                    raise ForbiddenDependency(
                        f"Dynamic execution/import in {path.name}:{node.lineno}"
                    )
            checked.append(str(path.relative_to(root)))
    return checked


def _audit(event, args):
    if ACTIVE.get() and (
        event.startswith("socket.")
        or event.startswith("subprocess.")
        or event in {"os.system", "os.posix_spawn", "urllib.Request"}
    ):
        raise ForbiddenDependency(f"fly-only forbids {event}")


sys.addaudithook(_audit)


@contextmanager
def fly_only():
    verify_generation_tree()
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        caller = (globals or {}).get("__name__", "")
        if ACTIVE.get() and (
            name.split(".")[0] in LEGACY
            or name in TEACHER_MODULES
            or (
                caller.startswith(("flybrain.", "models."))
                and name.split(".")[0] in FORBIDDEN
            )
        ):
            raise ForbiddenDependency(f"fly-only import denied: {name}")
        return original_import(name, globals, locals, fromlist, level)

    token = ACTIVE.set(True)
    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import
        ACTIVE.reset(token)
