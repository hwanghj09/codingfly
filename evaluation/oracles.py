"""Execution grading with explicit observations, no code generation or correction."""

import ast

from evaluation.sandbox import run_python


def grade_code(code, expected):
    try:
        if not code.strip():
            raise SyntaxError("Empty program")
        ast.parse(code)
        compile(code, "<generated>", "exec")
    except (SyntaxError, ValueError, RecursionError) as error:
        return {"syntax_ok": False, "semantic_ok": False, "status": "syntax_error", "error": str(error)}
    result = run_python(code, variables=expected.get("variables", {}).keys(), calls=expected.get("calls", []))
    passed = (result["status"] == "ok" and result.get("stdout") == expected.get("stdout", "")
              and result.get("variables") == expected.get("variables", {})
              and result.get("returns") == expected.get("returns", []))
    return {"syntax_ok": True, "semantic_ok": passed, **result}
