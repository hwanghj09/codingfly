"""Only executed inside the bubblewrap namespace. Never exec generated code in a server."""

import ast
import contextlib
import io
import json
import os
import resource
import socket
import sys


ALLOWED = {"Module", "Assign", "AugAssign", "Expr", "Name", "Load", "Store", "Constant",
           "Call", "FunctionDef", "arguments", "arg", "Return", "BinOp", "UnaryOp", "BoolOp",
           "Compare", "If", "For", "While", "Break", "Continue", "Pass", "List", "Tuple",
           "Dict", "Set", "Subscript", "Slice", "Add", "Sub", "Mult", "Div", "FloorDiv",
           "Mod", "USub", "UAdd", "Not", "And", "Or", "Eq", "NotEq", "Lt", "LtE", "Gt",
           "GtE", "In", "NotIn", "keyword", "IfExp"}
BUILTINS = {"print": print, "range": range, "abs": abs, "len": len, "sum": sum, "min": min,
            "max": max, "sorted": sorted, "enumerate": enumerate, "bool": bool, "int": int,
            "str": str, "list": list}


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    request = json.loads(sys.stdin.read(65537))
    if request.get("probe"):
        # Fixed infrastructure diagnostics, separate from the AST-gated user path.
        with socket.socket() as probe:
            probe.settimeout(.1)
            try:
                probe.connect(("1.1.1.1", 53))
                network_blocked = False
            except OSError:
                network_blocked = True
        try:
            with open("/usr/.codingfly-write-probe", "w"):
                pass
            runtime_readonly = False
        except OSError as error:
            runtime_readonly = error.errno == 30  # EROFS, not merely filesystem permissions.
        print(json.dumps({"status": "ok", "network_blocked": network_blocked,
                          "runtime_readonly": runtime_readonly,
                          "host_workspace_hidden": not os.path.exists("/workspace"),
                          "host_home_hidden": not os.path.exists("/home"),
                          "environment_cleared": not any(name in os.environ for name in
                                                         ("HTTP_PROXY", "HTTPS_PROXY", "HOME", "PATH")),
                          "memory_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                          "cpu_limit_seconds": resource.getrlimit(resource.RLIMIT_CPU)[0],
                          "process_limit": resource.getrlimit(resource.RLIMIT_NPROC)[0]}))
        return
    code = request["code"]
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 12000:
        print(json.dumps({"status": "syntax_error", "error": "Empty or oversized code"}))
        return
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as error:
        print(json.dumps({"status": "syntax_error", "error": str(error)}))
        return
    for node in ast.walk(tree):
        if (type(node).__name__ not in ALLOWED
                or isinstance(node, ast.Name) and node.id.startswith("__")
                or isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name.startswith("__"))):
            print(json.dumps({"status": "policy_error", "error": type(node).__name__}))
            return
    environment = {"__builtins__": BUILTINS}
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            exec(compile(tree, "<generated>", "exec"), environment)
            returns = []
            for call in request.get("calls", []):
                returns.append(environment[call["name"]](*call.get("args", [])))
        # Only explicitly requested simple values cross the process boundary.
        variables = {name: environment.get(name) for name in request.get("variables", [])}
        result = {"status": "ok", "stdout": captured.getvalue(), "variables": variables,
                  "returns": returns}
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "runtime_error", "error": type(error).__name__ + ": " + str(error)}))


if __name__ == "__main__":
    main()
