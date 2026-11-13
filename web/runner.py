"""Restricted execution inside a disposable browser worker, separate from model state."""

import ast
import contextlib
import io
import json
import sys


ALLOWED = {
    "Module",
    "Assign",
    "AugAssign",
    "Expr",
    "Name",
    "Load",
    "Store",
    "Constant",
    "Call",
    "FunctionDef",
    "arguments",
    "arg",
    "Return",
    "BinOp",
    "UnaryOp",
    "BoolOp",
    "Compare",
    "If",
    "For",
    "While",
    "Break",
    "Continue",
    "Pass",
    "List",
    "Tuple",
    "Dict",
    "Set",
    "Subscript",
    "Slice",
    "Add",
    "Sub",
    "Mult",
    "Div",
    "FloorDiv",
    "Mod",
    "USub",
    "UAdd",
    "Not",
    "And",
    "Or",
    "Eq",
    "NotEq",
    "Lt",
    "LtE",
    "Gt",
    "GtE",
    "In",
    "NotIn",
    "keyword",
    "IfExp",
}


class Output(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 16000:
            raise ValueError("출력 한도 16,000자를 초과했습니다.")
        return super().write(text)


def bounded_range(*args):
    result = range(*args)
    if len(result) > 10000:
        raise ValueError("브라우저에서는 반복 범위를 10,000개 이내로 입력해 주세요.")
    return result


def execute(request):
    code = request.get("code")
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 12000:
        raise ValueError("실행할 코드가 없거나 너무 깁니다.")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if (
            type(node).__name__ not in ALLOWED
            or isinstance(node, ast.Name)
            and node.id.startswith("__")
            or isinstance(node, ast.FunctionDef)
            and (node.decorator_list or node.name.startswith("__"))
        ):
            raise ValueError(
                f"브라우저 실행에서 허용하지 않는 구문: {type(node).__name__}"
            )
    args = request.get("args", [])
    if not isinstance(args, list) or len(args) > 8 or len(json.dumps(args)) > 2048:
        raise ValueError("함수 입력은 2,048자 이내의 JSON 배열로 입력해 주세요.")
    environment = {
        "__builtins__": {
            "print": print,
            "range": bounded_range,
            "abs": abs,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "enumerate": enumerate,
            "bool": bool,
            "int": int,
            "str": str,
            "list": list,
        }
    }
    output, steps, value = Output(), 0, None

    def budget(frame, event, arg):
        nonlocal steps
        steps += 1
        if steps > 100000:
            raise TimeoutError("실행 단계 한도를 초과했습니다.")
        return budget

    previous = sys.gettrace()
    try:
        sys.settrace(budget)
        with contextlib.redirect_stdout(output):
            exec(compile(tree, "<browser-generated>", "exec"), environment)
            name = request.get("function_name")
            functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            if name:
                if name not in functions:
                    raise ValueError("생성된 함수 중에서 선택해 주세요.")
                value = environment[name](*args)
    finally:
        sys.settrace(previous)
    variables = {
        name: item
        for name, item in environment.items()
        if not name.startswith("__") and not callable(item)
    }
    result = {
        "stdout": output.getvalue(),
        "variables": variables,
        "return_value": value,
        "called": bool(request.get("function_name")),
    }
    text = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(text) > 20000:
        raise ValueError("실행 결과가 너무 큽니다.")
    return text
