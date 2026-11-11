import ast
import keyword
import re


MAX_SOURCE = 10_000
LOOP = re.compile(r"(\S+)\s*(?:\s+(?:in|IN)\s+|∈)\s*(.+)")
COUNTED_LOOP = re.compile(r"(\S+)\s*=\s*(.+?)\s*\.\.\s*(.+)")
COMMANDS = {
    "함수": "function", "FUNCTION": "function",
    "각": "each", "FOR": "each", "반복": "repeat",
    "만약": "if", "IF": "if", "아니면": "else", "ELSE": "else", "ELIF": "elif",
    "동안": "while", "WHILE": "while", "반환": "return", "RETURN": "return",
    "출력": "print", "PRINT": "print", "호출": "call", "CALL": "call",
    "대입": "assign", "SET": "assign", "누적": "accumulate",
}
SIMPLE = {"중단": "break", "BREAK": "break", "건너뛰기": "continue",
          "CONTINUE": "continue", "통과": "pass", "PASS": "pass"}


def identifier(text):
    if not text.isidentifier() or keyword.iskeyword(text):
        raise ValueError(f"변수 이름을 확인하세요. {text!r}")
    return text


def expression(text):
    text = text.strip()
    if not text:
        raise ValueError("값이나 조건을 입력하세요.")
    ast.parse(text, mode="eval")
    return text


def assignment(text, operator):
    arrow = "←" if "←" in text.partition("=")[0] else "="
    name, separator, value = text.partition(arrow)
    if not separator:
        raise ValueError("변수와 값을 ← 또는 =로 연결하세요.")
    return f"{identifier(name.strip())} {operator} {expression(value.strip())}"


def statement(line):
    if line.startswith("#") or line in SIMPLE:
        return SIMPLE.get(line, line)
    if line in ("아니면:", "ELSE:"):
        return "else:"
    name, *parts = line.split(maxsplit=1)
    command = COMMANDS.get(name)
    body = parts[0] if parts else ""
    if command == "function":
        header = f"def {body.removesuffix(':').rstrip()}:"
        tree = ast.parse(f"{header}\n    pass")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("함수 이름과 매개변수를 입력하세요.")
        return header
    if command in ("each", "repeat"):
        body = body.removesuffix(":").rstrip().removeprefix("EACH ")
        counted = COUNTED_LOOP.fullmatch(body) if command == "repeat" else None
        if counted:
            name, start, end = counted.groups()
            return f"for {identifier(name)} in range({expression(start)}, ({expression(end)}) + 1):"
        loop = LOOP.fullmatch(body)
        if not loop:
            raise ValueError("‘각 항목 ∈ 목록’ 또는 ‘반복 i = 시작 .. 끝’ 형식으로 입력하세요.")
        name, values = loop.groups()
        return f"for {identifier(name)} in {expression(values)}:"
    if command == "else":
        if body in ("", ":"):
            return "else:"
        prefix, *rest = body.split(maxsplit=1)
        if prefix not in ("만약", "IF") or not rest:
            raise ValueError("‘아니면’ 또는 ‘아니면 만약 조건’ 형식으로 입력하세요.")
        command, body = "elif", rest[0]
    if command in ("if", "elif", "while"):
        condition = body.removesuffix(":").rstrip()
        condition = condition.removesuffix(" THEN" if command != "while" else " DO")
        return f"{command} {expression(condition)}:"
    if command == "return":
        return f"return {expression(body)}" if body else "return"
    if command == "print":
        return f"print({expression(body)})" if body else "print()"
    if command == "call":
        if not body or not isinstance(ast.parse(body, mode="eval").body, ast.Call):
            raise ValueError("‘호출 함수(인자)’ 형식으로 입력하세요.")
        return body
    if command in ("assign", "accumulate"):
        return assignment(body, "+=" if command == "accumulate" else "=")
    if "←" in line:
        return assignment(line, "=")
    raise ValueError(f"지원하는 명령을 확인하세요. {name!r}")


def translate(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("의사코드를 입력하세요.")
    if len(source) > MAX_SOURCE:
        raise ValueError(f"의사코드는 {MAX_SOURCE:,}자까지 입력할 수 있습니다.")
    output = []
    for number, line in enumerate(source.splitlines(), 1):
        text = line.lstrip(" \t").rstrip()
        indent = line[:len(line) - len(line.lstrip(" \t"))]
        if not text:
            output.append("")
            continue
        try:
            if "\t" in indent:
                raise ValueError("들여쓰기에 공백을 사용하세요.")
            output.append(indent + statement(text))
        except SyntaxError as error:
            raise ValueError(f"{number}줄 · {error.msg}") from error
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{number}줄 · {error}") from error

    code = "\n".join(output) + "\n"
    try:
        compile(code, "<pseudocode>", "exec")
    except SyntaxError as error:
        raise ValueError(f"{error.lineno}줄 · {error.msg}") from error
    except (UnicodeError, RecursionError) as error:
        raise ValueError("코드에 처리할 수 없는 문자나 너무 깊게 중첩된 식이 있습니다.") from error
    return code
