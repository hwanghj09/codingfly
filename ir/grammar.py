"""Bounded lexical/structural validation, without inferring or repairing meaning."""

from dataclasses import dataclass
import json
import keyword

from ir.tokens import BINARY, COMMANDS, UNARY


class InvalidIR(ValueError):
    pass


@dataclass(frozen=True)
class Statement:
    command: str
    tokens: tuple[str, ...]
    line: int


def lex(line):
    result = []
    position = 0
    while position < len(line):
        char = line[position]
        if char.isspace():
            position += 1
            continue
        if char in "()":
            result.append(char)
            position += 1
        elif char == '"':
            try:
                value, end = json.JSONDecoder().raw_decode(line[position:])
            except ValueError as error:
                raise InvalidIR("Invalid JSON string") from error
            if not isinstance(value, str):
                raise InvalidIR("Expected string")
            result.append(line[position:position + end])
            position += end
        else:
            end = position + 1
            while end < len(line) and not line[end].isspace() and line[end] not in '()"':
                end += 1
            result.append(line[position:end])
            position = end
    return result


def identifier(token):
    if (not token.isidentifier() or keyword.iskeyword(token)
            or token in COMMANDS | BINARY | UNARY | {"RANGE", "LIST", "TRUE", "FALSE"}):
        raise InvalidIR(f"Invalid identifier: {token}")


def expression(tokens, position=0, depth=0):
    if position >= len(tokens) or depth > 32:
        raise InvalidIR("Missing expression or excessive expression depth")
    token = tokens[position]
    if token != "(":
        if token in {"TRUE", "FALSE"} or token.startswith('"'):
            return position + 1
        if token.isascii() and token.removeprefix("-").isdecimal():
            if len(token) > 64:
                raise InvalidIR("Integer is too large")
            return position + 1
        identifier(token)
        return position + 1
    if position + 1 >= len(tokens):
        raise InvalidIR("Missing expression operator")
    operator = tokens[position + 1]
    position += 2
    if operator in BINARY | UNARY:
        for _ in range(2 if operator in BINARY else 1):
            position = expression(tokens, position, depth + 1)
    elif operator in {"LIST", "CALL"}:
        if operator == "CALL":
            if position >= len(tokens):
                raise InvalidIR("Missing callee")
            identifier(tokens[position])
            position += 1
        while position < len(tokens) and tokens[position] != ")":
            position = expression(tokens, position, depth + 1)
    else:
        raise InvalidIR(f"Unknown expression operator: {operator}")
    if position >= len(tokens) or tokens[position] != ")":
        raise InvalidIR("Missing closing parenthesis or wrong expression arity")
    return position + 1


def validate(source):
    if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > 8192:
        raise InvalidIR("IR must contain 1..8192 UTF-8 bytes")
    lines = source.splitlines()
    if len(lines) > 256:
        raise InvalidIR("Too many IR lines")
    statements, stack = [], []
    for line_number, line in enumerate(lines, 1):
        tokens = lex(line)
        if not tokens:
            continue
        command, *args = tokens
        if command not in COMMANDS:
            raise InvalidIR(f"Line {line_number}: unknown command {command}")
        consumed = 0
        if command == "FUNC":
            if stack or not args:
                raise InvalidIR("FUNC requires a name at top level")
            for token in args:
                identifier(token)
            if len(set(args)) != len(args):
                raise InvalidIR("Duplicate function name/parameter")
            consumed = len(args)
            stack.append([command, False])
        elif command in {"LET", "ADD"}:
            if not args:
                raise InvalidIR("Missing assignment target")
            identifier(args[0])
            consumed = expression(args, 1)
        elif command in {"PRINT", "RETURN", "IF"}:
            if command == "RETURN" and not any(block[0] == "FUNC" for block in stack):
                raise InvalidIR("RETURN outside FUNC")
            consumed = expression(args)
            if command == "IF":
                stack.append([command, False])
        elif command == "FOR":
            if len(args) < 4 or args[1] != "RANGE":
                raise InvalidIR("FOR requires name RANGE start stop")
            identifier(args[0])
            consumed = expression(args, expression(args, 2))
            stack.append([command, False])
        elif command == "ELSE":
            if not stack or stack[-1] != ["IF", False]:
                raise InvalidIR("ELSE requires one unmatched IF")
            stack[-1][1] = True
        elif command in {"END", "END_FUNC"}:
            if not stack or (stack[-1][0] == "FUNC") != (command == "END_FUNC"):
                raise InvalidIR("Wrong block terminator")
            stack.pop()
        elif command == "CALL":
            if not args:
                raise InvalidIR("CALL requires a name")
            identifier(args[0])
            consumed = 1
            while consumed < len(args):
                consumed = expression(args, consumed)
        if consumed != len(args):
            raise InvalidIR(f"Line {line_number}: wrong command arity")
        if len(stack) > 32:
            raise InvalidIR("Excessive block depth")
        statements.append(Statement(command, tuple(args), line_number))
    if stack:
        raise InvalidIR("Unclosed block")
    return tuple(statements)
