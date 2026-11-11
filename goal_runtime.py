import ast
import operator
from dataclasses import dataclass


LIMIT = 512
UNBOUND = object()
BUILTINS = {"len": len, "list": list, "sum": sum, "sorted": sorted, "min": min, "max": max, "str": str}
METHODS = {"sum": "sum()", "sorted": "sorted()"}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Eq: operator.eq,
             ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
             ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: operator.contains}


@dataclass(frozen=True)
class Value:
    data: object
    methods: frozenset[str] = frozenset()


class Returned(Exception):
    def __init__(self, value):
        self.value = value


def value(data, methods=frozenset()):
    if isinstance(data, int) and abs(data) > 1_000_000_000:
        raise ValueError("숫자가 실행 검사 범위를 넘었습니다.")
    if isinstance(data, (str, list, tuple, range)) and len(data) > LIMIT:
        raise ValueError("문자열 또는 자료의 크기가 실행 검사 범위를 넘었습니다.")
    return Value(data, methods)


class GoalRuntime:
    def __init__(self):
        self.remaining = 15_000
        self.output = ""
        self.methods = set()
        self.output_methods = []
        self.functions = {}
        self.globals = {}
        self.depth = 0

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("실행 검사에서 반복 또는 호출 한도에 도달했습니다.")

    def binary(self, op, left, right):
        if type(op) not in OPERATORS:
            raise ValueError("실행 검사에서 지원하지 않는 연산입니다.")
        a, b = left.data, right.data
        if isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In)):
            self.check_print_size(a)
            self.check_print_size(b)
        if isinstance(op, ast.Mult):
            if isinstance(a, int):
                a, b = b, a
            if isinstance(a, (str, list, tuple)) and isinstance(b, int) and len(a) * max(0, b) > LIMIT:
                raise ValueError("반복한 자료의 크기가 한도를 넘었습니다.")
        if isinstance(op, ast.Mod) and not (isinstance(a, int) and isinstance(b, int)):
            raise ValueError("나머지 연산은 정수에 사용할 수 있습니다.")
        result = OPERATORS[type(op)](b, a) if isinstance(op, ast.In) else OPERATORS[type(op)](a, b)
        return value(result, left.methods | right.methods)

    def expression(self, node, env, context):
        self.tick()
        match node:
            case ast.Constant(value=data) if type(data) in (int, str, bool, type(None)):
                return value(data)
            case ast.Name(id=name):
                if name not in env or env[name].data is UNBOUND:
                    raise ValueError(f"정의되지 않은 변수 {name!r}")
                return env[name]
            case ast.List(elts=items) | ast.Tuple(elts=items):
                values = [self.expression(item, env, context) for item in items]
                data = [item.data for item in values]
                return value(tuple(data) if isinstance(node, ast.Tuple) else data,
                             frozenset().union(*(item.methods for item in values)))
            case ast.UnaryOp(op=op, operand=operand) if type(op) in (ast.USub, ast.UAdd, ast.Not):
                result = self.expression(operand, env, context)
                apply = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}[type(op)]
                return value(apply(result.data), result.methods)
            case ast.BinOp(left=left, op=op, right=right):
                return self.binary(op, self.expression(left, env, context), self.expression(right, env, context))
            case ast.Compare(left=left, ops=[op], comparators=[right]):
                return self.binary(op, self.expression(left, env, context), self.expression(right, env, context))
            case ast.IfExp(test=test, body=body, orelse=other):
                decision = self.expression(test, env, context)
                result = self.expression(body if decision.data else other, env, context)
                return value(result.data, decision.methods | result.methods)
            case ast.Subscript(value=sequence, slice=ast.Slice(lower=lower, upper=upper, step=step)):
                data = self.expression(sequence, env, context)
                bounds = [self.expression(bound, env, context) if bound else value(None) for bound in (lower, upper, step)]
                return value(data.data[slice(*(bound.data for bound in bounds))],
                             data.methods | frozenset().union(*(bound.methods for bound in bounds)))
            case ast.Subscript(value=sequence, slice=index) if not isinstance(index, ast.Slice):
                data, position = self.expression(sequence, env, context), self.expression(index, env, context)
                return value(data.data[position.data], data.methods | position.methods)
            case ast.Call(func=ast.Name(id=name), args=args, keywords=keywords):
                values = []
                for argument in args:
                    if isinstance(argument, ast.Starred):
                        sequence = self.expression(argument.value, env, context)
                        context |= sequence.methods
                        values.extend(value(item, sequence.methods) for item in sequence.data)
                    else:
                        values.append(self.expression(argument, env, context))
                    if len(values) > LIMIT:
                        raise ValueError("함수 인자가 너무 많습니다.")
                options = {option.arg: self.expression(option.value, env, context) for option in keywords}
                return self.call(name, values, options, env, context)
            case ast.Call(func=ast.Attribute(value=receiver, attr="index"), args=[argument], keywords=[]):
                sequence = self.expression(receiver, env, context)
                target = self.expression(argument, env, context)
                if type(sequence.data) not in (list, tuple, str):
                    raise ValueError("index()는 리스트, 튜플, 문자열에 사용할 수 있습니다.")
                self.check_print_size(sequence.data)
                self.check_print_size(target.data)
                return value(sequence.data.index(target.data), sequence.methods | target.methods)
            case ast.ListComp(elt=element, generators=[ast.comprehension(target=ast.Name(id=name), iter=items, ifs=tests, is_async=0)]):
                sequence = self.expression(items, env, context)
                methods = sequence.methods | {"리스트 내포"}
                result = []
                local = env.copy()
                for item in sequence.data:
                    self.tick()
                    local[name] = value(item, sequence.methods)
                    decisions = [self.expression(test, local, context) for test in tests]
                    methods |= frozenset().union(*(decision.methods for decision in decisions))
                    if all(decision.data for decision in decisions):
                        result_item = self.expression(element, local, context)
                        result.append(result_item.data)
                        methods |= result_item.methods
                return value(result, methods)
        raise ValueError(f"실행 검사에서 지원하지 않는 표현식입니다. ({type(node).__name__})")

    def call(self, name, values, options, env, context):
        if name in env:
            raise ValueError(f"함수 이름 {name!r}이 값으로 바뀌었습니다.")
        data = [item.data for item in values]
        methods = frozenset().union(*(item.methods for item in [*values, *options.values()]))
        kwargs = {key: item.data for key, item in options.items()}
        if name in self.functions and not kwargs:
            parameters, statements, locals_ = self.functions[name]
            if len(values) != len(parameters) or self.depth >= 16:
                raise ValueError("함수 인자 개수 또는 호출 깊이를 확인하세요.")
            self.depth += 1
            function_context = context | {"함수"}
            try:
                local = {**self.globals, **dict.fromkeys(locals_, value(UNBOUND)), **dict(zip(parameters, values))}
                self.block(statements, local, function_context)
            except Returned as returned:
                return value(returned.value.data, returned.value.methods | function_context)
            finally:
                self.depth -= 1
            return value(None, function_context)
        if name == "print" and kwargs.keys() <= {"sep", "end"}:
            separator = kwargs.get("sep", " ")
            ending = kwargs.get("end", "\n")
            if not isinstance(separator, str) or not isinstance(ending, str):
                raise ValueError("출력 구분자와 끝 문자는 문자열이어야 합니다.")
            for item in data:
                self.check_print_size(item)
            lines = [str(item) for item in data]
            if len(self.output) + sum(map(len, lines)) + len(separator) * max(0, len(lines) - 1) + len(ending) > 4096:
                raise ValueError("출력 길이가 검사 한도를 넘었습니다.")
            chunk = separator.join(lines) + ending
            self.output += chunk
            self.methods.update(methods | context)
            if chunk:
                self.output_methods.append(methods | context)
            return value(None)
        if name == "range" and not kwargs:
            return value(range(*data), methods)
        if name in BUILTINS and (not kwargs or name == "sorted" and kwargs.keys() <= {"reverse"}):
            if name in ("str", "sorted", "min", "max"):
                for item in data:
                    self.check_print_size(item)
            tags = {METHODS[name]} if name in METHODS else set()
            return value(BUILTINS[name](*data, **kwargs), methods | tags)
        raise ValueError(f"실행 검사에서 호출할 수 없는 함수입니다. {name!r}")

    def check_print_size(self, item):
        # Check shared nested containers before str() expands their representation.
        self.tick()
        if isinstance(item, (list, tuple)):
            for child in item:
                self.check_print_size(child)

    def block(self, statements, env, context):
        for node in statements:
            self.tick()
            match node:
                case ast.Assign(targets=[ast.Name(id=name)], value=expression):
                    result = self.expression(expression, env, context)
                    env[name] = value(result.data, result.methods | context | {"변수"})
                case ast.AugAssign(target=ast.Name(id=name), op=op, value=expression) if name in env:
                    result = self.binary(op, env[name], self.expression(expression, env, context))
                    env[name] = value(result.data, result.methods | context)
                case ast.Expr(value=expression):
                    self.expression(expression, env, context)
                case ast.If(test=test, body=body, orelse=other):
                    condition = self.expression(test, env, context)
                    self.block(body if condition.data else other, env, context | condition.methods)
                case ast.For(target=ast.Name(id=name), iter=items, body=body, orelse=[]):
                    sequence = self.expression(items, env, context)
                    loop = context | sequence.methods | {"for"}
                    for item in sequence.data:
                        self.tick()
                        env[name] = value(item, loop)
                        self.block(body, env, loop)
                case ast.While(test=test, body=body, orelse=[]):
                    while (condition := self.expression(test, env, context)).data:
                        self.block(body, env, context | condition.methods | {"while"})
                case ast.FunctionDef(name=name, args=parameters, body=body, decorator_list=[]) if self.depth == 0:
                    if (parameters.posonlyargs or parameters.kwonlyargs or parameters.defaults
                            or parameters.vararg or parameters.kwarg or name in {*BUILTINS, "print", "range"}):
                        raise ValueError("함수 정의의 인자 또는 이름을 확인하세요.")
                    locals_ = set()
                    for statement in body:
                        for child in ast.walk(statement):
                            if isinstance(child, ast.FunctionDef):
                                raise ValueError("중첩 함수는 실행 검사에서 지원하지 않습니다.")
                            targets = child.targets if isinstance(child, ast.Assign) else (
                                [child.target] if isinstance(child, (ast.For, ast.AugAssign)) else [])
                            locals_.update(target.id for target in targets if isinstance(target, ast.Name))
                    self.functions[name] = ([arg.arg for arg in parameters.args], body, locals_)
                    env.pop(name, None)
                case ast.Return(value=expression) if self.depth:
                    raise Returned(self.expression(expression, env, context) if expression else value(None))
                case ast.Pass():
                    continue
                case _:
                    raise ValueError(f"실행 검사에서 지원하지 않는 문장입니다. ({type(node).__name__})")


def inspect_output(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 4_000:
        raise ValueError("검사할 Python 코드는 4,000자까지 입력할 수 있습니다.")
    runtime = GoalRuntime()
    try:
        tree = ast.parse(code)
        compile(tree, "<answer>", "exec")
        runtime.block(tree.body, runtime.globals, frozenset())
    except (SyntaxError, ArithmeticError, TypeError, LookupError, RecursionError) as error:
        raise ValueError(f"실행 검사 오류. {error}") from error
    return runtime.output, runtime.methods, runtime.output_methods
