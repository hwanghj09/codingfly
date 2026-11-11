import ast
import io
import random
import tokenize
from dataclasses import dataclass
from string import Formatter

from pseudocode import COMMANDS, SIMPLE, statement, translate
from reward import RewardMemory


@dataclass(frozen=True)
class Operation:
    lessons: tuple[int, ...] = ()
    template: str = ""
    aliases: tuple[str, ...] = ()


OPERATIONS = {
    "대입": Operation((1,)),
    "호출": Operation((2, 3, 4)),
    "만약": Operation((5, 6)),
    "같음": Operation((7,), "{} = {} == {}"),
    "반환": Operation((8,)),
    "목록": Operation((9, 10), "{} = [{}]"),
    "각": Operation((11,)),
    "함수": Operation((12,)),
    "아니면": Operation((13,)),
    "아니면 만약": Operation((14,), aliases=("ELSE IF", "ELIF")),
    "동안": Operation((15,)),
    "조건값": Operation((16,), "{} = {} if {} else {}"),
    "모두": Operation((17,), "{} = {} and {}"),
    "기본값": Operation((18,), "{} = {} or {}"),
    "반전": Operation((19,), "{} = not ({})"),
    "동일객체": Operation((20,), "{} = {} is {}"),
    "클래스": Operation((21,), "class {}:"),
    "시도": Operation((22,), "try:"),
    "예외": Operation((23,), "except {}:"),
    "마무리": Operation((24,), "finally:"),
    "자원": Operation((25,), "with {} as {}:"),
    "별명모듈": Operation((26,), "import {} as {}"),
    "가져오기": Operation((27,), "import {}"),
    "가져올곳": Operation((28,), "from {} import {}"),
    "짧은함수": Operation((29,), "{} = lambda {}: {}"),
    "내보내기": Operation((30,), "yield {}"),
    "오류": Operation((31,), "raise {}"),
    "검증": Operation((32,), "assert {}"),
    "통과": Operation((33,)),
    "중단": Operation((34,)),
    "건너뛰기": Operation((35,)),
    "삭제": Operation((36,), "del {}"),
    "사전": Operation((37, 38, 39), "{} = {{{}: {}}}"),
    "데코레이터": Operation((40,), "@{}"),
    "반환형함수": Operation((41,), "def {} -> {}:"),
    "가변인자함수": Operation((42,), "def {}(*{}):"),
    "사전합치기": Operation((43,), "{} = {{**{}, **{}}}"),
    "대입하며검사": Operation((44,), "if ({} := {}):"),
    "기다리기": Operation((45,), "await {}"),
    "비동기함수": Operation((46,), "async def {}:"),
    "패턴비교": Operation((47,), "match {}:"),
    "패턴": Operation((48,), "case {}:"),
    "누적": Operation(),
    "반복": Operation(),
    "출력": Operation(),
}


def arguments(body):
    if not body:
        return []
    depth = 0
    start = 0
    parts = []
    for token in tokenize.generate_tokens(io.StringIO(body).readline):
        if token.type != tokenize.OP:
            continue
        if token.string in "([{":
            depth += 1
        elif token.string in ")]}":
            depth -= 1
        elif token.string == "|" and depth == 0 and body[token.start[1] - 1:token.end[1] + 1] == " | ":
            parts.append(body[start:token.start[1]].strip())
            start = token.end[1]
    return [*parts, body[start:].strip()]


def instruction(line):
    for phrase in ("아니면 만약", "ELSE IF"):
        if line.startswith(phrase + " "):
            return phrase, line[len(phrase):].strip()
    cue, *body = line.split(maxsplit=1)
    if cue.isidentifier() and body and body[0].startswith("←"):
        return "대입", line
    return cue.removesuffix(":"), body[0] if body else ""


class ProgramModel:
    def __init__(self, fly):
        actions = {cue: cue for cue in OPERATIONS}
        aliases = {**COMMANDS, **SIMPLE}
        for cue, operation in OPERATIONS.items():
            if not operation.template and cue != "아니면 만약":
                actions.update({alias: cue for alias, action in aliases.items() if action == aliases[cue]})
            actions.update(dict.fromkeys(operation.aliases, cue))
        self.cues = frozenset(actions)
        examples = [(0, cue, action) for cue, action in actions.items()]
        self.memory = RewardMemory(fly, examples, examples, {}, project=fly.project)
        rng = random.Random(42)
        learned = []
        self.attempts = 0
        self.bonus = 0
        for action in OPERATIONS:
            learned.append(action)
            for cue, expected in actions.items():
                if expected != action:
                    continue
                for _ in learned:
                    _, reward = self.memory.attempt(cue, expected, learned, rng)
                    self.attempts += 1
                    if reward > 0:
                        self.bonus += reward - 1
                        break

    def render(self, line, used):
        if line.startswith("#"):
            return line
        cue, body = instruction(line)
        if cue not in self.cues:
            raise ValueError(f"학습 목록에 없는 명령입니다. {cue!r}")
        action = self.memory.predict(cue)
        if action is None:
            raise ValueError("모델이 Python 구문을 고르지 못했습니다.")
        operation = OPERATIONS[action]
        used.update(operation.lessons)
        if not operation.template:
            return statement(f"{action} {body}".rstrip())
        if operation.template.endswith(":"):
            body = body.removesuffix(":").rstrip()
        try:
            parts = arguments(body)
        except tokenize.TokenError as error:
            raise ValueError(f"{action}의 인자를 확인하세요.") from error
        expected = sum(field is not None for _, field, _, _ in Formatter().parse(operation.template))
        if len(parts) != expected or any(not part for part in parts):
            raise ValueError(f"{action}에 필요한 인자 {expected}개를 ‘ | ’로 구분하세요.")
        return operation.template.format(*parts)

    def generate(self, source):
        used = set()
        code = translate(source, lambda line: self.render(line, used))
        return {"code": code, "lessons": sorted(used)}

    def evaluate(self, checks):
        correct = 0
        covered = set()
        for check in checks:
            try:
                result = self.generate(check["source"])
            except ValueError:
                continue
            if ast.dump(ast.parse(result["code"])) == ast.dump(ast.parse(check["python"])):
                correct += 1
                covered.update(result["lessons"])
        return {"correct": correct, "total": len(checks), "covered": sorted(covered),
                "attempts": self.attempts, "bonus": self.bonus}
