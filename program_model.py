import ast
import io
import json
import random
import re
import tokenize
from dataclasses import dataclass
from string import Formatter

from algorithm_model import AlgorithmModel
from fly import HERE
from pseudocode import COMMANDS, SIMPLE, identifier, statement, translate
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
    "사전갱신": Operation(template="{}[{}] = {}"),
    "데코레이터": Operation((40,), "@{}"),
    "반환형함수": Operation((41,), "def {} -> {}:"),
    "가변인자함수": Operation((42,), "def {}(*{}):"),
    "사전합치기": Operation((43,), "{} = {{**{}, **{}}}"),
    "대입하며검사": Operation((44,), "if ({} := {}):"),
    "기다리기": Operation((45,), "await {}"),
    "비동기함수": Operation((46,), "async def {}:"),
    "패턴비교": Operation((47,), "match {}:"),
    "패턴": Operation((48,), "case {}:"),
    "목록내포": Operation((49,), "{} = [{} for {} in {}]"),
    "조건내포": Operation((50,), "{} = [{} for {} in {} if {}]"),
    "집합내포": Operation((51,), "{} = {{{} for {} in {}}}"),
    "사전내포": Operation((52,), "{} = {{{}: {} for {} in {}}}"),
    "생성식": Operation((53,), "{} = ({} for {} in {})"),
    "조각": Operation((54,), "{} = {}[{}:{}:{}]"),
    "여러변수": Operation((55,), "{} = {}"),
    "별표풀기": Operation((56,), "{}, *{} = {}"),
    "키워드가변함수": Operation((57,), "def {}(**{}):"),
    "위치전용함수": Operation((58,), "def {}({}, /, {}):"),
    "키워드전용함수": Operation((59,), "def {}({}, *, {}):"),
    "타입대입": Operation((60,), "{}: {} = {}"),
    "범위비교": Operation((61,), "{} = {} <= {} < {}"),
    "포함": Operation((62,), "{} = {} in {}"),
    "미포함": Operation((63,), "{} = {} not in {}"),
    "다름": Operation((64,), "{} = {} != {}"),
    "나눈몫": Operation((65,), "{} = {} // {}"),
    "나머지": Operation((66,), "{} = {} % {}"),
    "거듭제곱": Operation((67,), "{} = {} ** {}"),
    "문자열서식": Operation((68,), "{} = f{}"),
    "연쇄대입": Operation((69,), "{} = {} = {}"),
    "예외별명": Operation((70,), "except {} as {}:"),
    "원인예외": Operation((71,), "raise {} from {}"),
    "위임생성": Operation((72,), "yield from {}"),
    "순회완료": Operation((73,), "else:"),
    "동시순회": Operation((74,), "for {}, {} in zip({}, {}):"),
    "번호순회": Operation((75,), "for {}, {} in enumerate({}, {}):"),
    "패턴조건": Operation((76,), "case {} if {}:"),
    "여러패턴": Operation((77,), "case {} | {}:"),
    "비동기순회": Operation((78,), "async for {} in {}:"),
    "비동기자원": Operation((79,), "async with {} as {}:"),
    "비동기내포": Operation((80,), "{} = [{} async for {} in {}]"),
    "누적": Operation(),
    "반복": Operation(),
    "출력": Operation(),
}

CONSTRUCT_LESSONS = json.loads((HERE / "construct_lessons.json").read_text(encoding="utf-8"))
for lesson in CONSTRUCT_LESSONS:
    if "action" in lesson:
        OPERATIONS[lesson["action"]] = Operation(template=lesson["template"])


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


STEP_NUMBER = re.compile(r"^\d+(?:\.\d+)*[.)]\s+")
STEP_PHRASE = re.compile(r"([가-힣][가-힣 ]*):(?:\s+(.*)|$)")
BRANCH = {"아니면", "아니면 만약"}


def render_step(action, body):
    if action in ("대입", "누적"):
        parts = arguments(body)
        if len(parts) == 2 and all(parts):
            target, value = parts
            operator, node = ("=", ast.Assign) if action == "대입" else ("+=", ast.AugAssign)
            code = f"{target} {operator} {value}"
            tree = ast.parse(code).body
            if len(tree) == 1 and isinstance(tree[0], node):
                return code
            raise ValueError("대상과 값을 확인하세요.")
    if action == "각":
        parts = arguments(body)
        if len(parts) == 2:
            body = " ∈ ".join(parts)
    return statement(f"{action} {body}".rstrip())


class ProgramModel:
    def __init__(self, fly):
        self.algorithm = AlgorithmModel(fly)
        actions = {cue: cue for cue in OPERATIONS}
        aliases = {**COMMANDS, **SIMPLE}
        for cue, operation in OPERATIONS.items():
            if not operation.template and cue != "아니면 만약":
                actions.update({alias: cue for alias, action in aliases.items() if action == aliases[cue]})
            actions.update(dict.fromkeys(operation.aliases, cue))
        self.cues = frozenset(actions)
        self.curriculum = len({lesson for operation in OPERATIONS.values() for lesson in operation.lessons})
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

    def render(self, line, used, indent, siblings, dictionaries):
        if line.startswith("#"):
            return line
        step = STEP_NUMBER.match(line)
        if step:
            line = line[step.end():]
        for deeper in [level for level in siblings if level > indent]:
            del siblings[deeper]
        cue, body = instruction(line)
        phrase = STEP_PHRASE.fullmatch(line)
        if phrase and phrase[1] not in self.cues and (step or phrase[2] or cue not in self.cues):
            action, code = self.decode_step(phrase[1], phrase[2] or "", siblings.get(indent))
            siblings[indent] = action
            used.update(OPERATIONS[action].lessons)
            return code
        if cue not in self.cues:
            try:
                fields = arguments(line)
                if len(fields) == 3 and all(fields):
                    name = fields[0]
                    cue = "사전갱신" if name in dictionaries else "사전"
                    fields[0] = dictionaries.setdefault(name, identifier("_".join(name.split())))
                    body = " | ".join(fields)
                elif isinstance(ast.parse(line, mode="eval").body, ast.Call):
                    cue, body = "호출", line
            except (SyntaxError, tokenize.TokenError):
                pass
            if cue not in self.cues:
                raise ValueError(f"학습 목록에 없는 명령입니다. {cue!r}")
        action = self.memory.predict(cue)
        if action is None:
            raise ValueError("모델이 Python 구문을 고르지 못했습니다.")
        siblings[indent] = action
        operation = OPERATIONS[action]
        used.update(operation.lessons)
        if action == "출력":
            body = dictionaries.get(body, body)
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

    def decode_step(self, phrase, body, previous):
        for action, score in self.algorithm.ranks(phrase):
            if score < 0.42:
                break
            if action in BRANCH and previous not in ("만약", "아니면 만약"):
                continue
            try:
                return action, render_step(action, body)
            except (SyntaxError, ValueError, tokenize.TokenError):
                continue
        raise ValueError(f"‘{phrase}’ 단계의 명령과 인자를 확인하세요.")

    def generate(self, source):
        used = set()
        siblings = {}
        dictionaries = {}
        code = translate(source, lambda line, indent: self.render(line, used, indent, siblings, dictionaries))
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

    def construct_results(self):
        courses = []
        for lesson in CONSTRUCT_LESSONS:
            examples = []
            for example in lesson["examples"]:
                try:
                    code = self.generate(example["source"])["code"]
                    correct = ast.dump(ast.parse(code)) == ast.dump(ast.parse(example["python"]))
                except ValueError as error:
                    code, correct = str(error), False
                examples.append({"source": example["source"], "code": code, "correct": correct})
            courses.append({"group": lesson["group"], "title": lesson["title"],
                            "description": lesson["description"], "examples": examples})
        return courses
