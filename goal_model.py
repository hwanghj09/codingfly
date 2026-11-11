import ast
import re
from dataclasses import dataclass
from textwrap import indent

from goal_runtime import inspect_output


METHOD_NAMES = {
    "함수": "함수", "함수 호출": "함수", "변수": "변수", "반복문": "반복문",
    "for": "for", "for문": "for", "for 문": "for",
    "while": "while", "while문": "while", "while 문": "while",
    "리스트 내포": "리스트 내포", "리스트 컴프리헨션": "리스트 내포",
    "sum": "sum()", "sum()": "sum()", "sorted": "sorted()", "sorted()": "sorted()",
}
PROCESS = re.compile(r"(.+?)(?:을|를)?\s*(?:이용해서|사용해서|이용해|사용해|이용하여|사용하여|써서)\s+(.+)")
PROCESS_SHORT = re.compile(r"(" + "|".join(map(re.escape, METHOD_NAMES)) + r")(?:으로|로)\s+(.+)")
PRINT_GOAL = re.compile(r"(.+?)\s*(?:출력해\s*(?:줘|주세요)?|출력(?:하기|하라|하세요)?)\s*[.!]?")
RANGE_GOAL = re.compile(r"(-?\d+)\s*부터\s*(-?\d+)\s*까지(?:의)?\s*(.*?)")
LIST_GOAL = re.compile(r"(\[.*\])\s*(.*)")


@dataclass(frozen=True)
class Goal:
    output: str
    plans: tuple[str, ...]
    required: tuple[str, ...] = ()


def text_plans(text, count=1):
    direct = f"출력 {text!r}" if count == 1 else f"호출 print(*([{text!r}] * {count}), sep='\\n')"
    variable = f"문장 ← {text!r}\n출력 문장" if count == 1 else f"문장 ← {text!r}\n호출 print(*([문장] * {count}), sep='\\n')"
    return (direct, f"각 항목 ∈ [{text!r}] * {count}\n    출력 항목", variable,
            f"횟수 ← 0\n동안 횟수 < {count}\n    출력 {text!r}\n    누적 횟수 ← 1",
            f"항목들 ← [{text!r} for 항목 in range({count})]\n호출 print(*항목들, sep='\\n')")


def range_goal(start, stop, operation):
    if max(abs(start), abs(stop)) > 100_000 or not 0 <= stop - start <= 99:
        raise ValueError("시작과 끝 사이에 최대 100개의 정수를 입력하세요.")
    items = list(range(start, stop + 1))
    tail = operation.removesuffix("를").removesuffix("을").strip()
    if tail in ("합", "합계", "총합", "더해서", "합산해서", "모두 더해서"):
        return str(sum(items)), (
            f"출력 sum(range({start}, {stop + 1}))",
            f"합계 ← 0\n각 숫자 ∈ range({start}, {stop + 1})\n    누적 합계 ← 숫자\n출력 합계",
            f"합계 ← 0\n숫자 ← {start}\n동안 숫자 <= {stop}\n    누적 합계 ← 숫자\n    누적 숫자 ← 1\n출력 합계",
            f"출력 sum([숫자 for 숫자 in range({start}, {stop + 1})])",
        )
    if tail in ("짝수만", "짝수", "홀수만", "홀수"):
        condition = f"숫자 % 2 == {int(tail.startswith('홀수'))}"
        selected = [number for number in items if number % 2 == int(tail.startswith("홀수"))]
    elif tail in ("", "숫자", "숫자 한 줄씩", "한 줄씩", "정수"):
        condition, selected = None, items
    else:
        raise ValueError("정수 범위는 전체 출력, 짝수·홀수 출력, 합계 출력을 지원합니다.")
    expected = "\n".join(map(str, selected))
    body = f"만약 {condition}\n    출력 숫자" if condition else "출력 숫자"
    filtered = f" if {condition}" if condition else ""
    if not selected:
        gather = f"조건값 추가 | [숫자] | {condition} | []\n누적 선택 ← 추가"
        print_empty = "호출 print(*선택, sep='\\n', end='')"
        return expected, (
            f"선택 ← []\n각 숫자 ∈ range({start}, {stop + 1})\n{indent(gather, '    ')}\n{print_empty}",
            f"선택 ← []\n숫자 ← {start}\n동안 숫자 <= {stop}\n{indent(gather, '    ')}\n    누적 숫자 ← 1\n{print_empty}",
            f"선택 ← [숫자 for 숫자 in range({start}, {stop + 1}){filtered}]\n{print_empty}",
        )
    return expected, (
        f"각 숫자 ∈ range({start}, {stop + 1})\n{indent(body, '    ')}",
        f"숫자 ← {start}\n동안 숫자 <= {stop}\n{indent(body, '    ')}\n    누적 숫자 ← 1",
        f"호출 print(*[숫자 for 숫자 in range({start}, {stop + 1}){filtered}], sep='\\n')",
    )


def list_goal(literal, operation):
    try:
        items = ast.literal_eval(literal)
    except (ValueError, SyntaxError) as error:
        raise ValueError("[3, 1, 2]처럼 정수 목록을 입력하세요.") from error
    if not isinstance(items, list) or len(items) > 100 or any(type(item) is not int or abs(item) > 100_000 for item in items):
        raise ValueError("−100,000부터 100,000 사이의 정수를 최대 100개 입력하세요.")
    reverse = "내림차순" in operation
    if re.fullmatch(r"(?:을|를)?\s*(?:(?:오름차순|내림차순)(?:으로)?\s*)?정렬(?:해서|하여|한 결과를)?", operation):
        direct = f"출력 sorted({items!r}, reverse={reverse})"
        selector = "max" if reverse else "min"
        setup = f"남은값 ← {items!r}\n정렬값 ← []"
        step = (f"    선택 ← {selector}(남은값)\n    위치 ← 남은값.index(선택)\n"
                "    누적 정렬값 ← [선택]\n    남은값 ← 남은값[:위치] + 남은값[위치 + 1:]\n출력 정렬값")
        return str(sorted(items, reverse=reverse)), (
            direct, f"{setup}\n각 항목 ∈ range(len(남은값))\n{step}", f"{setup}\n동안 남은값\n{step}")
    if re.fullmatch(r"(?:의\s*)?(?:합계|총합|합)(?:을|를)?", operation):
        return str(sum(items)), (f"출력 sum({items!r})",
                                 f"합계 ← 0\n각 값 ∈ {items!r}\n    누적 합계 ← 값\n출력 합계")
    raise ValueError("정수 목록은 정렬 결과와 합계 출력을 지원합니다.")


def parse_goal(source):
    if not isinstance(source, str) or not source.strip() or len(source) > 600:
        raise ValueError("목적을 600자 이내로 입력하세요.")
    source = source.strip()
    required = ()
    process = PROCESS.fullmatch(source) or PROCESS_SHORT.fullmatch(source)
    if process:
        methods, source = process.groups()
        names = [part.strip() for part in re.split(r"\s*(?:,|그리고|및|와\s+|과\s+)\s*", methods)]
        if any(name not in METHOD_NAMES for name in names):
            raise ValueError("방법으로 함수, 변수, 반복문, for문, while문, 리스트 내포, sum(), sorted()를 지정할 수 있습니다.")
        required = tuple(dict.fromkeys(METHOD_NAMES[name] for name in names))
    match = PRINT_GOAL.fullmatch(source)
    if not match:
        raise ValueError("‘hello를 출력해’ 또는 ‘for문을 이용해서 1부터 5까지의 합계를 출력해’처럼 입력하세요.")
    text = match[1].strip()
    if span := RANGE_GOAL.fullmatch(text):
        output, plans = range_goal(int(span[1]), int(span[2]), span[3])
    elif series := LIST_GOAL.fullmatch(text):
        output, plans = list_goal(*series.groups())
    else:
        repeated = re.fullmatch(r"(.+?)\s+(\d+)\s*(?:번|회)(?:\s+반복해서)?", text)
        count = int(repeated[2]) if repeated else 1
        text = repeated[1] if repeated else text
        text = re.sub(r"[을를]\s*$", "", text).strip()
        if text.startswith(('"', "'")):
            try:
                text = ast.literal_eval(text)
            except (ValueError, SyntaxError) as error:
                raise ValueError("출력할 문자열의 따옴표를 확인하세요.") from error
        elif not text.isascii() and re.search(r"\s", text):
            raise ValueError("문장 전체를 그대로 출력하려면 따옴표로 감싸세요. 정수 범위와 목록의 합계·정렬도 입력할 수 있습니다.")
        if not isinstance(text, str) or len(text) > 500 or not 1 <= count <= 100 or (len(text) + 1) * count > 4096:
            raise ValueError("문자열은 500자, 반복은 100회, 전체 출력은 4,096자 이내로 입력하세요.")
        output, plans = "\n".join([text] * count), text_plans(text, count)
    return Goal(output, plans, required)


def judge(goal, code):
    try:
        output, methods, chunks = inspect_output(code)
    except ValueError as error:
        return {"accepted": False, "output": "", "methods": [], "missing": list(goal.required), "error": str(error)}
    missing = [name for name in goal.required if any(
        name not in used and not (name == "반복문" and used & {"for", "while"}) for used in chunks or [methods])]
    matches = output in (goal.output, goal.output + "\n")
    return {"accepted": matches and not missing, "output": output, "methods": sorted(methods), "missing": missing,
            "error": "" if matches else "출력 결과가 목적과 다릅니다."}


def solve(model, source, answer=None):
    goal = parse_goal(source)
    if answer is not None:
        if not isinstance(answer, str):
            raise ValueError("검사할 답안은 Python 코드 문자열이어야 합니다.")
        return {"expected": goal.output, "required": goal.required, "answers": [{"code": answer, **judge(goal, answer)}]}
    answers = []
    for plan in goal.plans:
        wrapped = f"함수 해결()\n{indent(plan, '    ')}\n호출 해결()"
        for candidate in (plan, wrapped):
            try:
                code = model.generate(candidate)["code"]
                answers.append({"code": code, **judge(goal, code)})
            except ValueError as error:
                answers.append({"code": "", "accepted": False, "output": "", "methods": [],
                                "missing": list(goal.required), "error": str(error)})
    return {"expected": goal.output, "required": goal.required, "answers": answers}
