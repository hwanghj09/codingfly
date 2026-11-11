import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


INSTRUCTIONS = (
    "Write a complete Python 3 solution to the user's programming request. Interpret Korean and English. "
    "Follow every explicitly requested algorithm, library, and method. Preserve the requested variable and "
    "function names. If asked for a function, define it without example calls unless requested. Return the result "
    "and print only if asked. If asked for "
    "a program, include required imports and complete runnable code. Use concise readable code. "
    "Reply with Python code only, without Markdown or explanations outside the code."
)
LOCAL_MODEL = "qwen2.5-coder-3b"
LOCAL_URL = "http://127.0.0.1:8766/v1/chat/completions"
FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def request_json(payload):
    request = Request(LOCAL_URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    local = build_opener(ProxyHandler({}))
    try:
        with local.open(request, timeout=90) as response:
            data = response.read(512_001)
    except HTTPError as error:
        raise ValueError(f"언어 모델 서버에서 생성 요청을 처리하지 못했습니다. (HTTP {error.code})") from error
    except (URLError, TimeoutError) as error:
        raise ValueError("로컬 코드 모델에 연결하지 못했습니다. llama-server가 포트 8766에서 실행 중인지 확인하세요.") from error
    if len(data) > 512_000:
        raise ValueError("언어 모델 응답이 너무 큽니다.")
    try:
        return json.loads(data)
    except ValueError as error:
        raise ValueError("언어 모델 응답을 읽지 못했습니다.") from error


def syntax_check(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 12_000:
        raise ValueError("검사할 Python 코드는 1자 이상 12,000자 이하여야 합니다.")
    try:
        compile(code, "<generated>", "exec")
    except (SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError(f"Python 문법 검사에 실패했습니다. {error}") from error


def generate_code(goal):
    try:
        result = request_json({
            "model": LOCAL_MODEL, "messages": [{"role": "system", "content": INSTRUCTIONS},
                                               {"role": "user", "content": goal}],
            "max_tokens": 1200, "temperature": 0.1,
        })
        code = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError("언어 모델에서 Python 답안을 받지 못했습니다.") from error
    if not isinstance(code, str):
        raise ValueError("언어 모델에서 Python 답안을 받지 못했습니다.")
    code = code.strip()
    if fenced := FENCED_CODE.fullmatch(code):
        code = fenced[1]
    code += "\n"
    syntax_check(code)
    return code, "로컬 Qwen2.5-Coder 3B"
