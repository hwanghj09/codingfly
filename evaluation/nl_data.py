"""Korean supervised utterances; paraphrase and literal/name groups do not cross splits."""

import json
from evaluation.curriculum_v2 import group_cases
from ir.grammar import lex
from flybrain.sequence_io import lexical_spans


NL_DATA_VERSION = "nl-quoted-literal-paraphrase-and-name-holdout-v2"


def utterances(case):
    source = case["source"]
    shape = case["shape"]
    lines = source.splitlines()
    tokens = lex(lines[0])
    if shape == "let":
        _, x, a = tokens
        return [
            f"{x} 에 {a} 을 저장해",
            f"변수 {x} 값을 {a} 로 설정해",
            f"{a} 을 {x} 에 저장해",
            f"{x} 값을 {a} 로 정해",
            f"변수 {x} 에 {a} 을 저장해",
            f"변수 {x} 에 {a} 를 저장해",
        ]
    if shape in {"print", "string"}:
        a = tokens[1]
        return [
            f"{a} 을 출력해",
            f"{a} 를 화면에 출력해",
            f"화면에 {a} 출력해",
            f"값 {a} 을 출력해",
            f"{a} 값을 출력해",
            f"화면에 {a} 값을 출력해",
        ]
    if shape == "let_print":
        _, x, a = tokens
        return [
            f"{x} 에 {a} 을 저장하고 {x} 를 출력해",
            f"변수 {x} 값을 {a} 로 정하고 출력해",
            f"{a} 을 변수 {x} 에 저장한 다음 출력해",
            f"{x} 값을 {a} 로 설정하고 {x} 를 출력해",
            f"변수 {x} 에 {a} 을 저장하고 출력해",
            f"변수 {x} 에 {a} 를 저장하고 출력해",
        ]
    if shape == "add_print":
        _, x, a = tokens
        b = lex(lines[1])[2]
        return [
            f"{x} 에 {a} 을 저장하고 {b} 을 더한 다음 출력해",
            f"변수 {x} 값을 {a} 로 설정하고 {b} 을 더해서 출력해",
            f"{a} 을 {x} 에 저장하고 {b} 만큼 증가시켜 출력해",
            f"{x} 값을 {a} 로 정하고 {b} 을 더해 출력해",
            f"변수 {x} 에 {a} 을 저장한 뒤 {b} 을 더해서 출력해",
            f"변수 {x} 에 {a} 를 저장한 뒤 {b} 를 더해서 출력해",
        ]
    if shape == "identity":
        _, f, p = tokens
        return [
            f"{p} 를 그대로 반환하는 함수 {f} 를 만들어",
            f"함수 {f} 는 {p} 를 받아 그대로 반환해",
            f"{p} 를 반환할 함수 {f} 를 정의해",
            f"함수 {f} 를 만들고 인자 {p} 를 반환해",
            f"함수 {f} 는 인자 {p} 를 그대로 반환해",
            f"함수 {f} 는 인자 {p} 를 받아 그대로 반환해",
        ]
    if shape == "func_add":
        _, f, p, q = tokens
        return [
            f"{p} 와 {q} 의 합을 반환하는 함수 {f} 를 만들어",
            f"함수 {f} 는 {p} 와 {q} 를 받아 합을 반환해",
            f"{p} 더하기 {q} 를 반환할 함수 {f} 를 정의해",
            f"함수 {f} 의 인자 {p} 와 {q} 를 더해 반환해",
            f"함수 {f} 는 인자 {p} 와 {q} 의 합을 반환해",
            f"함수 {f} 는 두 인자 {p} 와 {q} 의 합을 반환해",
        ]
    if shape == "for_print":
        _, i, _, a, b = tokens
        return [
            f"{i} 로 {a} 이상 {b} 미만을 순서대로 출력해",
            f"반복 변수 {i} 로 {a} 부터 {b} 미만까지 출력해",
            f"{a} 이상 {b} 미만의 수를 변수 {i} 로 출력해",
            f"변수 {i} 를 써서 {a} 이상 {b} 미만을 출력해",
            f"반복 변수 {i} 로 {a} 이상 {b} 미만을 출력해",
            f"반복 변수 {i} 로 {a} 이상 {b} 미만의 수를 출력해",
        ]
    raise ValueError("Unsupported teacher intent")


def nl_cases():
    supported = {
        "let",
        "print",
        "let_print",
        "add_print",
        "string",
        "identity",
        "func_add",
        "for_print",
    }
    splits = {name: [] for name in ("train", "validation", "test")}
    for group in range(24):
        split = "train" if group < 16 else "validation" if group < 20 else "test"
        forms = range(4) if split == "train" else [4] if split == "validation" else [5]
        for case in group_cases(group + 100):
            if case["shape"] not in supported:
                continue
            for form in forms:
                sentence = utterances(case)[form]
                names = {
                    token
                    for line in case["source"].splitlines()
                    for token in lex(line)
                    if token.isidentifier() and token[0].islower()
                }
                raw = sentence.encode()
                pieces = []
                previous = 0
                for span in lexical_spans(raw):
                    word = raw[span.start : span.end].decode()
                    pieces.append(raw[previous : span.start].decode())
                    pieces.append(
                        json.dumps(word, ensure_ascii=False) if word in names else word
                    )
                    previous = span.end
                pieces.append(raw[previous:].decode())
                sentence = "".join(pieces)
                splits[split].append(
                    {
                        "id": f"nl-{group}-{case['shape']}-{form}",
                        "family": f"nl-name-{group}-paraphrase-{form}",
                        "shape": case["shape"],
                        "source": sentence,
                        "target": case["source"],
                        "expected": case["expected"],
                        "supported": True,
                        "paraphrase": form,
                    }
                )
        for form in forms:
            unknown = [
                f"도시 {group} 의 날씨를 알려줘",
                f"사진 {group} 을 그려줘",
                f"음악 {group} 을 연주해",
                f"파일 {group} 을 삭제해",
                f"사진 {group} 을 편집해",
                f"사진 {group} 에 색을 입혀줘",
            ][form]
            splits[split].append(
                {
                    "id": f"nl-{group}-unsupported-{form}",
                    "family": f"nl-name-{group}-paraphrase-{form}",
                    "shape": "unsupported",
                    "source": unknown,
                    "target": "UNKNOWN\n",
                    "expected": {},
                    "supported": False,
                    "paraphrase": form,
                }
            )
    return splits
