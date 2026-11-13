"""Offline supervision with public v2 phrases and new compositional phrase splits."""

import json

from evaluation.curriculum_v2 import group_cases
from evaluation.nl_data import utterances
from flybrain.sequence_io import lexical_spans
from ir.grammar import lex


INITIAL_DATA_VERSION = "nl-public-v2-augmentation-and-new-compositions-v3"
CONTRASTIVE_DATA_VERSION = "nl-public-v2-contrastive-adverb-and-new-compositions-v3.1"
NL_DATA_VERSION = "nl-public-v2-contrastive-morphology-compositions-v3.2"
SUPPORTED = {
    "let",
    "print",
    "string",
    "let_print",
    "add_print",
    "identity",
    "func_add",
    "for_print",
}


def quote_names(sentence, source):
    names = {
        token
        for line in source.splitlines()
        for token in lex(line)
        if token.isidentifier() and token[0].islower()
    }
    raw = sentence.encode()
    pieces, previous = [], 0
    for span in lexical_spans(raw):
        word = raw[span.start : span.end].decode()
        pieces.extend(
            [
                raw[previous : span.start].decode(),
                json.dumps(word, ensure_ascii=False) if word in names else word,
            ]
        )
        previous = span.end
    pieces.append(raw[previous:].decode())
    return "".join(pieces)


def composition_forms(case, split):
    lines = case["source"].splitlines()
    tokens = lex(lines[0])
    shape = case["shape"]
    if shape == "let":
        _, x, a = tokens
        validation = [f"변수 {x} 값을 {a} 로 정해", f"{a} 를 변수 {x} 에 저장해"]
        test = [
            f"변수 {x} 에 값 {a} 를 저장해",
            f"값 {a} 을 {x} 에 저장해",
            f"지금 {a} 을 변수 {x} 에 저장해",
        ]
    elif shape in {"print", "string"}:
        a = tokens[1]
        validation = [f"화면에 값 {a} 를 출력해", f"지금 값 {a} 를 화면에 출력해"]
        test = [
            f"값 {a} 를 화면에 출력해",
            f"지금 화면에 값 {a} 을 출력해",
            f"{a} 을 화면에 출력해",
        ]
    elif shape == "let_print":
        _, x, a = tokens
        validation = [
            f"변수 {x} 값을 {a} 로 설정하고 {x} 를 출력해",
            f"{a} 를 {x} 에 저장한 다음 출력해",
        ]
        test = [
            f"변수 {x} 에 값 {a} 를 저장하고 출력해",
            f"값 {a} 을 {x} 에 저장하고 {x} 를 출력해",
            f"지금 {x} 에 {a} 를 저장한 다음 출력해",
        ]
    elif shape == "add_print":
        _, x, a = tokens
        b = lex(lines[1])[2]
        validation = [
            f"변수 {x} 에 {a} 을 저장하고 {b} 를 더해 출력해",
            f"{a} 를 {x} 에 저장한 뒤 {b} 만큼 증가시켜 출력해",
        ]
        test = [
            f"변수 {x} 에 값 {a} 를 저장한 뒤 {b} 를 더해서 출력해",
            f"값 {a} 을 {x} 에 저장하고 {b} 을 더해 출력해",
            f"지금 {a} 을 변수 {x} 에 저장하고 {b} 를 더해서 출력해",
        ]
    elif shape == "identity":
        _, f, p = tokens
        validation = [
            f"함수 {f} 를 정의하고 인자 {p} 를 그대로 반환해",
            f"{p} 를 받아 그대로 반환하는 함수 {f} 를 만들어",
        ]
        test = [
            f"인자 {p} 를 그대로 반환하는 함수 {f} 를 정의해",
            f"지금 함수 {f} 를 만들고 {p} 를 그대로 반환해",
            f"함수 {f} 는 {p} 를 받아 반환해",
        ]
    elif shape == "func_add":
        _, f, p, q = tokens
        validation = [
            f"함수 {f} 를 정의하고 {p} 와 {q} 의 합을 반환해",
            f"{p} 와 {q} 를 받아 합을 반환하는 함수 {f} 를 만들어",
        ]
        test = [
            f"인자 {p} 와 {q} 의 합을 반환하는 함수 {f} 를 정의해",
            f"지금 함수 {f} 는 {p} 와 {q} 의 합을 반환해",
            f"함수 {f} 를 만들고 두 인자 {p} 와 {q} 를 더해 반환해",
        ]
    elif shape == "for_print":
        _, i, _, a, b = tokens
        validation = [
            f"변수 {i} 로 {a} 부터 {b} 미만까지 출력해",
            f"{a} 이상 {b} 미만을 반복 변수 {i} 로 출력해",
        ]
        test = [
            f"변수 {i} 로 {a} 이상 {b} 미만의 수를 순서대로 출력해",
            f"지금 {a} 이상 {b} 미만의 수를 반복 변수 {i} 로 출력해",
            f"반복 변수 {i} 를 써서 {a} 부터 {b} 미만까지 출력해",
        ]
    else:
        raise ValueError("Unsupported v3 curriculum shape")
    return validation if split == "validation" else test


def data_version(profile):
    if profile not in {"initial", "contrastive", "morphology"}:
        raise ValueError("Unknown v3 supervision profile")
    return (
        INITIAL_DATA_VERSION
        if profile == "initial"
        else CONTRASTIVE_DATA_VERSION
        if profile == "contrastive"
        else NL_DATA_VERSION
    )


def nl_cases_v3(*, profile="morphology"):
    data_version(profile)
    splits = {name: [] for name in ("train", "validation", "test")}
    for split, groups in (
        ("train", range(400, 404)),
        ("validation", range(500, 504)),
        ("test", range(600, 604)),
    ):
        for group in groups:
            for case in group_cases(group):
                if case["shape"] not in SUPPORTED:
                    continue
                base = utterances(case)
                if (
                    split == "train"
                    and profile == "morphology"
                    and case["shape"] in {"identity", "func_add"}
                ):
                    base += [
                        sentence.replace("만들고", "정의하고").replace(
                            "만들어", "정의해"
                        )
                        for sentence in list(base)
                        if "만들고" in sentence or "만들어" in sentence
                    ]
                forms = (
                    sorted(
                        {
                            prefix + sentence.replace(" 을 ", particle)
                            for sentence in base
                            for particle in (" 을 ", " 를 ")
                            for prefix in ("", "지금 ")
                        }
                    )
                    if split == "train"
                    else composition_forms(case, split)
                )
                for form, sentence in enumerate(forms):
                    splits[split].append(
                        {
                            "id": f"nl-v3-{split}-{group}-{case['shape']}-{form}",
                            "family": f"v3-{split}-names-{group}",
                            "shape": case["shape"],
                            "source": quote_names(sentence, case["source"]),
                            "target": case["source"],
                            "expected": case["expected"],
                            "supported": True,
                            "phrase_split": split,
                        }
                    )
            unknown = (
                [
                    f"도시 {group} 의 날씨를 알려줘",
                    f"사진 {group} 을 그려줘",
                    f"음악 {group} 을 연주해",
                    f"파일 {group} 을 삭제해",
                    f"사진 {group} 을 편집해",
                    f"사진 {group} 에 색을 입혀줘",
                ]
                if split == "train"
                else [
                    f"지금 도시 {group} 의 날씨를 알려줘",
                    f"사진 {group} 을 화면에 그려줘",
                ]
                if split == "validation"
                else [
                    f"지금 사진 {group} 을 그려줘",
                    f"파일 {group} 을 지금 삭제해",
                    f"지금 음악 {group} 을 연주해",
                ]
            )
            if split == "train" and profile != "initial":
                # Position-only augmentation across every unsupported topic. This
                # removes the spurious association between the adverb and support.
                unknown += [
                    "지금 " + sentence
                    if "파일" in sentence
                    else sentence.rsplit(" ", 1)[0]
                    + " 지금 "
                    + sentence.rsplit(" ", 1)[1]
                    for sentence in list(unknown)
                ]
            for form, sentence in enumerate(unknown):
                splits[split].append(
                    {
                        "id": f"nl-v3-{split}-{group}-unsupported-{form}",
                        "family": f"v3-{split}-names-{group}",
                        "shape": "unsupported",
                        "source": sentence,
                        "target": "UNKNOWN\n",
                        "expected": {},
                        "supported": False,
                        "phrase_split": split,
                    }
                )
    return splits
