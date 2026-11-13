"""Explicit non-AI teachers and independent executable oracles for the v2 curriculum."""

import json
import random
import string

from evaluation.datasets import Case, split_cases
from flybrain.sequence_io import lexical_spans
from ir.grammar import validate


DATA_VERSION = "code-24-families-v2"


def teacher_actions(source, target, vocabulary):
    """Training-data tokenization only. Never imported by a generation model."""
    raw, expected = source.encode(), target.encode()
    spans = lexical_spans(raw)
    if len(spans) > vocabulary.max_spans:
        raise ValueError("Teacher exceeds COPY vocabulary")
    candidates = [
        (value, index) for index, value in enumerate(vocabulary.literals) if value
    ]
    candidates += [
        (raw[span.start : span.end], len(vocabulary.literals) + index)
        for index, span in enumerate(spans)
    ]
    if vocabulary.inner_copy:
        candidates += [
            (
                raw[span.start + 1 : span.end - 1],
                len(vocabulary.literals) + vocabulary.max_spans + index,
            )
            for index, span in enumerate(spans)
            if raw[span.start : span.start + 1] == b'"'
            and raw[span.end - 1 : span.end] == b'"'
            and span.end - span.start > 2
        ]
    candidates.sort(key=lambda item: (-len(item[0]), item[1]))
    actions, position = [], 0
    while position < len(expected):
        match = next(
            (
                (value, index)
                for value, index in candidates
                if expected.startswith(value, position)
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"No literal/COPY teacher for target byte {position}: {expected[position : position + 30]!r}"
            )
        value, index = match
        actions.append(index)
        position += len(value)
    return actions + [0]


def aligned_code_actions(source, target, vocabulary):
    """Supervised alignment only; inference never imports these command rules."""
    target_lines = iter(target.splitlines(keepends=True))
    actions = []
    for line in source.splitlines():
        command = line.split()[0] if line.split() else ""
        if command in {"END", "END_FUNC"}:
            actions.extend([vocabulary.pop, vocabulary.next_line])
            continue
        if not command:
            actions.append(vocabulary.next_line)
            continue
        if command == "ELSE":
            actions.append(vocabulary.pop)
        reference = next(target_lines)
        actions.extend(teacher_actions(line, reference, vocabulary)[:-1])
        if command in {"FUNC", "IF", "FOR", "ELSE"}:
            actions.append(vocabulary.push)
        actions.append(vocabulary.next_line)
    if next(target_lines, None) is not None:
        raise ValueError("Teacher line alignment mismatch")
    return actions + [0]


def group_cases(group, *, seed=271828):
    rng = random.Random(seed + group)

    def name(prefix):
        return (
            prefix
            + "_"
            + "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 12)))
        )

    x, y, f, p, q, total, i = [
        name(prefix) for prefix in ("v", "w", "fn", "p", "q", "acc", "i")
    ]
    a, b = group * 7 - 99, group % 7 + 1
    text = json.dumps(f"새 문자열 {group} !", ensure_ascii=False)
    cases = []

    def add(shape, source, target, expected):
        validate(source)
        compile(target, "<teacher>", "exec")
        cases.append(
            {
                "id": f"v2-{group}-{shape}",
                "family": f"v2-name-value-{group}",
                "shape": shape,
                "source": source,
                "target": target,
                "expected": expected,
            }
        )

    def module(shape, source, target, stdout="", variables=None):
        add(shape, source, target, {"stdout": stdout, "variables": variables or {}})

    def function(shape, source, target, calls, returns):
        add(
            shape,
            source,
            target,
            {
                "stdout": "",
                "variables": {},
                "calls": [{"name": f, "args": args} for args in calls],
                "returns": returns,
            },
        )

    module("let", f"LET {x} {a}\n", f"{x} = {a}\n", variables={x: a})
    module("print", f"PRINT {a}\n", f"print({a})\n", f"{a}\n")
    module(
        "let_print",
        f"LET {x} {a}\nPRINT {x}\n",
        f"{x} = {a}\nprint({x})\n",
        f"{a}\n",
        {x: a},
    )
    module(
        "add_print",
        f"LET {x} {a}\nADD {x} {b}\nPRINT {x}\n",
        f"{x} = {a}\n{x} += {b}\nprint({x})\n",
        f"{a + b}\n",
        {x: a + b},
    )
    module(
        "copy_assign",
        f"LET {x} {a}\nLET {y} {x}\nPRINT {y}\n",
        f"{x} = {a}\n{y} = {x}\nprint({y})\n",
        f"{a}\n",
        {x: a, y: a},
    )
    module("string", f"PRINT {text}\n", f"print({text})\n", f"새 문자열 {group} !\n")
    module(
        "two_prints",
        f"PRINT {a}\nPRINT {b}\n",
        f"print({a})\nprint({b})\n",
        f"{a}\n{b}\n",
    )
    for shape, operator, symbol, result in (
        ("add_expr", "ADD", "+", a + b),
        ("mul_expr", "MUL", "*", a * b),
        ("mod_expr", "MOD", "%", a % b),
    ):
        module(
            shape,
            f"PRINT ( {operator} {a} {b} )\n",
            f"print({a} {symbol} {b})\n",
            f"{result}\n",
        )
    function(
        "constant",
        f"FUNC {f}\nRETURN {a}\nEND_FUNC\n",
        f"def {f}():\n    return {a}\n",
        [[]],
        [a],
    )
    function(
        "identity",
        f"FUNC {f} {p}\nRETURN {p}\nEND_FUNC\n",
        f"def {f}({p}):\n    return {p}\n",
        [[-9], [0], [17]],
        [-9, 0, 17],
    )
    pairs = [[-3, 2], [0, 1], [5, 7], [11, -4]]
    for shape, operator, symbol, answers in (
        ("func_add", "ADD", "+", [u + v for u, v in pairs]),
        ("func_mul", "MUL", "*", [u * v for u, v in pairs]),
        ("func_mod", "MOD", "%", [u % v for u, v in pairs]),
    ):
        function(
            shape,
            f"FUNC {f} {p} {q}\nRETURN ( {operator} {p} {q} )\nEND_FUNC\n",
            f"def {f}({p}, {q}):\n    return {p} {symbol} {q}\n",
            pairs,
            answers,
        )
    function(
        "increment",
        f"FUNC {f} {p}\nRETURN ( ADD {p} {b} )\nEND_FUNC\n",
        f"def {f}({p}):\n    return {p} + {b}\n",
        [[-9], [0], [17]],
        [-9 + b, b, 17 + b],
    )
    function(
        "absolute",
        f"FUNC {f} {p}\nIF ( LT {p} 0 )\nRETURN ( NEG {p} )\nELSE\nRETURN {p}\nEND\nEND_FUNC\n",
        f"def {f}({p}):\n    if {p} < 0:\n        return -{p}\n    else:\n        return {p}\n",
        [[-17], [-1], [0], [8]],
        [17, 1, 0, 8],
    )
    function(
        "maximum",
        f"FUNC {f} {p} {q}\nIF ( GT {p} {q} )\nRETURN {p}\nELSE\nRETURN {q}\nEND\nEND_FUNC\n",
        f"def {f}({p}, {q}):\n    if {p} > {q}:\n        return {p}\n    else:\n        return {q}\n",
        pairs,
        [max(u, v) for u, v in pairs],
    )
    function(
        "even",
        f"FUNC {f} {p}\nRETURN ( EQ ( MOD {p} 2 ) 0 )\nEND_FUNC\n",
        f"def {f}({p}):\n    return {p} % 2 == 0\n",
        [[-3], [0], [2], [7]],
        [False, True, True, False],
    )
    module(
        "for_print",
        f"FOR {i} RANGE {a} {a + 3}\nPRINT {i}\nEND\n",
        f"for {i} in range({a}, {a + 3}):\n    print({i})\n",
        "".join(f"{n}\n" for n in range(a, a + 3)),
        {i: a + 2},
    )
    function(
        "sum_range",
        f"FUNC {f} {p}\nLET {total} 0\nFOR {i} RANGE 0 {p}\nADD {total} {i}\nEND\nRETURN {total}\nEND_FUNC\n",
        f"def {f}({p}):\n    {total} = 0\n    for {i} in range(0, {p}):\n        {total} += {i}\n    return {total}\n",
        [[0], [1], [2], [11]],
        [0, 0, 1, 55],
    )
    function(
        "even_sum",
        f"FUNC {f} {p}\nLET {total} 0\nFOR {i} RANGE 0 {p}\nIF ( EQ ( MOD {i} 2 ) 0 )\nADD {total} {i}\nEND\nEND\nRETURN {total}\nEND_FUNC\n",
        f"def {f}({p}):\n    {total} = 0\n    for {i} in range(0, {p}):\n        if {i} % 2 == 0:\n            {total} += {i}\n    return {total}\n",
        [[0], [1], [7], [17]],
        [0, 0, sum(range(0, 7, 2)), sum(range(0, 17, 2))],
    )
    module(
        "if_print",
        f"IF ( LT {a} {b} )\nPRINT {a}\nELSE\nPRINT {b}\nEND\n",
        f"if {a} < {b}:\n    print({a})\nelse:\n    print({b})\n",
        f"{min(a, b)}\n",
    )
    function(
        "count",
        f"FUNC {f} {p}\nLET {total} 0\nFOR {i} RANGE 0 {p}\nIF ( GT {i} 3 )\nADD {total} 1\nEND\nEND\nRETURN {total}\nEND_FUNC\n",
        f"def {f}({p}):\n    {total} = 0\n    for {i} in range(0, {p}):\n        if {i} > 3:\n            {total} += 1\n    return {total}\n",
        [[0], [2], [4], [9]],
        [0, 0, 0, 5],
    )
    return cases


def code_cases_v2(groups=24):
    records = [record for group in range(groups) for record in group_cases(group)]
    lookup = {record["id"]: record for record in records}
    split = split_cases(
        [Case(r["id"], r["family"], r["source"], r["target"]) for r in records],
        seed=314159,
    )
    return {
        name: [lookup[case["id"]] for case in cases] for name, cases in split.items()
    }


def composition_cases():
    """New combinations and range boundaries, never passed to fit or model selection."""
    records = []
    for group, kind in (
        (901, "conditional_sum"),
        (902, "even_count"),
        (903, "negative_start"),
        (904, "less_than_count"),
    ):
        by_shape = {c["shape"]: c for c in group_cases(group)}
        original = by_shape[
            "count"
            if kind in {"conditional_sum", "less_than_count"}
            else "even_sum"
            if kind == "even_count"
            else "sum_range"
        ]
        case = {
            **original,
            "id": f"composition-{kind}",
            "family": f"composition-{group}",
            "shape": kind,
        }
        if kind == "conditional_sum":
            tokens = case["source"].splitlines()[2].split()
            loop = tokens[1]
            add_line = case["source"].splitlines()[4]
            total = add_line.split()[1]
            case["source"] = case["source"].replace(
                f"ADD {total} 1", f"ADD {total} {loop}"
            )
            case["target"] = case["target"].replace(
                f"{total} += 1", f"{total} += {loop}"
            )
            expected = {
                **case["expected"],
                "returns": [
                    sum(range(4, call["args"][0])) for call in case["expected"]["calls"]
                ],
            }
        elif kind == "even_count":
            loop = case["source"].splitlines()[2].split()[1]
            total = case["source"].splitlines()[1].split()[1]
            case["source"] = case["source"].replace(
                f"ADD {total} {loop}", f"ADD {total} 1"
            )
            case["target"] = case["target"].replace(
                f"{total} += {loop}", f"{total} += 1"
            )
            expected = {
                **case["expected"],
                "returns": [
                    (call["args"][0] + 1) // 2 for call in case["expected"]["calls"]
                ],
            }
        elif kind == "negative_start":
            case["source"] = case["source"].replace("RANGE 0 ", "RANGE -2 ")
            case["target"] = case["target"].replace("range(0, ", "range(-2, ")
            expected = {
                **case["expected"],
                "returns": [
                    sum(range(-2, call["args"][0]))
                    for call in case["expected"]["calls"]
                ],
            }
        else:
            case["source"] = case["source"].replace("( GT ", "( LT ")
            case["target"] = case["target"].replace(" > 3:", " < 3:")
            expected = {
                **case["expected"],
                "returns": [
                    sum(i < 3 for i in range(call["args"][0]))
                    for call in case["expected"]["calls"]
                ],
            }
        case["expected"] = expected
        validate(case["source"])
        compile(case["target"], "<composition>", "exec")
        records.append(case)
    return records
