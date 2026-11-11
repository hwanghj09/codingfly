import argparse
import csv
import hashlib
import json
import keyword
import math
import random
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
BLANK = "□"
LEXEMES = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"|[A-Za-z_]\w*|\d+")


def normalize(code):
    # shortcut: Simple strings, names and integers cover these exercises. Use a lexer for richer snippets.
    def replace(match):
        token = match.group()
        if keyword.iskeyword(token) or keyword.issoftkeyword(token):
            return token
        if token[0] in "'\"":
            return '"s"'
        return "0" if token[0].isdigit() else "n"

    return LEXEMES.sub(replace, code)


class Fly:
    def __init__(self, tokens, graph_path=HERE / "flywire_sample.csv", *, extended_context=False):
        with graph_path.open(newline="", encoding="utf-8") as file:
            edges = [(row["pre_root_id"], row["post_root_id"], int(row["syn_count"]))
                     for row in csv.DictReader(file)]
        sources = {node: index for index, node in enumerate(sorted({a for a, _, _ in edges}))}
        targets = {node: index for index, node in enumerate(sorted({b for _, b, _ in edges}))}
        incoming = Counter()
        for _, target, count in edges:
            incoming[target] += count
        self.edges = [(sources[a], targets[b], count / incoming[b]) for a, b, count in edges]
        self.node_count = len(sources.keys() | targets.keys())
        self.input_size = len(sources)
        self.output_size = len(targets)
        self.extended_context = extended_context
        self.weights = {token: [0.0] * (self.output_size + 1) for token in tokens}

    def activity(self, example):
        return self.project(*map(normalize, example.split(BLANK)))

    def project(self, left, right=""):
        sizes = (1, 2, 3, 5, 8, 12, 20) if self.extended_context else (1, 2, 3, 5, 8)
        features = [f"{side}{size}:{fragment}"
                    for size in sizes
                    for side, fragment in (("left", left[-size:]), ("right", right[:size]))]
        features.append(f"start:{left[:3]}")
        if self.extended_context:
            left_lines = left.split("\n")
            right_lines = right.split("\n")
            current_line = left_lines[-1]
            features.append(f"line:{current_line[:4]}")
            features.extend(f"{side}:symbol:{symbol}" for side, fragment in (("left", current_line), ("right", right))
                            for symbol in sorted(set(fragment) & set("()[]{}=:,\n")))
            for side, lines in (("left", left_lines), ("right", right_lines)):
                first = lines[0].lstrip()
                local = (lines[-1] if side == "left" else lines[0]).rstrip()
                features.extend([f"{side}:head:{first[:4]}", f"{side}:end:{local[-1:]}"] * 3)
                for depth, line in enumerate(reversed(lines[:-1]) if side == "left" else lines[1:], 1):
                    if depth > 2:
                        break
                    features.extend([f"{side}:line:{depth}:{line.lstrip()[:5]}"] * 3)
        return self.propagate(features)

    def propagate(self, features):
        inputs = [0.0] * self.input_size
        for feature in features:
            digest = hashlib.blake2s(feature.encode(), digest_size=4).digest()
            index, sign = divmod(int.from_bytes(digest, "big"), 2)
            inputs[index % self.input_size] += 1 if sign else -1

        outputs = [0.0] * self.output_size
        for source, target, weight in self.edges:
            outputs[target] += inputs[source] * weight
        activity = [0.5, *outputs]
        scale = math.hypot(*activity)
        return [value / scale for value in activity]

    def scores(self, activity):
        active = [(index, value) for index, value in enumerate(activity) if value]
        return {token: sum(weights[index] * value for index, value in active)
                for token, weights in self.weights.items()}

    def predict(self, activity):
        scores = self.scores(activity)
        best = max(scores, key=scores.__getitem__)
        return best if scores[best] > 0 else None

    def train(self, examples):
        rng = random.Random(42)
        batch = [(self.activity(example), expected) for example, expected in examples]
        for _ in range(20):
            rng.shuffle(batch)
            for activity, expected in batch:
                predicted = self.predict(activity)
                if predicted == expected:
                    continue
                for index, value in enumerate(activity):
                    self.weights[expected][index] += value
                    if predicted is not None:
                        self.weights[predicted][index] -= value

def load_lessons():
    lessons = json.loads((HERE / "lessons.json").read_text(encoding="utf-8"))
    if not lessons:
        raise ValueError("수업을 한 개 이상 넣어 주세요.")
    train_examples = {example for lesson in lessons for example in lesson["train"]}
    for lesson in lessons:
        if not lesson["token"] or BLANK in lesson["token"] or not lesson["train"] or not lesson["check"]:
            raise ValueError(f"정답 기호와 학습·평가 예문을 넣어 주세요. {lesson['title']}")
        for example in lesson["train"] + lesson["check"]:
            if example.count(BLANK) != 1:
                raise ValueError(f"빈칸을 하나만 넣어 주세요. {example!r}")
            compile(example.replace(BLANK, lesson["token"]), lesson["title"], "exec")
        if train_examples.intersection(lesson["check"]):
            raise ValueError(f"학습 예문과 평가 예문이 겹칩니다. {lesson['title']}")
    return lessons


def evaluate(fly, lessons):
    examples = [(example, lesson["token"]) for lesson in lessons for example in lesson["check"]]
    correct = sum(fly.predict(fly.activity(example)) == expected for example, expected in examples)
    return f"{correct}/{len(examples)}"


def main():
    lessons = load_lessons()
    parser = argparse.ArgumentParser(description="초파리 연결망으로 Python 문법 빈칸을 학습합니다.")
    parser.add_argument("--through", type=int, default=1, choices=range(1, len(lessons) + 1),
                        help="몇 번째 수업까지 학습할지 선택합니다. 기본값은 1입니다.")
    parser.add_argument("--fill", help="□ 하나를 포함한 Python 예문을 입력합니다.")
    args = parser.parse_args()
    if args.fill is not None and args.fill.count(BLANK) != 1:
        parser.error("--fill 예문에 □를 하나만 넣어 주세요.")

    fly = Fly([lesson["token"] for lesson in lessons])
    training = []
    for index, lesson in enumerate(lessons[:args.through], start=1):
        before = evaluate(fly, [lesson])
        training.extend((example, lesson["token"]) for example in lesson["train"])
        fly.train(training)
        after = evaluate(fly, [lesson])
        cumulative = evaluate(fly, lessons[:index])
        print(f"{index}과 {lesson['title']} {lesson['token']!r} | 새 예문 {before} → {after} | 누적 {cumulative}")

    example = args.fill if args.fill is not None else lessons[args.through - 1]["check"][0]
    predicted = fly.predict(fly.activity(example))
    print(f"\n입력 {example!r}\n예측 {predicted!r}")
    if predicted is not None:
        print(example.replace(BLANK, predicted))


if __name__ == "__main__":
    main()
