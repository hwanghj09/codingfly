import json
import math
import random
import subprocess
import sys
import unittest
from collections import Counter
from itertools import accumulate

from algorithm_phrases import CHECKS, GENERALIZATION_CHECKS, training_phrases
from fly import HERE, Fly
from lab import GRAPH
from program_model import ProgramModel
from pseudocode import translate


def generated_cases(name):
    rng = random.Random(2026)
    for _ in range(64):
        values = [rng.randint(-9, 9) for _ in range(rng.randrange(13))]
        if name == "gcd":
            pair = [rng.randint(-10_000, 10_000), rng.randint(-10_000, 10_000)]
            yield pair, math.gcd(*pair)
        elif name == "search":
            ordered = sorted(set(values))
            target = rng.randint(-10, 10)
            yield [ordered, target], ordered.index(target) if target in ordered else -1
        elif name == "insertion_sort":
            yield [values], sorted(values)
        elif name == "prefix_sums":
            yield [values], list(accumulate(values))
        elif name == "merge_sorted":
            other = [rng.randint(-9, 9) for _ in range(rng.randrange(13))]
            yield [sorted(values), sorted(other)], sorted(values + other)
        elif name == "is_palindrome":
            text = "".join(rng.choice("ab한글") for _ in values)
            if rng.choice((True, False)):
                text += text[::-1]
            yield [text], text == text[::-1]
        elif name == "maximum_subarray":
            yield [values], max((sum(values[start:end]) for start in range(len(values))
                                 for end in range(start + 1, len(values) + 1)), default=0)
        elif name == "frequencies":
            yield [values], dict(Counter(values))
        else:
            raise ValueError(f"Unknown algorithm: {name}")


class PseudocodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fly = Fly([], GRAPH, extended_context=True)
        cls.model = ProgramModel(cls.fly)

    def test_nested_statements_generate_a_working_program(self):
        code = translate("""함수 classify(limit)
    결과 ← []
    반복 i = 1 .. limit
        만약 i % 6 == 0
            호출 결과.append('둘 다')
        아니면 만약 i % 2 == 0
            건너뛰기
        아니면:
            호출 결과.append(i)
    반환 결과

값 ← classify(6)""")
        scope = {}
        exec(code, scope)
        self.assertEqual(scope["값"], [1, 3, 5, "둘 다"])

    def test_english_loop_and_while_compose(self):
        source = """FUNCTION total(values):
    SET result = 0
    FOR EACH value IN values
        WHILE value > 0 DO
            누적 result ← value
            value ← value - 1
    RETURN result
answer ← total([2, 3])"""
        code = self.model.generate(source)["code"]
        scope = {}
        exec(code, scope)
        self.assertEqual(scope["answer"], 9)

    def test_whole_program_evaluation_covers_every_syntax_lesson(self):
        checks = [*self.checks("program_checks.json"), *self.checks("advanced_checks.json")]
        scores = self.model.evaluate(checks)
        self.assertEqual(scores["correct"], len(checks))
        self.assertEqual(scores["covered"], list(range(1, 81)))

        expected = [([0, (1, "짝수"), (3, "홀수"), (4, "영")], 3, True), (9, 6, {"계산": 3}), 11]
        for check, value in zip(checks, expected):
            with self.subTest(check=check["title"]):
                scope = {}
                exec(self.model.generate(check["source"])["code"], scope)
                self.assertEqual(scope["평가"], value)

        disconnected = Fly([], GRAPH, extended_context=True)
        disconnected.edges.clear()
        self.assertLess(ProgramModel(disconnected).evaluate(checks)["correct"], scores["correct"])

    def test_complete_syntax_lessons_and_examples(self):
        courses = self.model.construct_results()
        self.assertEqual(len(courses), 100)
        self.assertEqual(sum(len(course["examples"]) for course in courses), 300)
        by_title = {course["title"]: course for course in courses}
        self.assertEqual(len(by_title), len(courses))
        for course in courses:
            with self.subTest(lesson=course["title"]):
                self.assertTrue(all(example["correct"] for example in course["examples"]))

        self.assertEqual(self.run_generated(by_title["print(값)"]["examples"][0]["source"]), "안녕하세요\n")
        for title, assertion in (
            ("sorted(자료)", "assert 결과 == [1, 3, 4]"),
            ("list.append(값)", "assert 대상 == [1, 2, 3]"),
            ("str.split(구분자)", "assert 결과 == ['a', 'b', 'c']"),
            ("dict.get(키, 기본값)", "assert 결과 == 2"),
            ("json.loads(문자열)", "assert 결과 == {'이름': '민수', '점수': 90}"),
        ):
            with self.subTest(lesson=title):
                self.run_generated(by_title[title]["examples"][0]["source"], f"\n{assertion}\n")

    @staticmethod
    def checks(name):
        return json.loads((HERE / name).read_text(encoding="utf-8"))

    def run_generated(self, source, assertions=""):
        code = self.model.generate(source)["code"]
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code + assertions],
                                capture_output=True, encoding="utf-8", timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_phrase_accuracy_on_unseen_wording(self):
        cases = [(phrase, action) for split in (CHECKS, GENERALIZATION_CHECKS)
                 for action, phrases in split.items() for phrase in phrases]
        self.assertTrue({phrase for phrase, _ in cases}.isdisjoint(
            phrase for phrases in training_phrases().values() for phrase in phrases))
        for contrast, expected in ((False, 54), (True, len(cases))):
            self.assertEqual(sum(self.model.algorithm.ranks(phrase, contrast=contrast)[0][0] == action
                                 for phrase, action in cases), expected)

    def test_numbered_algorithms_match_expected_behavior(self):
        for check in [*self.checks("algorithm_checks.json"), *self.checks("algorithm_transfer.json")]:
            with self.subTest(algorithm=check["title"]):
                cases = [*check["cases"], *generated_cases(check["name"])]
                assertions = (f"\nfor args, expected in {cases!r}:\n"
                              f"    assert {check['name']}(*args) == expected, args\n")
                self.run_generated(check["source"], assertions)

    def test_supplied_examples_produce_expected_output(self):
        for check in self.checks("example_checks.json"):
            with self.subTest(example=check["title"]):
                self.assertEqual(self.run_generated(check["source"]), check["output"])

    def test_errors_identify_the_input_line(self):
        for source in ("x ← 1\n미지원 명령", "함수 f()\n반환 1", "x ← 1\n값 ← 2; 출력 3"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, r"^2줄"):
                translate(source)
