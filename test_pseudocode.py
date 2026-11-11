import json
import unittest

from fly import HERE, Fly
from lab import GRAPH
from program_model import ProgramModel
from pseudocode import translate


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
        checks = json.loads((HERE / "program_checks.json").read_text(encoding="utf-8"))
        scores = self.model.evaluate(checks)
        self.assertEqual(scores["correct"], len(checks))
        self.assertEqual(scores["covered"], list(range(1, 49)))

        expected = [([0, (1, "짝수"), (3, "홀수"), (4, "영")], 3, True), (9, 6, {"계산": 3}), 11]
        for check, value in zip(checks, expected):
            with self.subTest(check=check["title"]):
                scope = {}
                exec(self.model.generate(check["source"])["code"], scope)
                self.assertEqual(scope["평가"], value)

        disconnected = Fly([], GRAPH, extended_context=True)
        disconnected.edges.clear()
        self.assertLess(ProgramModel(disconnected).evaluate(checks)["correct"], scores["correct"])

    def test_errors_identify_the_input_line(self):
        for source in ("x ← 1\n미지원 명령", "함수 f()\n반환 1", "x ← 1\n값 ← 2; 출력 3"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, r"^2줄"):
                translate(source)
