import unittest

from pseudocode import translate


class PseudocodeTest(unittest.TestCase):
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
        code = translate("""FUNCTION total(values):
    SET result = 0
    FOR EACH value IN values
        WHILE value > 0 DO
            누적 result ← value
            value ← value - 1
    RETURN result
answer ← total([2, 3])""")
        scope = {}
        exec(code, scope)
        self.assertEqual(scope["answer"], 9)

    def test_errors_identify_the_input_line(self):
        for source in ("x ← 1\n미지원 명령", "함수 f()\n반환 1", "x ← 1\n값 ← 2; 출력 3"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, r"^2줄"):
                translate(source)
