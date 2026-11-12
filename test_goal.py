import subprocess
import sys
import unittest
from unittest.mock import patch

from code_generator import generate_code
from fly import Fly
from goal_model import parse_goal, solve
from lab import GRAPH
from program_model import ProgramModel


class GoalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ProgramModel(Fly([], GRAPH, extended_context=True))

    def test_different_solutions_produce_the_same_output(self):
        result = solve(self.model, "hello를 출력해")
        direct, function = result["answers"][:2]
        self.assertTrue(direct["accepted"] and function["accepted"])
        self.assertEqual(direct["output"], function["output"])
        self.assertNotEqual(direct["code"], function["code"])
        self.assertIn("함수", function["methods"])

    def test_assignment_checks_final_values_instead_of_prints(self):
        request = "두 변수 a,b를 각각 1, 3으로 지정해"
        result = solve(self.model, request)
        self.assertEqual(result["kind"], "변수 값")
        self.assertEqual(result["expected"], "a = 1, b = 3")
        self.assertTrue(all(answer["accepted"] for answer in result["answers"]))
        self.assertTrue(solve(self.model, request, "a, b = 1, 3")["answers"][0]["accepted"])
        self.assertFalse(solve(self.model, request, "print(1, 3)")["answers"][0]["accepted"])

    def test_sum_function_checks_the_returned_value(self):
        request = "1부터 10까지 합을 구하는 함수식 만들어"
        result = solve(self.model, request)
        self.assertEqual(result["expected"], "55")
        self.assertEqual(result["kind"], "함수 반환값")
        self.assertTrue(all(answer["accepted"] for answer in result["answers"]))
        self.assertTrue(solve(self.model, request, "def total():\n    return 55")["answers"][0]["accepted"])
        self.assertFalse(solve(self.model, request, "def total():\n    print(55)")["answers"][0]["accepted"])
        printing = solve(self.model, "함수를 이용해서 1부터 10까지 출력해")
        self.assertEqual(printing["kind"], "출력")
        self.assertEqual(printing["expected"], "\n".join(map(str, range(1, 11))))

    def test_general_generation_is_not_graded_as_correct(self):
        goal = "목록에서 중복을 없애고 원래 순서를 유지하는 함수를 작성해"
        code = "def unique(items):\n    return list(dict.fromkeys(items))\n"
        with patch("goal_model.generate_code", return_value=(code, "로컬 모델")) as generator:
            result = solve(self.model, goal)
        generator.assert_called_once_with(goal)
        self.assertEqual(result["answers"][0]["code"], code)
        self.assertIsNone(result["answers"][0]["accepted"])

    def test_local_model_code_block_with_surrounding_prose(self):
        code = "def rank(items):\n    return sorted(items, key=lambda item: item['score'], reverse=True)\n"
        content = f"다음과 같이 작성할 수 있습니다.\n\n```python\n{code}```\n\n점수가 높은 순서로 반환합니다."
        response = {"choices": [{"message": {"content": content}}]}
        with patch("code_generator.request_json", return_value=response):
            self.assertEqual(generate_code("점수로 내림차순 정렬하는 함수를 작성해")[0], code)

    def test_requested_method_must_contribute_to_every_output(self):
        goal = "함수를 이용해서 hello를 출력해"
        cases = {
            "print('hello')": False,
            "def f():\n    pass\nf()\nprint('hello')": False,
            "def f():\n    print('h', end='')\nf()\nprint('ello')": False,
            "def f():\n    return 'hello'\nprint(f())": True,
            "def f():\n    print('hello')\nf()": True,
            "def f():\n    print('wrong')\nf()": False,
        }
        for code, expected in cases.items():
            with self.subTest(code=code):
                self.assertEqual(solve(self.model, goal, code)["answers"][0]["accepted"], expected)
        self.assertFalse(solve(self.model, "for문으로 hello를 출력해",
                               "for x in [1]:\n    pass\nprint('hello')")["answers"][0]["accepted"])

    def test_generated_candidates_match_real_python(self):
        goals = {
            "for문을 이용해서 1부터 5까지의 합계를 출력해": "15\n",
            "while문으로 hello를 3번 출력해": "hello\nhello\nhello\n",
            "for문을 이용해서 1부터 1까지 짝수만 출력해": "",
            "함수와 리스트 내포를 사용해서 1부터 3까지 출력해": "1\n2\n3\n",
            "while문을 사용해서 [3, 1, 2, 1]을 내림차순으로 정렬해서 출력해": "[3, 2, 1, 1]\n",
            "변수를 이용해서 'hello'를 출력해": "hello\n",
        }
        for goal, expected in goals.items():
            with self.subTest(goal=goal):
                answers = [answer for answer in solve(self.model, goal)["answers"] if answer["accepted"]]
                self.assertTrue(answers)
                for answer in answers:
                    result = subprocess.run([sys.executable, "-X", "utf8", "-c", answer["code"]],
                                            capture_output=True, encoding="utf-8", timeout=5, check=True)
                    self.assertEqual(result.stdout, expected)

    def test_unknown_method_and_unbounded_or_unsupported_code_fail(self):
        with self.assertRaisesRegex(ValueError, "방법"):
            parse_goal("재귀를 이용해서 hello를 출력해")
        with self.assertRaisesRegex(ValueError, "따옴표"):
            parse_goal("파일에서 내용을 읽어서 출력해")
        for code in ("while True:\n    pass", "import math\nprint('hello')", "print = 0\nprint('hello')",
                     "x = 'hello'\ndef f():\n    print(x)\n    x = 'other'\nf()"):
            self.assertFalse(solve(self.model, "hello를 출력해", code)["answers"][0]["accepted"])
