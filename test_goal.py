import subprocess
import sys
import unittest

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
