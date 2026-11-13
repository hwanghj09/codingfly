import json
import sys
import unittest
import zipfile
import io

from web.build import model_archive, verified_release
from web.runner import execute


class BrowserReleaseTest(unittest.TestCase):
    def test_release_uses_evaluated_weights_without_teachers_or_test_answers(self):
        release = verified_release()
        archive = model_archive(release)
        self.assertEqual(archive, model_archive(release))
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            names = z.namelist()
            self.assertIn("checkpoints/fly-nl.json", names)
            self.assertIn("checkpoints/fly-code.json", names)
            self.assertIn("docs/IR_SPEC.md", names)
            self.assertIn("models/pipeline.py", names)
            self.assertFalse(
                any(
                    "train_" in name
                    or "dataset" in name
                    or name.startswith("evaluation/")
                    for name in names
                )
            )
            self.assertEqual(
                json.loads(z.read("browser_manifest.json"))["evaluation"]["correct"], 96
            )

    def test_browser_runner_observations_and_explicit_function_call(self):
        actual = json.loads(execute({"code": "합계 = -13\n합계 += 4\nprint(합계)\n"}))
        self.assertEqual(actual["stdout"], "-9\n")
        self.assertEqual(actual["variables"], {"합계": -9})
        actual = json.loads(
            execute(
                {
                    "code": "def add(a,b):\n    return a+b\n",
                    "function_name": "add",
                    "args": [7, 8],
                }
            )
        )
        self.assertEqual(actual["return_value"], 15)
        self.assertTrue(actual["called"])

    def test_browser_runner_blocks_imports_attributes_and_restores_budget(self):
        previous = sys.gettrace()
        for code in ("import js\n", "print((1).__class__)\n"):
            with self.assertRaises(ValueError):
                execute({"code": code})
        with self.assertRaises(TimeoutError):
            execute({"code": "while True:\n    pass\n"})
        self.assertIs(previous, sys.gettrace())
        with self.assertRaises(ValueError):
            execute({"code": 'print("x"*20000)\n'})
        with self.assertRaises(ValueError):
            execute({"code": "for i in range(10001):\n    pass\n"})
