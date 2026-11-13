import unittest
from unittest.mock import patch

from evaluation.oracles import grade_code
from evaluation.sandbox import isolation_status, run_python


class SandboxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.available = run_python("print(1)")["status"] == "ok"

    def require_sandbox(self):
        if not self.available:
            self.skipTest("OS-isolated bubblewrap Python runtime unavailable; no fallback")

    def test_missing_os_isolation_never_falls_back_to_host_execution(self):
        with patch("evaluation.sandbox.shutil.which", return_value=None):
            self.assertEqual(run_python("print(1)")["status"], "sandbox_unavailable")

    def test_observations_and_function_boundary_inputs(self):
        self.require_sandbox()
        result = run_python("x=7\nprint(x)\ndef f(n):\n    return n*(n+1)//2\n",
                            variables=["x"], calls=[{"name": "f", "args": [n]} for n in (0, 1, 10, 101)])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stdout"], "7\n")
        self.assertEqual(result["variables"], {"x": 7})
        self.assertEqual(result["returns"], [0, 1, 55, 5151])

    def test_actual_network_filesystem_environment_and_resource_isolation(self):
        self.require_sandbox()
        status = isolation_status()
        for check in ("network_blocked", "runtime_readonly", "host_workspace_hidden", "host_home_hidden",
                      "environment_cleared"):
            self.assertTrue(status[check], status)
        self.assertEqual(status["process_limit"], 1)
        self.assertEqual(status["cpu_limit_seconds"], 1)
        self.assertEqual(status["memory_limit_bytes"], 256 * 1024 * 1024)

    def test_import_io_and_attribute_escape_are_rejected(self):
        self.require_sandbox()
        for code in ("import socket", "import os", 'open("/workspace/secret")', "().__class__"):
            with self.subTest(code=code):
                self.assertIn(run_python(code)["status"], {"policy_error", "runtime_error"})

    def test_infinite_loop_and_output_bomb_are_bounded(self):
        self.require_sandbox()
        self.assertEqual(run_python("while True:\n    pass", timeout=.15)["status"], "timeout")
        self.assertEqual(run_python('print("x" * 70000)')["status"], "output_limit")

    def test_syntax_is_insufficient_and_wrong_values_are_not_repaired(self):
        self.require_sandbox()
        expected = {"stdout": "9\n", "variables": {"x": 9}}
        passed = grade_code("x=9\nprint(x)\n", expected)
        wrong = grade_code("x=8\nprint(x)\n", expected)
        self.assertTrue(passed["semantic_ok"])
        self.assertTrue(wrong["syntax_ok"])
        self.assertFalse(wrong["semantic_ok"])
        self.assertEqual(wrong["stdout"], "8\n")
