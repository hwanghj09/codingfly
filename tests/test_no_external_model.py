import builtins
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from flybrain.graph import Graph
from flybrain.policy import ForbiddenDependency, fly_only, verify_generation_tree
from models.fly_code import FlyCode


ROOT = Path(__file__).resolve().parents[1]


class NoExternalModelTest(unittest.TestCase):
    def test_fly_only_inference_cannot_fall_back_to_qwen(self):
        model = FlyCode(Graph.from_csv(ROOT / "flywire_lab.csv"))
        with patch("code_generator.request_json", side_effect=AssertionError("external model called")) as qwen:
            generation = model.generate("LET x 1\n", max_actions=4)
            self.assertIn(generation.status, {"eos", "invalid_copy", "max_tokens"})
            qwen.assert_not_called()

    def test_import_and_network_guards_restore_normal_context(self):
        original = builtins.__import__
        with fly_only():
            with self.assertRaises(ForbiddenDependency):
                __import__("code_generator")
            with self.assertRaises(ForbiddenDependency):
                socket.socket()
        self.assertIs(builtins.__import__, original)
        with socket.socket():
            pass

    def test_static_generation_dependency_contract(self):
        checked = verify_generation_tree()
        self.assertIn("models/fly_code.py", checked)
        for source in ("from goal_model import parse_goal", "import re", "import requests", "exec('x=1')",
                       "from program_model import ProgramModel", "import importlib"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "models"
                path.mkdir()
                (path / "bad.py").write_text(source)
                with self.assertRaises(ForbiddenDependency):
                    verify_generation_tree(directory)
