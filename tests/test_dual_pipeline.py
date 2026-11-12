from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flybrain.graph import Graph
from flybrain.policy import ForbiddenDependency, verify_generation_tree
from models.fly_code import Generation
from models.fly_code_v2 import FlyCodeV2
from models.fly_nl import FlyNL
from models.pipeline import FlyPipeline
from models.train_nl import ir_equal, ir_structure


ROOT = Path(__file__).resolve().parents[1]


class DualPipelineTest(unittest.TestCase):
    def setUp(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        self.a, self.b = FlyNL(graph), FlyCodeV2(graph)
        self.pipeline = FlyPipeline(self.a, self.b)

    def test_invalid_and_uncertain_ir_never_reaches_code_model(self):
        for source, status in [("UNKNOWN\n", "uncertain"), ("LET\n", "invalid_ir")]:
            with patch.object(
                self.a, "generate", return_value=Generation(source, "eos", (0,), ())
            ):
                with patch.object(
                    self.b, "generate", side_effect=AssertionError("B must not run")
                ):
                    result = self.pipeline.run("request")
                    self.assertEqual(result["status"], status)
                    self.assertEqual(result["code"], "")

    def test_valid_but_wrong_code_remains_wrong_and_is_never_repaired(self):
        with patch.object(
            self.a, "generate", return_value=Generation("PRINT 9\n", "eos", (0,), ())
        ):
            with patch.object(
                self.b,
                "generate",
                return_value=Generation("print(8)\n", "eos", (0,), ()),
            ):
                result = self.pipeline.run(
                    "request", expected={"stdout": "9\n", "variables": {}}
                )
                if (
                    result.get("verification", {}).get("status")
                    == "sandbox_unavailable"
                ):
                    self.skipTest("OS isolation unavailable")
                self.assertEqual(result["status"], "test_failure")
                self.assertEqual(result["code"], "print(8)\n")
                self.assertEqual(result["verification"]["stdout"], "8\n")

    def test_independent_roles_synapses_and_trace_identifiers(self):
        with self.assertRaises(ValueError):
            FlyPipeline(self.b, self.a)
        with patch.object(
            self.a, "generate", return_value=Generation("UNKNOWN\n", "eos", (0,), ())
        ):
            first = self.pipeline.run("request")
            second = self.pipeline.run("request")
        self.assertNotEqual(first["trace_id"], second["trace_id"])
        self.assertFalse(np.shares_memory(self.a.brain.weights, self.b.brain.weights))
        self.a.brain.weights = self.b.brain.weights
        with self.assertRaises(ValueError):
            FlyPipeline(self.a, self.b)

    def test_ir_shape_and_exact_values_are_distinct_metrics(self):
        self.assertEqual(ir_structure("LET x 2\n"), ir_structure("LET unseen 87\n"))
        self.assertFalse(ir_equal("LET x 2\n", "LET x 87\n"))
        self.assertTrue(ir_equal("LET  x  2\n", "LET x 2\n"))

    def test_teacher_imports_cannot_enter_generation_path(self):
        for statement in (
            "from evaluation.curriculum_v2 import aligned_code_actions",
            "from models.train_nl import nl_teacher",
        ):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "models"
                target.mkdir()
                (target / "bad.py").write_text(statement)
                with self.assertRaises(ForbiddenDependency):
                    verify_generation_tree(directory)
