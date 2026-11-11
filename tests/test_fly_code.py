import json
from pathlib import Path
import tempfile
import unittest

from evaluation.code_data import code_cases
from flybrain.decoder import whitespace_spans
from flybrain.graph import Graph
from models.fly_code import FlyCode
from models.train_code import require_p2


ROOT = Path(__file__).resolve().parents[1]


class FlyCodeTest(unittest.TestCase):
    def test_supervised_data_decodes_to_valid_ir_python_pairs(self):
        model = FlyCode(Graph.from_csv(ROOT / "flywire_lab.csv"))
        for split in code_cases().values():
            for case in split:
                source = case["source"].encode()
                code = b"".join(model.decoder.emit(a, source, whitespace_spans(source)) for a in case["actions"])
                self.assertEqual(code.decode(), case["target"])
                compile(code, "<teacher>", "exec")

    def test_training_and_restored_autoregressive_trace_match(self):
        graph = Graph.from_csv(ROOT / "flywire_lab.csv")
        model = FlyCode(graph)
        example = code_cases()["train"][0]
        before = model.brain.weights_hash
        for stimulus, action in model.teacher_examples(example["source"], example["actions"]):
            model.train_example(stimulus, action)
        self.assertNotEqual(model.brain.weights_hash, before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "code.json"
            contract = {"data": "unit-test"}
            model.save(path, contract=contract)
            restored, _ = FlyCode.restore(path, graph, contract=contract)
            self.assertEqual(model.generate("PRINT 99", record=True), restored.generate("PRINT 99", record=True))

    def test_failed_p2_report_cannot_start_p3(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"phase": "P2", "gate": {"passed": False}}))
            with self.assertRaisesRegex(ValueError, "stop before P3"):
                require_p2(path)
            path.write_text(json.dumps({"phase": "P2", "gate": {"passed": True}, "runs": []}))
            with self.assertRaisesRegex(ValueError, "five paired"):
                require_p2(path)

    def test_generation_bounds_and_bad_teacher_sequences(self):
        model = FlyCode(Graph.from_csv(ROOT / "flywire_lab.csv"))
        for actions in ([], [8, 1], [0, 8, 0]):
            with self.assertRaises(ValueError):
                model.teacher_examples("PRINT 1", actions)
        with self.assertRaises(ValueError):
            model.generate("", max_actions=4)
        self.assertEqual(model.generate(" ".join(["x"] * 20)).status, "input_limit")
