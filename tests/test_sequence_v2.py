from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation.curriculum_v2 import group_cases, aligned_code_actions
from evaluation.oracles import grade_code
from flybrain.batch import EdgeAdam, loss_gradient
from flybrain.clock_encoder import ClockEncoder
from flybrain.dynamics import Brain
from flybrain.graph import Graph
from flybrain.plasticity import supervised_step
from flybrain.sequence_io import lexical_spans
from flybrain.sequence_state import SequenceState
from models.fly_code_v2 import CODE_VOCABULARY, FlyCodeV2
from models.neural_sequence import NeuralSequence


ROOT = Path(__file__).resolve().parents[1]


class BatchedGradientTest(unittest.TestCase):
    def test_sparse_batch_matches_scalar_probabilities_and_gradients(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        brain = Brain(graph, 4)
        encoder = ClockEncoder(64, brain.input_nodes)
        stimuli = np.array([encoder.encode(b"LET x 3", step, 0) for step in range(3)])
        targets = np.array([0, 2, 1])
        _, gradient, probabilities = loss_gradient(brain, stimuli, targets)
        reference = []
        for i, stimulus in enumerate(stimuli):
            scalar = Brain(graph, 4)
            np.testing.assert_allclose(
                probabilities[i], scalar.forward(stimulus)[0], rtol=1e-12
            )
            supervised_step(scalar, stimulus, int(targets[i]))
            reference.append(-scalar.eligibility)
        np.testing.assert_allclose(
            gradient, np.mean(reference, axis=0), rtol=1e-11, atol=1e-13
        )

    def test_optimizer_frozen_and_exact_resume(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        brain = Brain(graph, 4)
        encoder = ClockEncoder(64, brain.input_nodes)
        stimuli = np.array([encoder.encode(b"PRINT 3", 0, 0)])
        optimizer = EdgeAdam(brain)
        before = brain.weights_hash
        optimizer.update(stimuli, np.array([1]), frozen=True)
        self.assertEqual(before, brain.weights_hash)
        self.assertEqual(optimizer.step, 0)
        optimizer.update(stimuli, np.array([1]))
        clone = Brain(graph, 4)
        clone.weights = brain.weights.copy()
        resumed = EdgeAdam(clone)
        resumed.restore(optimizer.state())
        optimizer.update(stimuli, np.array([2]))
        resumed.update(stimuli, np.array([2]))
        np.testing.assert_array_equal(brain.weights, clone.weights)


class SequenceV2Test(unittest.TestCase):
    def test_lexical_copy_preserves_quotes_and_adjacent_expression_boundaries(self):
        raw = 'PRINT "안녕, 친구"\nIF (LT 값 -3)\n'.encode()
        values = [raw[s.start : s.end].decode() for s in lexical_spans(raw)]
        self.assertEqual(
            values, ["PRINT", '"안녕, 친구"', "IF", "(", "LT", "값", "-3", ")"]
        )

    def test_fixed_encoder_excludes_readouts_and_uses_real_output_history(self):
        brain = Brain(Graph.from_csv(ROOT / "flywire_sample.csv"), 4)
        encoder = ClockEncoder(64, brain.input_nodes)
        first = encoder.encode(b"PRINT 37", 3, 1, 2, [1, 2])
        self.assertTrue(
            np.array_equal(first, encoder.encode(b"PRINT 37", 3, 1, 2, [1, 2]))
        )
        self.assertFalse(
            np.array_equal(first, encoder.encode(b"PRINT 37", 3, 1, 2, [1, 3]))
        )
        self.assertFalse(first[list(brain.output_nodes)].any())

    def test_twenty_four_teachers_are_executable_with_boundary_function_inputs(self):
        cases = group_cases(91)
        self.assertEqual(len(cases), 24)
        for case in cases:
            actions = aligned_code_actions(
                case["source"], case["target"], CODE_VOCABULARY
            )
            state = SequenceState(case["source"].encode().splitlines(), CODE_VOCABULARY)
            actual = b"".join(state.apply(action)[0] for action in actions).decode()
            self.assertEqual(actual, case["target"])
            grade = grade_code(actual, case["expected"])
            if grade["status"] == "sandbox_unavailable":
                self.skipTest("OS isolation unavailable")
            self.assertTrue(grade["semantic_ok"], (case["shape"], grade))

    def test_neural_digit_trace_and_checkpoint_role_isolation(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        model = FlyCodeV2(graph)
        cut = model.generate("PRINT 17", transmission=False, record=True)
        self.assertEqual((cut.code, cut.status), ("", "premature_eos"))
        self.assertEqual(len(cut.trace), CODE_VOCABULARY.digits)
        for event in cut.trace:
            scores = event["neural"]["probabilities"]
            self.assertEqual(
                event["chosen"], max(range(len(scores)), key=scores.__getitem__)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "code.json"
            model.save(path, contract={"test": "v2"})
            resumed, _ = NeuralSequence.restore(
                path, graph, CODE_VOCABULARY, role="fly-code", contract={"test": "v2"}
            )
            self.assertEqual(
                model.generate("PRINT 17", max_actions=8),
                resumed.generate("PRINT 17", max_actions=8),
            )
            with self.assertRaises(ValueError):
                NeuralSequence.restore(
                    path, graph, CODE_VOCABULARY, role="fly-nl", contract={"test": "v2"}
                )
