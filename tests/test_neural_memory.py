from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.nl_data import nl_cases
from flybrain.graph import Graph
from flybrain.sequence_state import SequenceState
from models.fly_code_v2 import FlyCodeV2, CODE_VOCABULARY
from models.fly_nl import FlyNL, NL_VOCABULARY
from models.train_nl import nl_teacher


ROOT = Path(__file__).resolve().parents[1]


class NeuralMemoryTest(unittest.TestCase):
    def test_both_v2_models_never_call_teachers_or_qwen(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        with patch(
            "models.train_nl.nl_teacher", side_effect=AssertionError("teacher bypass")
        ) as teacher:
            with patch(
                "code_generator.request_json",
                side_effect=AssertionError("external model"),
            ) as qwen:
                FlyNL(graph).generate("37 을 출력해", max_actions=2)
                FlyCodeV2(graph).generate("PRINT 37", max_actions=2)
                teacher.assert_not_called()
                qwen.assert_not_called()

    def test_ir_input_invariant_to_unseen_payloads_and_values_only_travel_by_copy(self):
        brain = FlyCodeV2(Graph.from_csv(ROOT / "flywire_sample.csv"))
        first = brain.encoder.encode(b"LET alpha 123", 1, 0)
        second = brain.encoder.encode("LET 처음_보는_이름 -783".encode(), 1, 0)
        np.testing.assert_array_equal(first, second)
        # UNKNOWN is an output transport marker, not a reserved IR identifier.
        np.testing.assert_array_equal(
            first, brain.encoder.encode(b"LET UNKNOWN -783", 1, 0)
        )
        raw = "LET 처음_보는_이름 -783".encode()
        state = SequenceState([raw], CODE_VOCABULARY)
        self.assertEqual(
            state.apply(len(CODE_VOCABULARY.literals) + 1)[0].decode(), "처음_보는_이름"
        )

    def test_memory_context_never_selects_or_inserts_ir_text(self):
        v = NL_VOCABULARY
        state = SequenceState(['변수 "새_이름" 에 37 을 저장해'.encode()], v)
        # Context 0 is only a numeric input stimulus. The driver knows no meaning.
        emitted = []
        for action in [
            v.context_start,
            v.bind,
            len(v.literals) + v.max_spans + 1,
            v.begin_text,
            v.literals.index(b"LET"),
            v.register_start,
            v.next_line,
            0,
        ]:
            value, _ = state.apply(action)
            emitted.append(value)
        self.assertEqual(b"".join(emitted).decode(), "LET새_이름")
        # No spaces, correction, or template are inserted despite malformed IR.

    def test_all_teacher_memory_bindings_replay_exactly_without_model_rules(self):
        for case in nl_cases()["train"][:36]:
            actions = nl_teacher(case["source"], case["target"], NL_VOCABULARY)
            state = SequenceState(case["source"].encode().splitlines(), NL_VOCABULARY)
            output = b"".join(state.apply(action)[0] for action in actions).decode()
            self.assertEqual(output, case["target"])

    def test_pointer_candidates_support_new_positions_and_never_reuse_bound_pointer(
        self,
    ):
        v = NL_VOCABULARY
        state = SequenceState(['변수 "새이름" 에 37 을 저장해'.encode()], v)
        state.apply(v.context_start + 1)
        state.apply(v.bind)
        chosen = len(v.literals) + v.max_spans + 1
        self.assertIn(chosen, state.copy_candidates())
        state.apply(chosen)
        self.assertNotIn(chosen, state.copy_candidates())
        self.assertIn(len(v.literals) + 3, state.copy_candidates())

    def test_no_graph_signal_cannot_generate_ir_from_context_decoder(self):
        model = FlyNL(Graph.from_csv(ROOT / "flywire_sample.csv"))
        result = model.generate(
            "37 을 출력해", transmission=False, max_actions=8, record=True
        )
        self.assertEqual(result.code, "")
        self.assertNotEqual(result.status, "eos")
        context = [event for event in result.trace if event["decision"] == "context"]
        self.assertEqual(len(context), NL_VOCABULARY.context_slots)
        self.assertTrue(all(event["positive_score"] == 0.5 for event in context))

    def test_register_stack_and_cursor_errors_remain_explicit_failures(self):
        v = CODE_VOCABULARY
        with self.assertRaisesRegex(ValueError, "stack_error"):
            SequenceState([b"END"], v).apply(v.pop)
        state = SequenceState([b"PRINT 1"], v)
        state.apply(v.next_line)
        with self.assertRaisesRegex(ValueError, "cursor_overrun"):
            state.apply(v.next_line)
        with self.assertRaisesRegex(ValueError, "invalid_copy"):
            state.copy(5, len(v.literals))
