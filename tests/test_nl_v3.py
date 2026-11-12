import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation.accuracy_v3 import require_validation
from evaluation.nl_data_v3 import nl_cases_v3
from flybrain.clock_encoder import ClockEncoder
from flybrain.graph import Graph
from flybrain.policy import ForbiddenDependency, fly_only, verify_generation_tree
from flybrain.sequence_state import SequenceState
from models.fly_nl import FlyNL, NL_VOCABULARY
from models.neural_sequence import NeuralSequence
from models.train_nl import nl_teacher
from models.train_nl_v3 import require_unopened


class NLVersion3Test(unittest.TestCase):
    def test_context_is_invariant_to_word_order_but_pointer_io_stays_unchanged(self):
        graph = Graph.from_csv(
            Path(__file__).resolve().parents[1] / "flywire_sample.csv"
        )
        legacy, model = FlyNL(graph), FlyNL(graph, context_bag=True)
        first = SequenceState(
            ["지금 도시 500 의 날씨를 알려줘".encode()], NL_VOCABULARY
        )
        second = SequenceState(
            ["도시 991 의 날씨를 지금 알려줘".encode()], NL_VOCABULARY
        )
        for candidate in range(8):
            np.testing.assert_array_equal(
                first.context_stimulus(model.encoder, candidate),
                second.context_stimulus(model.encoder, candidate),
            )
            self.assertFalse(
                np.array_equal(
                    first.context_stimulus(legacy.encoder, candidate),
                    second.context_stimulus(legacy.encoder, candidate),
                )
            )
        state = SequenceState(['변수 "합계" 에 -13 을 저장해'.encode()], NL_VOCABULARY)
        state.apply(NL_VOCABULARY.context_start + 1)
        state.apply(NL_VOCABULARY.bind)
        for candidate in state.copy_candidates():
            np.testing.assert_array_equal(
                state.pointer_stimulus(legacy.encoder, candidate),
                state.pointer_stimulus(model.encoder, candidate),
            )
        self.assertFalse(
            first.context_stimulus(model.encoder, 0)[
                list(model.brain.output_nodes)
            ].any()
        )

    def test_versioned_checkpoint_restores_both_context_views_and_rejects_bad_headers(
        self,
    ):
        graph = Graph.from_csv(
            Path(__file__).resolve().parents[1] / "flywire_sample.csv"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            for invariant in (False, True):
                model = FlyNL(graph, context_bag=invariant)
                model.save(path, contract={"study": "test-v3"})
                resumed, _ = NeuralSequence.restore(
                    path,
                    graph,
                    NL_VOCABULARY,
                    role="fly-nl",
                    contract={"study": "test-v3"},
                )
                self.assertEqual(resumed.encoder.context_bag, invariant)
                self.assertEqual(
                    model.generate("37 을 출력해", max_actions=8, record=True),
                    resumed.generate("37 을 출력해", max_actions=8, record=True),
                )
            path.write_text('{"payload":[]}')
            with self.assertRaisesRegex(ValueError, "Malformed sequence checkpoint"):
                NeuralSequence.restore(
                    path, graph, NL_VOCABULARY, role="fly-nl", contract={}
                )

    def test_context_view_has_no_readout_without_graph_transmission(self):
        graph = Graph.from_csv(
            Path(__file__).resolve().parents[1] / "flywire_sample.csv"
        )
        model = FlyNL(graph, context_bag=True)
        result = model.generate(
            '변수 "합계" 에 -13 를 저장해',
            transmission=False,
            max_actions=8,
            record=True,
        )
        context = [e for e in result.trace if e["decision"] == "context"]
        self.assertEqual(len(context), 8)
        self.assertTrue(all(e["positive_score"] == 0.5 for e in context))
        self.assertEqual(result.code, "")
        self.assertNotEqual(result.status, "eos")

    def test_literal_masked_phrases_names_and_sources_do_not_cross_splits(self):
        splits = nl_cases_v3()
        masked, families, sources = {}, {}, {}
        for name, cases in splits.items():
            masked[name] = {
                hashlib.sha256(
                    ClockEncoder.features(c["source"].encode(), False, True)[
                        0
                    ].tobytes()
                ).hexdigest()
                for c in cases
            }
            families[name] = {c["family"] for c in cases}
            sources[name] = {c["source"] for c in cases}
            self.assertEqual(len(sources[name]), len(cases))
            self.assertTrue(any(not c["supported"] for c in cases))
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            self.assertFalse(masked[first] & masked[second])
            self.assertFalse(families[first] & families[second])
            self.assertFalse(sources[first] & sources[second])

    def test_every_augmented_teacher_copies_exact_source_values_and_ir(self):
        for cases in nl_cases_v3().values():
            for case in cases:
                actions = nl_teacher(case["source"], case["target"], NL_VOCABULARY)
                state = SequenceState(
                    case["source"].encode().splitlines(), NL_VOCABULARY
                )
                self.assertEqual(
                    b"".join(state.apply(action)[0] for action in actions).decode(),
                    case["target"],
                )

    def test_contrastive_augmentation_changes_only_training_negatives(self):
        initial, expanded = (
            nl_cases_v3(profile="initial"),
            nl_cases_v3(profile="contrastive"),
        )
        self.assertEqual(initial["validation"], expanded["validation"])
        self.assertEqual(initial["test"], expanded["test"])
        for split in (initial, expanded):
            self.assertEqual(sum(c["supported"] for c in split["train"]), 480)
        self.assertEqual(len(expanded["train"]) - len(initial["train"]), 24)
        self.assertTrue(
            any(not c["supported"] and "지금" in c["source"] for c in expanded["train"])
        )
        final = nl_cases_v3()
        self.assertEqual(final["validation"], initial["validation"])
        self.assertEqual(final["test"], initial["test"])

    def test_opening_evaluation_irreversibly_seals_training_and_reopening(self):
        for marker in ("hidden.json", "evaluation-started.json", "report.json"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / marker).write_text(json.dumps({"test_evaluated": True}))
                with self.assertRaisesRegex(ValueError, "opened its test"):
                    require_unopened(path)

    def test_accuracy_gate_keeps_test_sealed_when_uncertainty_or_semantics_fail(self):
        for supported, unknown in ((0.94, 1), (1, 0.89)):
            with self.assertRaisesRegex(ValueError, "test remains sealed"):
                require_validation(
                    {
                        "validation": {
                            "supported_semantic_exact_rate": supported,
                            "unknown_correct_rate": unknown,
                        }
                    }
                )
        require_validation(
            {
                "validation": {
                    "supported_semantic_exact_rate": 1,
                    "unknown_correct_rate": 1,
                }
            }
        )

    def test_new_data_and_training_modules_are_blocked_from_generation(self):
        for module in ("evaluation.nl_data_v3", "models.train_nl_v3"):
            with fly_only():
                with self.assertRaises(ForbiddenDependency):
                    __import__(module)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "models"
                target.mkdir()
                (target / "bypass.py").write_text(f"import {module}\n")
                with self.assertRaises(ForbiddenDependency):
                    verify_generation_tree(directory)


if __name__ == "__main__":
    unittest.main()
