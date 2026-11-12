from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from flybrain.graph import Graph
from flybrain.retry_input import RetryDecoding, feedback_drive
from evaluation.retries import retry_cases, visible_oracle
from models.fly_code import Generation
from models.fly_code_v2 import FlyCodeV2
from models.fly_nl import FlyNL
from models.pipeline import FlyPipeline
from models.retry import failure_signal, failed_checks, run_retries


ROOT = Path(__file__).resolve().parents[1]


def generation(text):
    return Generation(text, "eos", (1, 0), ())


def grade(passed, **extra):
    return {
        "status": "ok",
        "syntax_ok": True,
        "semantic_ok": passed,
        "stdout": "9\n" if passed else "8\n",
        "variables": {},
        "returns": [],
        **extra,
    }


class RetryTest(unittest.TestCase):
    def setUp(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        self.a, self.b = FlyNL(graph), FlyCodeV2(graph)
        self.pipeline = FlyPipeline(self.a, self.b)
        self.expected = {"stdout": "9\n", "variables": {}}

    def test_supervised_b_failure_reuses_a_and_never_exposes_gold_to_generator(self):
        with patch.object(
            self.a, "generate", return_value=generation("PRINT 9\n")
        ) as a:
            with patch.object(
                self.b,
                "generate",
                side_effect=[generation("print(8)\n"), generation("print(9)\n")],
            ) as b:
                with patch(
                    "models.pipeline.grade_code",
                    side_effect=[grade(False), grade(True)],
                ):
                    result = run_retries(
                        self.pipeline,
                        "request",
                        expected=self.expected,
                        expected_ir="PRINT 9\n",
                        feedback_mode="supervised",
                    )
        self.assertEqual(result["status"], "ok")
        self.assertEqual((a.call_count, b.call_count), (1, 2))
        self.assertTrue(result["retry"]["attempts"][1]["reused_a"])
        signal = result["retry"]["attempts"][0]["failure_feedback"]
        self.assertEqual(signal["stage"], "B")
        self.assertEqual(signal["failed_checks"], [{"kind": "stdout", "actual": "8\n"}])
        for call in a.call_args_list + b.call_args_list:
            self.assertNotIn("expected_ir", call.kwargs)
            self.assertNotIn("expected", call.kwargs)
        self.assertNotIn("PRINT 9", str(b.call_args_list[1].kwargs))

    def test_a_semantic_error_regenerates_a_and_retranslates_without_b_penalty(self):
        with patch.object(
            self.a,
            "generate",
            side_effect=[generation("PRINT 8\n"), generation("PRINT 9\n")],
        ):
            with patch.object(
                self.b,
                "generate",
                side_effect=[generation("print(8)\n"), generation("print(9)\n")],
            ) as b:
                with patch(
                    "models.pipeline.grade_code",
                    side_effect=[grade(False), grade(True)],
                ):
                    result = run_retries(
                        self.pipeline,
                        "request",
                        expected=self.expected,
                        expected_ir="PRINT 9\n",
                        feedback_mode="supervised",
                    )
        attempts = result["retry"]["attempts"]
        self.assertEqual(attempts[0]["failure_feedback"]["stage"], "A")
        self.assertEqual(attempts[1]["nl_decoding"]["context_rank"], 1)
        self.assertFalse(attempts[1]["reused_a"])
        self.assertEqual(
            b.call_args_list[1].kwargs, {"record": False, "max_actions": 512}
        )

    def test_weak_and_equal_budget_control_do_not_localize_semantic_failure(self):
        for mode in ("weak", "resample"):
            with patch.object(
                self.a, "generate", return_value=generation("PRINT 9\n")
            ) as a:
                with patch.object(
                    self.b, "generate", return_value=generation("print(8)\n")
                ) as b:
                    with patch("models.pipeline.grade_code", return_value=grade(False)):
                        result = run_retries(
                            self.pipeline,
                            "request",
                            expected=self.expected,
                            feedback_mode=mode,
                        )
            self.assertEqual((a.call_count, b.call_count), (3, 3))
            self.assertEqual(result["code"], "print(8)\n")
            self.assertEqual(result["retry"]["attempts_used"], 3)
            self.assertEqual(result["retry"]["learning_updates"], 0)
            self.assertTrue(result["retry"]["weights_unchanged"])
            self.assertEqual(a.call_args_list[2].kwargs["decoding"].context_rank, 2)
            self.assertEqual(
                a.call_args_list[1].kwargs["feedback"] is None, mode == "resample"
            )

    def test_success_unknown_and_infrastructure_failure_stop_immediately(self):
        for source, status, verification in [
            ("UNKNOWN\n", "uncertain", None),
            ("PRINT 9\n", "ok", grade(True)),
            ("PRINT 9\n", "test_failure", grade(False, status="sandbox_unavailable")),
        ]:
            with patch.object(self.a, "generate", return_value=generation(source)) as a:
                with patch.object(
                    self.b, "generate", return_value=generation("print(9)\n")
                ):
                    with patch("models.pipeline.grade_code", return_value=verification):
                        result = run_retries(
                            self.pipeline, "request", expected=self.expected
                        )
            self.assertEqual(result["status"], status)
            self.assertEqual(result["retry"]["attempts_used"], 1)
            self.assertEqual(a.call_count, 1)

    def test_invalid_ir_never_invokes_b_and_semantic_blame_needs_supervision(self):
        with patch.object(self.a, "generate", return_value=generation("LET\n")):
            with patch.object(
                self.b, "generate", side_effect=AssertionError("Invalid IR reached B")
            ):
                result = run_retries(
                    self.pipeline,
                    "request",
                    expected=self.expected,
                    feedback_mode="supervised",
                )
        self.assertEqual(result["status"], "invalid_ir")
        self.assertEqual(
            result["retry"]["attempts"][0]["failure_feedback"]["stage"], "A"
        )
        self.assertEqual(
            failure_signal(
                {"status": "test_failure", "ir": "PRINT 8\n"},
                self.expected,
                mode="supervised",
            )["stage"],
            "unlocalized",
        )

    def test_oracle_checks_report_failed_inputs_and_actuals_without_answers(self):
        expected = {
            "stdout": "secret-answer",
            "variables": {"x": 9000},
            "calls": [{"name": "f", "args": [-3]}],
            "returns": [12345],
        }
        actual = grade(False, variables={"x": 1}, returns=[0])
        checks = failed_checks(actual, expected)
        self.assertEqual([c["kind"] for c in checks], ["stdout", "variable", "return"])
        self.assertIn("-3", checks[-1]["input"])
        self.assertNotIn("secret-answer", str(checks))
        self.assertNotIn("9000", str(checks))
        self.assertNotIn("12345", str(checks))

    def test_retry_bounds_and_no_oracle_fail_before_generation(self):
        for kwargs in (
            {"max_attempts": 0},
            {"max_attempts": 4},
            {"max_attempts": True},
            {"feedback_mode": "repair"},
            {"seed": -1},
            {"max_attempts": 2, "expected": None},
            {"expected_ir": "PRINT 9\n"},
        ):
            options = {"expected": self.expected, **kwargs}
            with patch.object(
                self.a, "generate", side_effect=AssertionError("Bounds not checked")
            ):
                with self.assertRaises(ValueError):
                    run_retries(self.pipeline, "request", **options)

    def test_feedback_bytes_change_only_inputs_and_are_deterministic(self):
        base = {"reward": -1, "issue": "runtime_error", "previous_actions": [3, 5, 0]}
        first, identity = feedback_drive(self.a.encoder, base)
        repeated, repeated_identity = feedback_drive(self.a.encoder, deepcopy(base))
        changed, _ = feedback_drive(self.a.encoder, {**base, "issue": "syntax_error"})
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(identity, repeated_identity)
        self.assertFalse(np.array_equal(first, changed))
        self.assertFalse(first[list(self.a.brain.output_nodes)].any())
        with self.assertRaises(ValueError):
            feedback_drive(self.a.encoder, {"error": "x" * 17000})

    def test_recorded_digit_is_actual_sample_and_cut_graph_has_no_learned_choices(self):
        decoding = RetryDecoding(temperature=0.3, seed=42)
        first = self.b.generate(
            "PRINT 9",
            decoding=decoding,
            feedback={"reward": -1},
            max_actions=5,
            record=True,
            transmission=False,
        )
        repeated = self.b.generate(
            "PRINT 9",
            decoding=decoding,
            feedback={"reward": -1},
            max_actions=5,
            record=True,
            transmission=False,
        )
        self.assertEqual(first, repeated)
        action = 0
        for event in first.trace[: self.b.vocabulary.digits]:
            self.assertEqual(event["neural"]["probabilities"], [0.5, 0.5])
            action = 2 * action + event["chosen"]
            self.assertIsNotNone(event["feedback_sha256"])
        self.assertEqual(
            first.trace[self.b.vocabulary.digits - 1]["selected_action"], action
        )

    def test_default_generation_stays_identical_and_feedback_reaches_brain(self):
        before = self.b.brain.weights_hash
        old = self.b.generate("PRINT 9", max_actions=2)
        explicit = self.b.generate("PRINT 9", max_actions=2, decoding=RetryDecoding())
        self.assertEqual(old, explicit)
        seen = []
        original = self.b.brain.forward

        def capture(stimulus, **kwargs):
            seen.append(stimulus.copy())
            return original(stimulus, **kwargs)

        with patch.object(self.b.brain, "forward", side_effect=capture):
            self.b.generate("PRINT 9", max_actions=1)
            baseline = seen[0]
            seen.clear()
            self.b.generate("PRINT 9", max_actions=1, feedback={"reward": -1})
        self.assertFalse(np.array_equal(baseline, seen[0]))
        self.assertEqual(before, self.b.brain.weights_hash)

    def test_hidden_function_boundaries_are_not_available_to_retry_selection(self):
        expected = {
            "stdout": "",
            "variables": {},
            "calls": [{"name": "f", "args": [1]}, {"name": "f", "args": [-9]}],
            "returns": [1, -9],
        }
        public = visible_oracle(expected)
        self.assertEqual(public["calls"], [{"name": "f", "args": [1]}])
        self.assertEqual(public["returns"], [1])
        self.assertEqual(len(expected["calls"]), 2)

    def test_retry_cohorts_are_disjoint_and_preserve_known_baseline(self):
        from evaluation.nl_data import nl_cases

        cohorts = retry_cases()
        original = nl_cases()["test"]
        self.assertEqual(
            [c["source"] for c in cohorts["known_v2"]], [c["source"] for c in original]
        )
        self.assertEqual(len(cohorts["fresh_literals"]), 36)
        self.assertEqual(sum(c["supported"] for c in cohorts["fresh_literals"]), 32)
        self.assertFalse(
            {c["source"] for c in cohorts["known_v2"]}
            & {c["source"] for c in cohorts["fresh_literals"]}
        )
