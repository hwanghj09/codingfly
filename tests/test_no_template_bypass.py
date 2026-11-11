from pathlib import Path
import unittest
from unittest.mock import patch

from flybrain.decoder import ActionDecoder, whitespace_spans
from flybrain.graph import Graph
from models.fly_code import FlyCode


ROOT = Path(__file__).resolve().parents[1]


class NoTemplateBypassTest(unittest.TestCase):
    def test_neural_choices_alone_determine_unrepaired_output(self):
        model = FlyCode(Graph.from_csv(ROOT / "flywire_lab.csv"))
        # Every recorded choice is independently recoverable from neuron output.
        generation = model.generate("LET unfamiliar 987\n", max_actions=8, record=True)
        for action, event in zip(generation.actions, generation.trace):
            scores = event["neural"]["probabilities"]
            self.assertEqual(action, max(range(len(scores)), key=scores.__getitem__))
            self.assertEqual(event["neural"]["weights_sha256"], model.brain.weights_hash)
        raw, prefix = b"LET unfamiliar 987\n", b""
        spans = whitespace_spans(raw)
        for action in generation.actions:
            if action == 0:
                break
            try:
                prefix += model.decoder.emit(action, raw, spans)
            except ValueError:
                break
        self.assertEqual(generation.code, prefix.decode())

    def test_legacy_parser_and_translator_are_never_used(self):
        model = FlyCode(Graph.from_csv(ROOT / "flywire_lab.csv"))
        with patch("goal_model.parse_goal", side_effect=AssertionError("regex NLP used")) as parse:
            with patch("program_model.ProgramModel.generate", side_effect=AssertionError("template used")) as render:
                generation = model.generate("PRINT 314\n", max_actions=3, transmission=False)
                self.assertEqual((generation.code, generation.status, generation.actions), ("", "eos", (0,)))
                parse.assert_not_called()
                render.assert_not_called()

    def test_copy_preserves_unseen_unicode_names_and_invalid_positions_fail(self):
        decoder = ActionDecoder()
        source = "LET 처음_보는_이름 -987654\n".encode()
        spans = whitespace_spans(source)
        self.assertEqual(decoder.emit(8, source, spans).decode(), "처음_보는_이름")
        self.assertEqual(decoder.emit(9, source, spans), b"-987654")
        with self.assertRaisesRegex(ValueError, "invalid_copy"):
            decoder.emit(22, source, spans)
