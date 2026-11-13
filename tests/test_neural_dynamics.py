from pathlib import Path
import unittest

import numpy as np

from flybrain.dynamics import Brain, DynamicsConfig
from flybrain.encoder import ByteEncoder
from flybrain.graph import Graph


ROOT = Path(__file__).resolve().parents[1]


class NeuralDynamicsTest(unittest.TestCase):
    def setUp(self):
        self.graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        self.brain = Brain(self.graph, 4, seed=12)
        self.encoder = ByteEncoder(64, self.brain.input_nodes, seed=12)

    def test_recurrent_activity_and_reproducibility(self):
        stimulus = self.encoder.encode("파리".encode(), b"prefix")
        first = self.brain.forward(stimulus, record=True)
        other = Brain(self.graph, 4, seed=12).forward(stimulus, record=True)
        self.assertTrue(np.array_equal(first[0], other[0]))
        self.assertEqual(first[2], other[2])
        self.assertNotEqual(first[2]["activity"][0], first[2]["activity"][-1])
        self.assertNotEqual(first[2]["output_rates"], [0.] * 4)
        self.assertEqual(self.graph.topology_hash, first[2]["graph_sha256"])
        self.assertEqual(self.brain.weights_hash, first[2]["weights_sha256"])

    def test_no_input_and_disconnected_transmission_have_no_output_signal(self):
        probabilities, _, _ = self.brain.forward(np.zeros(64))
        np.testing.assert_array_equal(probabilities, [.25] * 4)
        stimulus = self.encoder.encode(b"hello")
        cut, _, _ = self.brain.forward(stimulus, transmission=False)
        np.testing.assert_array_equal(cut, [.25] * 4)
        self.assertFalse(set(self.brain.input_nodes) & set(self.brain.output_nodes))

    def test_bad_stimuli_and_direct_readout_stimulation_are_rejected(self):
        for stimulus in ([1.], np.full(64, float("nan")), np.ones(64)):
            with self.assertRaises(ValueError):
                self.brain.forward(stimulus)

    def test_isolated_neurons_stay_silent(self):
        graph = Graph(("1", "2", "3", "4", "5"), ((0, 1, 1), (2, 3, 1)), "test")
        brain = Brain(graph, 2)
        _, _, trace = brain.forward(np.zeros(5), record=True)
        self.assertEqual(trace["active_neurons"], 0)
        self.assertEqual(brain.state[4], 0.)

    def test_long_activity_is_finite_bounded_and_zero_input_decays(self):
        stimulus = self.encoder.encode(b"A")
        for _ in range(100):
            self.brain.forward(stimulus, reset=False)
        self.assertTrue(np.isfinite(self.brain.state).all())
        self.assertLessEqual(np.abs(self.brain.state).max(), 1.)
        for _ in range(100):
            self.brain.forward(np.zeros(64), reset=False, transmission=False)
        self.assertLess(np.abs(self.brain.state).max(), 1e-10)

    def test_encoder_boundaries_position_and_limits(self):
        self.assertFalse(np.array_equal(self.encoder.encode(b"ab"), self.encoder.encode(b"ba")))
        self.assertFalse(np.array_equal(self.encoder.encode(b"a", b"b"), self.encoder.encode(b"ab")))
        with self.assertRaises(ValueError):
            self.encoder.encode(b"x" * 5000)
        for arguments in ({"steps": 1}, {"leak": 0}, {"gain": float("nan")}, {"max_weight": -1}):
            with self.assertRaises(ValueError):
                DynamicsConfig(**arguments)
