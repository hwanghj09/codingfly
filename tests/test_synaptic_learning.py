import math
from pathlib import Path
import random
import unittest

import numpy as np

from evaluation.datasets import toy_cases
from evaluation.toy import accuracy
from flybrain.dynamics import Brain
from flybrain.encoder import ByteEncoder
from flybrain.graph import Graph
from flybrain.plasticity import supervised_step


ROOT = Path(__file__).resolve().parents[1]


class SynapticLearningTest(unittest.TestCase):
    def setUp(self):
        self.graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        self.brain = Brain(self.graph, 4)
        self.encoder = ByteEncoder(64, self.brain.input_nodes)

    def test_analytic_gradient_agrees_with_numerical_loss_derivative(self):
        stimulus = self.encoder.encode(b"A\x12")
        original = self.brain.weights.copy()
        supervised_step(self.brain, stimulus, 2, learning_rate=.001)
        gradient = -self.brain.eligibility.copy()
        self.brain.weights = original.copy()
        for index in np.argsort(np.abs(gradient))[-8:]:
            epsilon = 1e-6
            self.brain.weights[index] = original[index] + epsilon
            plus = -math.log(self.brain.forward(stimulus)[0][2])
            self.brain.weights[index] = original[index] - epsilon
            minus = -math.log(self.brain.forward(stimulus)[0][2])
            self.brain.weights[index] = original[index]
            self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * epsilon), places=6)

    def test_only_existing_edges_learn_and_frozen_state_is_unchanged(self):
        before = self.brain.weights_hash
        topology = self.graph.topology_hash
        stimulus = self.encoder.encode(b"B\x07")
        frozen_result = supervised_step(self.brain, stimulus, 1, frozen=True)
        self.assertEqual(before, self.brain.weights_hash)
        self.assertEqual(frozen_result["changed_edges"], 0)
        result = supervised_step(self.brain, stimulus, 1)
        self.assertGreater(result["changed_edges"], 0)
        self.assertNotEqual(before, self.brain.weights_hash)
        self.assertEqual(topology, self.graph.topology_hash)
        self.assertEqual(len(self.brain.weights), len(self.graph.edges))
        self.assertTrue(np.all(self.brain.signs * self.brain.weights >= 0))
        self.assertTrue(np.all(np.abs(self.brain.weights) <= self.brain.config.max_weight))

    def test_learning_improves_a_locked_unseen_noise_holdout(self):
        splits = toy_cases()
        encoded = {name: [(self.encoder.encode(bytes.fromhex(case["source"])), int(case["target"]))
                          for case in cases] for name, cases in splits.items()}
        frozen = Brain(self.graph, 4)
        order, rng = list(range(len(encoded["train"]))), random.Random(0)
        for _ in range(30):
            rng.shuffle(order)
            for i in order:
                stimulus, expected = encoded["train"][i]
                supervised_step(self.brain, stimulus, expected)
        self.assertGreaterEqual(accuracy(self.brain, encoded["test"]), .75)
        self.assertGreater(accuracy(self.brain, encoded["test"]), accuracy(frozen, encoded["test"]))
        self.assertEqual(accuracy(self.brain, encoded["test"], transmission=False), .25)
