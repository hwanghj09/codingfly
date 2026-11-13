import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flybrain.checkpoint import load, save
from flybrain.dynamics import Brain
from flybrain.encoder import ByteEncoder
from flybrain.graph import Graph, digest
from flybrain.plasticity import supervised_step


ROOT = Path(__file__).resolve().parents[1]


class CheckpointTest(unittest.TestCase):
    def test_restore_and_next_update_are_bit_exact(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        brain = Brain(graph, 4, seed=23)
        stimulus = ByteEncoder(64, brain.input_nodes, seed=23).encode(b"C\x10")
        supervised_step(brain, stimulus, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "code.json"
            contract = {"ir": "0.1", "dataset": "test"}
            save(brain, path, role="fly-code", contract=contract, training_state={"epoch": 1})
            restored, state = load(path, graph, role="fly-code", contract=contract)
            self.assertEqual(state, {"epoch": 1})
            np.testing.assert_array_equal(brain.state, restored.state)
            np.testing.assert_array_equal(brain.eligibility, restored.eligibility)
            np.testing.assert_array_equal(brain.forward(stimulus, reset=False)[0],
                                          restored.forward(stimulus, reset=False)[0])
            supervised_step(brain, stimulus, 1)
            supervised_step(restored, stimulus, 1)
            self.assertEqual(brain.weights_hash, restored.weights_hash)
            self.assertEqual(brain.steps_seen, restored.steps_seen)

    def test_cross_role_data_and_corrupt_checkpoints_are_rejected(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        brain = Brain(graph, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save(brain, path, role="fly-code", contract={"data": "a"})
            for role, contract in (("fly-nl", {"data": "a"}), ("fly-code", {"data": "b"})):
                with self.assertRaises(ValueError):
                    load(path, graph, role=role, contract=contract)
            wrong_graph = Graph(graph.node_ids, graph.edges[:-1], graph.source_hash)
            with self.assertRaises(ValueError):
                load(path, wrong_graph, role="fly-code", contract={"data": "a"})
            record = json.loads(path.read_text())
            record["payload"]["weights"][0] += .1
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load(path, graph, role="fly-code", contract={"data": "a"})
            record["payload"]["weights"][0] = -2.
            record["sha256"] = digest(record["payload"])
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "constraints"):
                load(path, graph, role="fly-code", contract={"data": "a"})

    def test_two_brains_never_share_mutable_synapses(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        a, b = Brain(graph, 4, seed=1), Brain(graph, 4, seed=2)
        before = b.weights_hash
        supervised_step(a, ByteEncoder(64, a.input_nodes).encode(b"x"), 1)
        self.assertEqual(before, b.weights_hash)
        self.assertFalse(np.shares_memory(a.weights, b.weights))
