from collections import Counter
import csv
from pathlib import Path
import tempfile
import unittest

from evaluation.ablation import random_graph, rewire
from flybrain.graph import Graph


ROOT = Path(__file__).resolve().parents[1]


class GraphIntegrityTest(unittest.TestCase):
    def test_real_subsets_preserve_every_direction_count_and_identity(self):
        for filename, nodes, edges in (("flywire_sample.csv", 64, 168), ("flywire_lab.csv", 512, 1403),
                                       ("flywire_lab_2048.csv", 2048, 5340)):
            with self.subTest(filename=filename):
                graph = Graph.from_csv(ROOT / filename)
                with (ROOT / filename).open() as stream:
                    original = {(r["pre_root_id"], r["post_root_id"]): int(r["syn_count"])
                                for r in csv.DictReader(stream)}
                reconstructed = {(graph.node_ids[a], graph.node_ids[b]): count for a, b, count in graph.edges}
                self.assertEqual(reconstructed, original)
                self.assertEqual((len(graph.node_ids), len(graph.edges)), (nodes, edges))
                self.assertEqual(graph.synapse_count, sum(original.values()))
                self.assertEqual(graph.topology_hash, Graph.from_csv(ROOT / filename).topology_hash)

    def csv_graph(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.csv"
            path.write_text(data)
            return Graph.from_csv(path)

    def test_duplicate_rows_aggregate_but_reverse_and_self_edges_remain(self):
        graph = self.csv_graph("pre_root_id,post_root_id,syn_count\n1,2,3\n1,2,4\n2,1,5\n2,2,2\n")
        self.assertEqual(graph.edges, ((0, 1, 7), (1, 0, 5), (1, 1, 2)))
        self.assertEqual(graph.synapse_count, 14)

    def test_malformed_data_fail_closed(self):
        header = "pre_root_id,post_root_id,syn_count\n"
        for row in ("", "1,2,0\n", "1,2,-1\n", "1,2,nan\n", "1,2,1.5\n", "x,2,5\n",
                    "1,,3\n", "1,2\n", "1,2,3,4\n", "01,2,3\n", "0,2,3\n"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.csv_graph(header + row)
        with self.assertRaises(ValueError):
            self.csv_graph("pre,post,count\n1,2,3\n")

    def test_rewiring_preserves_directed_degrees_and_counts(self):
        original = Graph.from_csv(ROOT / "flywire_sample.csv")
        modified = rewire(original, 42)
        self.assertEqual(original.node_ids, modified.node_ids)
        self.assertNotEqual(original.topology_hash, modified.topology_hash)
        self.assertEqual(modified, rewire(original, 42))
        for index in (0, 1):
            self.assertEqual(Counter(edge[index] for edge in original.edges),
                             Counter(edge[index] for edge in modified.edges))
        self.assertEqual(original.synapse_count, modified.synapse_count)
        random = random_graph(original, 42)
        self.assertEqual(len(random.edges), len(original.edges))
        self.assertEqual(len({(a, b) for a, b, _ in random.edges}), len(original.edges))

    def test_csr_csc_roundtrip_and_unreachable_output_diagnostics(self):
        graph = Graph(("1", "2", "3", "4", "5"), ((0, 1, 7), (1, 0, 2), (2, 3, 4)), "fixture")
        for incoming in (False, True):
            sparse = graph.sparse_index(incoming=incoming)
            reconstructed = []
            for node in range(len(graph.node_ids)):
                for position in range(sparse.offsets[node], sparse.offsets[node + 1]):
                    neighbor, edge_id = sparse.neighbors[position], sparse.edge_ids[position]
                    pair = (neighbor, node) if incoming else (node, neighbor)
                    reconstructed.append((*pair, graph.edges[edge_id][2]))
            self.assertEqual(sorted(reconstructed), list(graph.edges))
        report = graph.connectivity([0], [1, 3])
        self.assertEqual(report["unreachable_output_ids"], ["4"])
        self.assertEqual(report["isolated_neuron_ids"], ["5"])
