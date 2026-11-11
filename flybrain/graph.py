"""Validated immutable graph; duplicate directed rows are summed, self edges retained."""

import csv
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SparseIndex:
    offsets: tuple[int, ...]
    neighbors: tuple[int, ...]
    edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class Graph:
    node_ids: tuple[str, ...]
    # Single common index, sorted by (pre, post).
    edges: tuple[tuple[int, int, int], ...]
    source_hash: str
    source_version: str = "unspecified CSV; no biological provenance asserted"
    selection: str = "all rows in supplied CSV; no additional threshold"

    def __post_init__(self):
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("Graph requires unique neuron IDs")
        seen = set()
        for pre, post, count in self.edges:
            if (type(pre) is not int or type(post) is not int or type(count) is not int
                    or not 0 <= pre < len(self.node_ids) or not 0 <= post < len(self.node_ids)
                    or count <= 0 or (pre, post) in seen):
                raise ValueError("Invalid or duplicate directed edge")
            seen.add((pre, post))

    @classmethod
    def from_csv(cls, path):
        path = Path(path)
        counts = Counter()
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["pre_root_id", "post_root_id", "syn_count"]:
                raise ValueError("Expected pre_root_id,post_root_id,syn_count CSV header")
            for line, row in enumerate(reader, 2):
                try:
                    pre, post, count = row["pre_root_id"], row["post_root_id"], row["syn_count"]
                    if (None in row or not pre or not post
                            or any(not node.isascii() or not node.isdecimal()
                                   or not 0 < int(node) < 2 ** 64 or str(int(node)) != node
                                   for node in (pre, post))):
                        raise ValueError("invalid neuron ID")
                    number = int(count)
                    if number <= 0 or str(number) != count:
                        raise ValueError("invalid synapse count")
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid graph CSV row {line}") from error
                counts[pre, post] += number
        if not counts:
            raise ValueError("Empty graph CSV")
        nodes = tuple(sorted({node for pair in counts for node in pair}, key=int))
        index = {node: i for i, node in enumerate(nodes)}
        edges = tuple(sorted((index[pre], index[post], count)
                             for (pre, post), count in counts.items()))
        known_subsets = {"flywire_sample.csv": 30_000, "flywire_lab.csv": 150_000,
                         "flywire_lab_2048.csv": 500_000}
        metadata = {}
        if path.resolve().parent == Path(__file__).resolve().parents[1] and path.name in known_subsets:
            metadata = {"source_version": "FlyWire Codex FAFB v783 (repository README provenance)",
                        "selection": f"upstream first {known_subsets[path.name]} rows; syn_count >= 5; "
                        "self edges excluded upstream; BFS from 720575940624547622; no further filtering"}
        return cls(nodes, edges, hashlib.sha256(path.read_bytes()).hexdigest(), **metadata)

    @property
    def topology_hash(self):
        return digest({"nodes": self.node_ids, "edges": self.edges})

    @property
    def synapse_count(self):
        return sum(count for _, _, count in self.edges)

    def metadata(self):
        return {"source_sha256": self.source_hash, "graph_sha256": self.topology_hash,
                "source_version": self.source_version, "selection": self.selection,
                "node_ids": self.node_ids, "nodes": len(self.node_ids), "edges": len(self.edges),
                "syn_count_total": self.synapse_count,
                "duplicates": "sum same direction", "self_edges": "retain if present"}

    def sparse_index(self, *, incoming=False):
        """CSR for outgoing edges, CSC for incoming; original edge IDs keep counts aligned."""
        axis = 1 if incoming else 0
        order = sorted(range(len(self.edges)), key=lambda i: (self.edges[i][axis], self.edges[i][1 - axis]))
        counts = Counter(edge[axis] for edge in self.edges)
        offsets = [0]
        for i in range(len(self.node_ids)):
            offsets.append(offsets[-1] + counts[i])
        return SparseIndex(tuple(offsets), tuple(self.edges[i][1 - axis] for i in order), tuple(order))

    def connectivity(self, inputs, outputs):
        csr = self.sparse_index()
        seen, queue = set(inputs), deque(inputs)
        while queue:
            node = queue.popleft()
            for target in csr.neighbors[csr.offsets[node]:csr.offsets[node + 1]]:
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        connected = {node for a, b, _ in self.edges for node in (a, b)}
        return {"isolated_neuron_ids": [self.node_ids[i] for i in range(len(self.node_ids)) if i not in connected],
                "unreachable_output_ids": [self.node_ids[i] for i in outputs if i not in seen],
                "reachable_neurons": len(seen)}

    def groups(self, output_count):
        """Fixed anatomical readout: highest indegree, ID order breaks ties."""
        incoming = Counter(post for _, post, _ in self.edges)
        if not 1 < output_count <= len(incoming):
            raise ValueError("Not enough connected output neurons")
        outputs = tuple(sorted(incoming, key=lambda node: (-incoming[node], node))[:output_count])
        output_set = set(outputs)
        inputs = tuple(i for i in range(len(self.node_ids)) if i not in output_set)
        if not inputs:
            raise ValueError("Input and output groups must be disjoint")
        return inputs, outputs
