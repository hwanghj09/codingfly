"""Controls preserve node IDs and edge count; rewiring also preserves degree sequence."""

import random

from flybrain.graph import Graph


def rewire(graph, seed):
    rng = random.Random(seed)
    edges = list(graph.edges)
    occupied = {(a, b) for a, b, _ in edges}
    swaps = 0
    for _ in range(len(edges) * 20):
        i, j = rng.sample(range(len(edges)), 2)
        a, b, c = edges[i]
        x, y, z = edges[j]
        if a == x or b == y or (a, y) in occupied or (x, b) in occupied:
            continue
        occupied.remove((a, b))
        occupied.remove((x, y))
        occupied.update(((a, y), (x, b)))
        edges[i], edges[j] = (a, y, c), (x, b, z)
        swaps += 1
    if not swaps:
        raise ValueError("Graph cannot be rewired with directed degree-preserving swaps")
    return Graph(graph.node_ids, tuple(sorted(edges)), graph.source_hash, graph.source_version,
                 f"directed double-edge swaps; seed={seed}; swaps={swaps}")


def random_graph(graph, seed):
    rng = random.Random(seed)
    n, m = len(graph.node_ids), len(graph.edges)
    pairs = rng.sample(range(n * n), m)
    edges = tuple(sorted((pair // n, pair % n, edge[2]) for pair, edge in zip(pairs, graph.edges)))
    return Graph(graph.node_ids, edges, graph.source_hash, graph.source_version,
                 f"uniform directed pairs without replacement; seed={seed}; self edges allowed")
