"""Leaky recurrent rate model; artificial dynamics, not a biophysical simulation."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from flybrain.graph import digest


@dataclass(frozen=True)
class DynamicsConfig:
    steps: int = 4
    leak: float = 0.65
    gain: float = 2.0
    output_scale: float = 6.0
    max_weight: float = 3.0
    inhibitory_fraction: float = 0.0

    def __post_init__(self):
        if (type(self.steps) is not int or not 2 <= self.steps <= 128
                or not all(math.isfinite(x) for x in (self.leak, self.gain, self.output_scale,
                                                      self.max_weight, self.inhibitory_fraction))
                or not 0 < self.leak <= 1 or not 0 < self.gain <= 10
                or not 0 < self.output_scale <= 100 or not 0 < self.max_weight <= 10
                or not 0 <= self.inhibitory_fraction <= 1):
            raise ValueError("Invalid dynamics configuration")


class Brain:
    def __init__(self, graph, output_count, *, seed=0, config=None):
        self.graph, self.seed = graph, seed
        self.config = config or DynamicsConfig()
        self.input_nodes, self.output_nodes = graph.groups(output_count)
        self.pre = np.array([a for a, _, _ in graph.edges], dtype=np.int64)
        self.post = np.array([b for _, b, _ in graph.edges], dtype=np.int64)
        counts = np.array([c for _, _, c in graph.edges], dtype=np.float64)
        incoming = np.bincount(self.post, weights=counts, minlength=len(graph.node_ids))
        rng = np.random.default_rng(seed)
        # Sign is a documented artificial assumption, never inferred neurotransmitter data.
        neuron_sign = np.where(rng.random(len(graph.node_ids)) < self.config.inhibitory_fraction, -1., 1.)
        self.signs = neuron_sign[self.pre]
        self.initial_weights = self.signs * np.minimum(counts / incoming[self.post], self.config.max_weight)
        self.weights = self.initial_weights.copy()
        self.state = np.zeros(len(graph.node_ids), dtype=np.float64)
        self.eligibility = np.zeros(len(graph.edges), dtype=np.float64)
        self.updates = 0
        self.steps_seen = 0
        # Edge endpoints are immutable and never optimized.
        self.pre.flags.writeable = self.post.flags.writeable = False
        self.initial_weights.flags.writeable = self.signs.flags.writeable = False

    @property
    def weights_hash(self):
        return digest(self.weights.tolist())

    def reset(self):
        self.state.fill(0.)

    def forward(self, stimulus, *, reset=True, transmission=True, record=False):
        stimulus = np.asarray(stimulus, dtype=np.float64)
        if stimulus.shape != self.state.shape or not np.isfinite(stimulus).all():
            raise ValueError("Invalid neuron stimulus")
        if np.any(stimulus[list(self.output_nodes)]):
            raise ValueError("Direct output stimulation is forbidden")
        if reset:
            self.reset()
        cfg = self.config
        tape, history = [], []
        for _ in range(cfg.steps):
            previous = self.state.copy()
            current = np.bincount(self.post, weights=self.weights * previous[self.pre],
                                  minlength=len(self.state)) if transmission else np.zeros_like(self.state)
            proposal = np.tanh(stimulus + cfg.gain * current)
            self.state = (1. - cfg.leak) * previous + cfg.leak * proposal
            tape.append((previous, proposal))
            if record:
                history.append(self.state.tolist())
        logits = cfg.output_scale * self.state[list(self.output_nodes)]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        trace = None
        if record:
            trace = {"graph_sha256": self.graph.topology_hash, "weights_sha256": self.weights_hash,
                     "config": asdict(cfg), "seed": self.seed, "updates": self.updates,
                     "output_neuron_ids": [self.graph.node_ids[i] for i in self.output_nodes],
                     "output_rates": self.state[list(self.output_nodes)].tolist(),
                     "probabilities": probabilities.tolist(), "activity": history,
                     "transmission": transmission,
                     "active_neurons": int(np.count_nonzero(np.abs(self.state) > 1e-6)),
                     "saturated_neurons": int(np.count_nonzero(np.abs(self.state) > .99))}
        return probabilities, tape, trace
