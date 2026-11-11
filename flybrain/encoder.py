"""Fixed positional byte stimuli. No vocabulary, language parsing or learned embeddings."""

import hashlib
import math

import numpy as np


ENCODER_VERSION = "positional-bytes-v1"


class ByteEncoder:
    def __init__(self, node_count, input_nodes, *, seed=0, fanout=8, max_bytes=4096):
        self.node_count = node_count
        self.input_nodes = tuple(input_nodes)
        if not self.input_nodes or fanout < 1 or max_bytes < 1:
            raise ValueError("Invalid encoder configuration")
        self.seed, self.fanout, self.max_bytes = seed, fanout, max_bytes

    def encode(self, *fields):
        stimuli = np.zeros(self.node_count, dtype=np.float64)
        total = sum(len(field) for field in fields)
        if total > self.max_bytes or any(not isinstance(field, bytes) for field in fields):
            raise ValueError("Encoder requires bounded byte fields")
        # Each field has distinct boundaries, including an explicit empty-field marker.
        for field_index, field in enumerate(fields):
            for position, value in enumerate([256, *field, 257]):
                key = f"{self.seed}:{field_index}:{position}:{value}".encode("ascii")
                data = hashlib.shake_256(key).digest(self.fanout * 8)
                for slot in range(self.fanout):
                    number = int.from_bytes(data[slot * 8:(slot + 1) * 8], "big")
                    neuron = self.input_nodes[(number >> 1) % len(self.input_nodes)]
                    stimuli[neuron] += 1.0 if number & 1 else -1.0
        stimuli /= math.sqrt(max(1, total + 2 * len(fields)))
        return stimuli
