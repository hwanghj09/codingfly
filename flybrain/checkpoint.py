"""Versioned JSON checkpoints, strict identity validation and atomic writes."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from flybrain.dynamics import Brain, DynamicsConfig
from flybrain.encoder import ENCODER_VERSION
from flybrain.graph import digest


FORMAT_VERSION = 1


def save(brain, path, *, role, contract, training_state=None):
    if role not in ("toy", "fly-code", "fly-nl"):
        raise ValueError("Unknown model role")
    payload = {"format": FORMAT_VERSION, "role": role, "graph": brain.graph.metadata(),
               "config": asdict(brain.config), "seed": brain.seed,
               "output_nodes": brain.output_nodes, "encoder": ENCODER_VERSION,
               "contract": contract, "weights": brain.weights.tolist(),
               "initial_weights": brain.initial_weights.tolist(), "signs": brain.signs.tolist(),
               "state": brain.state.tolist(), "eligibility": brain.eligibility.tolist(),
               "updates": brain.updates, "steps_seen": brain.steps_seen,
               "training_state": training_state or {}}
    record = {"payload": payload, "sha256": digest(payload)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = stream.name
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load(path, graph, *, role, contract):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = record["payload"]
        if record["sha256"] != digest(payload):
            raise ValueError("Checkpoint checksum mismatch")
        if (payload["format"] != FORMAT_VERSION or payload["role"] != role
                or payload["encoder"] != ENCODER_VERSION or payload["contract"] != contract
                or payload["graph"]["graph_sha256"] != graph.topology_hash
                or payload["graph"]["source_sha256"] != graph.source_hash):
            raise ValueError("Checkpoint graph/role/data/IR/encoder mismatch")
        brain = Brain(graph, len(payload["output_nodes"]), seed=payload["seed"],
                      config=DynamicsConfig(**payload["config"]))
        if (list(brain.output_nodes) != payload["output_nodes"]
                or brain.initial_weights.tolist() != payload["initial_weights"]
                or brain.signs.tolist() != payload["signs"]):
            raise ValueError("Checkpoint anatomical initialization mismatch")
        for name in ("weights", "state", "eligibility"):
            values = np.asarray(payload[name], dtype=np.float64)
            if values.shape != getattr(brain, name).shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid checkpoint array: {name}")
            setattr(brain, name, values.copy())
        if (np.any(brain.weights * brain.signs < 0)
                or np.any(np.abs(brain.weights) > brain.config.max_weight)
                or np.any(np.abs(brain.state) > 1)):
            raise ValueError("Checkpoint violates neuron/weight constraints")
        for name in ("updates", "steps_seen"):
            value = payload[name]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid checkpoint counters")
            setattr(brain, name, value)
        return brain, payload["training_state"]
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed checkpoint") from error
