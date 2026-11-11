"""P3 experiment: autoregressive Python actions selected only by graph activity."""

from dataclasses import dataclass
import time

from flybrain.checkpoint import load, save
from flybrain.decoder import ActionDecoder, DECODER_VERSION, EOS, whitespace_spans
from flybrain.dynamics import Brain
from flybrain.encoder import ByteEncoder
from flybrain.graph import digest
from flybrain.plasticity import supervised_step
from flybrain.policy import fly_only


@dataclass(frozen=True)
class Generation:
    code: str
    status: str
    actions: tuple[int, ...]
    trace: tuple[dict, ...]


class FlyCode:
    def __init__(self, graph, *, seed=0, max_spans=16, config=None):
        self.decoder = ActionDecoder(max_spans)
        self.brain = Brain(graph, self.decoder.size, seed=seed, config=config)
        self.encoder = ByteEncoder(len(graph.node_ids), self.brain.input_nodes, seed=seed, max_bytes=4098)

    def stimulus(self, source, prefix, step):
        return self.encoder.encode(source, prefix, step.to_bytes(2, "little"))

    def teacher_examples(self, source, actions):
        """Training only; target sequences come from explicit supervised datasets."""
        source = source.encode("utf-8")
        spans = whitespace_spans(source)
        if len(spans) > self.decoder.max_spans or not actions or actions[-1] != EOS or EOS in actions[:-1]:
            raise ValueError("Invalid teacher sequence")
        prefix, examples = b"", []
        for step, expected in enumerate(actions):
            examples.append((self.stimulus(source, prefix, step), expected))
            prefix += self.decoder.emit(expected, source, spans)
        return examples

    def train_example(self, stimulus, expected, *, learning_rate=.03, frozen=False):
        return supervised_step(self.brain, stimulus, expected, learning_rate=learning_rate, frozen=frozen)

    def generate(self, source, *, max_actions=64, timeout=5., record=False, transmission=True):
        if not isinstance(source, str) or not 1 <= max_actions <= 1024 or not 0 < timeout <= 60:
            raise ValueError("Invalid generation bounds")
        encoded = source.encode("utf-8")
        if not encoded or len(encoded) > 2048:
            raise ValueError("Source must contain 1..2048 bytes")
        spans = whitespace_spans(encoded)
        if len(spans) > self.decoder.max_spans:
            return Generation("", "input_limit", (), ())
        prefix, actions, trace = b"", [], []
        deadline = time.monotonic() + timeout
        status = "max_tokens"
        with fly_only():
            for step in range(max_actions):
                if time.monotonic() > deadline:
                    status = "timeout"
                    break
                probabilities, _, activity = self.brain.forward(self.stimulus(encoded, prefix, step),
                                                               record=record, transmission=transmission)
                action = self.decoder.choose(probabilities)
                actions.append(action)
                if record:
                    # All output rates/probabilities remain auditable. A digest records
                    # the complete time history without multi-megabyte per-action logs.
                    activity["activity_sha256"] = digest(activity.pop("activity"))
                    trace.append({"step": step, "action": self.decoder.labels()[action],
                                  "neural": activity})
                if action == EOS:
                    status = "eos"
                    break
                try:
                    prefix += self.decoder.emit(action, encoded, spans)
                except ValueError:
                    status = "invalid_copy"
                    break
                if len(prefix) > 2048:
                    status = "output_limit"
                    break
        try:
            code = prefix.decode("utf-8")
        except UnicodeDecodeError:
            code, status = prefix.decode("utf-8", errors="replace"), "invalid_utf8"
        return Generation(code, status, tuple(actions), tuple(trace))

    def save(self, path, *, contract, training_state=None):
        complete_contract = {**contract, "decoder": DECODER_VERSION, "max_spans": self.decoder.max_spans}
        save(self.brain, path, role="fly-code", contract=complete_contract, training_state=training_state)

    @classmethod
    def restore(cls, path, graph, *, contract, max_spans=16):
        complete_contract = {**contract, "decoder": DECODER_VERSION, "max_spans": max_spans}
        brain, state = load(path, graph, role="fly-code", contract=complete_contract)
        model = cls(graph, seed=brain.seed, max_spans=max_spans, config=brain.config)
        if len(brain.output_nodes) != model.decoder.size:
            raise ValueError("Checkpoint output vocabulary mismatch")
        model.brain = brain
        return model, state
