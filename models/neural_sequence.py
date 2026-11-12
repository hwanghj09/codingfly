"""Neural literal/COPY transducer with graph-selected cursor and memory controls."""

import time

import numpy as np

from flybrain.checkpoint import load, save
from flybrain.clock_encoder import ClockEncoder
from flybrain.dynamics import Brain, DynamicsConfig
from flybrain.graph import digest
from flybrain.policy import fly_only
from flybrain.sequence_io import lexical_spans
from flybrain.sequence_state import SequenceState
from models.fly_code import Generation


class NeuralSequence:
    def __init__(self, graph, vocabulary, *, role, seed=0, config=None):
        if role not in {"fly-code", "fly-nl"}:
            raise ValueError("Invalid sequence role")
        self.role, self.vocabulary = role, vocabulary
        self.brain = Brain(
            graph,
            vocabulary.radix,
            seed=seed,
            config=config
            or DynamicsConfig(inhibitory_fraction=0.2, sign_constraint=False),
        )
        self.encoder = self.make_encoder(graph, self.brain.input_nodes, seed, role)

    @staticmethod
    def make_encoder(graph, inputs, seed, role):
        return ClockEncoder(
            len(graph.node_ids),
            inputs,
            seed=seed,
            source_position_decay=role == "fly-code",
            ir_syntax=role == "fly-code",
            opaque_literals=role == "fly-nl",
        )

    def teacher_decisions(self, source, actions):
        state = SequenceState(source.encode().splitlines(), self.vocabulary)
        if not actions or actions[-1] != 0 or 0 in actions[:-1]:
            raise ValueError("Invalid teacher actions")
        for action in actions:
            if type(action) is not int or not 0 <= action < self.vocabulary.size:
                raise ValueError("Invalid teacher vocabulary action")
            if state.needs_context:
                if not self.vocabulary.context_start <= action < self.vocabulary.bind:
                    raise ValueError("Teacher must choose a context")
                for candidate in range(self.vocabulary.context_slots):
                    yield (
                        state.context_stimulus(self.encoder, candidate),
                        int(action == self.vocabulary.context_start + candidate),
                    )
            elif state.binding:
                candidates = state.copy_candidates()
                if action not in candidates:
                    raise ValueError("Invalid teacher pointer")
                for candidate in candidates:
                    yield (
                        state.pointer_stimulus(self.encoder, candidate),
                        int(action == candidate),
                    )
            else:
                partial = 0
                for digit in range(self.vocabulary.digits):
                    expected = (
                        action
                        // self.vocabulary.radix ** (self.vocabulary.digits - digit - 1)
                        % self.vocabulary.radix
                    )
                    yield state.stimulus(self.encoder, digit, partial), expected
                    partial = self.vocabulary.radix * partial + expected
            _, status = state.apply(action)
            if status and status != "eos":
                raise ValueError("Invalid teacher termination")

    def teacher_decision_count(self, source, actions):
        state = SequenceState(source.encode().splitlines(), self.vocabulary)
        count = 0
        for action in actions:
            count += (
                self.vocabulary.context_slots
                if state.needs_context
                else len(state.copy_candidates())
                if state.binding
                else self.vocabulary.digits
            )
            state.apply(action)
        return count

    def generate(
        self, source, *, max_actions=512, timeout=30.0, record=False, transmission=True
    ):
        if (
            not isinstance(source, str)
            or not source
            or len(source.encode()) > 8192
            or not 1 <= max_actions <= 4096
            or not 0 < timeout <= 60
        ):
            raise ValueError("Invalid sequence input or bounds")
        state = SequenceState(source.encode().splitlines(), self.vocabulary)
        output, trace = b"", []
        status, deadline = "max_tokens", time.monotonic() + timeout
        with fly_only():
            for _ in range(max_actions):
                raw = state.source()
                if (
                    len(lexical_spans(raw)) > self.vocabulary.max_spans
                    or len(raw) > 2048
                ):
                    status = "input_limit"
                    break

                def read(stimulus, event):
                    if time.monotonic() > deadline:
                        raise TimeoutError("timeout")
                    probabilities, _, activity = self.brain.forward(
                        stimulus, record=record, transmission=transmission
                    )
                    if record:
                        activity["activity_sha256"] = digest(activity.pop("activity"))
                        trace.append(
                            {
                                "cursor": state.cursor,
                                "phase": state.phase,
                                "depth": state.depth,
                                "context": state.context,
                                "text_mode": state.text_mode,
                                "register_count": len(state.registers),
                                **event,
                                "chosen": int(np.argmax(probabilities)),
                                "positive_score": float(probabilities[1]),
                                "neural": activity,
                            }
                        )
                    return probabilities

                try:
                    if state.needs_context:
                        scores = [
                            float(
                                read(
                                    state.context_stimulus(self.encoder, candidate),
                                    {"decision": "context", "candidate": candidate},
                                )[1]
                            )
                            for candidate in range(self.vocabulary.context_slots)
                        ]
                        action = self.vocabulary.context_start + int(np.argmax(scores))
                    elif state.binding:
                        candidates = state.copy_candidates()
                        if not candidates:
                            raise ValueError("invalid_binding")
                        scores = [
                            float(
                                read(
                                    state.pointer_stimulus(self.encoder, candidate),
                                    {"decision": "pointer", "candidate": candidate},
                                )[1]
                            )
                            for candidate in candidates
                        ]
                        action = candidates[int(np.argmax(scores))]
                    else:
                        action = 0
                        for digit in range(self.vocabulary.digits):
                            probabilities = read(
                                state.stimulus(self.encoder, digit, action),
                                {
                                    "decision": "digit",
                                    "digit": digit,
                                    "partial": action,
                                },
                            )
                            action = self.vocabulary.radix * action + int(
                                np.argmax(probabilities)
                            )
                    emitted, termination = state.apply(action)
                    output += emitted
                except TimeoutError:
                    status = "timeout"
                    break
                except ValueError as error:
                    status = str(error)
                    break
                if termination:
                    status = termination
                    break
                if len(output) > 12000:
                    status = "output_limit"
                    break
                if state.phase >= 128:
                    status = "phase_limit"
                    break
        return Generation(output.decode(), status, tuple(state.history), tuple(trace))

    def checkpoint_contract(self, contract):
        return {
            **contract,
            "sequence_encoder": self.encoder.version,
            "vocabulary": self.vocabulary.metadata(),
            "source_position_decay": self.encoder.source_position_decay,
            "ir_syntax": self.encoder.ir_syntax,
            "history_scope": "current input line",
        }

    def save(self, path, *, contract, training_state=None):
        save(
            self.brain,
            path,
            role=self.role,
            contract=self.checkpoint_contract(contract),
            training_state=training_state,
            encoder_version=self.encoder.version,
        )

    @classmethod
    def restore(cls, path, graph, vocabulary, *, role, contract):
        model = NeuralSequence(graph, vocabulary, role=role)
        brain, state = load(
            path,
            graph,
            role=role,
            contract=model.checkpoint_contract(contract),
            encoder_version=model.encoder.version,
        )
        if len(brain.output_nodes) != vocabulary.radix:
            raise ValueError("Sequence readout mismatch")
        model.brain = brain
        model.encoder = model.make_encoder(graph, brain.input_nodes, brain.seed, role)
        return model, state
