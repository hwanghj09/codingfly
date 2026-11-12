"""Clocked byte population coding. Every slot uses the same nonsemantic hash rule."""

from functools import lru_cache
import hashlib

import numpy as np

from flybrain.sequence_io import IR_LITERALS, lexical_spans


CLOCK_ENCODER_VERSION = "clocked-line-byte-population-v2.6"
NL_ENCODER_VERSION = "clocked-nl-pointer-bytes-v3.0"
IR_SYNTAX_IDS = {
    value: index + 1
    for index, value in enumerate(IR_LITERALS)
    if value and not value.isspace() and value != b"UNKNOWN"
}


class ClockEncoder:
    def __init__(
        self,
        node_count,
        input_nodes,
        *,
        seed=0,
        fanout=16,
        source_position_decay=False,
        ir_syntax=False,
        opaque_literals=False,
    ):
        self.node_count, self.input_nodes = node_count, tuple(input_nodes)
        self.seed, self.fanout = seed, fanout
        self.source_position_decay = source_position_decay
        self.ir_syntax = ir_syntax
        self.opaque_literals = opaque_literals

    @property
    def version(self):
        return NL_ENCODER_VERSION if self.opaque_literals else CLOCK_ENCODER_VERSION

    @staticmethod
    @lru_cache(maxsize=256)
    def features(source, ir_syntax=False, opaque_literals=False):
        spans = lexical_spans(source)
        values = [(1, 0, len(spans))]
        fingerprints = [len(spans)]
        for index, span in enumerate(spans):
            token = source[span.start : span.end]
            quoted = len(token) >= 2 and token[:1] == b'"' and token[-1:] == b'"'
            numeric = token.removeprefix(b"-").isdigit()
            literal = opaque_literals and (quoted or numeric)
            # An opaque hash of raw bytes, not a vocabulary ID or word meaning.
            fingerprint = (
                IR_SYNTAX_IDS.get(token, 0)
                if ir_syntax
                else (0xF077AC1 if quoted else 0xF077AC2)
                if literal
                else int.from_bytes(
                    hashlib.blake2b(token, digest_size=8).digest(), "little"
                )
            )
            fingerprints.append(
                fingerprint ^ (((index + 1) * 0xD6E8FEB86659FD93) & ((1 << 64) - 1))
            )
            values.extend(
                [
                    (index + 2, 0, 256),
                    (index + 2, 1 if literal else len(token) + 1, 257),
                ]
            )
            if not literal:
                values.extend(
                    (index + 2, position + 1, value)
                    for position, value in enumerate(token)
                )
        rows = np.array(values, dtype=np.uint64)
        result = (
            rows[:, 0] * np.uint64(0x9E3779B97F4A7C15)
            ^ rows[:, 1] * np.uint64(0xBF58476D1CE4E5B9)
            ^ rows[:, 2]
        )
        result.flags.writeable = False
        opaque = np.array(fingerprints, dtype=np.uint64)
        opaque.flags.writeable = False
        return opaque, result

    @staticmethod
    def mix(values):
        values = values ^ (values >> np.uint64(30))
        values = values * np.uint64(0xBF58476D1CE4E5B9)
        values = values ^ (values >> np.uint64(27))
        values = values * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))

    @staticmethod
    @lru_cache(maxsize=256)
    def nl_features(source):
        bag, literals, bytes_view = [], [], []
        for index, span in enumerate(lexical_spans(source)):
            token = source[span.start : span.end]
            quoted = len(token) >= 2 and token[:1] == b'"' and token[-1:] == b'"'
            numeric = token.removeprefix(b"-").isdigit()
            fingerprint = (
                (0xF077AC1 if quoted else 0xF077AC2)
                if quoted or numeric
                else int.from_bytes(
                    hashlib.blake2b(token, digest_size=8).digest(), "little"
                )
            )
            bag.append(fingerprint)
            if quoted or numeric:
                literals.append(
                    fingerprint ^ (((index + 1) * 0xD6E8FEB86659FD93) & ((1 << 64) - 1))
                )
            else:
                bytes_view.extend(
                    (((j + 1) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)) ^ value
                    for j, value in enumerate(token)
                )
        return tuple(
            np.array(values or [0], dtype=np.uint64)
            for values in (bag, literals, bytes_view)
        )

    def project(self, features, clock, weights=None):
        offsets = np.arange(1, self.fanout + 1, dtype=np.uint64) * np.uint64(
            0x94D049BB133111EB
        )
        clock_key = self.mix(np.array([clock], dtype=np.uint64))[0]
        numbers = self.mix(features[:, None] ^ offsets[None, :] ^ clock_key)
        indices = (
            ((numbers >> np.uint64(1)) % len(self.input_nodes)).ravel().astype(np.int64)
        )
        signs = (numbers & np.uint64(1)).astype(np.int64).ravel() * 2 - 1
        weights = np.ones(len(features)) if weights is None else np.asarray(weights)
        signs = signs * np.repeat(weights, self.fanout)
        drive = np.zeros(self.node_count, dtype=np.float64)
        drive[list(self.input_nodes)] = np.bincount(
            indices, weights=signs, minlength=len(self.input_nodes)
        )
        return drive / float(np.linalg.norm(weights))

    def encode(
        self,
        source,
        step,
        digit,
        partial=0,
        history=(),
        depth=0,
        *,
        contextual=False,
        context=-1,
        binding=False,
        text_mode=True,
        register_count=0,
    ):
        if (
            not isinstance(source, bytes)
            or not source
            or len(source) > 2048
            or not 0 <= step < 1024
            or not 0 <= digit < 8
            or not 0 <= partial < 65536
            or not 0 <= depth <= 16
        ):
            raise ValueError("Invalid clocked input")
        clock = (
            self.seed * 0x9E3779B97F4A7C15
            ^ step * 0xBF58476D1CE4E5B9
            ^ digit * 0x94D049BB133111EB
            ^ partial * 0xD6E8FEB86659FD93
            ^ depth * 0x8EBC6AF09C88C6E3
        ) & ((1 << 64) - 1)
        if contextual:
            clock ^= (
                (context + 1) * 0x589965CC75374CC3
                ^ int(binding) * 0xA0761D6478BD642F
                ^ int(text_mode) * 0xE7037ED1A0B428DB
                ^ register_count * 0x1D8E4E27C47D124F
            ) & ((1 << 64) - 1)
        opaque, raw = self.features(source, self.ir_syntax, self.opaque_literals)
        weights = (
            np.r_[2.0, 4.0 / np.arange(1, len(opaque), dtype=np.float64) ** 1.5]
            if self.source_position_decay
            else None
        )
        drive = self.project(opaque, clock, weights)
        if self.opaque_literals:
            bag, literal_positions, word_bytes = self.nl_features(source)
            drive = (
                self.project(bag, clock)
                + 0.75 * self.project(literal_positions, clock ^ 0x589965CC75374CC3)
                + 0.1 * drive
                + 0.1 * self.project(word_bytes, clock ^ 0x1D8E4E27C47D124F)
            )
        elif not self.ir_syntax:
            drive += 0.02 * self.project(raw, clock ^ 0xE7037ED1A0B428DB)
        # A bounded, fixed-strength context channel uses actual earlier actions.
        # It does not reinsert variable-length copied text or decode source meaning.
        values = []
        for position, action in enumerate(history[-4:]):
            if type(action) is not int or not 0 <= action < 65536:
                raise ValueError("Invalid output history")
            values.append((position + 1) * 65536 + action)
        if history:
            drive += 0.1 * self.project(
                np.array(values, dtype=np.uint64), clock ^ 0xA0761D6478BD642F
            )
        return drive.astype(np.float32)
