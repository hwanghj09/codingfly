"""Fixed neural action decoding and literal span assembly, with no grammar repair."""

from dataclasses import dataclass

import numpy as np


LITERALS = (b"", b" ", b"\n", b"=", b"print", b"(", b")")
EOS = 0
DECODER_VERSION = "literal-or-whitespace-span-v1"


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def whitespace_spans(source):
    """Byte boundaries only; never classifies command names or argument roles."""
    spans = []
    position = 0
    whitespace = b" \t\r\n\v\f"
    while position < len(source):
        if source[position] in whitespace:
            position += 1
            continue
        end = position + 1
        while end < len(source) and source[end] not in whitespace:
            end += 1
        spans.append(Span(position, end))
        position = end
    return tuple(spans)


class ActionDecoder:
    def __init__(self, max_spans=16):
        if not 1 <= max_spans <= 128:
            raise ValueError("Invalid span vocabulary size")
        self.max_spans = max_spans
        self.size = len(LITERALS) + max_spans

    def choose(self, probabilities):
        if len(probabilities) != self.size or not np.isfinite(probabilities).all():
            raise ValueError("Invalid neural output")
        # No grammar mask, nearest-template lookup, or default-on-failure.
        return int(np.argmax(probabilities))

    def emit(self, action, source, spans):
        if not 0 <= action < self.size:
            raise ValueError("Invalid output action")
        if action < len(LITERALS):
            return LITERALS[action]
        index = action - len(LITERALS)
        if index >= len(spans):
            raise ValueError("invalid_copy")
        span = spans[index]
        return source[span.start:span.end]

    def labels(self):
        return ["EOS", "SPACE", "NEWLINE", "=", "print", "(", ")"] + [
            f"COPY({i})" for i in range(self.max_spans)]
