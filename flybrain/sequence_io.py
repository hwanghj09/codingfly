"""Fixed lexical byte spans and literal/COPY actions shared by both neural roles."""

from dataclasses import dataclass

from flybrain.decoder import Span


PYTHON_LITERALS = (
    b"",
    b" ",
    b"\n",
    b"    ",
    b"=",
    b"print",
    b"(",
    b")",
    b":",
    b",",
    b"def",
    b"return",
    b"if",
    b"else",
    b"for",
    b"in",
    b"range",
    b"+=",
    b"-=",
    b"+",
    b"-",
    b"*",
    b"%",
    b"==",
    b"!=",
    b"<",
    b">",
    b"<=",
    b">=",
    b"True",
    b"False",
    b"and",
    b"or",
    b"not",
    b"[",
    b"]",
    b"//",
    b"/",
)
IR_LITERALS = tuple(
    token.encode()
    for token in (
        "",
        " ",
        "\n",
        "LET",
        "PRINT",
        "ADD",
        "FUNC",
        "END_FUNC",
        "FOR",
        "RANGE",
        "END",
        "IF",
        "ELSE",
        "RETURN",
        "CALL",
        "(",
        ")",
        "EQ",
        "NE",
        "LT",
        "LE",
        "GT",
        "GE",
        "MOD",
        "SUB",
        "MUL",
        "DIV",
        "NOT",
        "NEG",
        "AND",
        "OR",
        "LIST",
        "TRUE",
        "FALSE",
        "UNKNOWN",
    )
)
SEQUENCE_IO_VERSION = "lexical-span-transducer-actions-v2"


def lexical_spans(source):
    spans, position = [], 0
    boundaries = b"()[],:"
    whitespace = b" \t\r\n\v\f"
    while position < len(source):
        if source[position] in whitespace:
            position += 1
            continue
        end = position + 1
        if source[position] == 34:
            while end < len(source):
                if source[end] == 92:
                    end += 2
                    continue
                if source[end] == 34:
                    end += 1
                    break
                end += 1
        elif source[position] not in boundaries:
            while (
                end < len(source) and source[end] not in whitespace + boundaries + b'"'
            ):
                end += 1
        spans.append(Span(position, min(end, len(source))))
        position = end
    return tuple(spans)


@dataclass(frozen=True)
class Vocabulary:
    literals: tuple[bytes, ...]
    max_spans: int = 23
    radix: int = 2
    inner_copy: bool = False
    context_slots: int = 0
    register_slots: int = 0

    def __post_init__(self):
        if (
            not self.literals
            or self.literals[0] != b""
            or len(set(self.literals)) != len(self.literals)
            or not 1 <= self.max_spans <= 256
            or self.radix not in (2, 4)
            or not 0 <= self.context_slots <= 16
            or not 0 <= self.register_slots <= 16
            or self.register_slots
            and not self.context_slots
            or self.size > 256
        ):
            raise ValueError("Invalid fixed output vocabulary")

    @property
    def size(self):
        return self.register_start + self.register_slots

    @property
    def next_line(self):
        return len(self.literals) + self.max_spans * (2 if self.inner_copy else 1)

    @property
    def push(self):
        return self.next_line + 1

    @property
    def pop(self):
        return self.next_line + 2

    @property
    def context_start(self):
        return self.next_line + 3

    @property
    def bind(self):
        return self.context_start + self.context_slots

    @property
    def begin_text(self):
        return self.bind + 1

    @property
    def register_start(self):
        return self.bind + (2 if self.context_slots else 0)

    @property
    def digits(self):
        count, capacity = 1, self.radix
        while capacity < self.size:
            count, capacity = count + 1, capacity * self.radix
        return count

    def emit(self, action, source, spans):
        if type(action) is not int or not 0 <= action < self.size:
            raise ValueError("invalid_action")
        if action < len(self.literals):
            return self.literals[action]
        if action >= self.next_line:
            raise ValueError("control_action")
        index = action - len(self.literals)
        inner = self.inner_copy and index >= self.max_spans
        if inner:
            index -= self.max_spans
        if index >= len(spans):
            raise ValueError("invalid_copy")
        span = spans[index]
        if inner:
            if (
                source[span.start : span.start + 1] != b'"'
                or source[span.end - 1 : span.end] != b'"'
            ):
                raise ValueError("invalid_inner_copy")
            return source[span.start + 1 : span.end - 1]
        return source[span.start : span.end]

    def metadata(self):
        metadata = {
            "version": SEQUENCE_IO_VERSION,
            "literals": [v.hex() for v in self.literals],
            "max_spans": self.max_spans,
            "radix": self.radix,
            "digits": self.digits,
        }
        if self.inner_copy:
            metadata["inner_copy"] = True
        if self.context_slots:
            metadata.update(
                context_slots=self.context_slots,
                register_slots=self.register_slots,
                memory_protocol="context-bind-text-v1",
            )
        return metadata
