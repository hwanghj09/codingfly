"""Literal/pointer I/O registers. No language keywords or command-to-code mappings."""


class SequenceState:
    def __init__(self, lines, vocabulary):
        self.lines, self.vocabulary = lines, vocabulary
        self.cursor, self.phase, self.depth = 0, 0, 0
        self.context, self.binding = -1, False
        self.text_mode = not bool(vocabulary.context_slots)
        self.registers, self.history = [], []

    def source(self):
        if self.cursor >= len(self.lines):
            return b"\x00"
        if self.vocabulary.context_slots and self.context >= 0 and not self.binding:
            return b"\x02" if self.text_mode else b"\x03"
        return self.lines[self.cursor] or b"\x01"

    @property
    def needs_context(self):
        return bool(
            self.vocabulary.context_slots and self.context < 0 and not self.text_mode
        )

    def context_stimulus(self, encoder, candidate):
        return encoder.encode(
            self.source(), 0, 0, contextual=True, context=candidate, text_mode=False
        )

    def copy_candidates(self):
        from flybrain.sequence_io import lexical_spans

        if not 0 <= self.cursor < len(self.lines):
            raise ValueError("invalid_copy")
        raw = self.lines[self.cursor]
        v = self.vocabulary
        used = {action for cursor, action in self.registers if cursor == self.cursor}
        candidates = []
        for index, span in enumerate(lexical_spans(raw)):
            if index >= v.max_spans:
                break
            token = raw[span.start : span.end]
            full = len(v.literals) + index
            if full not in used:
                candidates.append(full)
            if v.inner_copy and token[:1] == b'"' and token[-1:] == b'"':
                inner = len(v.literals) + v.max_spans + index
                if inner not in used:
                    candidates.append(inner)
        return candidates

    def pointer_stimulus(self, encoder, action):
        import numpy as np
        from flybrain.sequence_io import lexical_spans

        v = self.vocabulary
        raw = self.lines[self.cursor]
        spans = lexical_spans(raw)
        offset = action - len(v.literals)
        inner = offset >= v.max_spans
        position = offset % v.max_spans
        if position >= len(spans):
            raise ValueError("invalid_copy")

        def kind(index):
            token = raw[spans[index].start : spans[index].end]
            return (
                2
                if token[:1] == b'"' and token[-1:] == b'"'
                else 1
                if token.removeprefix(b"-").isdigit()
                else 0
            )

        datatype = kind(position)
        rank = (
            1
            + sum(
                (candidate - len(v.literals)) % v.max_spans < position
                and ((candidate - len(v.literals)) >= v.max_spans) == inner
                and kind((candidate - len(v.literals)) % v.max_spans) == datatype
                for candidate in self.copy_candidates()
            )
            if datatype
            else 0
        )
        window = b" ".join(
            raw[span.start : span.end]
            for span in spans[max(0, position - 2) : position + 3]
        )
        drive = encoder.encode(
            window,
            0,
            0,
            contextual=True,
            context=self.context,
            binding=True,
            text_mode=False,
            register_count=len(self.registers),
        ).astype(np.float64)
        metadata = np.array(
            [0x100 + (2 if inner else 1), 0x200 + datatype, 0x300 + rank],
            dtype=np.uint64,
        )
        clock = (
            (self.context + 1) * 0x9E3779B97F4A7C15
            ^ (len(self.registers) + 1) * 0xBF58476D1CE4E5B9
        ) & ((1 << 64) - 1)
        drive += 2 * encoder.project(metadata, clock)
        return drive.astype(np.float32)

    def stimulus(self, encoder, digit, partial):
        context = self.history[-min(self.phase, 4) :] if self.phase else []
        return encoder.encode(
            self.source(),
            self.phase,
            digit,
            partial,
            context,
            self.depth,
            contextual=bool(self.vocabulary.context_slots),
            context=self.context,
            binding=self.binding,
            text_mode=self.text_mode,
            register_count=len(self.registers),
        )

    def copy(self, cursor, action):
        from flybrain.sequence_io import lexical_spans

        if not 0 <= cursor < len(self.lines):
            raise ValueError("invalid_copy")
        raw = self.lines[cursor]
        return self.vocabulary.emit(action, raw, lexical_spans(raw))

    def apply(self, action):
        v = self.vocabulary
        if type(action) is not int or not 0 <= action < v.size:
            raise ValueError("invalid_action")
        self.history.append(action)
        if action == 0:
            status = (
                "eos"
                if self.cursor == len(self.lines)
                and self.depth == 0
                and self.text_mode
                and not self.binding
                else "premature_eos"
            )
            return b"", status
        if self.binding:
            if (
                not len(v.literals) <= action < v.next_line
                or len(self.registers) >= v.register_slots
            ):
                raise ValueError("invalid_binding")
            self.copy(self.cursor, action)
            self.registers.append((self.cursor, action))
            self.binding = False
            self.phase += 1
            return b"", None
        if action == v.next_line:
            self.cursor += 1
            self.phase = 0
            if self.cursor > len(self.lines):
                raise ValueError("cursor_overrun")
            return b"", None
        self.phase += 1
        if action in (v.push, v.pop):
            self.depth += 1 if action == v.push else -1
            if not 0 <= self.depth <= 16:
                raise ValueError("stack_error")
            return b"", None
        if v.context_slots:
            if v.context_start <= action < v.bind:
                if self.text_mode:
                    raise ValueError("invalid_context")
                self.context = action - v.context_start
                return b"", None
            if action == v.bind:
                if self.text_mode or self.context < 0:
                    raise ValueError("invalid_binding")
                self.binding = True
                return b"", None
            if action == v.begin_text:
                if self.text_mode or self.context < 0:
                    raise ValueError("invalid_context")
                self.text_mode = True
                self.phase = 0
                return b"", None
            if v.register_start <= action < v.size:
                index = action - v.register_start
                if not self.text_mode or index >= len(self.registers):
                    raise ValueError("invalid_register")
                return self.copy(*self.registers[index]), None
            if not self.text_mode:
                raise ValueError("invalid_header")
        if action < len(v.literals):
            return v.literals[action], None
        if self.cursor >= len(self.lines):
            raise ValueError("invalid_copy")
        return self.copy(self.cursor, action), None
