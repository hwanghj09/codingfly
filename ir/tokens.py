"""Frozen IR contract, shared by datasets, validators and checkpoints."""

import hashlib
from pathlib import Path


IR_VERSION = "0.1"
COMMANDS = frozenset({"FUNC", "LET", "PRINT", "ADD", "IF", "ELSE", "FOR", "RETURN",
                      "CALL", "END", "END_FUNC"})
BINARY = frozenset({"ADD", "SUB", "MUL", "DIV", "MOD", "EQ", "NE", "LT", "LE", "GT", "GE",
                    "AND", "OR"})
UNARY = frozenset({"NOT", "NEG"})


def spec_hash():
    return hashlib.sha256((Path(__file__).resolve().parents[1] / "docs/IR_SPEC.md").read_bytes()).hexdigest()
