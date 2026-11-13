"""Canonical IR text, never Python text."""

from ir.grammar import validate


def canonicalize(source):
    return "\n".join(" ".join((statement.command, *statement.tokens))
                     for statement in validate(source)) + "\n"
