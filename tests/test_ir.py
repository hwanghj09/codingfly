import unittest

from ir.grammar import InvalidIR, validate
from ir.serializer import canonicalize
from ir.tokens import spec_hash


class IRTest(unittest.TestCase):
    def test_nested_blocks_and_prefix_expressions(self):
        source = ("FUNC sum_even\nLET total 0\nFOR i RANGE 1 11\n"
                  "IF (EQ (MOD i 2) 0)\nADD total i\nELSE\nPRINT i\nEND\nEND\n"
                  "RETURN total\nEND_FUNC\n")
        parsed = validate(source)
        self.assertEqual(len(parsed), 11)
        self.assertEqual(canonicalize(canonicalize(source)), canonicalize(source))
        self.assertEqual(len(spec_hash()), 64)

    def test_json_strings_unicode_names_lists_and_calls(self):
        source = 'LET 이름 "안녕, 친구"\nPRINT (LIST 이름 -3 TRUE)\nCALL run (CALL other 1)\n'
        self.assertEqual(len(validate(source)), 3)
        self.assertEqual(validate(source)[0].tokens, ("이름", '"안녕, 친구"'))

    def test_incomplete_invalid_and_unsupported_ir_is_not_repaired(self):
        for source in ("", "PRINT", "PRINT 1 2", "FOR i RANGE 1\nEND", "FUNC f\nPRINT 1\nEND",
                       "IF TRUE\n", "END", "ELSE", "RETURN 1", "LET for 2", "LET IF 2",
                       "LET x (ADD 1)", "LET x (ADD 1 2 3)", 'PRINT "unterminated',
                       "IF TRUE\nELSE\nELSE\nEND", "PRINT 1.5", "MAGIC x"):
            with self.subTest(source=source), self.assertRaises(InvalidIR):
                validate(source)

    def test_bound_expression_and_block_depth(self):
        for source in ("PRINT " + "(NEG " * 40 + "1" + ")" * 40,
                       "IF TRUE\n" * 40 + "END\n" * 40, 'PRINT "' + "a" * 9000 + '"'):
            with self.assertRaises(InvalidIR):
                validate(source)
