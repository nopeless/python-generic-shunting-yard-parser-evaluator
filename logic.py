import itertools
from shunting_yard_generic_parser import Context, Parser, Tokenlib


def boolean_permutation(length):
    return itertools.product([True, False], repeat=length)


tokens = Tokenlib.load(
    Tokenlib.base, Tokenlib.logical, Tokenlib.arithmetic, Tokenlib.numeric
)

parser = Parser(tokens)

p = parser.parse("False v False")

print(p)

print(p.evaluate())
