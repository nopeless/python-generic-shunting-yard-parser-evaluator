from src.shunting_yard_generic_parser import *

import itertools

import pandas as pd


def boolean_permutation(length):
    return itertools.product([True, False], repeat=length)


tokens = Tokenlib.load(Tokenlib.base, Tokenlib.logical, Tokenlib.functional)

parser = Parser(tokens)


def generate_truth_table(inp):
    """
    Returns pandas dataframe
    """
    p = parser.parse(inp)

    variables = get_variables(p)

    headers = [*variables, inp]
    rows = []

    for perm in boolean_permutation(len(variables)):
        res = p.evaluate(LexicalScope({var: val for var, val in zip(variables, perm)}))
        rows.append([*perm, res])

    return pd.DataFrame(rows, columns=headers)


def print_truth_table(inp):
    s = generate_truth_table(inp).to_string(index=False)

    print(s)


print_truth_table("P => (Q => R)")
print_truth_table("(P => Q) => R")
