from src.shunting_yard_generic_parser import *

########################################################################
tokens = [
    # Whitespace
    Whitespace,
    # Control flow
    OpenParenthesis,
    CloseParenthesis,
    Separator,
    # Identifiers
    Parameter,
    Identifier,
    # Literals
    Number,
    Boolean,
    # Unary operators
    Negation,
    # Arithmetic operators (including Unary)
    Multiplication,
    Division,
    Addition,
    Subtraction,
]

parser = Parser(tokens)

tokenized = list(parser.tokenize("Hello(P) + Hello(P)"))

print(tokenized)

rpn = parser.parse_to_rpn(tokenized)

print(rpn)

ast = parser.rpn_to_ast(rpn)

print("expression is", ast)

print(ast.evaluate(LexicalScope({"P": False, "Q": True, "R": False})))

print(get_variables(ast))
