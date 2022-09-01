from src.shunting_yard_generic_parser import *

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

print(ast)

hello_func = parser.rpn_to_ast(parser.parse_to_rpn(parser.tokenize("$0 * 100")))

print("hello function is", hello_func)
print("expression is", ast)

print(
    ast.evaluate(
        LexicalScope(
            {"P": 3, "Hello": hello_func, "square": lambda x: x**2, "type": type}
        )
    )
)

print(get_variables(ast))
