from enum import Enum
import re

###########
# Library #
###########

# This library supports a modified version of the shunting yard algorithm
# It introduces a limited deterministic finite state machine to parse the input by keeping track of the last token
# You can consider it a context free LL(1) LR(1) parser

# Utils


def str_match(string: str, m: str, l=None):
    l = l or (lambda x: x)
    if len(string) == 0:
        return None
    if string.startswith(m):
        return l(string[: len(m)]), len(m)
    return None


def re_match(string, r, l=None):
    l = l or (lambda x: x.group(0))
    m = re.match(r, string)
    if m:
        return l(m), m.end()
    return None


class Context:
    def __init__(
        self, variables: dict = None, functions: dict = None, parameters: dict = None
    ):
        self.variables = variables or {}
        self.functions = functions or {}
        self.parameters = parameters or {}


class IToken:
    m_re = None
    m_str = None

    re_l = None
    str_l = None

    def __init__(self, value, start, end):
        self.value = value
        self.start = start
        self.end = end

    @classmethod
    def match(cls, string):
        """
        A match function returns
        (value, length, ?class)

        The third argument is when you want a token depending on the context of the next n tokens
        """
        if cls.m_re:
            return re_match(string, cls.m_re, cls.re_l)
        elif cls.m_str:
            return str_match(string, cls.m_str, cls.str_l)
        raise NotImplementedError()

    @classmethod
    def LL(cls, string, gen_func):
        return cls.match(string)

    def evaluate(self, ctx):
        raise NotImplementedError(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.value.__repr__()}>"


class WhitespaceToken(IToken):
    pass


class SeparatorToken(IToken):
    pass


class ISingleExpression(IToken):
    pass


class ConstantToken(ISingleExpression):
    pass


class IdentifierToken(ISingleExpression):
    pass


class Function(IToken):
    arity = 1
    l = None

    def exec(self, ctx, args):
        if self.l is not None:
            return self.l(*args)


class UnaryOperatorToken(Function):
    arity = 1


class Associativity(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class BinaryOperatorToken(Function):
    """
    Must provide a precedence value
    Associativity is left by default
    """

    precedence = None
    associativity = Associativity.LEFT
    arity = 2


class OpenParenthesisToken(IToken):
    pass


class CloseParenthesisToken(IToken):
    pass


class IExpression:
    def __init__(self, arguments):
        self.arguments = arguments

    def evaluate(self):
        raise NotImplementedError()


class ParseError(Exception):
    def __init__(self, err, start, end):
        self.err = err
        self.start = start
        self.end = end

    def __str__(self):
        return f"@[{self.start}, {self.end}]: {self.err}"


class UnparsableText(ParseError):
    pass


class UnparsableToken(ParseError):
    pass


class Parser:
    def __init__(self, tokens):
        for token in tokens:
            if not issubclass(token, IToken):
                raise Exception(token, "is not a token")
            if issubclass(token, BinaryOperatorToken):
                if token.precedence is None:
                    raise Exception(
                        f"class '{token.__name__}' has no precedence. BinaryOperatorToken must have a precedence"
                    )
        self.tokens = tokens

    @staticmethod
    def should_pop_op(stack, op):
        if not stack:
            return False

        top_op = stack[-1]

        if isinstance(top_op, OpenParenthesisToken):
            return False

        if getattr(op, "associativity", None) is None:
            raise ValueError(f"{op.__repr__()} has no associativity. Illegal argument")

        if op.associativity == Associativity.LEFT:
            return op.precedence < top_op.precedence
        elif op.associativity == Associativity.RIGHT:
            return op.precedence <= top_op.precedence
        raise ValueError(
            f"{op.__repr__()} has invalid associativity value {op.associativity.__repr__()}. Should be one of {Associativity.LEFT.__repr__()} or {Associativity.RIGHT.__repr__()}"
        )

    @staticmethod
    def resolve_ambiguous_token_to(token, target_cls):
        for base in token.__class__.__bases__:
            if issubclass(base, target_cls):
                return base(token.value, token.start, token.end)
        raise Exception(
            f"'{token.__class__.__name__}' parents all don't contain '{target_cls.__name__}'"
        )

    def tokenize(self, string):
        pointer = 0
        while string:
            for token in self.tokens:
                if m := token.LL(string, self.tokenize):
                    v, l = m[:2]
                    cls = token
                    if len(m) >= 3:
                        cls = m[2]
                    if not issubclass(cls, WhitespaceToken):
                        yield cls(v, pointer, pointer + l)
                    pointer += l
                    string = string[l:]
                    break
            else:
                raise UnparsableText(string, pointer, len(string))

    def parse_to_rpn(self, tokens):
        # Used to determine state
        last_token = None

        output = []
        operator_stack = []

        for token in tokens:
            print(f"on token {token.value}: ", end="")

            if isinstance(token, ISingleExpression):
                output.append(token)
            elif isinstance(token, OpenParenthesisToken):
                operator_stack.append(token)
            elif isinstance(token, CloseParenthesisToken):
                # Check for empty parentheses and function calls here
                if last_token is None:
                    raise UnparsableToken(
                        "Closing parenthesis at the start of an expression",
                        token.start,
                        token.end,
                    )
                while not isinstance(operator_stack[-1], OpenParenthesisToken):
                    output.append(operator_stack.pop())
                operator_stack.pop()

                # Parentheses are already removed here
                if isinstance(last_token, OpenParenthesisToken):
                    if len(operator_stack) == 0 or not (
                        r := isinstance(operator_stack[-1], Function)
                    ):
                        raise UnparsableToken(
                            "Empty expression", token.start, token.end
                        )

                    f = operator_stack.pop()

                    # Arity is 0
                    if r:
                        f.arity = 0

                    output.append(f)

            elif isinstance(token, Function):
                operator_stack.append(token)
            elif isinstance(token, SeparatorToken):
                while operator_stack and not isinstance(
                    operator_stack[-1], OpenParenthesisToken
                ):
                    output.append(operator_stack.pop())

                # This is the function
                operator_stack[-2].arity += 1
            elif isinstance(token, BinaryOperatorToken):
                # This is the "magical" part of the algorithm
                # If the last token is a binary operator, this is a unary operator
                if isinstance(last_token, BinaryOperatorToken) and isinstance(
                    token, UnaryOperatorToken
                ):
                    output.append(
                        self.resolve_ambiguous_token_to(token, UnaryOperatorToken)
                    )
                else:
                    if self.should_pop_op(operator_stack, token):
                        output.append(operator_stack.pop())

                    operator_stack.append(token)
            elif isinstance(token, UnaryOperatorToken):
                operator_stack.append(token)
            else:
                raise UnparsableToken(
                    f"{token.__repr__()} cannot be processed", token.start, token.end
                )
            print([v.value for v in output], [v.value for v in operator_stack])

            last_token = token

        # print(output, operator_stack)

        return [*output, *operator_stack[::-1]]

    def rpn_to_ast(self, rpn):
        stack = []
        for token in rpn:
            if isinstance(token, ISingleExpression):
                stack.append(token)
            elif isinstance(token, Function):
                args = [stack.pop() for _ in range(token.arity)]
                stack.append(Expression(token, args))

        if len(stack) != 1:
            raise Exception(f"Invalid RPN. Stack: ", stack)

        return stack[0]


class Expression:
    def __init__(self, op, args):
        self.op = op
        self.args = args

    def evaluate(self, ctx):
        return self.op.exec(ctx, [v.evaluate(ctx) for v in self.args])

    def __repr__(self):
        return f"{self.op.__repr__()}({', '.join(map(str, self.args))})"


# Built in tokens
# Some of these can be simplified using generics and using
# case-by-case evaluation to decrease developer headaches


class Number(ConstantToken):
    re_l = lambda x: float(x.group(0)) if "." in x.group(0) else int(x.group(0))
    m_re = r"[0-9]+(?:\.\d+)?"

    def evaluate(self, ctx):
        return self.value


class Boolean(ConstantToken):
    re_l = lambda x: x.group(0)[0].lower() == "t"
    m_re = r"T(?:rue|RUE)?|F(?:alse|ALSE)?"

    def evaluate(self, ctx):
        return self.value


class Whitespace(WhitespaceToken):
    m_re = r"\s+"


class Separator(SeparatorToken):
    m_re = r","


class Literal(ISingleExpression):
    m_str = r"[a-zA-Z_][a-zA-Z0-9_]*"

    def evaluate(self, ctx):
        return ctx.variables[self.value]


class Variable(IdentifierToken):
    def evaluate(self, ctx):
        return ctx.variables[self.value]


class Parameter(IdentifierToken):
    def evaluate(self, ctx):
        return ctx.parameters[self.value]


class IdentifierFunction(Function):
    def exec(self, ctx, args):
        print("args recieved", self, args)
        # TODO
        # Fix this later
        return (
            ctx.functions[self.value].evaluate(
                Context(
                    ctx.variables,
                    ctx.functions,
                    {f"${i}": v for i, v in enumerate(args)},
                )
            )
            if isinstance(ctx.functions[self.value], Expression)
            else ctx.functions[self.value](*args)
        )


class Identifier(IToken):
    m_re = r"[a-zA-Z_][a-zA-Z0-9_]*"

    @classmethod
    def LL(cls, string, gen):
        if m := re.match(cls.m_re, string):
            # is identifier
            token = next(gen(string[m.end() :]), None)
            constructor = (
                IdentifierFunction
                if not token is None and isinstance(token, OpenParenthesisToken)
                else Variable
            )
            return m.group(0), m.end(), constructor


class OpenParenthesis(OpenParenthesisToken):
    m_str = "("


class CloseParenthesis(CloseParenthesisToken):
    m_str = ")"


class Multiplication(BinaryOperatorToken):
    precedence = 1
    m_str = "*"

    def exec(self, ctx, args):
        return args[0] * args[1]


class Division(BinaryOperatorToken):
    precedence = 1
    m_str = "/"

    def exec(self, ctx, args):
        return args[0] / args[1]


class AdditionAsBinary(BinaryOperatorToken):
    precedence = 0

    def exec(self, ctx, args):
        return args[0] + args[1]


class AdditionAsUnary(UnaryOperatorToken):
    def exec(self, ctx, args):
        return +args[0]


class Addition(AdditionAsBinary, AdditionAsUnary):
    precedence = 0
    m_str = "+"


class SubtractionAsBinary(BinaryOperatorToken):
    precedence = 0

    def exec(self, ctx, args):
        return args[0] - args[1]


class SubtractionAsUnary(UnaryOperatorToken):
    def exec(self, ctx, args):
        return -args[0]


class Subtraction(SubtractionAsBinary, SubtractionAsUnary):
    precedence = 0
    m_str = "-"


class Negation(UnaryOperatorToken):
    m_re = r"!|~"

    def exec(self, ctx, args):
        return not args[0]


########################################################################


tokens = [
    # Whitespace
    Whitespace,
    # Control flow
    OpenParenthesis,
    CloseParenthesis,
    Separator,
    # Identifiers
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

tokenized = list(parser.tokenize("square(3)"))

print(tokenized)

rpn = parser.parse_to_rpn(tokenized)

print(rpn)

ast = parser.rpn_to_ast(rpn)

print(ast)

hello_func = parser.rpn_to_ast(parser.parse_to_rpn(parser.tokenize("1 + 1")))

print(hello_func)

print(ast.evaluate(Context({}, {"Hello": hello_func, "square": lambda x: x**2})))
