from enum import Enum
import re

from .scope import LexicalScope, NullScope

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


def pad_newlines_or_blank(string: str, padwith: str = "  "):
    if string:
        return "\n" + "\n".join([padwith + v for v in string.split("\n")]) + "\n"


def args_to_dict(args):
    return {f"${idx}": v for idx, v in enumerate(args)}


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

    def exec(self, scope, args):
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
            # print(f"on token {token.value}: ", end="")

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

                # Pop the open parenthesis
                operator_stack.pop()

                is_empty_expression = isinstance(last_token, OpenParenthesisToken)

                function_exists = False

                # Function does not exist
                if len(operator_stack) == 0 or not (
                    function_exists := isinstance(operator_stack[-1], Function)
                ):
                    # Check whether its an empty expression paren. If there is a function,allow it
                    if is_empty_expression:

                        raise UnparsableToken(
                            "Empty expression", token.start, token.end
                        )

                if function_exists:
                    f = operator_stack.pop()

                    # Arity is 0
                    if is_empty_expression:
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
            # print([v.value for v in output], [v.value for v in operator_stack])

            last_token = token

        # print(output, operator_stack)

        return [*output, *operator_stack[::-1]]

    def rpn_to_ast(self, rpn):
        stack = []
        for token in rpn:
            if isinstance(token, ISingleExpression):
                stack.append(token)
            elif isinstance(token, Function):
                args = [stack.pop() for _ in range(token.arity)][::-1]
                stack.append(Expression(token, args))

        if len(stack) != 1:
            raise Exception(f"Invalid RPN. Stack: ", stack)

        return Scope(stack)

    def parse(self, string):
        """
        One godly function
        """
        return self.rpn_to_ast(self.parse_to_rpn(self.tokenize(string)))


class Expression(IExpression):
    def __init__(self, op, args):
        self.op = op
        self.args = args

    def evaluate(self, scope):
        return self.op.exec(scope, [v.evaluate(scope) for v in self.args])

    def __repr__(self):
        return f"{self.op.__repr__()}({pad_newlines_or_blank(', '.join(map(str, self.args)))})"


# TODO write proper AST visitor later
def ast_visitor(ast, l):
    if isinstance(ast, IExpression):
        l(ast)
        for arg in ast.args:
            ast_visitor(arg, l)

    else:
        l(ast)


def get_variables(ast):
    variables = set()
    ast_visitor(
        ast,
        lambda x: variables.add(x.value) if isinstance(x, Variable) else None,
    )
    return [*variables]


class Scope(IExpression):
    def __init__(self, exps, scope=None):
        # Future proof
        self.exp = exps[0]
        self.scope = scope

    @property
    def args(self):
        return [self.exp]

    def evaluate(self, scope=None):
        return self.exp.evaluate(scope or self.scope or NullScope())

    def __repr__(self):
        return f"Scope({self.exp.__repr__()})"


# Built in tokens
# Some of these can be simplified using generics and using
# case-by-case evaluation to decrease developer headaches


class Number(ConstantToken):
    re_l = lambda x: float(x.group(0)) if "." in x.group(0) else int(x.group(0))
    m_re = r"[0-9]+(?:\.\d+)?"

    def evaluate(self, scope):
        return self.value


class Boolean(ConstantToken):
    re_l = lambda x: x.group(0)[0].lower() == "t"
    m_re = r"T(?:rue|RUE)?|F(?:alse|ALSE)?"

    def evaluate(self, scope):
        return self.value


class Whitespace(WhitespaceToken):
    m_re = r"\s+"


class Separator(SeparatorToken):
    m_re = r","


class Variable(IdentifierToken):
    m_re = r"[a-zA-Z_][a-zA-Z0-9_]*"

    def evaluate(self, scope):
        return scope.get_identifier(self.value)


# Acts almost like a Variable
class Parameter(Variable):
    m_re = r"\$(?:[1-9]\d*|0)"


class IdentifierFunction(Function):
    def exec(self, scope, args):
        return (
            scope.get_identifier(self.value).evaluate(
                scope.inner_scope(args_to_dict(args))
            )
            # Runtime check
            if isinstance(scope.get_identifier(self.value), IExpression)
            else scope.get_identifier(self.value)(*args)
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


class Power(BinaryOperatorToken):
    associativity = Associativity.RIGHT
    precedence = 4
    m_re = r"\*\*"

    def exec(self, scope, args):
        return args[0] ** args[1]


class Multiplication(BinaryOperatorToken):
    precedence = 3
    m_str = "*"

    def exec(self, scope, args):
        return args[0] * args[1]


class Division(BinaryOperatorToken):
    precedence = 3
    m_str = "/"

    def exec(self, scope, args):
        return args[0] / args[1]


class AdditionAsBinary(BinaryOperatorToken):
    precedence = 2

    def exec(self, scope, args):
        return args[0] + args[1]


class AdditionAsUnary(UnaryOperatorToken):
    def exec(self, scope, args):
        return +args[0]


class Addition(AdditionAsBinary, AdditionAsUnary):
    m_str = "+"


class SubtractionAsBinary(BinaryOperatorToken):
    precedence = 2

    def exec(self, scope, args):
        return args[0] - args[1]


class SubtractionAsUnary(UnaryOperatorToken):
    def exec(self, scope, args):
        return -args[0]


class Subtraction(SubtractionAsBinary, SubtractionAsUnary):
    m_str = "-"


class And(BinaryOperatorToken):
    precedence = 1
    m_re = r"&&|&|\^"

    def exec(self, scope, args):
        return args[0] and args[1]


class Or(BinaryOperatorToken):
    precedence = 0
    m_re = r"\|\||\||v"

    def exec(self, scope, args):
        return args[0] or args[1]


class Implies(BinaryOperatorToken):
    precedence = 0
    m_str = r"=>"

    def exec(self, scope, args):
        return not args[0] or args[1]


class Iff(BinaryOperatorToken):
    precedence = 0
    m_str = r"<=>"

    def exec(self, scope, args):
        return args[0] == args[1]


class Negation(UnaryOperatorToken):
    m_re = r"!|~"

    def exec(self, scope, args):
        return not args[0]


class ITokenCollection(Enum):
    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class FunctionalToken(ITokenCollection):
    PARAMETER = Parameter
    IDENTIFIER = Identifier


class BaseTokens(ITokenCollection):
    WHITESPACE = Whitespace


class ArithmeticTokens(ITokenCollection):
    ADDITION = Addition
    SUBTRACTION = Subtraction
    MULTIPLICATION = Multiplication
    DIVISION = Division
    OPEN_PARENTHESIS = OpenParenthesis
    CLOSE_PARENTHESIS = CloseParenthesis


class LogicalTokens(ITokenCollection):
    BOOLEAN = Boolean
    NEGATION = Negation
    AND = And
    OR = Or
    IMPLIES = Implies
    IFF = Iff
    OPEN_PARENTHESIS = OpenParenthesis
    CLOSE_PARENTHESIS = CloseParenthesis


class NumericTokens(ITokenCollection):
    NUMBER = Number


class Tokenlib:
    arithmetic = ArithmeticTokens
    logical = LogicalTokens
    base = BaseTokens
    numeric = NumericTokens
    functional = FunctionalToken

    @staticmethod
    def load(*args):
        return list(set(t for arg in args for t in arg.list()))
