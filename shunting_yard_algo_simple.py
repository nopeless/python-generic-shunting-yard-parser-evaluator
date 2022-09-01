#####
# This is a reference file for those who
# Don't know the shunting yard algorithm
####

import re


class Token:
    def __init__(self, type, value, start, length):
        self.type = type
        self.value = value
        self.start = start
        self.length = length

    def __repr__(self):
        return f"Token<type={self.type.__repr__()} value={self.value.__repr__()}>"


OPERATOR_PRECEDENCE = {
    # Higher number => stronger associativity
    "mul": 1,
    "div": 1,
    "add": 0,
    "sub": 0,
}

print(OPERATOR_PRECEDENCE)

ASSOCIATIVITY = {
    "mul": "left",
    "add": "left",
}


def tokenize(string):
    tokens = []
    for idx, chr in enumerate(string):
        if re.match(r"\d", chr):
            tokens.append(Token("num", int(chr), idx, 1))
        elif re.match(r"\s", chr):
            continue
        elif re.match(r"\*", chr):
            tokens.append(Token("mul", chr, idx, 1))
        elif re.match(r"\+", chr):
            tokens.append(Token("add", chr, idx, 1))
        elif re.match(r"\(", chr):
            tokens.append(Token("left_paren", chr, idx, 1))
        elif re.match(r"\)", chr):
            tokens.append(Token("right_paren", chr, idx, 1))
        else:
            raise Exception(f"Unknown character: {chr}")
    return tokens


def should_pop_op(stack, op):
    if not stack:
        return False
    top_op = stack[-1]

    if ASSOCIATIVITY.get(top_op.type, None) is None:
        # Assume parenthesis
        return False

    a = ASSOCIATIVITY[op.type]

    if a == "left":
        return OPERATOR_PRECEDENCE[op.type] < OPERATOR_PRECEDENCE[top_op.type]
    elif a == "right":
        return OPERATOR_PRECEDENCE[op.type] <= OPERATOR_PRECEDENCE[top_op.type]
    raise Exception("Unknown associativity")


def to_rpn(tokens):
    output = []
    operator_stack = []
    for token in tokens:
        print(f"on token {token.value}: ", end="")
        if token.type == "num":
            output.append(token)
        elif OPERATOR_PRECEDENCE.get(token.type, None) is not None:
            if should_pop_op(operator_stack, token):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        elif token.type == "left_paren":
            operator_stack.append(token)
        elif token.type == "right_paren":
            while operator_stack[-1].type != "left_paren":
                output.append(operator_stack.pop())
            operator_stack.pop()

        print([v.value for v in output], [v.value for v in operator_stack])

    return [*output, *operator_stack[::-1]]


class Operator:
    def __init__(self, lhs, op, rhs):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def __repr__(self):
        return f"Operator<lhs={self.lhs.__repr__()} op={self.op.__repr__()} rhs={self.rhs.__repr__()}>"


def to_ast(rpn):
    stack = []
    for token in rpn:
        if token.type == "num":
            stack.append(token)
        elif OPERATOR_PRECEDENCE.get(token.type, None) is not None:
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(Operator(lhs, token, rhs))
    if len(stack) != 1:
        print(stack)
        raise Exception("Invalid RPN")
    return stack[0]


inp = "(1 + 2) * 3"

print("input: ", inp)

rpn = to_rpn(tokenize(inp))

print(rpn)
print(to_ast(rpn))
