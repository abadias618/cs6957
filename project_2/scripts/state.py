from typing import List, Set

class Token:
    def __init__(self, idx: int, word: str, pos: str):
        self.idx = idx # Unique index of the token
        self.word = word # Token string
        self.pos  = pos # Part of speech tag 

class DependencyEdge:
    def __init__(self, source: Token, target: Token, label:str):
        self.source = source  # Source token index
        self.target = target  # target token index
        self.label  = label  # dependency label
        pass


class ParseState:
    def __init__(self, stack: List[Token], parse_buffer: List[Token], dependencies: List[DependencyEdge]):
        self.stack = stack # A stack of token indices in the sentence. Assumption: the root token has index 0, the rest of the tokens in the sentence starts with 1.
        self.parse_buffer = parse_buffer  # A buffer of token indices
        self.dependencies = dependencies
        pass

    def add_dependency(self, source_token, target_token, label):
        self.dependencies.append(
            DependencyEdge(
                source=source_token,
                target=target_token,
                label=label,
            )
        )


def shift(state: ParseState) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues
    x = state.parse_buffer.pop(0)
    state.stack.append(x)
    pass


def left_arc(state: ParseState, label: str) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.
    rhs = state.stack.pop()
    lhs = state.stack.pop()
    state.add_dependency(rhs, lhs, label)
    state.stack.append(rhs)
    pass


def right_arc(state: ParseState, label: str) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.
    rhs = state.stack.pop()
    lhs = state.stack.pop()
    state.add_dependency(lhs, rhs, label)
    state.stack.append(lhs)
    pass



def is_final_state(state: ParseState, cwindow: int) -> bool:
    # TODO: Implemement this
    if len(state.parse_buffer) == 0 and len(state.stack) == 0:
        return True
    else:
        return False

def pad(vec: list, cwindow: int, type: str) -> list:
    padded_vec = []
    counter = cwindow

    if type == "postag":
        while counter > 0:
            if len(vec) > 0:
                padded_vec.append(vec.pop())
            else:
                padded_vec.append("NULL")
            counter -= 1

    elif type == "token":
        while counter > 0:
            if len(vec) > 0:
                padded_vec.append(vec.pop(0))
            else:
                padded_vec.append("[PAD]")
            counter -= 1
    
    return padded_vec

def is_action_valid(state: ParseState, action: str):
    a = action.split("_")
    
    if  len(a) > 1:
        if len(state.stack) < 2:
            return False
    else:
        if len(state.parse_buffer) < 1:
            return False
    return True