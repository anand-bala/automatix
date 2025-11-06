"""Weight functions for automata.

Weight functions are a fundamental abstraction in the automatix library that
map (input symbol, guard expression) pairs to semiring values. They implement
the lambda(x, Delta) function from weighted automata theory, where:

- x is the concrete input symbol (vector in state space)
- Delta is the guard expression on a transition
- lambda(x, Delta) is the semiring weight for that transition

Weight functions bridge automata (which define guards) and semirings (which
define output values), making them fundamental to the library's architecture.

Examples
--------
Create a simple constant weight function:

>>> def constant_weight(x, guard):
...     return 1.0
>>> wf = constant_weight

Use it with an automaton:

>>> from automatix import NFA
>>> from automatix.algebra import get_semiring
>>> maxplus = get_semiring("MaxPlus", backend="jax")
>>> nfa = NFA(...)  # doctest: +SKIP
>>> # Pass wf to make_automaton_operator  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Protocol, Union, runtime_checkable

from logic_asts.base import Expr

# Type aliases for weight function components
InputSymbol = Any
"""The concrete input data (vector in state space)."""

Guard = Union[str, Expr]
"""A guard expression (string or logic_asts.Expr)."""

SemiringValue = Any
"""A value in the target semiring (float, array, etc.)."""


@runtime_checkable
class WeightFunction(Protocol):
    """Weight function mapping (input, guard) to semiring value.

    A weight function implements lambda(x, Delta) from weighted automata theory:
    - Takes an input symbol x and guard expression Delta
    - Returns a weight in the target semiring
    - Used to compute transition weights in automaton operators

    Examples
    --------
    Simple constant weight function:

    >>> def constant(x, guard):
    ...     return 1.0

    Distance-based weight function:

    >>> def distance_weight(x, guard):
    ...     # Distance from x to satisfying guard
    ...     return compute_distance(x, guard)

    Predicate-based weight function:

    >>> def predicate_weight(x, guard):
    ...     # Evaluate guard with input x
    ...     return evaluate_guard(x, guard)
    """

    def __call__(self, x: InputSymbol, guard: Guard) -> SemiringValue: ...
