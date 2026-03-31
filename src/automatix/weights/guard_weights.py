"""Conversion of :class:`Guard` expressions to :class:`WeightFunction`\\ s.

Provides composable predicate classes that evaluate boolean guard expressions
using semiring operations (AND → multiply, OR → add).  These are plain Python
dataclasses with no JAX or Equinox dependency; users who want JIT compilation
or gradient support should wrap their predicates in an
:class:`equinox.Module` or :class:`torch.nn.Module` weight function.
"""

from __future__ import annotations

import functools
from abc import abstractmethod
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import cast

import logic_asts.base as exprs
from algebraic import Semiring

from automatix.spec import Guard


@dataclass
class AbstractPredicate[S: Semiring]:
    """A predicate is an effective Boolean alphabet over some domain.

    A predicate evaluates a condition on input data and returns a weight
    in a semiring.  Predicates can be combined using boolean operations
    (:class:`And`, :class:`Or`) which compose using semiring operations
    (multiplication for AND, addition for OR).

    Subclasses
    ----------
    Predicate
        Wraps a user-defined callable function.
    And
        Conjunction of predicates (semiring multiplication).
    Or
        Disjunction of predicates (semiring addition).
    """

    algebra: S

    @abstractmethod
    def __call__(self, x: object) -> object:
        """Evaluate the predicate on input *x*.

        Parameters
        ----------
        x :
            Input data.

        Returns
        -------
        object
            Weight in the target semiring.
        """
        ...


@dataclass
class Predicate[S: Semiring](AbstractPredicate[S]):
    """Wrapper for a user-defined predicate function.

    Parameters
    ----------
    algebra :
        The semiring algebra.
    fn :
        The predicate function mapping input → semiring weight.
    """

    fn: Callable[[object], object]

    def __call__(self, x: object) -> object:
        return self.fn(x)


@dataclass
class And[S: Semiring](AbstractPredicate[S]):
    """Conjunction of predicates (semiring multiplication).

    For a Boolean semiring this is logical AND; for MaxPlus/MinPlus this is
    addition of weights.

    Parameters
    ----------
    algebra :
        The semiring algebra.
    args :
        Sub-predicates whose results are multiplied together.
    """

    args: list[AbstractPredicate]

    def __call__(self, x: object) -> object:
        weights = [arg(x) for arg in self.args]
        return functools.reduce(self.algebra.mul, weights, self.algebra.one)


@dataclass
class Or[S: Semiring](AbstractPredicate[S]):
    """Disjunction of predicates (semiring addition).

    For a Boolean semiring this is logical OR; for MaxPlus/MinPlus this is
    the maximum/minimum of weights.

    Parameters
    ----------
    algebra :
        The semiring algebra.
    args :
        Sub-predicates whose results are summed together.
    """

    args: list[AbstractPredicate]

    def __call__(self, x: object) -> object:
        weights = [arg(x) for arg in self.args]
        return functools.reduce(self.algebra.add, weights, self.algebra.zero)


@dataclass
class ExprWeightFn[S: Semiring, AtomicPredicate: Hashable]:
    """A weight function recursively defined from atomic predicates.

    Evaluates guard expressions by:

    1. Converting the guard to NNF (negation normal form).
    2. Recursively evaluating atoms and their negations.
    3. Composing results with semiring operations (AND → multiply, OR → add).

    Results are memoised in ``cache`` for efficiency.

    Parameters
    ----------
    algebra :
        The semiring algebra for composing predicates.
    atoms :
        Predicates for positive atoms, keyed by atom name.
    neg_atoms :
        Predicates for negated atoms, keyed by atom name.
    """

    algebra: S
    atoms: dict[str, Predicate]
    neg_atoms: dict[str, Predicate]
    cache: dict[str, AbstractPredicate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache.update(self.atoms.items())
        self.cache.update((f"~{atom}", pred) for atom, pred in self.neg_atoms.items())
        self.cache.update(
            (expr, Predicate(self.algebra, lambda _: self.algebra.zero)) for expr in ("0", "FALSE", "False", "false")
        )
        self.cache.update(
            (expr, Predicate(self.algebra, lambda _: self.algebra.one)) for expr in ("1", "TRUE", "True", "true")
        )

    def add_expr(self, guard: Guard[AtomicPredicate]) -> AbstractPredicate:
        """Add a guard expression and return its weight predicate.

        Recursively builds and caches a predicate for *guard* using the atomic
        predicates and semiring operations.
        """
        expr = cast(Guard[AtomicPredicate], guard.to_nnf())
        expr_str = str(expr)
        guard_str = str(guard)
        if expr_str in self.cache:
            if guard_str not in self.cache:
                self.cache[guard_str] = self.cache[expr_str]
            return self.cache[expr_str]

        for subexpr in expr.iter_subtree():
            subexpr_str = str(subexpr)
            if subexpr_str in self.cache:
                continue
            match subexpr:
                case exprs.Literal(value):
                    self.cache[subexpr_str] = (
                        Predicate(self.algebra, lambda _: self.algebra.one)
                        if value
                        else Predicate(self.algebra, lambda _: self.algebra.zero)
                    )
                case exprs.Variable(name):
                    assert isinstance(name, str)
                    self.cache[subexpr_str] = self.atoms[name]
                case exprs.Not(arg):
                    self.cache[subexpr_str] = self.neg_atoms[str(arg)]
                case exprs.Or(args):
                    self.cache[subexpr_str] = Or(self.algebra, [self.cache[str(arg)] for arg in args])
                case exprs.And(args):
                    self.cache[subexpr_str] = And(self.algebra, [self.cache[str(arg)] for arg in args])

        return self.cache[expr_str]

    def __call__(self, x: object, guard: Guard[AtomicPredicate]) -> object:
        return self.add_expr(guard)(x)
