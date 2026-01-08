"""Conversion of `Guard` expressions to `WeightFunction`s"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Hashable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import logic_asts.base as exprs
from algebraic import Semiring
from jaxtyping import Array, Num, Scalar, ScalarLike

from automatix.spec import Guard


class AbstractPredicate[S: Semiring](eqx.Module, strict=True):
    """A predicate is an effective Boolean alphabet over some domain.

    A predicate evaluates a condition on input data and returns a weight
    in a semiring. Predicates can be combined using boolean operations
    (AND, OR) which compose using semiring operations (multiplication for AND,
    addition for OR).

    Subclasses
    ----------
    Predicate
        Wraps a user-defined callable function.
    And
        Conjunction of predicates (uses semiring multiplication).
    Or
        Disjunction of predicates (uses semiring addition).
    """

    algebra: eqx.AbstractVar[S]

    @abstractmethod
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        """Evaluate the predicate on input x.

        Parameters
        ----------
        x : Array
            Input data (vector in domain).

        Returns
        -------
        Scalar
            Weight in the target semiring.
        """
        ...


class Predicate[S: Semiring](AbstractPredicate[S]):
    """Wrapper for a user-defined predicate function.

    This class wraps a callable function into a predicate that can be
    composed with other predicates using boolean operations.

    Attributes
    ----------
    fn : Callable
        The predicate function mapping Array -> Scalar (weight).
    """

    algebra: S
    fn: Callable[[Num[Array, "..."]], Scalar]

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        return self.fn(x)


class And[S: Semiring](AbstractPredicate[S]):
    """Conjunction of predicates.

    Combines multiple predicates using semiring multiplication (otimes).
    This implements the AND operation: the weight is the semiring product
    of the weights of all arguments.

    For a Boolean semiring, this is logical AND.
    For MaxPlus, this is addition of weights.
    For MinPlus, this is addition of weights.
    """

    algebra: S
    args: list[AbstractPredicate]

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        weights: list[Scalar] = [arg(x) for arg in self.args]
        weights_array = jnp.asarray(weights)
        one_typed = jnp.asarray(self.algebra.one, dtype=weights_array.dtype)
        return cast(Scalar, jax.lax.reduce(weights_array, one_typed, self.algebra.mul, (0,)))


class Or[S: Semiring](AbstractPredicate[S]):
    """Disjunction of predicates.

    Combines multiple predicates using semiring addition (oplus).
    This implements the OR operation: the weight is the semiring sum
    of the weights of all arguments.

    For a Boolean semiring, this is logical OR.
    For MaxPlus, this is the maximum of weights.
    For MinPlus, this is the minimum of weights.

    """

    algebra: S
    args: list[AbstractPredicate]

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        weights: list[Scalar] = [arg(x) for arg in self.args]
        weights_array = jnp.asarray(weights)
        zero_typed = jnp.asarray(self.algebra.zero, dtype=weights_array.dtype)
        return cast(Scalar, jax.lax.reduce(weights_array, zero_typed, self.algebra.add, (0,)))


class ExprWeightFn[S: Semiring, AtomicPredicate: Hashable](eqx.Module):
    """A weight function recursively defined from predicates.

    This bridges the atomic predicate-based approach to guard evaluation with the new
    weight function abstraction. It evaluates guard expressions by:
    1. Converting the guard to NNF (negation normal form)
    2. Recursively evaluating atoms and their negations
    3. Composing results with semiring operations (AND -> multiply, OR -> add)

    Attributes
    ----------
    algebra : S
        The semiring algebra for composing predicates.
    atoms : dict[str, Predicate]
        Predicates for positive atoms.
    neg_atoms : dict[str, Predicate]
        Predicates for negated atoms.
    """

    algebra: S
    atoms: dict[str, Predicate]
    neg_atoms: dict[str, Predicate]

    cache: dict[str, AbstractPredicate] = eqx.field(default_factory=dict)

    def __post_init__(self) -> None:
        # Populate the cache with the atoms and the neg atoms, and literal True and literal False
        self.cache.update(self.atoms.items())
        # self.cache.update((exprs.Variable(atom), pred) for atom, pred in self.atoms.items())

        self.cache.update((f"~{atom}", pred) for atom, pred in self.neg_atoms.items())
        # self.cache.update((~exprs.Variable(atom), pred) for atom, pred in self.neg_atoms.items())

        self.cache.update(
            (expr, Predicate(self.algebra, lambda _: jnp.asarray(self.algebra.zero)))
            for expr in (
                "0",
                "FALSE",
                "False",
                "false",
                # exprs.Literal(False)
            )
        )
        self.cache.update(
            (expr, Predicate(self.algebra, lambda _: jnp.asarray(self.algebra.one)))
            for expr in (
                "1",
                "TRUE",
                "True",
                "true",
                # exprs.Literal(True)
            )
        )

    def add_expr(self, guard: Guard[AtomicPredicate]) -> AbstractPredicate:
        """Add a guard expression and return its weight predicate.

        Recursively evaluates the guard using cached atoms and semiring operations.
        """
        # Parse string guards to Expr if needed
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
                        # Broadcastable ONE for True
                        Predicate(self.algebra, lambda _: jnp.asarray(self.algebra.one))
                        if value
                        # Broadcastable ZERO for False
                        else Predicate(self.algebra, lambda _: jnp.asarray(self.algebra.zero))
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

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."], guard: Guard[AtomicPredicate]) -> Array | ScalarLike:
        return self.add_expr(guard)(x)
