"""Predicates for automata: effective Boolean alphabets over domains.

A predicate is a function that evaluates a condition on input data and returns
a weight in a semiring. Predicates form the building blocks for guard evaluation
in automata and can be composed using semiring operations.

This module defines:
- AbstractPredicate: Base class for all predicates
- Predicate: Wrapper for user-defined predicate functions
- And: Conjunction of predicates (semiring multiplication)
- Or: Disjunction of predicates (semiring addition)

Predicates are semiring-aware: they compose using semiring operations so that
the same predicate definitions work across different semirings (Boolean, MaxPlus,
MinPlus, etc.).

Examples
--------
Define atomic predicates:

>>> from automatix.predicates import Predicate
>>> import jax.numpy as jnp
>>> inside_box = Predicate(lambda x: 1.0 if jnp.all(x > 0) else 0.0)
>>> outside_box = Predicate(lambda x: 0.0 if jnp.all(x > 0) else 1.0)

Use with a weight function factory:

>>> from automatix import make_atomic_predicate_weight_function
>>> from automatix.algebra.backends.jax_ import MaxPlusSemiring
>>> weight_fn = make_atomic_predicate_weight_function(
...     atoms={"box": inside_box},
...     neg_atoms={"box": outside_box},
...     semiring=MaxPlusSemiring,
... )
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Type, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import logic_asts.base as exprs
from jaxtyping import Array, Num, Scalar
from logic_asts.base import Expr

from automatix.algebra._compat import normalize_semiring
from automatix.algebra.kernels import AlgebraicStructure
from automatix.algebra.spec import AbstractSemiring
from automatix.weights import Guard, InputSymbol, SemiringValue


class AbstractPredicate(eqx.Module, strict=True):
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


class Predicate(AbstractPredicate):
    """Wrapper for a user-defined predicate function.

    This class wraps a callable function into a predicate that can be
    composed with other predicates using boolean operations.

    Attributes
    ----------
    fn : Callable
        The predicate function mapping Array -> Scalar (weight).
    """

    fn: Callable[[Num[Array, "..."]], Scalar]

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        return self.fn(x)


class And(AbstractPredicate):
    """Conjunction of predicates.

    Combines multiple predicates using semiring multiplication (otimes).
    This implements the AND operation: the weight is the semiring product
    of the weights of all arguments.

    For a Boolean semiring, this is logical AND.
    For MaxPlus, this is addition of weights.
    For MinPlus, this is addition of weights.

    Attributes
    ----------
    args : list[AbstractPredicate]
        The predicates to combine.
    semiring : Type[AbstractSemiring] | AlgebraicStructure
        The semiring defining multiplication (otimes).
        Can be either a class (MinPlusSemiring) or a kernel instance.
    """

    args: list[AbstractPredicate]
    semiring: Type[AbstractSemiring] | AlgebraicStructure

    def __post_init__(self) -> None:
        """Normalize semiring on construction."""
        object.__setattr__(self, "semiring", normalize_semiring(self.semiring))

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        weights: list[Scalar] = [arg(x) for arg in self.args]
        weights_array = jnp.asarray(weights)
        if self.semiring.prod is not None:
            return self.semiring.prod(weights_array, axis=None)
        else:
            return cast(Scalar, jax.lax.reduce(weights_array, self.semiring.one, self.semiring.mul, (0,)))


class Or(AbstractPredicate):
    """Disjunction of predicates.

    Combines multiple predicates using semiring addition (oplus).
    This implements the OR operation: the weight is the semiring sum
    of the weights of all arguments.

    For a Boolean semiring, this is logical OR.
    For MaxPlus, this is the maximum of weights.
    For MinPlus, this is the minimum of weights.

    Attributes
    ----------
    args : list[AbstractPredicate]
        The predicates to combine.
    semiring : Type[AbstractSemiring] | AlgebraicStructure
        The semiring defining addition (oplus).
        Can be either a class (MinPlusSemiring) or a kernel instance.
    """

    args: list[AbstractPredicate]
    semiring: Type[AbstractSemiring] | AlgebraicStructure

    def __post_init__(self) -> None:
        """Normalize semiring on construction."""
        object.__setattr__(self, "semiring", normalize_semiring(self.semiring))

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        weights: list[Scalar] = [arg(x) for arg in self.args]
        weights_array = jnp.asarray(weights)
        if self.semiring.sum is not None:
            return self.semiring.sum(weights_array, axis=None)
        else:
            return cast(Scalar, jax.lax.reduce(weights_array, self.semiring.zero, self.semiring.add, (0,)))


@dataclass(kw_only=True)
class ExprWeightFn:
    """A weight function recursively defined from predicates.

    This bridges the atomic predicate-based approach to guard evaluation with the new
    weight function abstraction. It evaluates guard expressions by:
    1. Converting the guard to NNF (negation normal form)
    2. Recursively evaluating atoms and their negations
    3. Composing results with semiring operations (AND -> multiply, OR -> add)

    Attributes
    ----------
    atoms : dict[str, Predicate]
        Predicates for positive atoms.
    neg_atoms : dict[str, Predicate]
        Predicates for negated atoms.
    semiring : Type[AbstractSemiring] | AlgebraicStructure
        The semiring for composing predicates.
        Can be either a class (MinPlusSemiring) or a kernel instance.

    Examples
    --------
    >>> from automatix.predicates import Predicate, ExprWeightFn
    >>> from automatix.algebra.backends.jax_ import MaxPlusSemiring
    >>> # Define atomic predicates
    >>> inside = Predicate(lambda x: -jnp.maximum(0, -distance_to_region(x)))
    >>> outside = Predicate(lambda x: -jnp.maximum(0, distance_to_region(x)))
    >>> # Create weight function
    >>> weight_fn = ExprWeightFn(
    ...     atoms={"region": inside},
    ...     neg_atoms={"region": outside},
    ...     semiring=MaxPlusSemiring,
    ... )
    """

    atoms: dict[str, Predicate]
    neg_atoms: dict[str, Predicate]
    semiring: Type[AbstractSemiring] | AlgebraicStructure

    cache: dict[str | Expr, AbstractPredicate] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize semiring
        object.__setattr__(self, "semiring", normalize_semiring(self.semiring))

        # Populate the cache with the atoms and the neg atoms, and literal True and literal False
        self.cache.update(self.atoms.items())
        self.cache.update((exprs.Variable(atom), pred) for atom, pred in self.atoms.items())

        self.cache.update((f"~{atom}", pred) for atom, pred in self.neg_atoms.items())
        self.cache.update((~exprs.Variable(atom), pred) for atom, pred in self.neg_atoms.items())

        self.cache.update(
            (expr, Predicate(lambda _: self.semiring.zeros(())))
            for expr in ("0", "FALSE", "False", "false", exprs.Literal(False))
        )
        self.cache.update(
            (expr, Predicate(lambda _: self.semiring.ones(()))) for expr in ("1", "TRUE", "True", "true", exprs.Literal(True))
        )

    def add_expr(self, guard: Guard) -> AbstractPredicate:
        # Parse string guards to Expr if needed
        if isinstance(guard, str):
            import logic_asts
            from lark.exceptions import LarkError

            try:
                expr = logic_asts.parse_expr(guard)
            except LarkError as e:
                raise ValueError(f"Unable to parse guard '{guard}' as a boolean expression") from e
        else:
            expr = guard

        expr = expr.to_nnf()
        if expr in self.cache:
            if guard not in self.cache:
                self.cache[guard] = self.cache[expr]
            return self.cache[expr]

        for subexpr in expr.iter_subtree():
            if subexpr in self.cache:
                continue
            match subexpr:
                case exprs.Literal(value):
                    self.cache[subexpr] = (
                        # Broadcastable ONE for True
                        Predicate(lambda _: self.semiring.ones(()))
                        if value
                        # Broadcastable ZERO for False
                        else Predicate(lambda _: self.semiring.zeros(()))
                    )
                case exprs.Variable(name):
                    assert isinstance(name, str)
                    self.cache[subexpr] = self.atoms[name]
                case exprs.Not(arg):
                    self.cache[subexpr] = self.neg_atoms[str(arg)]
                case exprs.Or(args):
                    self.cache[subexpr] = Or([self.cache[arg] for arg in args], self.semiring)
                case exprs.And(args):
                    self.cache[subexpr] = And([self.cache[arg] for arg in args], self.semiring)
            # duplicate it for strings too.
            self.cache[str(subexpr)] = self.cache[subexpr]

        return self.cache[expr]

    def __call__(self, x: InputSymbol, guard: Guard) -> SemiringValue:
        return self.add_expr(guard)(x)
