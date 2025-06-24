from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Type

import equinox as eqx
import jax.numpy as jnp
import logic_asts.base as exprs
from jaxtyping import Array, Num, Scalar
from logic_asts import Expr

from automatix.algebra.semiring.jax_backend import AbstractSemiring


class AbstractPredicate(eqx.Module, strict=True):
    """A predicate is an *effective Boolean alphabet* over some domain, e.g., real valued vectors, etc."""

    @abstractmethod
    def __call__(self, x: Num[Array, "..."]) -> Scalar: ...

    @classmethod
    def from_expr(
        cls,
        expr: Expr,
        *,
        atoms: dict[str, Predicate],
        neg_atoms: dict[str, Predicate],
        algebra: Type[AbstractSemiring],
    ) -> AbstractPredicate:
        """Convert a `logic_asts.Expr` to a callable predicate"""
        expr = expr.to_nnf()

        cache: dict[int, AbstractPredicate] = dict()
        for subexpr in expr.iter_subtree():
            ex_id = hash(subexpr)
            match subexpr:
                case exprs.Literal(value):
                    cache[ex_id] = (
                        # Broadcastable ONE for True
                        Predicate(lambda _: algebra.ones(()))
                        if value
                        # Broadcastable ZERO for False
                        else Predicate(lambda _: algebra.zeros(()))
                    )
                case exprs.Variable(name):
                    cache[ex_id] = atoms[name]
                case exprs.Not(arg):
                    cache[ex_id] = neg_atoms[str(arg)]
                case exprs.Or(lhs, rhs):
                    cache[ex_id] = Or([cache[hash(lhs)], cache[hash(rhs)]], algebra)
                case exprs.And(lhs, rhs):
                    cache[ex_id] = And([cache[hash(lhs)], cache[hash(rhs)]], algebra)

        return cache[hash(expr)]


class Predicate(AbstractPredicate):
    fn: Callable[[Num[Array, "..."]], Scalar]

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        return self.fn(x)


class And(AbstractPredicate):
    r"""Conjunction of two predicates.

    Given a semiring, the weights for two predicates \(a\) and \(b\) are computed as \(a \otimes b\).
    """

    args: list[AbstractPredicate]
    semiring: Type[AbstractSemiring]

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        weights: list[Scalar] = [arg(x) for arg in self.args]
        return self.semiring.prod(jnp.asarray(weights))


class Or(AbstractPredicate):
    r"""Disjunction of two predicates.

    Given a semiring, the weights for two predicates \(a\) and \(b\) are computed as \(a \oplus b\).
    """

    args: list[AbstractPredicate]
    semiring: Type[AbstractSemiring]

    @eqx.filter_jit
    def __call__(self, x: Num[Array, "..."]) -> Scalar:
        weights: list[Scalar] = [arg(x) for arg in self.args]
        return self.semiring.sum(jnp.asarray(weights))
