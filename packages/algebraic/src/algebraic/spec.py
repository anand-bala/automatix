"""Pure interface definitions for heirarchy of rings and lattices.

This module defines the abstract base classes that all algebraic implementations
must follow. These are pure interfaces with no implementation.
"""
# pyright: reportMissingParameterType=false
# ruff: noqa: ANN003

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Literal, Protocol, TypeGuard, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Num, Scalar, Shaped

type Axis = int | Sequence[int]
type MaybeAxis = None | Axis
type Shape = int | tuple[int, ...]

type UnaryOp = Callable[[Scalar | Array], Scalar | Array]
type BinaryOp = Callable[[Scalar | Array, Scalar | Array], Scalar | Array]
type VdotFn = Callable[[Num[Array, " n"], Num[Array, " n"]], Num[Array, ""]]
type MatmulFn = Callable[[Num[Array, "n k"], Num[Array, "k m"]], Num[Array, "n m"]]


@runtime_checkable
class IdentityFn(Protocol):
    def __call__(self, shape: Shape) -> Shaped[Array, " {shape}"]: ...


@runtime_checkable
class ReductionOp(Protocol):
    def __call__(self, a: Array, axis: MaybeAxis = None) -> Array: ...


type Property = Literal["idempotent_add", "idempotent_mul", "commutative", "simple", "complemented"] | str  # noqa: PYI051


class AlgebraicStructure(eqx.Module):
    properties: set[Property] = eqx.field(default_factory=set, kw_only=True, static=True)
    """Set of algebraic properties.
    Valid values: "idempotent_add", "idempotent_mul", "commutative", "simple", "has_negation" 
    """

    def is_idempotent_add(self) -> bool:
        """Check if a oplus a = a (additive idempotence)."""
        return "idempotent_add" in self.properties

    def is_idempotent_mul(self) -> bool:
        """Check if a otimes a = a (multiplicative idempotence)."""
        return "idempotent_mul" in self.properties

    def is_commutative(self) -> bool:
        """Check if a oplus b = b oplus a and a otimes b = b otimes a."""
        return "commutative" in self.properties

    def is_simple(self) -> bool:
        """Check if structure is simple (all properties hold)."""
        return "simple" in self.properties


class Semiring(AlgebraicStructure):
    """A simple runtime representation of an algebraic semiring."""

    add: BinaryOp = eqx.field(static=True)
    """Semiring addition operation (oplus)"""

    mul: BinaryOp = eqx.field(static=True)
    """Semiring multiplication (otimes)"""

    zeros: IdentityFn = eqx.field(static=True)
    """Additive identity of the semiring"""

    ones: IdentityFn = eqx.field(static=True)
    """Multiplicative identity of the semiring"""

    @cached_property
    def zero(self) -> Scalar:
        return self.zeros(())

    @cached_property
    def one(self) -> Scalar:
        return self.ones(())

    def __check_init__(self) -> None:
        if not jnp.isscalar(self.zero):
            raise ValueError("Semiring `zero` should be a scalar")
        if not jnp.isscalar(self.zero):
            raise ValueError("Semiring `zero` should be a scalar")


class BoundedDistributiveLattice(Semiring):
    """A bounded distributive lattice is a specialization of a semiring, where the `oplus` operator corresponds to `join` operator, `otimes` is the `meet` operator."""

    def __post_init__(self) -> None:
        self.properties |= {"idempotent_add", "idempotent_mul", "commutative", "simple"}

    @property
    def join(self) -> BinaryOp:
        r"""Lattice join operation (corresponds to $\oplus$)."""
        return self.add

    @property
    def meet(self) -> BinaryOp:
        r"""Lattice meet operation (corresponds to $\otimes$)."""
        return self.mul

    @property
    def top(self) -> Scalar | Array:
        """Top element of the lattice (multiplicative identity)."""
        return self.one

    @property
    def bottom(self) -> Scalar | Array:
        """Bottom element of the lattice (additive identity)."""
        return self.zero


class Ring(Semiring):
    """A ring is a semiring with the additional requirement that each element must have an additive inverse"""

    additive_inverse: UnaryOp = eqx.field(static=True)


class DeMorganAlgebra(BoundedDistributiveLattice):
    """
    A De Morgan Algebra is a bounded distributive lattice equipped with
    a complementation operator that is an involution (`~~a = a`) that follows De
    Morgan's laws.
    """

    complement: UnaryOp = eqx.field(static=True)


class HeytingAlgebra(BoundedDistributiveLattice):
    """
    A Heyting algebra is a bounded lattice equipped with a binary operation `a -> b`
    called implication such that `(c and a) <= b` is equivalent to `c <= (a -> b)`

    A Heyting algebra has a pseudo-complement such that `~a` is equivalent to `a -> 0`.
    """

    implication: BinaryOp = eqx.field(static=True)

    def complement(self, value: Scalar | Array) -> Scalar | Array:
        """Pseudo-complement in Heyting algebra."""
        return self.implication(value, self.zero)


class StoneAlgebra(BoundedDistributiveLattice):
    """
    A Stone Algebra is a bounded distributive lattice equipped with a pseudo-complement
    such that `~a or ~~a = 1` (but is not necessarily an involution) but follows De
    Morgan's laws.
    """

    complement: UnaryOp = eqx.field(static=True)


class BooleanAlgebra(DeMorganAlgebra):
    """
    A full Boolean algebra, i.e., the operators with complementation follow:

    1. De Morgan's Laws
    2. The law of excluded middle (`~x or x = 1`)
    3. The law of noncontradiction (`~x and x = 0`)

    This, by extension, satisfies the contracts of `Ring`, `StoneAlgebra`, and `HeytingAlgebra`.
    """

    def additive_inverse(self, a: Scalar | Array) -> Scalar | Array:
        return self.complement(a)

    def implication(self, a: Scalar | Array, b: Scalar | Array) -> Scalar | Array:
        r"""Boolean implication ($a \to b$ = $\neg a \lor b$)."""
        return self.add(self.complement(a), b)


# Type guards for runtime type narrowing


def is_ring(algebra: object) -> TypeGuard[Ring]:
    """Type guard to check if algebra is a Ring (has additive_inverse).

    Returns True for Ring instances and BooleanAlgebra (which satisfies Ring contract).
    """
    return isinstance(algebra, (Ring, BooleanAlgebra))


def is_demorgan_algebra(algebra: object) -> TypeGuard[DeMorganAlgebra]:
    """Type guard to check if algebra is a DeMorgan algebra (has complement with De Morgan laws).

    Returns True for DeMorganAlgebra instances (including BooleanAlgebra subclasses).
    """
    return isinstance(algebra, DeMorganAlgebra)


def is_heyting_algebra(algebra: object) -> TypeGuard[HeytingAlgebra]:
    """Type guard to check if algebra is a Heyting algebra (has implication).

    Returns True for HeytingAlgebra instances and BooleanAlgebra (which satisfies Heyting contract).
    """
    return isinstance(algebra, (HeytingAlgebra, BooleanAlgebra))


def is_stone_algebra(algebra: object) -> TypeGuard[StoneAlgebra]:
    """Type guard to check if algebra is a Stone algebra (has pseudo-complement).

    Returns True for StoneAlgebra, DeMorganAlgebra, and BooleanAlgebra instances.
    """
    return isinstance(algebra, (StoneAlgebra, DeMorganAlgebra))


def is_boolean_algebra(algebra: object) -> TypeGuard[BooleanAlgebra]:
    """Type guard to check if algebra is a Boolean algebra.

    Returns True for BooleanAlgebra instances.
    """
    return isinstance(algebra, BooleanAlgebra)


def has_complement(algebra: object) -> TypeGuard[DeMorganAlgebra | HeytingAlgebra | StoneAlgebra]:
    """Type guard to check if algebra has a complement operation.

    Returns True for algebras with complement: DeMorganAlgebra, HeytingAlgebra, StoneAlgebra, or BooleanAlgebra.
    """
    return isinstance(algebra, (DeMorganAlgebra, HeytingAlgebra, StoneAlgebra))
