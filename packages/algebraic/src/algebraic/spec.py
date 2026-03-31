"""Pure interface definitions for hierarchy of rings and lattices.

This module defines the abstract base classes that all algebraic implementations
must follow. These are pure interfaces with no implementation.
"""
# pyright: reportMissingParameterType=false
# ruff: noqa: ANN003

from __future__ import annotations

import dataclasses
from typing import Literal, TypeGuard

from algebraic._better_abc import BetterABCMeta, frozen
from algebraic.types import Array, BinaryOp, Number, Scalar, UnaryOp

type Property = Literal["idempotent_add", "idempotent_mul", "commutative", "simple", "complemented"] | str  # noqa: PYI051


@frozen()
class AlgebraicStructure(metaclass=BetterABCMeta):
    properties: set[Property] = dataclasses.field(default_factory=set, kw_only=True)
    """Set of algebraic properties.
    Valid values: "idempotent_add", "idempotent_mul", "commutative", "simple", "has_negation"
    """

    def is_idempotent_add(self) -> bool:
        r"""Check if :math:`a \oplus a = a` (additive idempotence)."""
        return "idempotent_add" in self.properties

    def is_idempotent_mul(self) -> bool:
        r"""Check if :math:`a \otimes a = a` (multiplicative idempotence)."""
        return "idempotent_mul" in self.properties

    def is_commutative(self) -> bool:
        r"""Check if :math:`a \oplus b = b \oplus a` and :math:`a \otimes b = b \otimes a`."""
        return "commutative" in self.properties

    def is_simple(self) -> bool:
        """Check if structure is simple (all properties hold)."""
        return "simple" in self.properties


@frozen()
class Semiring(AlgebraicStructure):
    """A simple runtime representation of an algebraic semiring."""

    add: BinaryOp
    r"""Semiring addition operation (:math:`\oplus`)."""

    mul: BinaryOp
    r"""Semiring multiplication (:math:`\otimes`)."""

    zero: Number
    """Additive identity of the semiring."""

    one: Number
    """Multiplicative identity of the semiring."""

    def __post_init__(self) -> None:
        from algebraic.types import is_scalar

        if not is_scalar(self.zero):
            raise ValueError(f"Semiring `zero` should be a scalar, got {self.zero}")
        if not is_scalar(self.one):
            raise ValueError(f"Semiring `one` should be a scalar, got {self.one}")


@frozen
class BoundedDistributiveLattice(Semiring):
    r"""A bounded distributive lattice.

    A specialization of a :class:`Semiring` where the :math:`\oplus` operator
    corresponds to the *join* (:math:`\lor`) and :math:`\otimes` corresponds to
    the *meet* (:math:`\land`).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "properties",
            self.properties | {"idempotent_add", "idempotent_mul", "commutative", "simple"},
        )

    @property
    def join(self) -> BinaryOp:
        r"""Lattice join operation (corresponds to :math:`\oplus`)."""
        return self.add

    @property
    def meet(self) -> BinaryOp:
        r"""Lattice meet operation (corresponds to :math:`\otimes`)."""
        return self.mul

    @property
    def top(self) -> Scalar | Array:
        """Top element of the lattice (multiplicative identity)."""
        return self.one

    @property
    def bottom(self) -> Scalar | Array:
        """Bottom element of the lattice (additive identity)."""
        return self.zero


@frozen()
class Ring(Semiring):
    """A ring is a semiring with the additional requirement that each element must have an additive inverse"""

    additive_inverse: UnaryOp


@frozen()
class DeMorganAlgebra(BoundedDistributiveLattice):
    """
    A De Morgan Algebra is a bounded distributive lattice equipped with
    a complementation operator that is an involution (`~~a = a`) that follows De
    Morgan's laws.
    """

    complement: UnaryOp


@frozen()
class HeytingAlgebra(BoundedDistributiveLattice):
    """
    A Heyting algebra is a bounded lattice equipped with a binary operation `a -> b`
    called implication such that `(c and a) <= b` is equivalent to `c <= (a -> b)`

    A Heyting algebra has a pseudo-complement such that `~a` is equivalent to `a -> 0`.
    """

    implication: BinaryOp

    def complement(self, value: Scalar | Array) -> Scalar | Array:
        """Pseudo-complement in Heyting algebra."""
        return self.implication(value, self.zero)


@frozen()
class StoneAlgebra(BoundedDistributiveLattice):
    """
    A Stone Algebra is a bounded distributive lattice equipped with a pseudo-complement
    such that `~a or ~~a = 1` (but is not necessarily an involution) but follows De
    Morgan's laws.
    """

    complement: UnaryOp


@frozen()
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
    """Check if *algebra* is a :class:`Ring` (has ``additive_inverse``).

    Parameters
    ----------
    algebra : object
        The algebraic structure to test.

    Returns
    -------
    bool
        ``True`` for :class:`Ring` instances and :class:`BooleanAlgebra`
        (which satisfies the Ring contract).
    """
    return isinstance(algebra, (Ring, BooleanAlgebra))


def is_demorgan_algebra(algebra: object) -> TypeGuard[DeMorganAlgebra]:
    """Check if *algebra* is a :class:`DeMorganAlgebra`.

    Parameters
    ----------
    algebra : object
        The algebraic structure to test.

    Returns
    -------
    bool
        ``True`` for :class:`DeMorganAlgebra` instances (including
        :class:`BooleanAlgebra` subclasses).
    """
    return isinstance(algebra, DeMorganAlgebra)


def is_heyting_algebra(algebra: object) -> TypeGuard[HeytingAlgebra]:
    """Check if *algebra* is a :class:`HeytingAlgebra` (has ``implication``).

    Parameters
    ----------
    algebra : object
        The algebraic structure to test.

    Returns
    -------
    bool
        ``True`` for :class:`HeytingAlgebra` instances and
        :class:`BooleanAlgebra` (which satisfies the Heyting contract).
    """
    return isinstance(algebra, (HeytingAlgebra, BooleanAlgebra))


def is_stone_algebra(algebra: object) -> TypeGuard[StoneAlgebra]:
    """Check if *algebra* is a :class:`StoneAlgebra` (has pseudo-complement).

    Parameters
    ----------
    algebra : object
        The algebraic structure to test.

    Returns
    -------
    bool
        ``True`` for :class:`StoneAlgebra`, :class:`DeMorganAlgebra`,
        and :class:`BooleanAlgebra` instances.
    """
    return isinstance(algebra, (StoneAlgebra, DeMorganAlgebra))


def is_boolean_algebra(algebra: object) -> TypeGuard[BooleanAlgebra]:
    """Check if *algebra* is a :class:`BooleanAlgebra`.

    Parameters
    ----------
    algebra : object
        The algebraic structure to test.

    Returns
    -------
    bool
        ``True`` for :class:`BooleanAlgebra` instances.
    """
    return isinstance(algebra, BooleanAlgebra)


def has_complement(algebra: object) -> TypeGuard[DeMorganAlgebra | HeytingAlgebra | StoneAlgebra]:
    """Check if *algebra* has a complement operation.

    Parameters
    ----------
    algebra : object
        The algebraic structure to test.

    Returns
    -------
    bool
        ``True`` for algebras with complement: :class:`DeMorganAlgebra`,
        :class:`HeytingAlgebra`, :class:`StoneAlgebra`, or
        :class:`BooleanAlgebra`.
    """
    return isinstance(algebra, (DeMorganAlgebra, HeytingAlgebra, StoneAlgebra))
