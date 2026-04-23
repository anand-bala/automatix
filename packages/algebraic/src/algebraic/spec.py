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
    """Base class for all algebraic structures.

    Attributes
    ----------
    properties : set of Property
        Runtime-queryable set of algebraic properties. Valid values are
        ``"idempotent_add"``, ``"idempotent_mul"``, ``"commutative"``,
        ``"simple"``, and ``"complemented"``.
    """

    properties: frozenset[Property] = dataclasses.field(default_factory=frozenset, kw_only=True)

    def is_idempotent_add(self) -> bool:
        r"""Check whether :math:`a \oplus a = a` (additive idempotence).

        Returns
        -------
        bool
            ``True`` if ``"idempotent_add"`` is in :attr:`properties`.
        """
        return "idempotent_add" in self.properties

    def is_idempotent_mul(self) -> bool:
        r"""Check whether :math:`a \otimes a = a` (multiplicative idempotence).

        Returns
        -------
        bool
            ``True`` if ``"idempotent_mul"`` is in :attr:`properties`.
        """
        return "idempotent_mul" in self.properties

    def is_commutative(self) -> bool:
        r"""Check whether both operations are commutative.

        That is, :math:`a \oplus b = b \oplus a` and
        :math:`a \otimes b = b \otimes a`.

        Returns
        -------
        bool
            ``True`` if ``"commutative"`` is in :attr:`properties`.
        """
        return "commutative" in self.properties

    def is_simple(self) -> bool:
        """Check whether the structure is marked simple.

        Returns
        -------
        bool
            ``True`` if ``"simple"`` is in :attr:`properties`.
        """
        return "simple" in self.properties


@frozen()
class Semiring(AlgebraicStructure):
    r"""A runtime representation of an algebraic semiring :math:`(S, \oplus, \otimes, 0, 1)`.

    Attributes
    ----------
    add : BinaryOp
        Semiring addition :math:`\oplus`. Must be associative, commutative,
        and have :attr:`zero` as its identity.
    mul : BinaryOp
        Semiring multiplication :math:`\otimes`. Must be associative,
        distribute over :attr:`add`, and have :attr:`one` as its identity.
        :attr:`zero` must absorb under multiplication.
    zero : Number
        Additive identity (:math:`0`). Must be a scalar.
    one : Number
        Multiplicative identity (:math:`1`). Must be a scalar.
    """

    add: BinaryOp  # type: ignore[misc]
    mul: BinaryOp  # type: ignore[misc]
    zero: Number  # type: ignore[misc]
    one: Number  # type: ignore[misc]

    def __post_init__(self) -> None:
        from algebraic.types import is_scalar

        if not is_scalar(self.zero):
            raise ValueError(f"Semiring `zero` should be a scalar, got {self.zero}")
        if not is_scalar(self.one):
            raise ValueError(f"Semiring `one` should be a scalar, got {self.one}")


@frozen
class BoundedDistributiveLattice(Semiring):  # type: ignore[misc]
    r"""A bounded distributive lattice.

    A specialization of :class:`Semiring` where :math:`\oplus` is the lattice
    *join* (:math:`\lor`) and :math:`\otimes` is the *meet* (:math:`\land`).
    Both operations are idempotent and commutative, and the structure has a
    greatest element (:attr:`top`) and a least element (:attr:`bottom`).

    This class automatically sets ``"idempotent_add"``, ``"idempotent_mul"``,
    ``"commutative"``, and ``"simple"`` in :attr:`~AlgebraicStructure.properties`.
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
        r"""Lattice join (:math:`\lor`), an alias for :attr:`~Semiring.add`."""
        return self.add

    @property
    def meet(self) -> BinaryOp:
        r"""Lattice meet (:math:`\land`), an alias for :attr:`~Semiring.mul`."""
        return self.mul

    @property
    def top(self) -> Scalar | Array:
        """Greatest element of the lattice, an alias for :attr:`~Semiring.one`."""
        return self.one

    @property
    def bottom(self) -> Scalar | Array:
        """Least element of the lattice, an alias for :attr:`~Semiring.zero`."""
        return self.zero


@frozen()
class Ring(Semiring):  # type: ignore[misc]
    r"""A semiring extended with an additive inverse.

    Every element :math:`a` must have an inverse :math:`-a` such that
    :math:`a \oplus (-a) = 0`.

    Attributes
    ----------
    additive_inverse : UnaryOp
        Unary operation returning the additive inverse of its argument.
    """

    additive_inverse: UnaryOp  # type: ignore[misc]


@frozen()
class DeMorganAlgebra(BoundedDistributiveLattice):  # type: ignore[misc]
    r"""A bounded distributive lattice with a De Morgan complement.

    Extends :class:`BoundedDistributiveLattice` with a unary ``complement``
    operation that is an involution (:math:`\neg \neg a = a`) and satisfies
    De Morgan's laws:

    .. math::

        \neg (a \lor b) = \neg a \land \neg b
        \qquad
        \neg (a \land b) = \neg a \lor \neg b

    Attributes
    ----------
    complement : UnaryOp
        Unary complementation operation.
    """

    complement: UnaryOp  # type: ignore[misc]


@frozen()
class HeytingAlgebra(BoundedDistributiveLattice):  # type: ignore[misc]
    r"""A bounded lattice with an implication operation.

    Extends :class:`BoundedDistributiveLattice` with a binary *implication*
    :math:`a \to b` satisfying the adjunction:

    .. math::

        (c \land a) \leq b \iff c \leq (a \to b)

    A pseudo-complement is derived as :math:`\neg a = a \to 0`.

    Attributes
    ----------
    implication : BinaryOp
        Binary implication operation :math:`a \to b`.
    """

    implication: BinaryOp  # type: ignore[misc]

    def complement(self, value: Scalar | Array) -> Scalar | Array:
        r"""Pseudo-complement, defined as :math:`\neg a = a \to 0`.

        Parameters
        ----------
        value : Scalar or Array
            The element to complement.

        Returns
        -------
        Scalar or Array
            ``self.implication(value, self.zero)``.
        """
        return self.implication(value, self.zero)


@frozen()
class StoneAlgebra(BoundedDistributiveLattice):  # type: ignore[misc]
    r"""A bounded distributive lattice with a pseudo-complement satisfying Stone's law.

    Extends :class:`BoundedDistributiveLattice` with a unary ``complement``
    that need not be an involution but must satisfy:

    .. math::

        \neg a \lor \neg \neg a = 1

    Attributes
    ----------
    complement : UnaryOp
        Pseudo-complement operation.
    """

    complement: UnaryOp  # type: ignore[misc]


@frozen()
class BooleanAlgebra(DeMorganAlgebra):  # type: ignore[misc]
    r"""A full Boolean algebra.

    Extends :class:`DeMorganAlgebra` so that complementation satisfies:

    1. **De Morgan's laws** -- :math:`\neg (a \lor b) = \neg a \land \neg b`.
    2. **Excluded middle** -- :math:`\neg a \lor a = 1`.
    3. **Non-contradiction** -- :math:`\neg a \land a = 0`.

    By satisfying all three laws, a :class:`BooleanAlgebra` also fulfils the
    contracts of :class:`Ring`, :class:`StoneAlgebra`, and
    :class:`HeytingAlgebra`.
    """

    def additive_inverse(self, a: Scalar | Array) -> Scalar | Array:
        r"""Additive inverse, implemented as :math:`\neg a` (complement).

        Parameters
        ----------
        a : Scalar or Array
            The element to negate.

        Returns
        -------
        Scalar or Array
            ``self.complement(a)``.
        """
        return self.complement(a)

    def implication(self, a: Scalar | Array, b: Scalar | Array) -> Scalar | Array:
        r"""Boolean implication :math:`a \to b = \neg a \lor b`.

        Parameters
        ----------
        a : Scalar or Array
            Antecedent.
        b : Scalar or Array
            Consequent.

        Returns
        -------
        Scalar or Array
            ``self.add(self.complement(a), b)``.
        """
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
