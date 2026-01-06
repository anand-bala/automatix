"""Pure interface definitions for heirarchy of rings and lattices.

This module defines the abstract base classes that all algebraic implementations
must follow. These are pure interfaces with no implementation.
"""
# pyright: reportMissingParameterType=false
# ruff: noqa: ANN003

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

import equinox as eqx
from jaxtyping import Array, Num, Scalar, ScalarLike

type Axis = int | Sequence[int]
type MaybeAxis = None | Axis
type Shape = int | tuple[int, ...]

type UnaryOp = Callable[[Scalar | Array], Scalar | Array]
type BinaryOp = Callable[[Scalar | Array, Scalar | Array], Scalar | Array]
type VdotFn = Callable[[Num[Array, " n"], Num[Array, " n"]], Num[Array, ""]]
type MatmulFn = Callable[[Num[Array, "n k"], Num[Array, "k m"]], Num[Array, "n m"]]


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

    def has_negation(self) -> bool:
        """Check if structure has a negation operation."""
        return getattr(self, "complement", None) is not None


class Semiring(AlgebraicStructure):
    """A simple runtime representation of an algebraic semiring."""

    add: BinaryOp = eqx.field(static=True)
    """Semiring addition operation (oplus)"""

    mul: BinaryOp = eqx.field(static=True)
    """Semiring multiplication (otimes)"""

    zero: Scalar | Array
    """Additive identity of the semiring"""

    one: Scalar | Array
    """Multiplicative identity of the semiring"""


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
    """

    def implication(self, a: Scalar | Array, b: Scalar | Array) -> Scalar | Array:
        r"""Boolean implication ($a \to b$ = $\neg a \lor b$)."""
        return self.add(self.complement(a), b)


class PolynomialSemiring[PolynomialRepr, K: Semiring](eqx.Module):
    """
    A polynomial semiring is formed from the set of polynomials with one or more
    indeterminants with coefficients in the underlying semiring (`algebra`).

    In general, such polynomials are defined over rings or fields, but they generalize
    well to semirings, especially in the context of automata.

    The variables of the polynomial are indexed by integers up to `degree`.
    """

    algebra: K
    """The underlying algebra to define the polynomial on. Must be a semiring or a specialization of a semiring"""
    degree: int = eqx.field(static=True)
    """Maximum degree of the multilinear polynomial."""

    def __post_init__(self) -> None:  # noqa: B027
        pass

    @property
    def zero(self) -> PolynomialRepr:
        return self.constant(self.algebra.zero)

    @property
    def one(self) -> PolynomialRepr:
        return self.constant(self.algebra.one)

    # @property
    # def add(self) -> BinaryOp:
    #     return self._add

    # @property
    # def mul(self) -> BinaryOp[PolynomialRepr]:
    #     return self._mul

    @abstractmethod
    def variable(self, i: int, coefficient: None | ScalarLike | Array = None) -> PolynomialRepr:
        """Create polynomial representing a single variable x_i."""

    @abstractmethod
    def constant(self, value: ScalarLike | Array) -> PolynomialRepr:
        """Create a constant polynomial"""

    @abstractmethod
    def _add(self, a: PolynomialRepr, b: PolynomialRepr) -> PolynomialRepr:
        """Add two polynomials with respect to their underlying algebra"""

    @abstractmethod
    def _mul(self, a: PolynomialRepr, b: PolynomialRepr) -> PolynomialRepr:
        """Multiply two polynomials with respect to their underlying algebra"""

    @abstractmethod
    def evaluate(self, poly: PolynomialRepr, point: Mapping[int, ScalarLike | Array]) -> PolynomialRepr:
        """Evaluate polynomial at a point."""

    @abstractmethod
    def compose(
        self,
        poly: PolynomialRepr,
        replacements: Mapping[int, PolynomialRepr],
    ) -> PolynomialRepr:
        """Compose polynomial with multiple substitutions.

        Returns p(x_1 <- q_1, ..., x_n <- q_n) where only specified indices are replaced.

        Note
        ----
        The composition should be performed simultaneuously. If not, this is a bug.
        """


class MultilinearPolynomialAlgebra[PolynomialRepr, K: BoundedDistributiveLattice](PolynomialSemiring[PolynomialRepr, K]):
    """
    A multilinear polynomial over multiple variables/indeterminants has the maximum
    power of each indeterminant to be 1, i.e., each term is linear with respect to each
    variable in the term.

    Thus, to algebraically represent these, we need to guarantee that multiplication of
    two identical variables are idempotent so as to maintain the "square-free"
    monomials.

    Otherwise, these polynomials are identical to the usual.
    """
