"""Pure interface definitions for heirarchy of rings and lattices.

This module defines the abstract base classes that all algebraic implementations
must follow. These are pure interfaces with no implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from jaxtyping import Array, Num
from typing_extensions import overload

type Axis = int | Sequence[int]
type MaybeAxis = None | Axis
type Shape = int | Sequence[int]

type UnaryOp = Callable[[Array], Array]
type BinaryOp = Callable[[Array, Array], Array]
type VdotFn = Callable[[Num[Array, " n"], Num[Array, " n"]], Num[Array, ""]]
type MatmulFn = Callable[[Num[Array, "n k"], Num[Array, "k m"]], Num[Array, "n m"]]


@runtime_checkable
class ReductionOp(Protocol):
    def __call__(self, a: Array, axis: MaybeAxis = None) -> Array: ...


# type ReductionOp =


type Property = Literal["idempotent_add", "idempotent_mul", "commutative", "simple", "complemented"] | str  # noqa: PYI051


@dataclass
class AlgebraicStructure:
    properties: set[Property] = field(default_factory=set, kw_only=True, metadata=dict(static=True))
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
        return getattr(self, "negate", None) is not None

    def __post_init__(self) -> None:
        pass


@dataclass
class Semiring(AlgebraicStructure):
    """A simple runtime representation of an algebraic semiring."""

    add: BinaryOp
    """Semiring addition operation (oplus)"""

    mul: BinaryOp
    """Semiring multiplication (otimes)"""

    zero: Array
    """Additive identity of the semiring"""

    one: Array
    """Multiplicative identity of the semiring"""


@dataclass
class BoundedDistributiveLattice(Semiring):
    """A bounded distributive lattice is a specialization of a semiring, where the `oplus` operator corresponds to `join` operator, `otimes` is the `meet` operator."""

    @property
    def join(self) -> BinaryOp:
        return self.add

    @property
    def meet(self) -> BinaryOp:
        return self.mul

    @property
    def top(self) -> Array:
        return self.one

    @property
    def bottom(self) -> Array:
        return self.zero

    def __post_init__(self) -> None:
        super().__post_init__()

        self.properties |= {"idempotent_add", "idempotent_mul", "commutative", "simple"}


@dataclass
class Ring(Semiring):
    """A ring is a semiring with the additional requirement that each element must have an additive inverse"""

    additive_inverse: UnaryOp


@dataclass
class DeMorganAlgebra(BoundedDistributiveLattice):
    """
    A De Morgan Algebra is a bounded distributive lattice equipped with
    a complementation operator that is an involution (`~~a = a`) that follows De
    Morgan's laws.
    """

    complement: UnaryOp

    def __post_init__(self) -> None:
        super().__post_init__()

        self.properties |= {"complemented"}


@dataclass
class HeytingAlgebra(BoundedDistributiveLattice):
    """
    A Heyting algebra is a bounded lattice equipped with a binary operation `a -> b`
    called implication such that `(c and a) <= b` is equivalent to `c <= (a -> b)`

    A Heyting algebra has a pseudo-complement such that `~a` is equivalent to `a -> 0`.
    """

    implication: BinaryOp

    def complement(self, value: Array) -> Array:
        return self.implication(value, self.zero)


@dataclass
class StoneAlgebra(BoundedDistributiveLattice):
    """
    A Stone Algebra is a bounded distributive lattice equipped with a pseudo-complement
    such that `~a or ~~a = 1` (but is not necessarily an involution) but follows De
    Morgan's laws.
    """

    complement: UnaryOp


@dataclass
class BooleanAlgebra(DeMorganAlgebra):
    """
    A full Boolean algebra, i.e., the operators with complementation follow:

    1. De Morgan's Laws
    2. The law of excluded middle (`~x or x = 1`)
    3. The law of noncontradiction (`~x and x = 0`)
    """

    def implication(self, a: Array, b: Array) -> Array:
        return self.add(self.complement(a), b)


@dataclass
class BiModule[S: Semiring](ABC):
    """
    A bimodule is a generalization of a vector space over a semiring (ususally a ring,
    but we relax it in this library).

    They are equipped with an operator that is analogous to matrix multiplication (or
    the inner product), and since they are *bimodules*, the operation can be performed
    from either side.

    In this library, we use `BiModule` as a generic interface for the various tensor
    libraries to operate on the algebraic structures.
    """

    algebra: S
    """The underlying algebra to define the module on. Must be a semiring or a specialization of a semiring"""

    sum: ReductionOp | None = None
    """Sum reduction (potentially along a specific axis or set of axes) using semiring addition (+)"""

    prod: ReductionOp | None = None
    """Product reduction (potentially along a specific axis or set of axes) using semiring multiplication (*)"""

    def add(self, x: Array, y: Array) -> Array:
        return self.algebra.add(x, y)

    def mul(self, x: Array, y: Array) -> Array:
        return self.algebra.mul(x, y)

    @overload
    @abstractmethod
    def zeros(self, shape: int) -> Num[Array, " {shape}"]:
        """Return an array of given shape filled with the additive identity (zero)"""

    @overload
    @abstractmethod
    def zeros(self, shape: Sequence[int]) -> Num[Array, " {*shape}"]:
        """Return an array of given shape filled with the additive identity (zero)"""

    # @abstractmethod
    # def zeros(self, shape: Shape) -> Num[Array, " {shape}"]:
    #     """Return an array of given shape filled with the additive identity (zero)"""

    @overload
    @abstractmethod
    def ones(self, shape: int) -> Num[Array, " {shape}"]:
        """Return an array of given shape filled with the multiplicative identity (one)"""

    @overload
    @abstractmethod
    def ones(self, shape: Sequence[int]) -> Num[Array, " {*shape}"]:
        """Return an array of given shape filled with the multiplicative identity (one)"""

    # @abstractmethod
    # def ones(self, shape: Shape) -> Num[Array, " {shape}"]:
    #     """Return an array of given shape filled with the multiplicative identity (one)"""

    @abstractmethod
    def vdot(self, a: Num[Array, " n"], b: Num[Array, " n"]) -> Num[Array, ""]:
        """Compute the dot product of two 1D arrays using the semiring.

        Computes: sum_i (a_i * b_i) using semiring operations.

        Parameters
        ----------
        a : Num[Array, " n"]
            First input array.
        b : Num[Array, " n"]
            Second input array.

        Returns
        -------
        Num[Array, ""]
            Scalar result of the semiring dot product.
        """

    @abstractmethod
    def matmul(self, a: Num[Array, "n k"], b: Num[Array, "k m"]) -> Num[Array, "n m"]:
        """Compute matrix-semiring product of two arrays.

        This uses vdot (which uses semiring operations) for each row-column pair.

        Parameters
        ----------
        a : Num[Array, "n k"]
            First matrix.
        b : Num[Array, "k m"]
            Second matrix.

        Returns
        -------
        Num[Array, "n m"]
            Result of semiring matrix multiplication.
        """

    @abstractmethod
    def tensordot(
        self,
        a: Array,
        b: Array,
        axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    ) -> Array:
        """
        Compute tensor dot product over a semiring.

        Args:
            a: First tensor
            b: Second tensor
            axes: Specification of axes to contract:
                - int: contract last `axes` axes of a with first `axes` axes of b
                - tuple of lists: ([axes_a], [axes_b]) specifying which axes to contract

        Returns:
            Contracted tensor
        """

    @abstractmethod
    def transpose(self, a: Array, axes: Sequence[int] | None = None) -> Array:
        """Returns an array with axes transposed."""
