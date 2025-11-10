"""Pure interface definitions for semiring operations and related abstractions.

This module defines the abstract base classes that all semiring implementations
must follow. These are pure interfaces with no implementation.
"""
# ruff: noqa: ANN401

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Mapping, Protocol, TypeVar, runtime_checkable

import jax
from jaxtyping import Array, Num
from typing_extensions import ClassVar, Self, TypeAlias

if TYPE_CHECKING:
    from automatix.algebra.kernels import AlgebraicStructure

# Type variables for semiring elements
S = TypeVar("S")

# Type aliases for JAX arrays
Axis: TypeAlias = None | int | tuple[int, ...]
Shape: TypeAlias = int | tuple[int, ...]


class AbstractSemiring(ABC):
    """Base semiring interface for array-based operations.

    A semiring is an algebraic structure (S, +, *, 0, 1) where:
    - + (add) is associative and commutative with identity 0
    - * (multiply) is associative with identity 1
    - * distributes over +
    - 0 * x = 0 for all x

    All operations work on arrays and are designed to be JIT-compilable.
    """

    @staticmethod
    @abstractmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        """Return an array of given shape filled with the additive identity (zero).

        Parameters
        ----------
        shape : Shape
            Shape of the new array, e.g., `(2, 3)` or `2`.

        Returns
        -------
        Num[Array, "..."]
            Array filled with the semiring's zero element.
        """

    @staticmethod
    @abstractmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        """Return an array of given shape filled with the multiplicative identity (one).

        Parameters
        ----------
        shape : Shape
            Shape of the new array, e.g., `(2, 3)` or `2`.

        Returns
        -------
        Num[Array, "..."]
            Array filled with the semiring's one element.
        """

    @classmethod
    @abstractmethod
    def add(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        """Element-wise semiring addition (+).

        Parameters
        ----------
        x1 : Num[Array, " n"]
            First input array.
        x2 : Num[Array, " n"]
            Second input array.

        Returns
        -------
        Num[Array, " n"]
            Result of semiring addition.
        """

    @classmethod
    @abstractmethod
    def multiply(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        """Element-wise semiring multiplication (*).

        Parameters
        ----------
        x1 : Num[Array, " n"]
            First input array.
        x2 : Num[Array, " n"]
            Second input array.

        Returns
        -------
        Num[Array, " n"]
            Result of semiring multiplication.
        """

    @classmethod
    @abstractmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        """Sum reduction using semiring addition (+).

        Parameters
        ----------
        a : Num[Array, " ..."]
            Input array.
        axis : Axis, optional
            Axis or axes along which to sum. Default is None (sum all).

        Returns
        -------
        Num[Array, " ..."]
            Result of semiring sum.
        """

    @classmethod
    @abstractmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        """Product reduction using semiring multiplication (*).

        Parameters
        ----------
        a : Num[Array, " ..."]
            Input array.
        axis : Axis, optional
            Axis or axes along which to multiply. Default is None (product all).

        Returns
        -------
        Num[Array, " ..."]
            Result of semiring product.
        """

    @classmethod
    def vdot(cls, a: Num[Array, " n"], b: Num[Array, " n"]) -> Num[Array, ""]:
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
        return cls.sum(cls.multiply(a, b))

    @classmethod
    def matmul(cls, a: Num[Array, "n k"], b: Num[Array, "k m"]) -> Num[Array, "n m"]:
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
        mv = jax.vmap(cls.vdot, (0, None), 0)
        mm = jax.vmap(mv, (None, 1), 1)
        c: Num[Array, "n m"] = jax.jit(mm)(a, b)
        return c

    @classmethod
    def to_kernel(cls) -> AlgebraicStructure:
        """Convert this semiring class to a GPU-optimized kernel.

        Returns an AlgebraicStructure instance that encodes the semiring's
        operations in a form suitable for JAX/GPU computation.

        The default implementation creates a kernel from the class's methods.
        Subclasses can override to provide custom kernels.

        Returns
        -------
        AlgebraicStructure
            Kernel representation of this semiring.

        """
        from automatix.algebra.kernels import AlgebraicStructure

        # Create kernel from class methods
        kernel = AlgebraicStructure(
            add=cls.add,
            mul=cls.multiply,
            zero=cls.zeros(()),  # Scalar
            one=cls.ones(()),  # Scalar
            sum=cls.sum,  # type: ignore[arg-type]
            prod=cls.prod,  # type: ignore[arg-type]
            negate=cls.negate if hasattr(cls, "negate") else None,
            properties=frozenset(cls._get_properties()),
        )

        return kernel

    @classmethod
    def _get_properties(cls) -> set[str]:
        """Get the set of algebraic properties for this semiring.

        Returns
        -------
        set[str]
            Set of property names (e.g., {"idempotent_add", "commutative"}).
        """
        props = set()
        if cls.is_additively_idempotent:
            props.add("idempotent_add")
        if cls.is_multiplicatively_idempotent:
            props.add("idempotent_mul")
        if cls.is_commutative:
            props.add("commutative")
        if cls.is_simple:
            props.add("simple")
        return props

    # Class variables for semiring properties
    is_additively_idempotent: ClassVar[bool] = False
    is_multiplicatively_idempotent: ClassVar[bool] = False
    is_commutative: ClassVar[bool] = False
    is_simple: ClassVar[bool] = False


class AbstractNegation(ABC):
    """Interface for negation operation (~).

    A negation is an involution on the semiring: ~(~x) = x.
    """

    @classmethod
    @abstractmethod
    def negate(cls, x: Num[Array, "*size"]) -> Num[Array, "*size"]:
        """Apply negation (~) to an array element-wise.

        Parameters
        ----------
        x : Num[Array, "*size"]
            Input array.

        Returns
        -------
        Num[Array, "*size"]
            Negated array.
        """


class AbstractDeMorganAlgebra(AbstractSemiring, AbstractNegation):
    """Interface for De Morgan algebras.

    A De Morgan algebra is a semiring with negation where the semiring operations
    are idempotent and commutative.

    Properties:
    - Additive idempotence: x + x = x
    - Multiplicative idempotence: x * x = x
    - Commutativity: x + y = y + x, x * y = y * x
    """

    is_additively_idempotent: ClassVar[bool] = True
    is_multiplicatively_idempotent: ClassVar[bool] = True
    is_commutative: ClassVar[bool] = True
    is_simple: ClassVar[bool] = True


# Polynomial abstractions (unchanged from abc.py)


class AbstractPolynomial(ABC, Generic[S]):
    """A polynomial with coefficients and the value of variables in `S`, where `S` is a semiring."""

    @property
    @abstractmethod
    def support(self) -> set[str]:
        """Return the list of variables with non-zero coefficients in the polynomial"""
        ...

    @property
    @abstractmethod
    def context(self) -> "PolynomialManager[Self, S]":
        """Return the reference to the current polynomial context manager"""

    @abstractmethod
    def declare(self, var: str) -> Self:
        """Declare a variable for the polynomial."""

    @abstractmethod
    def top(self) -> Self:
        """Return the multiplicative identity of the polynomial ring"""

    @abstractmethod
    def bottom(self) -> Self:
        """Return the additive identity of the polynomial ring"""

    @abstractmethod
    def is_bottom(self) -> bool:
        """Returns `True` if the Polynomial is just the additive identity in the ring."""

    @abstractmethod
    def is_top(self) -> bool:
        """Returns `True` if the Polynomial is just the multiplicative identity in the ring."""

    @abstractmethod
    def const(self, value: S) -> Self:
        """Return a new constant polynomial with value"""

    @abstractmethod
    def let(self, mapping: Mapping[str, S | Self]) -> Self:
        """Substitute variables with constants or other polynomials."""

    @abstractmethod
    def eval(self, mapping: Mapping[str, S]) -> S:
        """Evaluate the polynomial with the given variable values.

        !!! note

            Asserts that all variables that form the support of the polynomial are used.
        """

    @abstractmethod
    def negate(self) -> Self:
        """return the negation of the polynomial"""

    @abstractmethod
    def add(self, other: S | Self) -> Self:
        """Return the addition (with appropriate ring) of two polynomials."""

    @abstractmethod
    def multiply(self, other: S | Self) -> Self:
        """Return the multiplication (with appropriate ring) of two polynomials."""

    def __add__(self, other: S | Self) -> Self:
        return self.add(other)

    def __radd__(self, other: S | Self) -> Self:
        return self.add(other)

    def __mul__(self, other: S | Self) -> Self:
        return self.multiply(other)

    def __rmul__(self, other: S | Self) -> Self:
        return self.multiply(other)

    def __call__(self, mapping: Mapping[str, S | Self]) -> S | Self:
        return self.let(mapping)


_Poly = TypeVar("_Poly")


class PolynomialManager(ABC, Generic[_Poly, S]):
    """Context manager for polynomials.

    This context allows polynomials represented as decision diagrams to share
    their structure and, thus, minimize the memory footprint of all the polynomials
    used in the system.
    """

    @property
    @abstractmethod
    def top(self) -> _Poly:
        """Return the multiplicative identity of the polynomial ring"""

    @property
    @abstractmethod
    def bottom(self) -> _Poly:
        """Return the additive identity of the polynomial ring"""

    @abstractmethod
    def is_bottom(self, poly: _Poly) -> bool:
        """Returns `True` if the Polynomial is just the additive identity in the ring."""

    @abstractmethod
    def is_top(self, poly: _Poly) -> bool:
        """Returns `True` if the Polynomial is just the multiplicative identity in the ring."""

    @abstractmethod
    def const(self, value: S) -> _Poly:
        """Return a constant in the polynomial"""

    @abstractmethod
    def var(self, name: str) -> _Poly:
        """Get the monomial for the variable with the given name"""

    @abstractmethod
    def declare(self, var: str) -> _Poly:
        """Declare a variable with the given name"""

    @abstractmethod
    def let(self, poly: _Poly, mapping: Mapping[str, S | _Poly]) -> _Poly:
        """Substitute variables with constants or other polynomials."""

    @abstractmethod
    def negate(self, poly: _Poly) -> _Poly:
        """return the negation of the polynomial"""

    @abstractmethod
    def add(self, lhs: _Poly, rhs: _Poly) -> _Poly:
        """Return the addition (with appropriate ring) of two polynomials."""

    @abstractmethod
    def multiply(self, lhs: _Poly, rhs: _Poly) -> _Poly:
        """Return the multiplication (with appropriate ring) of two polynomials."""


# Weight function abstractions for Phase 2


@runtime_checkable
class GuardWeightFunction(Protocol):
    """Protocol for guard-based weight functions.

    A guard-based weight function maps a guard condition to a weight value.
    The weight function is semiring-agnostic: it outputs values in any domain.
    The semiring validates that outputs match the expected domain at construction time.

    Examples
    --------
    Guard functions typically take a guard (expressed as a predicate or expression)
    and return a weight. The actual semiring is specified when attaching this
    function to an automaton.

    Notes
    -----
    Weight functions must be JAX-compatible (pytrees) if they contain arrays.
    They should support jax.jit and jax.vmap for efficient computation.
    """

    def __call__(self, guard: Any) -> Any:
        """Evaluate the weight function for a given guard.

        Parameters
        ----------
        guard : Any
            A guard condition (typically a predicate or expression).

        Returns
        -------
        Any
            The weight value. The domain (real numbers, tropical, log-domain, etc.)
            depends on the guard and the semiring this function is used with.
        """
        ...


@runtime_checkable
class GlobalWeightFunction(Protocol):
    """Protocol for global weight functions.

    A global weight function assigns a constant weight to all transitions
    in an automaton. Unlike guard-based functions, it does not depend on
    individual guards.

    The weight function is semiring-agnostic: it outputs a value in any domain.
    The semiring validates that outputs match the expected domain at construction time.

    Examples
    --------
    Global weight functions are useful for uniform weighting schemes, such as:
    - Assigning the same weight to all transitions
    - Learnable global scaling factors
    - Numerical stability constants (e.g., log-domain offsets)

    Notes
    -----
    Weight functions must be JAX-compatible (pytrees) if they contain arrays.
    They should support jax.jit and jax.vmap for efficient computation.
    """

    def __call__(self) -> Any:
        """Evaluate the global weight function.

        Returns
        -------
        Any
            The weight value. The domain (real numbers, tropical, log-domain, etc.)
            depends on the semiring this function is used with.
        """
        ...


class WeightFunctionValidator(ABC):
    """Abstract base class for weight function validators.

    Validators ensure that weight functions produce values in the expected
    domain for a given semiring. This implements the runtime validation pattern.

    Each semiring should provide a validator that checks:
    - Output shape and type
    - Value ranges (e.g., finite, non-negative)
    - Domain-specific constraints (e.g., log-domain numbers)
    """

    @classmethod
    @abstractmethod
    def validate(cls, weight: Any) -> bool:
        """Validate that a weight value is compatible with this semiring.

        Parameters
        ----------
        weight : Any
            A weight value to validate.

        Returns
        -------
        bool
            True if the weight is valid for this semiring, False otherwise.

        Raises
        ------
        ValueError
            With a descriptive error message if validation fails.
        """
        ...

    @classmethod
    def validate_many(cls, weights: list[Any]) -> bool:
        """Validate multiple weight values.

        Parameters
        ----------
        weights : list[Any]
            A list of weight values to validate.

        Returns
        -------
        bool
            True if all weights are valid.

        Raises
        ------
        ValueError
            With a descriptive error message if any validation fails.
        """
        for i, weight in enumerate(weights):
            try:
                cls.validate(weight)
            except ValueError as e:
                raise ValueError(f"Weight at index {i} failed validation: {e}") from e
        return True
