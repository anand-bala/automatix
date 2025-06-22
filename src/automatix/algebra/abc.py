from abc import ABC, abstractmethod
from typing import Generic, Mapping, TypeVar

from typing_extensions import ClassVar, Self

S = TypeVar("S")


class AbstractSemiring(ABC, Generic[S]):
    """Base semiring class"""

    @staticmethod
    @abstractmethod
    def zero() -> S:
        """Return the additive identity"""

    @staticmethod
    @abstractmethod
    def one() -> S:
        """Return the multiplicative identity"""

    @staticmethod
    @abstractmethod
    def add(x: S, y: S) -> S:
        """Return the addition of two elements in the semiring"""

    @staticmethod
    @abstractmethod
    def multiply(x: S, y: S) -> S:
        """Return the multiplication of two elements in the semiring"""

    is_additively_idempotent: ClassVar[bool] = False
    is_multiplicatively_idempotent: ClassVar[bool] = False
    is_commutative: ClassVar[bool] = False
    is_simple: ClassVar[bool] = False


class AbstractNegation(ABC, Generic[S]):
    """A negation function on `S` is an involution on `S`"""

    @staticmethod
    @abstractmethod
    def negate(x: S) -> S:
        """An involution in the algebra"""


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


_Poly = TypeVar("_Poly", bound=AbstractPolynomial)


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
