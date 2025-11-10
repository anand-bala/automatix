"""Multilinear polynomial data structures over algebraic structures.

This module provides the core MultilinearPolynomial class for representing
and manipulating polynomials where indeterminants correspond to automaton states.

Key design decisions:
- Dense flat vector representation: coefficients stored as shape (2^q,)
- Monomial indexing via binary encoding: alpha tuple -> integer
- Immutable: use frozen dataclass to ensure coefficient arrays aren't mutated
"""

from typing import Optional, Tuple

import attrs
import jax.numpy as jnp
from attrs import frozen
from jaxtyping import Array, Num, Scalar

from automatix.algebra._compat import normalize_semiring
from automatix.algebra.kernels import AlgebraicStructure
from automatix.algebra.spec import AbstractSemiring


@frozen
class MultilinearPolynomial:
    r"""Multilinear polynomial over automaton states and a ring algebra.

    A multilinear polynomial over states {q_0, ..., q_{q-1}} is uniquely
    represented by a vector of 2^q coefficients, one per monomial.

    Each monomial corresponds to a subset of states, indexed by a binary
    tuple (alpha_0, alpha_1, ..., alpha_{q-1}) where alpha_i = 1 means
    "state q_i appears in this monomial".

    The polynomial represents:
        P(x_0, ..., x_{q-1}) = sum_{alpha in {0,1}^q} c_alpha * product_{i: alpha_i=1} x_i

    Attributes
    ----------
    algebra : AlgebraicStructure
        The ring algebra to interpret the polynomial over
    coefficients : Num[Array, " {2**num_states}"]
        Coefficient vector, where coefficients[alpha_as_int] is the
        ring value for the monomial indexed by alpha.
    num_states : int
        Number of automaton states (polynomial indeterminants).
    max_degree : Optional[int]
        If set, maximum degree constraint for this polynomial.
        None means no constraint (degree can be up to num_states).

    Examples
    --------
    For a 2-state polynomial with Boolean coefficients:

    >>> import jax.numpy as jnp
    >>> from automatix.algebra import create_boolean_kernel
    >>> poly = MultilinearPolynomial(
    ...     algebra=create_boolean_kernel(),
    ...     coefficients=jnp.array([0.0, 1.0, 1.0, 0.0]),
    ...     num_states=2,
    ...     max_degree=1
    ... )

    The polynomial represents: 0 + 1*x_0 + 1*x_1 + 0*x_0*x_1 = x_0 + x_1
    """

    algebra: AlgebraicStructure = attrs.field(converter=normalize_semiring)
    coefficients: Num[Array, " {2**num_states}"]
    num_states: int
    max_degree: Optional[int] = None

    @staticmethod
    def alpha_to_int(alpha: Tuple[int, ...]) -> int:
        r"""Convert binary tuple to monomial index.

        Parameters
        ----------
        alpha : Tuple[int, ...]
            Binary tuple (a_0, a_1, ..., a_{q-1}) where a_i in {0, 1}.

        Returns
        -------
        int
            Integer index: sum_i a_i * 2^i

        Examples
        --------
        >>> MultilinearPolynomial.alpha_to_int((0, 0, 0))
        0
        >>> MultilinearPolynomial.alpha_to_int((1, 0, 0))
        1
        >>> MultilinearPolynomial.alpha_to_int((0, 1, 0))
        2
        >>> MultilinearPolynomial.alpha_to_int((1, 1, 0))
        3
        >>> MultilinearPolynomial.alpha_to_int((1, 1, 1))
        7
        """
        return sum(a * (2**i) for i, a in enumerate(alpha))

    @staticmethod
    def int_to_alpha(index: int, num_states: int) -> Tuple[int, ...]:
        r"""Convert monomial index to binary tuple.

        Inverse of alpha_to_int. Converts an integer back to the binary
        representation of which states appear in the monomial.

        Parameters
        ----------
        index : int
            Monomial index in [0, 2^num_states).
        num_states : int
            Number of states (polynomial indeterminants).

        Returns
        -------
        Tuple[int, ...]
            Binary tuple (a_0, a_1, ..., a_{q-1}) with a_i in {0, 1}.

        Examples
        --------
        >>> MultilinearPolynomial.int_to_alpha(0, 3)
        (0, 0, 0)
        >>> MultilinearPolynomial.int_to_alpha(1, 3)
        (1, 0, 0)
        >>> MultilinearPolynomial.int_to_alpha(3, 3)
        (1, 1, 0)
        >>> MultilinearPolynomial.int_to_alpha(7, 3)
        (1, 1, 1)
        """
        alpha = []
        remaining = index
        for _ in range(num_states):
            alpha.append(remaining & 1)
            remaining >>= 1
        return tuple(alpha)

    def get_monomial(self, alpha: Tuple[int, ...]) -> Num[Array, ""]:
        r"""Get coefficient of a single monomial.

        Parameters
        ----------
        alpha : Tuple[int, ...]
            Binary tuple indexing the monomial.

        Returns
        -------
        Num[Array, ""]
            Scalar ring value for this monomial.
        """
        idx = self.alpha_to_int(alpha)
        return self.coefficients[idx]

    def set_monomial(self, alpha: Tuple[int, ...], value: Num[Array, ""]) -> "MultilinearPolynomial":
        r"""Create new polynomial with updated monomial coefficient.

        Returns a new polynomial (dataclass is frozen).

        Parameters
        ----------
        alpha : Tuple[int, ...]
            Binary tuple indexing the monomial.
        value : Num[Array, ""]
            New ring value.

        Returns
        -------
        MultilinearPolynomial
            New polynomial with updated coefficient.
        """
        idx = self.alpha_to_int(alpha)
        new_coeffs = self.coefficients.at[idx].set(value)
        return MultilinearPolynomial(
            algebra=self.algebra,
            coefficients=new_coeffs,
            num_states=self.num_states,
            max_degree=self.max_degree,
        )

    def degree(self) -> int:
        r"""Compute the degree of this polynomial.

        The degree is the maximum number of state indeterminants in any
        nonzero monomial.

        Returns
        -------
        int
            Maximum degree among all monomials (0 <= degree <= num_states).

        Notes
        -----
        This is a pure Python function that iterates through all monomials,
        so it's O(2^num_states). Use sparingly or cache results.
        """
        max_degree = 0
        for idx in range(len(self.coefficients)):
            if not jnp.isclose(self.coefficients[idx], self.algebra.zero):
                alpha = self.int_to_alpha(idx, self.num_states)
                monomial_degree = sum(alpha)
                max_degree = max(max_degree, monomial_degree)
        return max_degree

    @classmethod
    def zeros(
        cls,
        algebra: AlgebraicStructure | type[AbstractSemiring],
        num_states: int,
        max_degree: Optional[int] = None,
    ) -> "MultilinearPolynomial":
        r"""Create zero polynomial.

        Parameters
        ----------
        algebra : AlgebraicStructure | type[AbstractSemiring]
            Algebra to define the polynomial over.
        num_states : int
            Number of automaton states.
        max_degree : Optional[int]
            Maximum degree constraint, if any.

        Returns
        -------
        MultilinearPolynomial
            Polynomial with all coefficients = 0.
        """
        algebra = normalize_semiring(algebra)
        return cls(
            algebra=algebra,
            coefficients=algebra.zeros(2**num_states),
            num_states=num_states,
            max_degree=max_degree,
        )

    @classmethod
    def ones(
        cls, algebra: AlgebraicStructure | type[AbstractSemiring], num_states: int, max_degree: Optional[int] = None
    ) -> "MultilinearPolynomial":
        r"""Create constant polynomial (degree 0, coefficient 1).

        Parameters
        ----------
        algebra : AlgebraicStructure | type[AbstractSemiring]
            Algebra to define the polynomial over.
        num_states : int
            Number of automaton states.
        max_degree : Optional[int]
            Maximum degree constraint, if any.

        Returns
        -------
        MultilinearPolynomial
            Polynomial: 1 + 0*x_i for all i (only constant term nonzero).
        """
        algebra = normalize_semiring(algebra)
        coeffs = algebra.zeros(2**num_states)
        coeffs = coeffs.at[0].set(algebra.one)
        return cls(
            algebra=algebra,
            coefficients=coeffs,
            num_states=num_states,
            max_degree=max_degree,
        )

    @classmethod
    def from_monomial(
        cls,
        algebra: AlgebraicStructure | type[AbstractSemiring],
        num_states: int,
        alpha: Tuple[int, ...],
        coefficient: Scalar,
    ) -> "MultilinearPolynomial":
        r"""Create polynomial from a single monomial.

        Parameters
        ----------
        algebra : AlgebraicStructure | type[AbstractSemiring]
            Algebra to define the polynomial over.
        num_states : int
            Number of automaton states.
        alpha : Tuple[int, ...]
            Binary tuple indexing the monomial.
        coefficient : Num[Array, ""]
            Ring value for this monomial.

        Returns
        -------
        MultilinearPolynomial
            Polynomial with only this monomial nonzero.
        """
        algebra = normalize_semiring(algebra)
        coeffs = algebra.zeros(2**num_states)
        idx = cls.alpha_to_int(alpha)
        coeffs = coeffs.at[idx].set(coefficient)
        degree = sum(alpha)
        return cls(
            algebra=algebra,
            coefficients=coeffs,
            num_states=num_states,
            max_degree=degree,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MultilinearPolynomial(num_states={self.num_states}, "
            f"max_degree={self.max_degree}, coeffs_shape={self.coefficients.shape})"
        )
