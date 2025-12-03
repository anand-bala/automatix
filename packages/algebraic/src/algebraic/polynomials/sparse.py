"""Sparse polynomial representation using dictionary-based storage."""

from __future__ import annotations

import functools
import typing as ty
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import bitarray.util as ba_util
from bitarray import bitarray, frozenbitarray
from jaxtyping import Array, ScalarLike
from typing_extensions import final, override

from algebraic.spec import BoundedDistributiveLattice, MultilinearPolynomialAlgebra

type S = Array | ScalarLike


@final
@dataclass(frozen=True)
class SparsePolynomial(Mapping[frozenbitarray, Array | ScalarLike]):
    """Sparse polynomial represented as monomial -> coefficient mapping."""

    data: Mapping[frozenbitarray, Array | ScalarLike] = field(default_factory=dict)

    @override
    def __getitem__(self, monomial: bitarray | str | Iterable[int]) -> Array | ScalarLike:
        """Return the coefficient of the monomial with the given binary powers."""
        return self.data[frozenbitarray(monomial)]

    @override
    def __iter__(self) -> Iterator[frozenbitarray]:
        return iter(self.data)

    @override
    def __len__(self) -> int:
        return len(self.data)


@final
@dataclass
class SparsePolynomialAlgebra[K: BoundedDistributiveLattice](MultilinearPolynomialAlgebra[SparsePolynomial, K]):
    """Algebra for polynomials represented as `SparsePolynomial`

    Complexity
    ----------
    - Space: O(number of nonzero terms)
    - Product: O(|p| * |q|) where |p| is number of terms
    - Substitution: O(|p| * |q|) where q is the substituted polynomial
    - Evaluation: O(|p| * n) where n is num_vars

    Examples
    --------
    Create polynomial x_0 + x_1 over Boolean algebra:

    >>> from algebraic.tensor_algebra.jax import boolean_algebra
    >>> # Create a sparse polynomial algebra with at most 3 variables
    >>> alg = SparsePolynomialAlgebra(boolean_algebra().algebra, 3)
    >>> p = alg.variable(0) # equiv: True and x_0
    >>> q = alg.variable(1) # equiv: True and x_1
    >>> alg.mul(p, q)       # equiv: x_0 or x_1
    {frozenbitarray('100'): Array(True, dtype=bool), frozenbitarray('010'): Array(True, dtype=bool)}
    """

    @override
    def constant(self, value: S) -> SparsePolynomial:
        """
        Examples
        --------
        >>> from algebraic.tensor_algebra.jax import boolean_algebra
        >>> import jax.numpy as jnp
        >>> alg = SparsePolynomial(boolean_algebra(), degree=3)
        >>> p = alg.constant(jnp.array(True))
        >>> p['000']
        Array(True, dtype=bool)
        """
        zeros_idx = frozenbitarray(ba_util.zeros(self.degree))
        return SparsePolynomial({zeros_idx: value})

    @override
    def variable(self, i: int, coefficient: None | S = None) -> SparsePolynomial:
        """Create polynomial representing a single variable x_i."""
        monomial = ba_util.zeros(self.degree)
        monomial[i] = 1
        if coefficient is None:
            coefficient = ty.cast(S, self.algebra.one)
        assert coefficient is not None
        return SparsePolynomial({frozenbitarray(monomial): coefficient})

    @override
    def _add(self, a: SparsePolynomial, b: SparsePolynomial) -> SparsePolynomial:
        # This will essentially merge the two polynomials by adding the monomial coefficients where they are common, or using the additive identity where one isn't available.
        ret = {
            key: self.algebra.add(
                a.get(key, self.algebra.zero),
                b.get(key, self.algebra.zero),
            )
            for key in a.keys() | b.keys()
        }
        return SparsePolynomial(ret)

    @override
    def _mul(self, a: SparsePolynomial, b: SparsePolynomial) -> SparsePolynomial:
        r"""Multiply two polynomials.

        $(\sum_{S \in a} c_S x^S) \cdot (\sum_{T \in b} d_T x^T) = sum_{S,T} (c_S * d_T) x^{S \cup T}$
        """
        ret = {
            frozenbitarray(m_a | m_b):  # monomial powers are added
            self.algebra.mul(c_a, c_b)  # monomial coefficients are multiplied
            for (m_a, c_a) in a.items()
            for (m_b, c_b) in b.items()
        }
        return SparsePolynomial(ret)

    @override
    def evaluate(self, poly: SparsePolynomial, point: Array | Mapping[int, S]) -> SparsePolynomial:
        """Evaluate polynomial at a point.

        Examples
        --------
        >>> from algebraic.tensor_algebra.jax import boolean_algebra
        >>> import jax.numpy as jnp
        >>> alg = SparsePolynomialAlgebra(boolean_algebra(), 2)
        >>> x_0 = alg.variable(0)
        >>> x_1 = alg.variable(1)
        >>> p = alg.mul(x_0, x_1)  # x_0 AND x_1
        >>> p.evaluate(dict(enumerate(jnp.array([True, True]))))
        Array(True, dtype=bool)
        >>> p.evaluate(dict(enumerate(jnp.array([True, False]))))
        Array(False, dtype=bool)
        """
        # Make a mask for the monomials that are affected by this
        mask = ba_util.zeros(self.degree)
        if isinstance(point, Mapping):
            mask[list(point.keys())] = True
        else:
            # all monomials are affected...
            pass

        # We will loop through the terms in the polynomial, not operating on terms that don't pass the mask
        # only valid for Mapping
        if isinstance(point, Mapping):
            result = SparsePolynomial(
                {
                    monomial: coeff
                    for monomial, coeff in poly.items()
                    # Filter out monomial terms that aren't affected by the substitution
                    if ba_util.count_and(monomial, mask) == 0
                }
            )
        else:
            result = self.zero
        # Now, we will process the affected terms
        for monomial in (key for key in poly.keys() if key not in result):
            coeff = poly[monomial]
            # the new term will be the substituted bits set to 0
            new_monom = frozenbitarray(monomial & ~mask)
            # use the algebra to multiply substituted terms for the new coefficient
            new_coeff = functools.reduce(
                self.algebra.mul,
                (point[idx] for idx, deg in enumerate(monomial & mask) if deg == 1),
                coeff,
            )

            result = self.add(result, SparsePolynomial({new_monom: new_coeff}))

        return result

    @override
    def compose(
        self,
        poly: SparsePolynomial,
        replacements: Mapping[int, SparsePolynomial],
    ) -> SparsePolynomial:
        """Compose polynomial with multiple substitutions.

        Returns p(x_1 <- q_1, ..., x_n <- q_n) where only specified indices are replaced.

        Note
        ----
        The composition should be performed simultaneously. If not, this is a bug.
        """
        # Similar to evaluate
        # Make a mask for the monomials that are affected by this
        mask = ba_util.zeros(self.degree)
        mask[list(replacements.keys())] = True

        # We will loop through the terms in the polynomial, not operating on terms that don't pass the mask
        result = SparsePolynomial(
            {
                monomial: coeff
                for monomial, coeff in poly.items()
                # Filter out monomial terms that aren't affected by the substitution
                if ba_util.count_and(monomial, mask) == 0
            }
        )
        # Now, we will process the affected terms
        for monomial in (key for key in poly.keys() if key not in result):
            # Create a monomial with the unsubstituted terms
            coeff = poly[monomial]
            term = SparsePolynomial({frozenbitarray(monomial & ~mask): coeff})
            # use the algebra to multiply substituted terms with the unsub term
            term = functools.reduce(
                self.mul,
                (replacements[idx] for idx, deg in enumerate(monomial & mask) if deg == 1),
                term,
            )

            result = self.add(result, term)

        return result

    def simplify(self, poly: SparsePolynomial) -> SparsePolynomial:
        """Remove all terms with coefficient 0"""
        raise NotImplementedError()

    def simplify_(self, poly: SparsePolynomial) -> SparsePolynomial:
        """In-place variant of `simplify`"""
        raise NotImplementedError()

    def isscalar(self, poly: SparsePolynomial) -> bool:
        return len(poly) == 0 or (len(poly) == 1 and poly.get(frozenbitarray(self.degree)) is not None)
