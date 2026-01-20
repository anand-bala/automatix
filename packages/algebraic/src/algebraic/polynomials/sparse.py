"""Sparse polynomial representation using dictionary-based storage."""
# mypy: disable-error-code="misc"

from __future__ import annotations

import functools
import typing
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping

import bitarray.util as ba_util
import equinox as eqx
import jax.numpy as jnp
from bitarray import bitarray, frozenbitarray
from jaxtyping import Array, Scalar, ScalarLike, Shaped
from typing_extensions import override

from algebraic.spec import BoundedDistributiveLattice as Lattice

type S = Array | ScalarLike
K = typing.TypeVar("K", bound=Lattice)


class SparsePolynomial(eqx.Module, Mapping[frozenbitarray, Array | Scalar], typing.Generic[K]):
    """Sparse polynomial represented as monomial -> coefficient mapping."""

    algebra: K
    num_vars: int = eqx.field(static=True)
    data: Mapping[frozenbitarray, Array | Scalar] = eqx.field(default_factory=dict)

    def __check_init__(self) -> None:
        if not isinstance(self.algebra, Lattice):
            raise TypeError("Multilinear polynomial representation is only supported over BoundedDistributiveLattice algebras")

    @override
    def __getitem__(self, monomial: bitarray | str | Iterable[int]) -> Array | Scalar:
        """Return the coefficient of the monomial with the given binary powers."""
        return self.data[frozenbitarray(monomial)]

    @override
    def __iter__(self) -> Iterator[frozenbitarray]:
        return iter(self.data)

    @override
    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def constant(value: Scalar, num_vars: int, algebra: K) -> SparsePolynomial[K]:
        """
        Examples
        --------
        >>> from algebraic.semirings import boolean_algebra
        >>> import jax.numpy as jnp
        >>> alg = boolean_algebra('ste')
        >>> p = SparsePolynomial.constant(jnp.array(True), num_vars=3, algebra=alg)
        >>> p['000']
        Array(True, dtype=bool)
        """
        zeros_idx = frozenbitarray(ba_util.zeros(num_vars))
        return SparsePolynomial(algebra, num_vars, {zeros_idx: value})

    @staticmethod
    def zero(num_vars: int, algebra: K) -> SparsePolynomial[K]:
        return SparsePolynomial.constant(algebra.zero, num_vars, algebra)

    @staticmethod
    def one(num_vars: int, algebra: K) -> SparsePolynomial[K]:
        return SparsePolynomial.constant(algebra.one, num_vars, algebra)

    @staticmethod
    def variable(index: int, num_vars: int, algebra: K) -> SparsePolynomial[K]:
        """Create polynomial representing a single variable x_i."""
        monomial = ba_util.zeros(num_vars)
        monomial[index] = 1
        coefficient = jnp.asarray(algebra.one)
        assert coefficient is not None
        return SparsePolynomial(algebra, num_vars, {frozenbitarray(monomial): coefficient})

    def __add__(self, other: SparsePolynomial[K] | Scalar) -> SparsePolynomial[K]:
        # This will essentially merge the two polynomials by adding the monomial coefficients where they are common, or using the additive identity where one isn't available.
        if jnp.isscalar(other):
            other = SparsePolynomial.constant(typing.cast(Scalar, other), self.num_vars, self.algebra)
        assert isinstance(other, SparsePolynomial)
        assert self.algebra == other.algebra
        assert self.num_vars == other.num_vars

        ret = {
            key: self.algebra.add(
                self.get(key, self.algebra.zero),
                other.get(key, self.algebra.zero),
            )
            for key in self.keys() | other.keys()
        }
        return SparsePolynomial(self.algebra, self.num_vars, dict(ret))

    def __mul__(self, other: SparsePolynomial[K] | Scalar) -> SparsePolynomial[K]:
        r"""Multiply two polynomials.

        $(\sum_{S \in a} c_S x^S) \cdot (\sum_{T \in b} d_T x^T) = sum_{S,T} (c_S * d_T) x^{S \cup T}$
        """
        if jnp.isscalar(other):
            other = SparsePolynomial.constant(typing.cast(Scalar, other), self.num_vars, self.algebra)
        assert isinstance(other, SparsePolynomial)
        assert self.algebra == other.algebra
        assert self.num_vars == other.num_vars
        ret: defaultdict[frozenbitarray, Scalar] = defaultdict(lambda: self.algebra.zero)  # initialize with additive identity
        for m_a, c_a in self.items():
            for m_b, c_b in other.items():
                new_monom = frozenbitarray(m_a | m_b)
                new_coeff = self.algebra.mul(c_a, c_b)
                ret[new_monom] = self.algebra.add(ret[new_monom], new_coeff)
        return SparsePolynomial(self.algebra, self.num_vars, dict(ret))

    def evaluate(self, point: Shaped[Array, " {self.num_vars}"] | Mapping[int, Scalar]) -> SparsePolynomial[K]:
        """Evaluate polynomial at a point.

        Examples
        --------
        >>> from algebraic.semirings import boolean_algebra
        >>> import jax.numpy as jnp
        >>> num_vars = 2
        >>> algebra = boolean_algebra(mode='logic')
        >>> x_0 = SparsePolynomial.variable(0, num_vars, algebra)
        >>> x_1 = SparsePolynomial.variable(1, num_vars, algebra)
        >>> p = x_0 * x_1  # x_0 AND x_1
        >>> e1 = p.evaluate(dict(enumerate(jnp.array([True, True]))))
        >>> e1.isscalar()
        True
        >>> e1[(0,0)]
        Array(True, dtype=bool...)
        >>> p.evaluate(dict(enumerate(jnp.array([True, False]))))['00']
        Array(False, dtype=bool...)
        """
        if isinstance(point, Mapping):
            replacements = {i: SparsePolynomial.constant(val, self.num_vars, self.algebra) for i, val in point.items()}  # ty: ignore[invalid-argument-type]
        else:
            assert eqx.is_array(point)
            replacements = {i: SparsePolynomial.constant(point[i], self.num_vars, self.algebra) for i in range(self.num_vars)}
        return self.compose(replacements)  # ty: ignore[invalid-argument-type]

    def compose(self, replacements: Mapping[int, SparsePolynomial[K]]) -> SparsePolynomial[K]:
        """Compose polynomial with multiple substitutions.

        Returns p(x_1 <- q_1, ..., x_n <- q_n) where only specified indices are replaced.

        Note
        ----
        The composition should be performed simultaneously. If not, this is a bug.
        """
        result = SparsePolynomial.zero(self.num_vars, self.algebra)  # initialize with additive identity

        def var_at(idx: int) -> SparsePolynomial[K]:
            return SparsePolynomial.variable(idx, self.num_vars, self.algebra)

        def as_const(val: Scalar) -> SparsePolynomial[K]:
            return SparsePolynomial.constant(val, self.num_vars, self.algebra)

        for monomial, coeff in self.items():
            # make a new term by replacing the monomial terms with either the replacement if it exists, or a plain variable.
            term = functools.reduce(
                lambda a, b: a * b,
                (replacements.get(idx, var_at(idx)) for idx, deg in enumerate(monomial) if deg == 1),
                as_const(coeff),
            )
            result = result + term

        return result

    def isscalar(self) -> bool:
        return len(self) == 0 or (len(self) == 1 and self.get(frozenbitarray(self.num_vars)) is not None)
