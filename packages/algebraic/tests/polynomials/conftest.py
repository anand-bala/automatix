"""Shared fixtures and utilities for polynomial tests."""
# ruff: noqa: ANN201, ANN202
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

if TYPE_CHECKING:
    import algebraic as alx
    from algebraic.polynomials.sparse import SparsePolynomial
    from jaxtyping import Array


@pytest.fixture
def bool_algebra() -> alx.BooleanAlgebra:
    """Boolean algebra for tests."""
    import algebraic as alx

    return alx.semirings.boolean_algebra(mode="logic")


@pytest.fixture
def tropical_minplus_algebra() -> alx.Semiring:
    """Max-min algebra (restricted to negative reals) - similar to tropical min-plus."""
    import algebraic as alx

    # This gives a lattice with: add=max, mul=min, zero=-inf, one=inf
    # While not exactly tropical min-plus, it's a lattice that tests similar properties
    return alx.semirings.tropical_semiring(minplus=True)


@pytest.fixture
def tropical_maxplus_algebra() -> alx.Semiring:
    """Max-min algebra (restricted to positive reals) - similar to tropical max-plus."""
    import algebraic as alx

    # This gives a lattice with: add=max, mul=min, zero=0, one=inf
    return alx.semirings.tropical_semiring(minplus=False)


@pytest.fixture
def maxmin_algebra() -> alx.DeMorganAlgebra:
    """Max-min algebra (full De Morgan algebra with complement)."""
    import algebraic as alx

    return alx.semirings.max_min_algebra()


class SparseHelper:
    def __init__(self, algebra, num_vars: int) -> None:
        self.algebra = algebra
        self.num_vars = num_vars

    def variable(self, index, coefficient=None):
        from algebraic.polynomials.sparse import SparsePolynomial

        if coefficient is None:
            return SparsePolynomial.variable(index, self.num_vars, self.algebra)
        # For custom coefficient, create variable and scale it
        var = SparsePolynomial.variable(index, self.num_vars, self.algebra)
        const = SparsePolynomial.constant(coefficient, self.num_vars, self.algebra)
        return var * const

    def constant(self, value):
        from algebraic.polynomials.sparse import SparsePolynomial

        return SparsePolynomial.constant(value, self.num_vars, self.algebra)

    def zero(self):
        from algebraic.polynomials.sparse import SparsePolynomial

        return SparsePolynomial.zero(self.num_vars, self.algebra)

    def one(self):
        from algebraic.polynomials.sparse import SparsePolynomial

        return SparsePolynomial.one(self.num_vars, self.algebra)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def evaluate(self, poly, point):
        return poly.evaluate(point)

    def compose(self, poly, replacements):
        return poly.compose(replacements)


# Polynomial helper fixtures (no longer "algebra" wrappers)
# These are just convenience helpers for creating polynomials in tests
@pytest.fixture
def sparse_helper() -> SparseHelper:
    """Helper for creating SparsePolynomial instances."""

    def _factory(algebra, num_vars):
        return SparseHelper(algebra, num_vars)

    return _factory


@pytest.fixture
def monomial_helper():
    """Helper for creating MonomialBasis instances."""
    from algebraic.polynomials.monomial_basis import MonomialBasis

    class MonomialHelper:
        def __init__(self, algebra, num_vars):
            self.algebra = algebra
            self.num_vars = num_vars

        def variable(self, index):
            return MonomialBasis.variable(index, self.num_vars, self.algebra)

        def constant(self, value):
            return MonomialBasis.constant(value, self.num_vars, self.algebra)

        def zero(self):
            return MonomialBasis.zero(self.num_vars, self.algebra)

        def one(self):
            return MonomialBasis.one(self.num_vars, self.algebra)

        def add(self, a, b):
            return a + b

        def mul(self, a, b):
            return a * b

        def evaluate(self, poly, point):
            return poly.evaluate(point)

        def compose(self, poly, replacements):
            return poly.compose(replacements)

        def from_sparse(self, sparse_poly):
            # Convert sparse to monomial by enumerating all monomials
            import algebraic.array.core as alge
            from bitarray.util import int2ba

            coeffs = alge.zeros((2,) * self.num_vars, self.algebra)
            for monomial, coeff in sparse_poly.items():
                # Convert bitarray to index tuple
                idx = tuple(int(bit) for bit in monomial)
                coeffs = coeffs.at[idx].set(coeff)
            return MonomialBasis(coeffs)

        def to_sparse(self, poly):
            from algebraic.polynomials.sparse import SparsePolynomial
            from bitarray import frozenbitarray
            from itertools import product

            result = {}
            # Enumerate all 2^n possible indices
            for idx in product([0, 1], repeat=self.num_vars):
                coeff = poly.coeffs[idx]
                # Only include non-zero coefficients - compare .data to avoid JIT issues
                if not jnp.allclose(coeff.data, self.algebra.zero):
                    monomial = frozenbitarray(idx)
                    result[monomial] = coeff.data  # Store the raw data

            return SparsePolynomial(self.algebra, self.num_vars, result)

    def _factory(algebra, num_vars):
        return MonomialHelper(algebra, num_vars)

    return _factory


@pytest.fixture
def rank_helper():
    """Helper for creating RankDecomposition instances."""
    from algebraic.polynomials.rank_decomp import RankDecomposition

    class RankHelper:
        def __init__(self, algebra, num_vars, max_rank=100, max_degree=None, max_replacement_degree=None):
            self.algebra = algebra
            self.num_vars = num_vars
            self.max_rank = max_rank
            self.max_degree = max_degree if max_degree is not None else num_vars
            self.max_replacement_degree = max_replacement_degree if max_replacement_degree is not None else self.max_degree

        def variable(self, index):
            return RankDecomposition.variable(
                index, self.num_vars, self.algebra, self.max_rank, self.max_degree, self.max_replacement_degree
            )

        def constant(self, value):
            return RankDecomposition.constant(
                value, self.num_vars, self.algebra, self.max_rank, self.max_degree, self.max_replacement_degree
            )

        @property
        def zero(self):
            return RankDecomposition.zero(
                self.num_vars, self.algebra, self.max_rank, self.max_degree, self.max_replacement_degree
            )

        def one(self):
            return RankDecomposition.one(
                self.num_vars, self.algebra, self.max_rank, self.max_degree, self.max_replacement_degree
            )

        def add(self, a, b):
            return a + b

        def mul(self, a, b):
            return a * b

        def evaluate(self, poly, point):
            return poly.evaluate(point)

        def compose(self, poly, replacements):
            return poly.compose(replacements)

    def _factory(algebra, num_vars, max_rank=100, max_degree=None, max_replacement_degree=None):
        return RankHelper(algebra, num_vars, max_rank, max_degree, max_replacement_degree)

    return _factory


# Hypothesis strategies
@st.composite
def small_degrees(draw: st.DrawFn):
    """Generate small degrees for quick testing (2-4 variables)."""
    return draw(st.integers(min_value=2, max_value=4))


@st.composite
def medium_degrees(draw: st.DrawFn):
    """Generate medium degrees (2-8 variables)."""
    return draw(st.integers(min_value=2, max_value=8))


@st.composite
def variable_indices(draw: st.DrawFn, degree: int):
    """Generate valid variable indices for given degree."""
    return draw(st.integers(min_value=0, max_value=degree - 1))


@st.composite
def boolean_scalars(draw: st.DrawFn):
    """Generate boolean scalar arrays."""
    return jnp.array(draw(st.booleans()))


@st.composite
def float_scalars(draw: st.DrawFn, min_value: float = -10.0, max_value: float = 10.0):
    """Generate float scalar arrays."""
    value = draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False))
    return jnp.array(value)


@st.composite
def evaluation_points_bool(draw: st.DrawFn, degree: int):
    """Generate boolean evaluation points."""
    return draw(arrays(jnp.bool_, degree, elements=st.booleans()))


@st.composite
def evaluation_points_float(draw: st.DrawFn, degree: int, min_value: float = -10.0, max_value: float = 10.0):
    """Generate float evaluation points."""
    return draw(
        arrays(
            jnp.float32,
            degree,
            elements=st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False),
        )
    )


@st.composite
def sparse_evaluation_points_bool(draw: st.DrawFn, degree: int, max_vars: int | None = None):
    """Generate sparse evaluation points (as dict) for boolean algebra."""
    if max_vars is None:
        max_vars = degree
    num_vars = draw(st.integers(min_value=0, max_value=min(max_vars, degree)))
    indices = draw(st.lists(st.integers(0, degree - 1), min_size=num_vars, max_size=num_vars, unique=True))
    values = [jnp.array(draw(st.booleans())) for _ in range(num_vars)]
    return dict(zip(indices, values))


@st.composite
def sparse_evaluation_points_float(
    draw: st.DrawFn, degree: int, max_vars: int | None = None, min_value: float = -10.0, max_value: float = 10.0
):
    """Generate sparse evaluation points (as dict) for float algebra."""
    if max_vars is None:
        max_vars = degree
    num_vars = draw(st.integers(min_value=0, max_value=min(max_vars, degree)))
    indices = draw(st.lists(st.integers(0, degree - 1), min_size=num_vars, max_size=num_vars, unique=True))
    values = [
        jnp.array(draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False)))
        for _ in range(num_vars)
    ]
    return dict(zip(indices, values))


@st.composite
def simple_sparse_polynomials_bool(draw: st.DrawFn, degree: int):
    """Generate simple sparse polynomials over boolean algebra.

    Simple means: zero, constant, single variable, or sum/product of two variables.
    """
    from algebraic.polynomials.sparse import SparsePolynomial
    from algebraic.semirings import boolean_algebra

    bool_alg = boolean_algebra(mode="logic")

    poly_type = draw(st.sampled_from(["zero", "one", "constant", "variable", "sum_vars", "product_vars"]))

    if poly_type == "zero":
        return SparsePolynomial.constant(bool_alg.zero, degree, bool_alg)
    elif poly_type == "one":
        return SparsePolynomial.constant(bool_alg.one, degree, bool_alg)
    elif poly_type == "constant":
        value = jnp.array(draw(st.booleans()))
        return SparsePolynomial.constant(value, degree, bool_alg)
    elif poly_type == "variable":
        idx = draw(st.integers(0, degree - 1))
        return SparsePolynomial.variable(idx, degree, bool_alg)
    elif poly_type == "sum_vars":
        idx1 = draw(st.integers(0, degree - 1))
        idx2 = draw(st.integers(0, degree - 1))
        return SparsePolynomial.variable(idx1, degree, bool_alg) + SparsePolynomial.variable(idx2, degree, bool_alg)
    else:  # product_vars
        idx1 = draw(st.integers(0, degree - 1))
        idx2 = draw(st.integers(0, degree - 1))
        return SparsePolynomial.variable(idx1, degree, bool_alg) * SparsePolynomial.variable(idx2, degree, bool_alg)


@st.composite
def simple_sparse_polynomials_float(draw: st.DrawFn, degree: int, min_value: float = -5.0, max_value: float = 5.0):
    """Generate simple sparse polynomials over max-min algebra."""
    from algebraic.polynomials.sparse import SparsePolynomial
    from algebraic.semirings import max_min_algebra

    # Use max-min algebra (positive reals) as lattice for testing
    tropical_alg = max_min_algebra(only="positive")

    poly_type = draw(st.sampled_from(["zero", "one", "constant", "variable", "sum_vars", "product_vars"]))

    if poly_type == "zero":
        return SparsePolynomial.constant(tropical_alg.zero, degree, tropical_alg)
    elif poly_type == "one":
        return SparsePolynomial.constant(tropical_alg.one, degree, tropical_alg)
    elif poly_type == "constant":
        value = jnp.array(draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False)))
        return SparsePolynomial.constant(value, degree, tropical_alg)
    elif poly_type == "variable":
        idx = draw(st.integers(0, degree - 1))
        return SparsePolynomial.variable(idx, degree, tropical_alg)
    elif poly_type == "sum_vars":
        idx1 = draw(st.integers(0, degree - 1))
        idx2 = draw(st.integers(0, degree - 1))
        return SparsePolynomial.variable(idx1, degree, tropical_alg) + SparsePolynomial.variable(idx2, degree, tropical_alg)
    else:  # product_vars
        idx1 = draw(st.integers(0, degree - 1))
        idx2 = draw(st.integers(0, degree - 1))
        return SparsePolynomial.variable(idx1, degree, tropical_alg) * SparsePolynomial.variable(idx2, degree, tropical_alg)


# Utility functions
def polynomials_equal_by_evaluation(
    poly1: SparsePolynomial,
    poly2: SparsePolynomial,
    test_points: list[Array | Mapping[int, Array]],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check if two polynomials are equal by evaluating at test points.

    Parameters
    ----------
    poly1 : SparsePolynomial
        First polynomial
    poly2 : SparsePolynomial
        Second polynomial
    test_points : list
        List of evaluation points (arrays or mappings)
    rtol : float
        Relative tolerance for comparison
    atol : float
        Absolute tolerance for comparison

    Returns
    -------
    bool
        True if polynomials evaluate to same values at all test points
    """
    import quax

    allclose = quax.quaxify(jnp.allclose)
    array_equal = quax.quaxify(jnp.array_equal)

    for point in test_points:
        result1 = poly1.evaluate(point)
        result2 = poly2.evaluate(point)

        # Extract the scalar value from the evaluated polynomial (constant term)
        val1 = list(result1.values())[0] if len(result1) > 0 else poly1.algebra.zero
        val2 = list(result2.values())[0] if len(result2) > 0 else poly2.algebra.zero

        # Handle scalar vs array comparisons
        if isinstance(val1, Array) and val1.shape == ():
            val1 = val1.item()
        if isinstance(val2, Array) and val2.shape == ():
            val2 = val2.item()

        # For boolean values, use quaxified array_equal
        if isinstance(val1, (bool, jnp.bool_)) or (isinstance(val1, Array) and val1.dtype == jnp.bool_):
            if not array_equal(val1, val2):
                return False
        else:
            # For numeric values, use quaxified allclose
            if not allclose(val1, val2, rtol=rtol, atol=atol):
                return False

    return True


def generate_random_test_points(degree: int, num_points: int, use_bool: bool = False) -> list[Array]:
    """Generate random test points for evaluation.

    Parameters
    ----------
    degree : int
        Number of variables
    num_points : int
        Number of test points to generate
    use_bool : bool
        If True, generate boolean points; otherwise float points

    Returns
    -------
    list[Array]
        List of test point arrays
    """
    import jax.random as jrandom

    key = jrandom.PRNGKey(42)
    points = []

    for _ in range(num_points):
        key, subkey = jrandom.split(key)
        if use_bool:
            point = jrandom.bernoulli(subkey, shape=(degree,))
        else:
            point = jrandom.uniform(subkey, shape=(degree,), minval=-5.0, maxval=5.0)
        points.append(point)

    return points
