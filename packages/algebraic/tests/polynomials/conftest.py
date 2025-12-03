"""Shared fixtures and utilities for polynomial tests."""
# ruff: noqa: ANN201, ANN202

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

if TYPE_CHECKING:
    from algebraic.polynomials.sparse import SparsePolynomial, SparsePolynomialAlgebra
    from jaxtyping import Array


# Algebra fixtures
@pytest.fixture
def bool_algebra():
    """Boolean algebra for tests."""
    from algebraic.tensor_algebra.jax import boolean_algebra

    return boolean_algebra(mode="logic").algebra


@pytest.fixture
def bool_module():
    """Boolean module for tests."""
    from algebraic.tensor_algebra.jax import boolean_algebra

    return boolean_algebra(mode="logic")


# Removed counting_algebra and counting_module fixtures
# These are NOT BoundedDistributiveLattice algebras
# Use tropical_maxplus_algebra, tropical_minplus_algebra, or maxmin_algebra instead


@pytest.fixture
def tropical_minplus_algebra():
    """Tropical min-plus algebra."""
    from algebraic.tensor_algebra.jax import tropical_semiring

    return tropical_semiring(minplus=True).algebra


@pytest.fixture
def tropical_minplus_module():
    """Tropical min-plus module."""
    from algebraic.tensor_algebra.jax import tropical_semiring

    return tropical_semiring(minplus=True)


@pytest.fixture
def tropical_maxplus_algebra():
    """Tropical max-plus algebra."""
    from algebraic.tensor_algebra.jax import tropical_semiring

    return tropical_semiring(minplus=False).algebra


@pytest.fixture
def tropical_maxplus_module():
    """Tropical max-plus module."""
    from algebraic.tensor_algebra.jax import tropical_semiring

    return tropical_semiring(minplus=False)


@pytest.fixture
def maxmin_algebra():
    """Max-min algebra."""
    from algebraic.tensor_algebra.jax import max_min_algebra

    return max_min_algebra().algebra


@pytest.fixture
def maxmin_module():
    """Max-min module."""
    from algebraic.tensor_algebra.jax import max_min_algebra

    return max_min_algebra()


# Polynomial algebra factories
@pytest.fixture
def sparse_alg_factory():
    """Factory for creating SparsePolynomialAlgebra with given algebra and degree."""
    from algebraic.spec import BoundedDistributiveLattice as Lattice

    def _factory[K](algebra: Lattice[K], degree: int):
        from algebraic.polynomials.sparse import SparsePolynomialAlgebra

        return SparsePolynomialAlgebra(algebra=algebra, degree=degree)

    return _factory


@pytest.fixture
def monomial_alg_factory():
    """Factory for creating MonomialBasisAlgebra with given module and degree."""
    from algebraic.tensor_algebra.jax import JaxBiModule, Lattice

    def _factory[K: Lattice](module: JaxBiModule[K], num_vars: int):
        from algebraic.polynomials.jax import MonomialBasisAlgebra

        return MonomialBasisAlgebra(num_vars=num_vars, module=module)

    return _factory


@pytest.fixture
def rank_alg_factory():
    """Factory for creating RankDecompositionAlgebra with given module and degree."""
    from algebraic.tensor_algebra.jax import JaxBiModule, Lattice

    def _factory[K: Lattice](module: JaxBiModule[K], num_vars: int):
        from algebraic.polynomials.jax import RankDecompositionAlgebra

        return RankDecompositionAlgebra(num_vars=num_vars, module=module)

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
    from algebraic.polynomials.sparse import SparsePolynomialAlgebra
    from algebraic.tensor_algebra.jax import boolean_algebra

    bool_alg = boolean_algebra(mode="logic").algebra
    alg = SparsePolynomialAlgebra(algebra=bool_alg, degree=degree)

    poly_type = draw(st.sampled_from(["zero", "one", "constant", "variable", "sum_vars", "product_vars"]))

    if poly_type == "zero":
        return alg.constant(bool_alg.zero)
    elif poly_type == "one":
        return alg.constant(bool_alg.one)
    elif poly_type == "constant":
        value = jnp.array(draw(st.booleans()))
        return alg.constant(value)
    elif poly_type == "variable":
        idx = draw(st.integers(0, degree - 1))
        return alg.variable(idx)
    elif poly_type == "sum_vars":
        idx1 = draw(st.integers(0, degree - 1))
        idx2 = draw(st.integers(0, degree - 1))
        return alg.add(alg.variable(idx1), alg.variable(idx2))
    else:  # product_vars
        idx1 = draw(st.integers(0, degree - 1))
        idx2 = draw(st.integers(0, degree - 1))
        return alg.mul(alg.variable(idx1), alg.variable(idx2))


@st.composite
def simple_sparse_polynomials_float(draw: st.DrawFn, degree: int, min_value: float = -5.0, max_value: float = 5.0):
    """Generate simple sparse polynomials over tropical max-plus algebra."""
    from algebraic.polynomials.sparse import SparsePolynomialAlgebra
    from algebraic.tensor_algebra.jax import tropical_semiring

    tropical_alg = tropical_semiring(minplus=False).algebra  # max-plus
    alg = SparsePolynomialAlgebra(algebra=tropical_alg, degree=degree)

    poly_type = draw(st.sampled_from(["zero", "one", "constant", "variable", "sum_vars", "product_vars"]))

    if poly_type == "zero":
        return alg.constant(tropical_alg.zero)
    elif poly_type == "one":
        return alg.constant(tropical_alg.one)
    elif poly_type == "constant":
        value = jnp.array(draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False)))
        return alg.constant(value)
    elif poly_type == "variable":
        idx = draw(st.integers(0, degree - 1))
        return alg.variable(idx)
    elif poly_type == "sum_vars":
        idx1 = draw(st.integers(0, degree - 1))
        idx2 = draw(st.integers(0, degree - 1))
        return alg.add(alg.variable(idx1), alg.variable(idx2))
    else:  # product_vars
        idx1 = draw(st.integers(0, degree - 1))
        idx2 = draw(st.integers(0, degree - 1))
        return alg.mul(alg.variable(idx1), alg.variable(idx2))


# Utility functions
def polynomials_equal_by_evaluation(
    alg1: SparsePolynomialAlgebra,
    poly1: SparsePolynomial,
    alg2: SparsePolynomialAlgebra,
    poly2: SparsePolynomial,
    test_points: list[Array | Mapping[int, Array]],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check if two polynomials are equal by evaluating at test points.

    Parameters
    ----------
    alg1 : SparsePolynomialAlgebra
        Algebra for first polynomial
    poly1 : SparsePolynomial
        First polynomial
    alg2 : SparsePolynomialAlgebra
        Algebra for second polynomial
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
    for point in test_points:
        val1 = alg1.evaluate(poly1, point)
        val2 = alg2.evaluate(poly2, point)

        # Handle scalar vs array comparisons
        if isinstance(val1, Array) and val1.shape == ():
            val1 = val1.item()
        if isinstance(val2, Array) and val2.shape == ():
            val2 = val2.item()

        # For boolean values, use exact equality
        if isinstance(val1, (bool, jnp.bool_)) or (isinstance(val1, Array) and val1.dtype == jnp.bool_):
            if not jnp.array_equal(val1, val2):
                return False
        else:
            # For numeric values, use tolerance
            if not jnp.allclose(val1, val2, rtol=rtol, atol=atol):
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
