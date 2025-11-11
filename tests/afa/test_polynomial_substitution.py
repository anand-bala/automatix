"""Comprehensive tests for polynomial multiplication and substitution.

Tests the core operations needed for AFA polynomial evaluation:
- polynomial_multiply: Distribute-all-terms approach with multilinear constraint
- polynomial_substitute: Successor polynomial substitution with like-term collection
- Sparse successor handling with default zero polynomial semantics
- Idempotence verification for distributive lattice semirings
"""

import logging

import jax.numpy as jnp
import pytest

from automatix.algebra import create_boolean_kernel
from automatix.algebra.backends.jax_ import MaxMinSemiring
from automatix.algebra.kernels import AlgebraicStructure
from automatix.algebra.polynomials.ring_polynomials import MultilinearPolynomial
from automatix.algebra.polynomials.substitution import polynomial_multiply, polynomial_substitute

logger = logging.getLogger(__name__)


class TestPolynomialMultiplication:
    """Test polynomial multiplication with distribute-all-terms semantics."""

    def test_multiply_constant_polynomials(self) -> None:
        """Test: 1 * 1 = 1 (constant polynomials)."""
        algebra = create_boolean_kernel()
        poly1 = MultilinearPolynomial.ones(algebra, 2)
        poly2 = MultilinearPolynomial.ones(algebra, 2)
        result = polynomial_multiply(poly1, poly2)

        # Result should be 1 (only constant term nonzero)
        assert jnp.allclose(result.coefficients[0], 1.0)
        for i in range(1, 4):
            assert jnp.allclose(result.coefficients[i], 0.0)

    def test_multiply_by_zero(self) -> None:
        """Test: P * 0 = 0 (any polynomial by zero)."""
        algebra = create_boolean_kernel()
        poly1 = MultilinearPolynomial.ones(algebra, 2)
        poly2 = MultilinearPolynomial.zeros(algebra, 2)
        result = polynomial_multiply(poly1, poly2)

        # Result should be zero polynomial
        for coeff in result.coefficients:
            assert jnp.allclose(coeff, 0.0)

    def test_multiply_by_identity(self) -> None:
        """Test: P * 1 = P (polynomial by constant 1)."""
        algebra = create_boolean_kernel()
        poly1 = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)  # x_0
        poly_ones = MultilinearPolynomial.ones(algebra, 2)  # 1
        result = polynomial_multiply(poly1, poly_ones)

        # Result should equal poly1
        assert jnp.allclose(result.coefficients, poly1.coefficients)

    def test_multiply_single_variables(self) -> None:
        """Test: x_0 * x_1 = x_0 * x_1 (cross variable multiplication)."""
        algebra = create_boolean_kernel()
        poly1 = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)  # x_0
        poly2 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)  # x_1
        result = polynomial_multiply(poly1, poly2)

        # Result should be x_0 * x_1 (index 3)
        assert jnp.allclose(result.coefficients[3], 1.0)
        for i in [0, 1, 2]:
            assert jnp.allclose(result.coefficients[i], 0.0)

    def test_multiply_cross_polynomial_boolean(self) -> None:
        """Test: (1 + x_0) * (1 + x_1) = 1 + x_0 + x_1 + x_0*x_1 (Boolean)."""
        algebra = create_boolean_kernel()
        # poly1 = 1 + x_0
        poly1 = MultilinearPolynomial.ones(algebra, 2)
        poly1 = poly1.set_monomial((1, 0), 1.0)  # Add x_0 term
        # poly2 = 1 + x_1
        poly2 = MultilinearPolynomial.ones(algebra, 2)
        poly2 = poly2.set_monomial((0, 1), 1.0)  # Add x_1 term

        result = polynomial_multiply(poly1, poly2)

        # Result should be 1 + x_0 + x_1 + x_0*x_1
        # Indices: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
        assert jnp.allclose(result.coefficients[0], 1.0)  # constant
        assert jnp.allclose(result.coefficients[1], 1.0)  # x_0
        assert jnp.allclose(result.coefficients[2], 1.0)  # x_1
        assert jnp.allclose(result.coefficients[3], 1.0)  # x_0*x_1

    def test_multiply_idempotence_same_variable(self) -> None:
        """Test: (1 + x_0) * (1 + x_0) = 1 + x_0 (multilinear idempotence)."""
        algebra = create_boolean_kernel()
        # poly = 1 + x_0
        poly = MultilinearPolynomial.ones(algebra, 2)
        poly = poly.set_monomial((1, 0), 1.0)

        result = polynomial_multiply(poly, poly)

        # In idempotent Boolean: (1 + x_0)^2 = 1 + x_0 + x_0 + x_0 = 1 + x_0
        # (since a + a = a in Boolean)
        assert jnp.allclose(result.coefficients[0], 1.0)  # constant
        assert jnp.allclose(result.coefficients[1], 1.0)  # x_0
        for i in [2, 3]:
            assert jnp.allclose(result.coefficients[i], 0.0)

    def test_multiply_maxmin_semiring(self) -> None:
        """Test multiplication with MaxMin semiring (max/min operations)."""
        algebra = MaxMinSemiring.to_kernel()
        # poly1 = 5 + 3*x_0 (using -inf as zero)
        coeffs1 = jnp.array([5.0, 3.0, -jnp.inf, -jnp.inf])
        poly1 = MultilinearPolynomial(algebra=algebra, coefficients=coeffs1, num_states=2)
        # poly2 = 4 + 2*x_1
        coeffs2 = jnp.array([4.0, -jnp.inf, 2.0, -jnp.inf])
        poly2 = MultilinearPolynomial(algebra=algebra, coefficients=coeffs2, num_states=2)

        result = polynomial_multiply(poly1, poly2)

        # MaxMin: multiply uses min, add uses max
        # (5 + 3*x_0) * (4 + 2*x_1)
        # Constant: min(5, 4) = 4
        # x_0: min(3, 4) = 3
        # x_1: min(5, 2) = 2
        # x_0*x_1: min(3, 2) = 2
        assert jnp.allclose(result.coefficients[0], 4.0)  # min(5,4)
        assert jnp.allclose(result.coefficients[1], 3.0)  # min(3,4)
        assert jnp.allclose(result.coefficients[2], 2.0)  # min(5,2)
        assert jnp.allclose(result.coefficients[3], 2.0)  # min(3,2)

    def test_multiply_different_num_states_error(self) -> None:
        """Test: Error when multiplying polynomials with different num_states."""
        algebra = create_boolean_kernel()
        poly1 = MultilinearPolynomial.ones(algebra, 2)
        poly2 = MultilinearPolynomial.ones(algebra, 3)

        with pytest.raises(ValueError, match="same number of states"):
            polynomial_multiply(poly1, poly2)

    def test_multiply_distributivity(self) -> None:
        """Test: P * (Q + R) = P*Q + P*R (distributivity)."""
        algebra = create_boolean_kernel()
        P = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)  # x_0
        Q = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)  # x_1
        R = MultilinearPolynomial.from_monomial(algebra, 2, (1, 1), 1.0)  # x_0*x_1

        # Left side: P * (Q + R)
        Q_plus_R = P.algebra.add(Q.coefficients, R.coefficients)
        Q_plus_R_poly = MultilinearPolynomial(
            algebra=algebra, coefficients=Q_plus_R, num_states=2
        )
        left_side = polynomial_multiply(P, Q_plus_R_poly)

        # Right side: P*Q + P*R
        PQ = polynomial_multiply(P, Q)
        PR = polynomial_multiply(P, R)
        right_side_coeffs = algebra.add(PQ.coefficients, PR.coefficients)
        right_side = MultilinearPolynomial(
            algebra=algebra, coefficients=right_side_coeffs, num_states=2
        )

        # Should be approximately equal
        assert jnp.allclose(left_side.coefficients, right_side.coefficients, atol=1e-5)


class TestPolynomialSubstitution:
    """Test polynomial substitution with successor polynomials."""

    def test_substitute_identity(self) -> None:
        """Test: P[x_0 <- x_0, x_1 <- x_1] = P (identity substitution)."""
        algebra = create_boolean_kernel()
        # Create a non-trivial polynomial: 1 + x_0 + x_1
        poly = MultilinearPolynomial.ones(algebra, 2)
        poly = poly.set_monomial((1, 0), 1.0)
        poly = poly.set_monomial((0, 1), 1.0)

        # Create identity successors
        x0 = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)
        x1 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)
        successors = {0: x0, 1: x1}

        result = polynomial_substitute(poly, successors)

        # Result should equal original polynomial
        assert jnp.allclose(result.coefficients, poly.coefficients, atol=1e-5)

    def test_substitute_single_variable(self) -> None:
        """Test: x_0[x_0 <- 1] = 1 (substitute variable with constant)."""
        algebra = create_boolean_kernel()
        poly = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)  # x_0
        const_one = MultilinearPolynomial.ones(algebra, 2)  # 1
        identity_x1 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)

        successors = {0: const_one, 1: identity_x1}
        result = polynomial_substitute(poly, successors)

        # Result should be 1
        expected = const_one.coefficients
        assert jnp.allclose(result.coefficients, expected, atol=1e-5)

    def test_substitute_with_zero_successor(self) -> None:
        """Test: x_0[x_0 <- 0] = 0 (substitute variable with zero)."""
        algebra = create_boolean_kernel()
        poly = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)  # x_0
        const_zero = MultilinearPolynomial.zeros(algebra, 2)
        identity_x1 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)

        successors = {0: const_zero, 1: identity_x1}
        result = polynomial_substitute(poly, successors)

        # Result should be 0
        for coeff in result.coefficients:
            assert jnp.allclose(coeff, 0.0)

    def test_substitute_polynomial_by_polynomial(self) -> None:
        """Test: (x_0 + x_1)[x_0 <- 1, x_1 <- 0] = 1 (polynomial substitution)."""
        algebra = create_boolean_kernel()
        # poly = x_0 + x_1
        poly = MultilinearPolynomial.zeros(algebra, 2)
        poly = poly.set_monomial((1, 0), 1.0)  # x_0
        poly = poly.set_monomial((0, 1), 1.0)  # x_1

        const_one = MultilinearPolynomial.ones(algebra, 2)
        const_zero = MultilinearPolynomial.zeros(algebra, 2)
        successors = {0: const_one, 1: const_zero}

        result = polynomial_substitute(poly, successors)

        # Result should be 1 + 0 = 1
        assert jnp.allclose(result.coefficients[0], 1.0)  # constant
        for i in range(1, 4):
            assert jnp.allclose(result.coefficients[i], 0.0)

    def test_substitute_missing_successor_logs_warning(self, caplog) -> None:
        """Test: Missing successor logs warning and defaults to zero."""
        algebra = create_boolean_kernel()
        poly = MultilinearPolynomial.ones(algebra, 2)

        # Only provide successor for state 0
        x0 = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)
        successors = {0: x0}  # Missing state 1

        with caplog.at_level(logging.WARNING):
            result = polynomial_substitute(poly, successors)

        # Should have logged a warning about missing state 1
        assert "No successor for state q_1" in caplog.text

    def test_substitute_three_variables(self) -> None:
        """Test substitution with three state variables."""
        algebra = create_boolean_kernel()
        # poly = x_0 + x_1 + x_2
        poly = MultilinearPolynomial.zeros(algebra, 3)
        poly = poly.set_monomial((1, 0, 0), 1.0)
        poly = poly.set_monomial((0, 1, 0), 1.0)
        poly = poly.set_monomial((0, 0, 1), 1.0)

        # Identity substitution
        x0 = MultilinearPolynomial.from_monomial(algebra, 3, (1, 0, 0), 1.0)
        x1 = MultilinearPolynomial.from_monomial(algebra, 3, (0, 1, 0), 1.0)
        x2 = MultilinearPolynomial.from_monomial(algebra, 3, (0, 0, 1), 1.0)
        successors = {0: x0, 1: x1, 2: x2}

        result = polynomial_substitute(poly, successors)

        # Should equal original
        assert jnp.allclose(result.coefficients, poly.coefficients, atol=1e-5)

    def test_substitute_maxmin_semiring(self) -> None:
        """Test substitution with MaxMin semiring (multiplicative is min)."""
        algebra = MaxMinSemiring.to_kernel()
        # poly = x_0 (index 1, meaning coefficient 1.0 for this monomial)
        poly = MultilinearPolynomial.zeros(algebra, 2)
        poly = poly.set_monomial((1, 0), 1.0)  # This means: coefficient is 1.0

        # Substitute x_0 with constant 2
        const_two = MultilinearPolynomial.zeros(algebra, 2)
        const_two = const_two.set_monomial((0, 0), 2.0)
        identity_x1 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)

        successors = {0: const_two, 1: identity_x1}
        result = polynomial_substitute(poly, successors)

        # Substituting x_0 into const_two (which is 2):
        # The monomial (1, 0) has coefficient 1.0
        # For this monomial, we compute: 1.0 * const_two = 1.0 * 2 = min(1.0, 2.0) = 1.0
        # Result should have constant term = 1.0 (from the min operation)
        assert jnp.allclose(result.coefficients[0], 1.0)


class TestIdempotenceProperties:
    """Test idempotence properties for distributive lattice semirings."""

    def test_additive_idempotence_boolean(self) -> None:
        """Test: P + P = P in Boolean semiring."""
        algebra = create_boolean_kernel()
        poly = MultilinearPolynomial.from_monomial(algebra, 2, (1, 1), 1.0)

        # Add polynomial to itself
        result_coeffs = algebra.add(poly.coefficients, poly.coefficients)
        result = MultilinearPolynomial(algebra=algebra, coefficients=result_coeffs, num_states=2)

        # In Boolean (OR), a + a = a
        assert jnp.allclose(result.coefficients, poly.coefficients)

    def test_multiplicative_idempotence_boolean(self) -> None:
        """Test: P * P simplifies correctly in Boolean semiring."""
        algebra = create_boolean_kernel()
        # poly = x_0
        poly = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)

        result = polynomial_multiply(poly, poly)

        # x_0 * x_0 = x_0 (multilinear constraint)
        assert jnp.allclose(result.coefficients, poly.coefficients)

    def test_idempotence_substitution_cycle(self) -> None:
        """Test: Idempotence preserved through substitution cycle."""
        algebra = create_boolean_kernel()
        # Start with polynomial P = x_0 + x_1
        poly = MultilinearPolynomial.zeros(algebra, 2)
        poly = poly.set_monomial((1, 0), 1.0)
        poly = poly.set_monomial((0, 1), 1.0)

        # Identity substitution twice
        x0 = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)
        x1 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)
        successors = {0: x0, 1: x1}

        result1 = polynomial_substitute(poly, successors)
        result2 = polynomial_substitute(result1, successors)

        # Should still equal original (idempotence)
        assert jnp.allclose(result2.coefficients, poly.coefficients, atol=1e-5)

    def test_idempotence_maxmin_semiring(self) -> None:
        """Test: MaxMin semiring idempotence (max is idempotent)."""
        algebra = MaxMinSemiring.to_kernel()
        coeffs = jnp.array([1.0, 2.0, 3.0, 4.0])
        poly = MultilinearPolynomial(algebra=algebra, coefficients=coeffs, num_states=2)

        # Add polynomial to itself
        result_coeffs = algebra.add(poly.coefficients, poly.coefficients)

        # In MaxMin (max), a + a = a (idempotent)
        assert jnp.allclose(result_coeffs, poly.coefficients)


class TestIntegration:
    """End-to-end integration tests for polynomial operations."""

    def test_polynomial_evaluation_cycle(self) -> None:
        """Test: Full cycle from polynomial definition to evaluation."""
        algebra = create_boolean_kernel()

        # Create polynomial: 1 + x_0 + x_1 + x_0*x_1
        poly = MultilinearPolynomial.ones(algebra, 2)
        poly = poly.set_monomial((1, 0), 1.0)
        poly = poly.set_monomial((0, 1), 1.0)
        poly = poly.set_monomial((1, 1), 1.0)

        # Substitute x_0 = x_1, x_1 = 1
        x1 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)
        const_one = MultilinearPolynomial.ones(algebra, 2)
        successors = {0: x1, 1: const_one}

        result = polynomial_substitute(poly, successors)

        # Result should be 1 + x_1 + 1 + x_1 = 1 + x_1 (due to idempotence)
        assert jnp.allclose(result.coefficients[0], 1.0)  # constant
        assert jnp.allclose(result.coefficients[2], 1.0)  # x_1 coefficient
        assert jnp.allclose(result.coefficients[1], 0.0)  # x_0 coefficient
        assert jnp.allclose(result.coefficients[3], 0.0)  # x_0*x_1 coefficient

    def test_complex_substitution_sequence(self) -> None:
        """Test: Complex sequence of multiplications and substitutions."""
        algebra = create_boolean_kernel()

        # Create two simple polynomials and multiply them
        P1 = MultilinearPolynomial.from_monomial(algebra, 2, (1, 0), 1.0)  # x_0
        P2 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)  # x_1
        P = polynomial_multiply(P1, P2)  # P = x_0 * x_1

        # Substitute: x_0 <- 1, x_1 <- 0
        const_one = MultilinearPolynomial.ones(algebra, 2)
        const_zero = MultilinearPolynomial.zeros(algebra, 2)
        successors = {0: const_one, 1: const_zero}

        result = polynomial_substitute(P, successors)

        # P[1/0] = (x_0*x_1)[1/0] = 1*0 = 0
        for coeff in result.coefficients:
            assert jnp.allclose(coeff, 0.0)

    def test_nested_polynomial_operations(self) -> None:
        """Test: Nested polynomial multiply and substitute operations."""
        algebra = create_boolean_kernel()

        # Create base polynomial: x_0 + x_1
        base = MultilinearPolynomial.zeros(algebra, 2)
        base = base.set_monomial((1, 0), 1.0)
        base = base.set_monomial((0, 1), 1.0)

        # Multiply with itself
        squared = polynomial_multiply(base, base)

        # Substitute x_0 <- 1
        const_one = MultilinearPolynomial.ones(algebra, 2)
        x1 = MultilinearPolynomial.from_monomial(algebra, 2, (0, 1), 1.0)
        successors = {0: const_one, 1: x1}

        result = polynomial_substitute(squared, successors)

        # (x_0 + x_1)^2 = x_0 + x_1 (due to idempotence)
        # Substituting x_0 <- 1: 1 + x_1
        assert jnp.allclose(result.coefficients[0], 1.0)  # constant
        assert jnp.allclose(result.coefficients[2], 1.0)  # x_1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_state_polynomial_multiply(self) -> None:
        """Test multiplication with single state polynomial."""
        algebra = create_boolean_kernel()
        poly1 = MultilinearPolynomial.from_monomial(algebra, 1, (1,), 1.0)  # x_0
        poly2 = MultilinearPolynomial.from_monomial(algebra, 1, (1,), 1.0)  # x_0

        result = polynomial_multiply(poly1, poly2)

        # x_0 * x_0 = x_0
        assert jnp.allclose(result.coefficients[1], 1.0)
        assert jnp.allclose(result.coefficients[0], 0.0)

    def test_large_state_space_polynomial(self) -> None:
        """Test polynomial with many states (2^8 = 256 coefficients)."""
        algebra = create_boolean_kernel()
        num_states = 8
        poly = MultilinearPolynomial.ones(algebra, num_states)

        # Should have 2^8 = 256 coefficients
        assert poly.coefficients.shape[0] == 256

        # Constant term should be 1, rest should be 0
        assert jnp.allclose(poly.coefficients[0], 1.0)
        for i in range(1, 256):
            assert jnp.allclose(poly.coefficients[i], 0.0)

    def test_all_zero_polynomial_multiply(self) -> None:
        """Test multiplication with all-zero polynomial."""
        algebra = create_boolean_kernel()
        poly1 = MultilinearPolynomial.ones(algebra, 2)
        poly2 = MultilinearPolynomial.zeros(algebra, 2)

        result = polynomial_multiply(poly1, poly2)

        # Any polynomial * zero = zero
        for coeff in result.coefficients:
            assert jnp.allclose(coeff, 0.0)

    def test_all_nonzero_coefficients_polynomial(self) -> None:
        """Test polynomial where all 2^q coefficients are nonzero."""
        algebra = create_boolean_kernel()
        # Create polynomial with all coefficients = 1
        coeffs = jnp.ones(4)
        poly = MultilinearPolynomial(algebra=algebra, coefficients=coeffs, num_states=2)

        # Multiply by itself
        result = polynomial_multiply(poly, poly)

        # All coefficients should be nonzero
        for coeff in result.coefficients:
            assert coeff > 0


__all__ = [
    "TestPolynomialMultiplication",
    "TestPolynomialSubstitution",
    "TestIdempotenceProperties",
    "TestIntegration",
    "TestEdgeCases",
]
