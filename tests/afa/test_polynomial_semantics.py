"""Tests for multilinear polynomial semantics and operations.

Tests cover:
1. MultilinearPolynomial data structure and indexing
2. Monomial basis conversion (alpha <-> int)
3. Batch operations kernel (scatter-add for coefficient accumulation)
4. Basic polynomial operations on multiple semirings
"""

import jax.numpy as jnp
import pytest

from automatix.algebra import AlgebraicStructure, create_boolean_kernel
from automatix.algebra.backends.jax_ import (
    LatticeAlgebra,
)
from automatix.algebra.backends.jax_kernels import (
    batch_accumulate_coefficients,
    batch_accumulate_with_multiplication,
    batch_evaluate_monomials,
)
from automatix.algebra.polynomials import MultilinearPolynomial


class TestAlphaToInt:
    """Test monomial basis indexing: alpha tuple -> integer."""

    def test_zero_tuple(self) -> None:
        """Test conversion of zero tuple (constant monomial)."""
        result = MultilinearPolynomial.alpha_to_int((0, 0, 0))
        assert result == 0

    def test_single_nonzero_q0(self) -> None:
        """Test conversion with only q_0 present."""
        result = MultilinearPolynomial.alpha_to_int((1, 0, 0))
        assert result == 1

    def test_single_nonzero_q1(self) -> None:
        """Test conversion with only q_1 present."""
        result = MultilinearPolynomial.alpha_to_int((0, 1, 0))
        assert result == 2

    def test_single_nonzero_q2(self) -> None:
        """Test conversion with only q_2 present."""
        result = MultilinearPolynomial.alpha_to_int((0, 0, 1))
        assert result == 4

    def test_two_nonzero(self) -> None:
        """Test conversion with two states present."""
        result = MultilinearPolynomial.alpha_to_int((1, 1, 0))
        assert result == 3  # 1*2^0 + 1*2^1 = 1 + 2 = 3

    def test_all_nonzero(self) -> None:
        """Test conversion with all states present."""
        result = MultilinearPolynomial.alpha_to_int((1, 1, 1))
        assert result == 7  # 1*2^0 + 1*2^1 + 1*2^2 = 1 + 2 + 4 = 7

    def test_long_tuple(self) -> None:
        """Test with larger number of states."""
        result = MultilinearPolynomial.alpha_to_int((1, 0, 1, 0, 1))
        assert result == 21  # 1 + 4 + 16 = 21

    def test_empty_tuple(self) -> None:
        """Test with empty tuple (constant)."""
        result = MultilinearPolynomial.alpha_to_int(())
        assert result == 0


class TestIntToAlpha:
    """Test monomial basis indexing: integer -> alpha tuple."""

    def test_zero_index(self) -> None:
        """Test conversion of index 0 (constant monomial)."""
        result = MultilinearPolynomial.int_to_alpha(0, 3)
        assert result == (0, 0, 0)

    def test_single_state_q0(self) -> None:
        """Test conversion to q_0 only."""
        result = MultilinearPolynomial.int_to_alpha(1, 3)
        assert result == (1, 0, 0)

    def test_single_state_q1(self) -> None:
        """Test conversion to q_1 only."""
        result = MultilinearPolynomial.int_to_alpha(2, 3)
        assert result == (0, 1, 0)

    def test_single_state_q2(self) -> None:
        """Test conversion to q_2 only."""
        result = MultilinearPolynomial.int_to_alpha(4, 3)
        assert result == (0, 0, 1)

    def test_two_states(self) -> None:
        """Test conversion to two states."""
        result = MultilinearPolynomial.int_to_alpha(3, 3)
        assert result == (1, 1, 0)

    def test_all_states(self) -> None:
        """Test conversion to all states."""
        result = MultilinearPolynomial.int_to_alpha(7, 3)
        assert result == (1, 1, 1)

    def test_roundtrip_conversion(self) -> None:
        """Test roundtrip: alpha -> int -> alpha."""
        original = (1, 0, 1, 1, 0)
        idx = MultilinearPolynomial.alpha_to_int(original)
        reconstructed = MultilinearPolynomial.int_to_alpha(idx, len(original))
        assert reconstructed == original


class TestMultilinearPolynomialCreation:
    """Test polynomial data structure creation and basic operations."""

    @pytest.fixture
    def algebra(self) -> AlgebraicStructure:
        return create_boolean_kernel()

    def test_zeros_polynomial(self, algebra: AlgebraicStructure) -> None:
        """Test creation of zero polynomial."""
        poly = MultilinearPolynomial.zeros(algebra, num_states=2)
        assert poly.num_states == 2
        assert poly.coefficients.shape == (4,)  # 2^2 = 4
        assert jnp.all(poly.coefficients == 0)

    def test_ones_polynomial(self, algebra: AlgebraicStructure) -> None:
        """Test creation of constant 1 polynomial."""
        poly = MultilinearPolynomial.ones(algebra, num_states=2)
        assert poly.num_states == 2
        assert poly.coefficients[0] == 1.0
        assert jnp.sum(poly.coefficients) == 1.0  # Only constant term

    def test_from_monomial_single_state(self, algebra: AlgebraicStructure) -> None:
        """Test creation from single monomial."""
        poly = MultilinearPolynomial.from_monomial(
            algebra,
            num_states=3,
            alpha=(1, 0, 0),
            coefficient=jnp.asarray(1.0),
        )
        assert poly.get_monomial((1, 0, 0)) == 1.0
        assert poly.get_monomial((0, 0, 0)) == 0.0

    def test_from_monomial_two_states(self, algebra: AlgebraicStructure) -> None:
        """Test creation from two-state monomial."""
        poly = MultilinearPolynomial.from_monomial(
            algebra,
            num_states=3,
            alpha=(1, 1, 0),
            coefficient=jnp.asarray(2.5),
        )
        assert poly.get_monomial((1, 1, 0)) == 2.5

    def test_set_monomial(self, algebra: AlgebraicStructure) -> None:
        """Test setting individual monomials."""
        poly = MultilinearPolynomial.zeros(algebra, num_states=2)
        poly_updated = poly.set_monomial((1, 0), jnp.asarray(5.0))

        # Original should be unchanged (frozen dataclass)
        assert poly.get_monomial((1, 0)) == 0.0

        # New polynomial should have the value
        assert poly_updated.get_monomial((1, 0)) == 5.0

    def test_get_monomial_range(self, algebra: AlgebraicStructure) -> None:
        """Test accessing monomials across full coefficient range."""
        poly = MultilinearPolynomial.from_monomial(algebra, 3, (1, 1, 1), jnp.asarray(7.0))

        # Test all 2^3 = 8 monomials
        for i in range(8):
            alpha = MultilinearPolynomial.int_to_alpha(i, 3)
            if i == 7:  # Only (1, 1, 1) should be nonzero
                assert poly.get_monomial(alpha) == 7.0
            else:
                assert poly.get_monomial(alpha) == 0.0


class TestBatchAccumulateCoefficients:
    """Test batch_accumulate_coefficients kernel with multiple semirings."""

    def test_simple_accumulation_lattice(self) -> None:
        """Test basic accumulation with LatticeAlgebra (max operation)."""
        coeffs = jnp.zeros(4)
        indices = jnp.array([0, 2, 2])  # Accumulate at 0 once, at 2 twice
        values = jnp.array([1.0, 0.5, 0.3])

        result = batch_accumulate_coefficients(LatticeAlgebra, coeffs, indices, values)

        # For LatticeAlgebra max: [1.0, 0.0, max(0.5, 0.3), 0.0]
        expected = jnp.array([1.0, 0.0, 0.5, 0.0])
        assert jnp.allclose(result, expected)

    def test_accumulation_with_existing_coefficients(self) -> None:
        """Test accumulation when current_coeffs already has values."""
        coeffs = jnp.array([0.2, 0.0, 0.1, 0.0])
        indices = jnp.array([0, 2])
        values = jnp.array([0.5, 0.4])

        result = batch_accumulate_coefficients(LatticeAlgebra, coeffs, indices, values)

        # Max operation: max(0.2, 0.5) = 0.5 at index 0, max(0.1, 0.4) = 0.4 at index 2
        expected = jnp.array([0.5, 0.0, 0.4, 0.0])
        assert jnp.allclose(result, expected)

    def test_accumulation_all_indices_same(self) -> None:
        """Test accumulating multiple values at same index."""
        coeffs = jnp.zeros(4)
        indices = jnp.array([1, 1, 1, 1])
        values = jnp.array([1.0, 2.0, 3.0, 4.0])

        result = batch_accumulate_coefficients(LatticeAlgebra, coeffs, indices, values)

        # Max of all values at index 1
        expected = jnp.array([0.0, 4.0, 0.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_accumulation_single_index(self) -> None:
        """Test accumulation with single index-value pair."""
        coeffs = jnp.array([1.0, 2.0, 3.0, 4.0])
        indices = jnp.array([2])
        values = jnp.array([0.5])

        result = batch_accumulate_coefficients(LatticeAlgebra, coeffs, indices, values)

        # Should update only index 2: max(3.0, 0.5) = 3.0
        expected = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_accumulation_with_multiplication(self) -> None:
        """Test batch_accumulate_with_multiplication."""
        coeffs = jnp.zeros(4)
        indices = jnp.array([0, 1])
        values = jnp.array([2.0, 3.0])
        multiplier = jnp.array(0.5)

        result = batch_accumulate_with_multiplication(LatticeAlgebra, coeffs, indices, values, multiplier)

        # After multiply: [0.5*2, 0.5*3] = [.5, .5] (semiring multiply is min for MaxMin)
        # min(0.5, 2.0) = 0.5 at index 0, min(0.5, 3.0) = 0.5 at index 1
        expected = jnp.array([0.5, 0.5, 0.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_evaluate_monomials(self) -> None:
        """Test batch_evaluate_monomials (gather operation)."""
        coeffs = jnp.array([1.0, 0.0, 0.5, 0.8])
        indices = jnp.array([0, 2, 3])

        result = batch_evaluate_monomials(LatticeAlgebra, coeffs, indices)

        expected = jnp.array([1.0, 0.5, 0.8])
        assert jnp.allclose(result, expected)


class TestPolynomialProperties:
    """Test polynomial degree and other properties."""

    @pytest.fixture
    def algebra(self) -> AlgebraicStructure:
        return create_boolean_kernel()

    def test_degree_constant_polynomial(self, algebra: AlgebraicStructure) -> None:
        """Test degree of constant polynomial."""
        poly = MultilinearPolynomial.ones(algebra, num_states=3)
        # Constant has degree 0 (no state variables)
        assert poly.degree() == 0

    def test_degree_single_variable(self, algebra: AlgebraicStructure) -> None:
        """Test degree with single state variable."""
        poly = MultilinearPolynomial.from_monomial(algebra, 3, (1, 0, 0), jnp.asarray(1.0))
        assert poly.degree() == 1

    def test_degree_two_variables(self, algebra: AlgebraicStructure) -> None:
        """Test degree with two state variables."""
        poly = MultilinearPolynomial.from_monomial(algebra, 3, (1, 1, 0), jnp.asarray(1.0))
        assert poly.degree() == 2

    def test_degree_all_variables(self, algebra: AlgebraicStructure) -> None:
        """Test degree with all state variables."""
        poly = MultilinearPolynomial.from_monomial(algebra, 3, (1, 1, 1), jnp.asarray(1.0))
        assert poly.degree() == 3

    def test_degree_zero_polynomial(self, algebra: AlgebraicStructure) -> None:
        """Test degree of zero polynomial."""
        poly = MultilinearPolynomial.zeros(algebra, num_states=3)
        assert poly.degree() == 0  # No nonzero monomials


class TestPolynomialRepr:
    """Test string representation for debugging."""

    @pytest.fixture
    def algebra(self) -> AlgebraicStructure:
        return create_boolean_kernel()

    def test_repr_simple(self, algebra: AlgebraicStructure) -> None:
        """Test string representation."""
        poly = MultilinearPolynomial.ones(algebra, num_states=2)
        repr_str = repr(poly)
        assert "MultilinearPolynomial" in repr_str
        assert "num_states=2" in repr_str
        assert "(4,)" in repr_str  # 2^2 = 4


class TestPolynomialEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def algebra(self) -> AlgebraicStructure:
        return create_boolean_kernel()

    def test_single_state_polynomial(self, algebra: AlgebraicStructure) -> None:
        """Test polynomial with single state."""
        poly = MultilinearPolynomial.ones(algebra, num_states=1)
        assert poly.coefficients.shape == (2,)  # 2^1 = 2

    def test_large_polynomial(self, algebra: AlgebraicStructure) -> None:
        """Test polynomial with many states (within reasonable bounds)."""
        poly = MultilinearPolynomial.zeros(algebra, num_states=10)
        # 2^10 = 1024
        assert poly.coefficients.shape == (1024,)

    def test_max_degree_constraint(self, algebra: AlgebraicStructure) -> None:
        """Test polynomial with max_degree constraint."""
        poly = MultilinearPolynomial.zeros(algebra, num_states=4, max_degree=2)
        assert poly.max_degree == 2
        assert poly.num_states == 4

    def test_frozen_dataclass_immutability(self, algebra: AlgebraicStructure) -> None:
        """Test that frozen dataclass prevents direct modification."""
        poly = MultilinearPolynomial.ones(algebra, num_states=2)

        # Attempting to modify coefficients should fail
        with pytest.raises((AttributeError, ValueError)):
            poly.coefficients = jnp.array([1.0, 2.0, 3.0, 4.0])  # type: ignore[misc]


# Integration tests
class TestPolynomialIntegration:
    """Integration tests combining multiple operations."""

    @pytest.fixture
    def algebra(self) -> type[LatticeAlgebra]:
        return LatticeAlgebra

    def test_create_and_query_polynomial(self, algebra: type[LatticeAlgebra]) -> None:
        """Test creating a polynomial and querying monomials."""
        # Create a 3-state polynomial with degree-2 constraint
        poly = MultilinearPolynomial.zeros(algebra, num_states=3, max_degree=2)

        # Set a few monomials
        poly = poly.set_monomial((1, 0, 0), jnp.asarray(1.0))
        poly = poly.set_monomial((1, 1, 0), jnp.asarray(2.5))
        poly = poly.set_monomial((0, 0, 1), jnp.asarray(3.7))

        # Query them back
        assert poly.get_monomial((1, 0, 0)) == 1.0
        assert poly.get_monomial((1, 1, 0)) == 2.5
        assert poly.get_monomial((0, 0, 1)) == 3.7
        assert poly.get_monomial((0, 0, 0)) == 0.0  # Unset monomial

    def test_batch_operations_with_polynomial(self, algebra: type[LatticeAlgebra]) -> None:
        """Test using batch operations on polynomial coefficients."""
        poly = MultilinearPolynomial.zeros(algebra, num_states=2)

        # Accumulate into polynomial coefficients
        # For 2-state polynomial, indices map to alpha as follows:
        # 0 -> (0,0), 1 -> (1,0), 2 -> (0,1), 3 -> (1,1)
        indices = jnp.array([1, 2, 3])
        values = jnp.array([0.5, 1.0, 0.8])

        new_coeffs = batch_accumulate_coefficients(
            LatticeAlgebra,
            poly.coefficients,
            indices,
            values,
        )

        # Create updated polynomial
        poly_updated = MultilinearPolynomial(
            algebra,
            coefficients=new_coeffs,
            num_states=2,
            max_degree=None,
        )

        assert poly_updated.get_monomial((0, 0)) == 0.0  # Index 0, not updated
        assert poly_updated.get_monomial((1, 0)) == 0.5  # Index 1, updated to 0.5
        assert poly_updated.get_monomial((0, 1)) == 1.0  # Index 2, updated to 1.0
        assert poly_updated.get_monomial((1, 1)) == 0.8  # Index 3, updated to 0.8
