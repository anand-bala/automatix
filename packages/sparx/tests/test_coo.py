"""Tests for COO (Coordinate List) sparse array format."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from sparx._coo.core import COO

# ==============================================================================
# Initialization Tests
# ==============================================================================


class TestCOOInitialization:
    """Test COO array initialization and validation."""

    @pytest.mark.parametrize(
        "data,coords,shape,expected_nnz,description",
        [
            # 1D sparse array with 3 non-zero elements
            (
                [1.0, 2.0, 3.0],
                [[0, 2, 4]],
                (5,),
                3,
                "1D array with scattered values",
            ),
            # 2D sparse array (3x3 matrix with 4 non-zero elements)
            (
                [1.0, 2.0, 3.0, 4.0],
                [[0, 0, 1, 2], [0, 2, 1, 2]],
                (3, 3),
                4,
                "2D array (sparse matrix)",
            ),
            # 3D sparse array
            (
                [5.0, 6.0],
                [[0, 1], [1, 0], [2, 2]],
                (2, 2, 3),
                2,
                "3D sparse tensor",
            ),
            # Empty sparse array
            (
                [],
                [[], []],
                (3, 3),
                0,
                "Empty 2D sparse array",
            ),
            # Single element
            (
                [42.0],
                [[0], [0]],
                (1, 1),
                1,
                "Single element array",
            ),
        ],
    )
    def test_valid_initialization(
        self,
        data: list[float],
        coords: list[list[int]],
        shape: tuple[int, ...],
        expected_nnz: int,
        description: str,
    ) -> None:
        """Test that valid COO arrays are initialized correctly."""
        coo = COO(data=data, coords=coords, shape=shape, allow_materialize=True)

        assert coo.nnz == expected_nnz, f"Failed for: {description}"
        assert coo.shape == shape, f"Failed for: {description}"
        assert len(coo) == shape[0], f"Failed for: {description}"
        assert coo.ndim == len(shape), f"Failed for: {description}"

    @pytest.mark.parametrize(
        "data,coords,shape,error_pattern,description",
        [
            # Mismatched data and coords lengths
            (
                [1.0, 2.0, 3.0],
                [[0, 1]],
                (3,),
                "data length does not match",
                "Too few coordinates for data",
            ),
            # Mismatched shape and coords dimensions
            (
                [1.0, 2.0],
                [[0, 1]],
                (3, 3),
                "Shape specified.*doesn't match.*shape of.*coords",
                "Shape ndim mismatch with coords",
            ),
        ],
    )
    def test_initialization_errors(
        self,
        data: list[float],
        coords: list[list[int]],
        shape: tuple[int, ...],
        error_pattern: str,
        description: str,
    ) -> None:
        """Test that invalid initialization raises appropriate errors."""
        with pytest.raises(ValueError, match=error_pattern):
            COO(data=data, coords=coords, shape=shape)


# ==============================================================================
# Prune Tests
# ==============================================================================


class TestCOOPrune:
    """Test the prune method for removing fill values."""

    @pytest.mark.parametrize(
        "data,coords,shape,expected_nnz_after,description",
        [
            # Array with explicit zeros
            (
                [1.0, 0.0, 2.0, 0.0, 3.0],
                [[0, 1, 2, 3, 4]],
                (5,),
                3,
                "Remove zeros from 1D array",
            ),
            # 2D array with zeros
            (
                [1.0, 0.0, 0.0, 2.0],
                [[0, 0, 1, 1], [0, 1, 0, 1]],
                (2, 2),
                2,
                "Remove zeros from 2D array",
            ),
            # No zeros to remove
            (
                [1.0, 2.0, 3.0],
                [[0, 1, 2]],
                (3,),
                3,
                "No zeros - should keep all elements",
            ),
            # All zeros
            (
                [0.0, 0.0, 0.0],
                [[0, 1, 2]],
                (3,),
                0,
                "All zeros - should become empty",
            ),
        ],
    )
    def test_prune_zeros(
        self,
        data: list[float],
        coords: list[list[int]],
        shape: tuple[int, ...],
        expected_nnz_after: int,
        description: str,
    ) -> None:
        """Test pruning zero values from sparse arrays."""
        coo = COO(data=data, coords=coords, shape=shape, allow_materialize=True)
        pruned = coo.prune()

        assert pruned.nnz == expected_nnz_after, f"Failed for: {description}"
        assert pruned.shape == shape, f"Shape should not change: {description}"

        # Verify no zeros remain in the pruned data
        if pruned.nnz > 0:
            assert jnp.all(pruned.data != 0.0), f"Zeros still present: {description}"

    @pytest.mark.parametrize(
        "data,coords,shape,prune_value,expected_nnz_after,description",
        [
            # Prune specific value
            (
                [1.0, -1.0, 2.0, -1.0],
                [[0, 1, 2, 3]],
                (4,),
                -1.0,
                2,
                "Remove -1.0 values",
            ),
            # Prune another specific value
            (
                [1.0, 5.0, 2.0, 5.0],
                [[0, 1, 2, 3]],
                (4,),
                5.0,
                2,
                "Remove 5.0 values",
            ),
        ],
    )
    def test_prune_custom_value(
        self,
        data: list[float],
        coords: list[list[int]],
        shape: tuple[int, ...],
        prune_value: float,
        expected_nnz_after: int,
        description: str,
    ) -> None:
        """Test pruning custom values from sparse arrays."""
        coo = COO(data=data, coords=coords, shape=shape, allow_materialize=True)
        pruned = coo.prune(value=prune_value)

        assert pruned.nnz == expected_nnz_after, f"Failed for: {description}"

        # Verify the pruned value is not present
        if pruned.nnz > 0:
            assert jnp.all(pruned.data != prune_value), f"Pruned value still present: {description}"


# ==============================================================================
# Materialize Tests
# ==============================================================================


class TestCOOMaterialize:
    """Test converting sparse COO arrays to dense arrays."""

    @pytest.mark.parametrize(
        "data,coords,shape,expected_dense,description",
        [
            # Simple 1D case
            (
                [1.0, 2.0, 3.0],
                [[0, 2, 4]],
                (5,),
                [1.0, 0.0, 2.0, 0.0, 3.0],
                "1D sparse to dense",
            ),
            # 2D matrix
            (
                [1.0, 2.0, 3.0, 4.0],
                [[0, 0, 1, 2], [0, 2, 1, 2]],
                (3, 3),
                [
                    [1.0, 0.0, 2.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 4.0],
                ],
                "2D sparse matrix to dense",
            ),
            # Empty sparse array
            (
                np.array([]),
                np.array([[], []], dtype=np.int32),
                (2, 2),
                [[0.0, 0.0], [0.0, 0.0]],
                "Empty sparse to dense zeros",
            ),
            # All elements filled
            (
                [1.0, 2.0, 3.0, 4.0],
                [[0, 0, 1, 1], [0, 1, 0, 1]],
                (2, 2),
                [[1.0, 2.0], [3.0, 4.0]],
                "Fully populated 2x2 matrix",
            ),
        ],
    )
    def test_materialize_to_dense(
        self,
        data: list[float],
        coords: list[list[int]],
        shape: tuple[int, ...],
        expected_dense: list[Any],
        description: str,
    ) -> None:
        """Test materializing sparse arrays to dense format."""
        coo = COO(data=data, coords=coords, shape=shape, allow_materialize=True)
        dense = coo.materialise()

        expected = jnp.array(expected_dense)
        assert jnp.allclose(dense, expected), f"Failed for: {description}"
        assert dense.shape == shape, f"Shape mismatch: {description}"

    def test_materialize_blocked_when_disabled(self) -> None:
        """Test that materialization is blocked when allow_materialize=False."""
        coo = COO(data=[1.0], coords=[[0]], shape=(2,), allow_materialize=False)

        with pytest.raises(ValueError, match="Refusing to materialize"):
            coo.materialise()


# ==============================================================================
# Transpose Tests
# ==============================================================================


class TestCOOTranspose:
    """Test transpose operations on COO arrays."""

    @pytest.mark.parametrize(
        "data,coords,shape,axes,expected_coords,expected_shape,description",
        [
            # Default transpose (reverse axes) for 2D
            (
                [1.0, 2.0, 3.0],
                [[0, 1, 2], [0, 1, 2]],
                (3, 3),
                None,
                [[0, 1, 2], [0, 1, 2]],
                (3, 3),
                "Transpose diagonal 2D matrix",
            ),
            # Explicit transpose of 2D matrix
            (
                [1.0, 2.0],
                [[0, 1], [0, 1]],
                (2, 3),
                (1, 0),
                [[0, 1], [0, 1]],
                (3, 2),
                "Transpose 2x3 to 3x2",
            ),
            # 3D transpose with permutation
            (
                [1.0, 2.0],
                [[0, 1], [0, 1], [0, 1]],
                (2, 3, 4),
                (2, 0, 1),
                [[0, 1], [0, 1], [0, 1]],
                (4, 2, 3),
                "3D permutation (2,0,1)",
            ),
            # Identity transpose (no-op)
            (
                [1.0, 2.0],
                [[0, 1], [0, 1]],
                (2, 2),
                (0, 1),
                [[0, 1], [0, 1]],
                (2, 2),
                "Identity permutation (no-op)",
            ),
        ],
    )
    def test_transpose_operations(
        self,
        data: list[float],
        coords: list[list[int]],
        shape: tuple[int, ...],
        axes: tuple[int, ...] | None,
        expected_coords: list[list[int]],
        expected_shape: tuple[int, ...],
        description: str,
    ) -> None:
        """Test various transpose operations."""
        import quax

        coo = COO(data=data, coords=coords, shape=shape, allow_materialize=True)

        transpose_fn = quax.quaxify(jnp.transpose)
        transposed = transpose_fn(coo, axes=axes)

        assert transposed.shape == expected_shape, f"Shape mismatch: {description}"
        assert transposed.nnz == coo.nnz, f"nnz should not change: {description}"
        # Data should be the same
        assert jnp.allclose(transposed.data, coo.data), f"Data changed: {description}"

    def test_transpose_materialized(self) -> None:
        """Test transpose by comparing materialized dense arrays."""
        # Create a simple non-symmetric matrix
        data = [1.0, 2.0, 3.0]
        coords = [[0, 0, 1], [0, 1, 0]]
        shape = (2, 3)

        coo = COO(data=data, coords=coords, shape=shape, allow_materialize=True)

        # Transpose the sparse array
        import quax

        transpose_fn = quax.quaxify(jnp.transpose)
        transposed = transpose_fn(coo, axes=(1, 0))

        # Compare with dense transpose
        dense = coo.materialise()
        dense_transposed = dense.T

        assert jnp.allclose(transposed.materialise(), dense_transposed)

    @pytest.mark.parametrize(
        "shape,axes,error_pattern,description",
        [
            # Repeated axis
            (
                (2, 3),
                (0, 0),
                "repeated axis",
                "Repeated axis in transpose",
            ),
            # Wrong number of axes (caught by jnp.transpose validation)
            (
                (2, 3),
                (0, 1, 2),
                "axis .* is out of bounds",
                "Too many axes",
            ),
        ],
    )
    def test_transpose_errors(
        self,
        shape: tuple[int, ...],
        axes: tuple[int, ...],
        error_pattern: str,
        description: str,
    ) -> None:
        """Test that invalid transpose operations raise errors."""
        import quax

        # Create coords with shape (ndim, 1) - one coordinate per dimension
        coords = [[0] for _ in range(len(shape))]
        coo = COO(data=[1.0], coords=coords, shape=shape, allow_materialize=True)

        transpose_fn = quax.quaxify(jnp.transpose)
        with pytest.raises(ValueError, match=error_pattern):
            transpose_fn(coo, axes=axes)


# ==============================================================================
# dot_general Tests
# ==============================================================================


class TestCOODotGeneral:
    """Test dot_general (generalized matrix/tensor multiplication)."""

    @pytest.mark.parametrize(
        "lhs_data,lhs_coords,lhs_shape,rhs_data,rhs_coords,rhs_shape,expected_data,expected_coords,expected_shape,description",
        [
            # Simple vector dot product: [1, 0, 2] . [3, 0, 4] = 11
            (
                [1.0, 2.0],
                [[0, 2]],
                (3,),
                [3.0, 4.0],
                [[0, 2]],
                (3,),
                [11.0],
                [[]],
                (),
                "Vector dot product",
            ),
            # Matrix-vector product
            # [[1, 2],   [3]   [7]
            #  [0, 0]] @ [2] = [0]
            (
                [1.0, 2.0],
                [[0, 0], [0, 1]],
                (2, 2),
                [3.0, 2.0],
                [[0, 1]],
                (2,),
                [7.0],
                [[0]],
                (2,),
                "Matrix-vector product",
            ),
            # Matrix-matrix product (small example)
            # [[1, 0],   [[2, 0],   [[2, 3],
            #  [0, 3]] @  [0, 1]] =  [0, 3]]
            (
                [1.0, 3.0],
                [[0, 1], [0, 1]],
                (2, 2),
                [2.0, 1.0],
                [[0, 1], [0, 1]],
                (2, 2),
                [2.0, 1.0, 3.0],
                [[0, 0, 1], [0, 1, 1]],
                (2, 2),
                "Matrix-matrix product",
            ),
            # Empty arrays
            (
                [],
                [[], []],
                (2, 2),
                [],
                [[], []],
                (2, 2),
                [],
                [[], []],
                (2, 2),
                "Empty arrays multiplication",
            ),
        ],
    )
    def test_dot_general_basic(
        self,
        lhs_data: list[float],
        lhs_coords: list[list[int]],
        lhs_shape: tuple[int, ...],
        rhs_data: list[float],
        rhs_coords: list[list[int]],
        rhs_shape: tuple[int, ...],
        expected_data: list[float],
        expected_coords: list[list[int]],
        expected_shape: tuple[int, ...],
        description: str,
    ) -> None:
        """Test basic dot_general operations."""
        import jax
        import quax

        lhs = COO(data=lhs_data, coords=lhs_coords, shape=lhs_shape, allow_materialize=True)
        rhs = COO(data=rhs_data, coords=rhs_coords, shape=rhs_shape, allow_materialize=True)

        dot_general_fn = quax.quaxify(jax.lax.dot_general)
        result = dot_general_fn(lhs, rhs, dimension_numbers=None)

        assert result.shape == expected_shape, f"Shape mismatch: {description}"

        # For non-empty results, check against expected
        if len(expected_data) > 0:
            # Materialize and compare (easier for testing)
            result_dense = result.materialise() if result.nnz > 0 else jnp.zeros(result.shape)
            lhs_dense = lhs.materialise()
            rhs_dense = rhs.materialise()

            # Compute expected result using dense arrays
            if len(lhs_shape) == 1 and len(rhs_shape) == 1:
                # Vector dot product
                expected_dense = jnp.dot(lhs_dense, rhs_dense)
            elif len(rhs_shape) == 1:
                # Matrix-vector
                expected_dense = jnp.dot(lhs_dense, rhs_dense)
            else:
                # Matrix-matrix or higher
                expected_dense = jnp.dot(lhs_dense, rhs_dense)

            assert jnp.allclose(result_dense, expected_dense), f"Value mismatch: {description}"

    def test_dot_general_with_dimension_numbers(self) -> None:
        """Test dot_general with explicit dimension_numbers."""
        import jax
        import quax

        # Create two 2D arrays for custom contraction
        # lhs: shape (2, 3), rhs: shape (3, 4)
        # Contract on axis 1 of lhs with axis 0 of rhs
        lhs = COO(data=[1.0, 2.0], coords=[[0, 1], [0, 0]], shape=(2, 3), allow_materialize=True)
        rhs = COO(data=[3.0, 4.0], coords=[[0, 1], [0, 0]], shape=(3, 4), allow_materialize=True)

        dimension_numbers = (((1,), (0,)), ((), ()))  # Contract axis 1 of lhs with axis 0 of rhs
        dot_general_fn = quax.quaxify(jax.lax.dot_general)
        result = dot_general_fn(lhs, rhs, dimension_numbers=dimension_numbers)

        assert result.shape == (2, 4), "Shape should be (2, 4) after contraction"

        # Verify using dense arrays
        lhs_dense = lhs.materialise()
        rhs_dense = rhs.materialise()
        expected_dense = jax.lax.dot_general(
            lhs_dense, rhs_dense, dimension_numbers=dimension_numbers
        )

        result_dense = result.materialise() if result.nnz > 0 else jnp.zeros(result.shape)
        assert jnp.allclose(result_dense, expected_dense)

    def test_dot_general_batched(self) -> None:
        """Test batched dot_general operation."""
        import jax
        import quax

        # Create batched arrays: shape (2, 3, 4) with batch dimension 0
        # Contract on the last dimension
        lhs = COO(
            data=[1.0, 2.0],
            coords=[[0, 1], [0, 0], [0, 1]],
            shape=(2, 3, 4),
            allow_materialize=True,
        )
        rhs = COO(
            data=[1.0, 2.0],
            coords=[[0, 1], [0, 0], [0, 1]],
            shape=(2, 4, 3),
            allow_materialize=True,
        )

        # Batch on dimension 0, contract dimension 2 of lhs with dimension 1 of rhs
        dimension_numbers = (((2,), (1,)), ((0,), (0,)))
        dot_general_fn = quax.quaxify(jax.lax.dot_general)
        result = dot_general_fn(lhs, rhs, dimension_numbers=dimension_numbers)

        # Verify shape
        assert result.shape == (2, 3, 3), "Batched result should have shape (2, 3, 3)"

        # Verify using dense arrays
        lhs_dense = lhs.materialise()
        rhs_dense = rhs.materialise()
        expected_dense = jax.lax.dot_general(
            lhs_dense, rhs_dense, dimension_numbers=dimension_numbers
        )

        result_dense = result.materialise() if result.nnz > 0 else jnp.zeros(result.shape)
        assert jnp.allclose(result_dense, expected_dense)

    @pytest.mark.parametrize(
        "lhs_shape,rhs_shape,dimension_numbers,error_pattern,description",
        [
            # Incompatible contraction dimensions
            (
                (2, 3),
                (4, 5),
                (((1,), (0,)), ((), ())),
                "Incompatible shapes.*Contracting dimensions must have matching sizes",
                "Mismatched contraction dimensions",
            ),
            # Incompatible batch dimensions
            (
                (2, 3, 4),
                (5, 3, 4),
                (((2,), (2,)), ((0,), (0,))),
                "Incompatible shapes.*Batch dimensions must have matching sizes",
                "Mismatched batch dimensions",
            ),
        ],
    )
    def test_dot_general_errors(
        self,
        lhs_shape: tuple[int, ...],
        rhs_shape: tuple[int, ...],
        dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
        error_pattern: str,
        description: str,
    ) -> None:
        """Test that incompatible dot_general operations raise errors."""
        import jax
        import quax

        # Create coords with shape (ndim, 1) for each array
        lhs_coords = [[0] for _ in range(len(lhs_shape))]
        rhs_coords = [[0] for _ in range(len(rhs_shape))]

        lhs = COO(data=[1.0], coords=lhs_coords, shape=lhs_shape, allow_materialize=True)
        rhs = COO(data=[1.0], coords=rhs_coords, shape=rhs_shape, allow_materialize=True)

        dot_general_fn = quax.quaxify(jax.lax.dot_general)
        with pytest.raises(ValueError, match=error_pattern):
            dot_general_fn(lhs, rhs, dimension_numbers=dimension_numbers)
