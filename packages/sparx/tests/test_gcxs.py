"""Tests for GCXS (Generalized Compressed Sparse) array format."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest
from sparx._compressed.core import GCXS
from sparx._coo.core import COO

# ==============================================================================
# Initialization Tests
# ==============================================================================


class TestGCXSInitialization:
    """Test GCXS array initialization and validation."""

    @pytest.mark.parametrize(
        "data,indices,indptr,compressed_axes,shape,expected_nnz,description",
        [
            # 2D CSR format (compressed_axes=(0,))
            (
                [1.0, 2.0, 3.0, 4.0],
                [0, 2, 1, 2],
                [0, 2, 3, 4],
                (0,),
                (3, 3),
                4,
                "2D CSR matrix",
            ),
            # 2D CSC format (compressed_axes=(1,))
            (
                [1.0, 2.0, 3.0],
                [0, 1, 0],
                [0, 1, 2, 3],
                (1,),
                (2, 3),
                3,
                "2D CSC matrix",
            ),
            # Empty sparse array (CSR)
            (
                [],
                [],
                [0, 0, 0],
                (0,),
                (2, 3),
                0,
                "Empty 2D CSR array",
            ),
            # Single element (CSR)
            (
                [42.0],
                [0],
                [0, 1],
                (0,),
                (1, 1),
                1,
                "Single element CSR array",
            ),
            # Larger CSR matrix
            (
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0, 1, 2, 0, 3],
                [0, 2, 3, 5],
                (0,),
                (3, 4),
                5,
                "3x4 CSR matrix with 5 elements",
            ),
        ],
    )
    def test_valid_initialization(
        self,
        data: list[float],
        indices: list[int],
        indptr: list[int],
        compressed_axes: tuple[int, ...],
        shape: tuple[int, ...],
        expected_nnz: int,
        description: str,
    ) -> None:
        """Test that valid GCXS arrays are initialized correctly."""
        gcxs = GCXS(
            data=data,
            indices=indices,
            indptr=indptr,
            compressed_axes=compressed_axes,
            shape=shape,
            allow_materialize=True,
        )

        assert gcxs.nnz == expected_nnz, f"Failed for: {description}"
        assert gcxs.shape == shape, f"Failed for: {description}"
        assert len(gcxs) == shape[0], f"Failed for: {description}"
        assert gcxs.ndim == len(shape), f"Failed for: {description}"
        assert gcxs.compressed_axes == compressed_axes, f"Failed for: {description}"

    @pytest.mark.parametrize(
        "data,indices,indptr,compressed_axes,shape,error_pattern,description",
        [
            # Mismatched data and indices lengths
            (
                [1.0, 2.0, 3.0],
                [0, 1],
                [0, 2, 3],
                (0,),
                (2, 3),
                "data length.*must equal indices length",
                "Too few indices for data",
            ),
            # Invalid indptr start
            (
                [1.0, 2.0],
                [0, 1],
                [1, 2, 3],
                (0,),
                (2, 2),
                "indptr must start at 0",
                "indptr doesn't start at 0",
            ),
            # Invalid indptr end
            (
                [1.0, 2.0],
                [0, 1],
                [0, 1, 3],
                (0,),
                (2, 2),
                "indptr\\[-1\\].*must equal len\\(data\\)",
                "indptr[-1] != len(data)",
            ),
            # Invalid compressed_axes
            (
                [1.0],
                [0],
                [0, 1],
                (2,),
                (2, 2),
                "compressed_axes.*invalid for shape",
                "compressed_axes out of bounds",
            ),
        ],
    )
    def test_initialization_errors(
        self,
        data: list[float],
        indices: list[int],
        indptr: list[int],
        compressed_axes: tuple[int, ...],
        shape: tuple[int, ...],
        error_pattern: str,
        description: str,
    ) -> None:
        """Test that invalid initialization raises appropriate errors."""
        with pytest.raises(ValueError, match=error_pattern):
            GCXS(
                data=data,
                indices=indices,
                indptr=indptr,
                compressed_axes=compressed_axes,
                shape=shape,
            )


# ==============================================================================
# COO Conversion Tests
# ==============================================================================


class TestGCXSCOOConversion:
    """Test conversion between GCXS and COO formats."""

    @pytest.mark.parametrize(
        "data,indices,indptr,compressed_axes,shape,description",
        [
            # CSR to COO and back
            (
                [1.0, 2.0, 3.0],
                [0, 2, 1],
                [0, 2, 3],
                (0,),
                (2, 3),
                "CSR to COO roundtrip",
            ),
            # CSC to COO and back
            (
                [1.0, 2.0, 3.0],
                [0, 1, 0],
                [0, 1, 2, 3],
                (1,),
                (2, 3),
                "CSC to COO roundtrip",
            ),
            # Empty array
            (
                [],
                [],
                [0, 0],
                (0,),
                (1, 3),
                "Empty array roundtrip",
            ),
        ],
    )
    def test_to_coo_from_coo_roundtrip(
        self,
        data: list[float],
        indices: list[int],
        indptr: list[int],
        compressed_axes: tuple[int, ...],
        shape: tuple[int, ...],
        description: str,
    ) -> None:
        """Test that to_coo and from_coo are inverses."""
        gcxs = GCXS(
            data=data,
            indices=indices,
            indptr=indptr,
            compressed_axes=compressed_axes,
            shape=shape,
            allow_materialize=True,
        )

        # Convert to COO
        coo = gcxs.to_coo()

        # Convert back to GCXS
        gcxs_reconstructed = GCXS.from_coo(coo, compressed_axes=compressed_axes)

        # Check that data is preserved
        assert gcxs_reconstructed.nnz == gcxs.nnz, f"Failed for: {description}"
        assert gcxs_reconstructed.shape == gcxs.shape, f"Failed for: {description}"
        assert jnp.allclose(gcxs_reconstructed.data, gcxs.data), f"Failed for: {description}"

    def test_coo_to_csr_conversion(self) -> None:
        """Test converting COO to CSR format."""
        # Create a COO array
        coords = [[0, 0, 1, 2], [0, 2, 1, 2]]
        data = [1.0, 2.0, 3.0, 4.0]
        coo = COO(data=data, coords=coords, shape=(3, 3), allow_materialize=True)

        # Convert to CSR
        csr = GCXS.from_coo(coo, compressed_axes=(0,))

        assert csr.shape == (3, 3)
        assert csr.nnz == 4
        assert csr.compressed_axes == (0,)

        # Verify materialized form matches
        assert jnp.allclose(csr.materialise(), coo.materialise())

    def test_coo_to_csc_conversion(self) -> None:
        """Test converting COO to CSC format."""
        # Create a COO array
        coords = [[0, 0, 1, 2], [0, 2, 1, 2]]
        data = [1.0, 2.0, 3.0, 4.0]
        coo = COO(data=data, coords=coords, shape=(3, 3), allow_materialize=True)

        # Convert to CSC
        csc = GCXS.from_coo(coo, compressed_axes=(1,))

        assert csc.shape == (3, 3)
        assert csc.nnz == 4
        assert csc.compressed_axes == (1,)

        # Verify materialized form matches
        assert jnp.allclose(csc.materialise(), coo.materialise())


# ==============================================================================
# Prune Tests
# ==============================================================================


class TestGCXSPrune:
    """Test the prune method for removing fill values."""

    @pytest.mark.parametrize(
        "data,indices,indptr,compressed_axes,shape,expected_nnz_after,description",
        [
            # CSR with explicit zeros
            (
                [1.0, 0.0, 2.0, 0.0, 3.0],
                [0, 1, 2, 3, 4],
                [0, 5],
                (0,),
                (1, 5),
                3,
                "Remove zeros from CSR",
            ),
            # CSC with zeros
            (
                [1.0, 0.0, 0.0, 2.0],
                [0, 1, 0, 1],
                [0, 2, 3, 4],
                (1,),
                (2, 3),
                2,
                "Remove zeros from CSC",
            ),
            # No zeros to remove
            (
                [1.0, 2.0, 3.0],
                [0, 1, 2],
                [0, 3],
                (0,),
                (1, 3),
                3,
                "No zeros - should keep all elements",
            ),
            # All zeros
            (
                [0.0, 0.0, 0.0],
                [0, 1, 2],
                [0, 3],
                (0,),
                (1, 3),
                0,
                "All zeros - should become empty",
            ),
        ],
    )
    def test_prune_zeros(
        self,
        data: list[float],
        indices: list[int],
        indptr: list[int],
        compressed_axes: tuple[int, ...],
        shape: tuple[int, ...],
        expected_nnz_after: int,
        description: str,
    ) -> None:
        """Test pruning zero values from sparse arrays."""
        gcxs = GCXS(
            data=data,
            indices=indices,
            indptr=indptr,
            compressed_axes=compressed_axes,
            shape=shape,
            allow_materialize=True,
        )
        pruned = gcxs.prune()

        assert pruned.nnz == expected_nnz_after, f"Failed for: {description}"
        assert pruned.shape == shape, f"Shape should not change: {description}"

        # Verify no zeros remain in the pruned data
        if pruned.nnz > 0:
            assert jnp.all(pruned.data != 0.0), f"Zeros still present: {description}"

    @pytest.mark.parametrize(
        "data,indices,indptr,compressed_axes,shape,prune_value,expected_nnz_after,description",
        [
            # Prune specific value from CSR
            (
                [1.0, -1.0, 2.0, -1.0],
                [0, 1, 2, 3],
                [0, 4],
                (0,),
                (1, 4),
                -1.0,
                2,
                "Remove -1.0 values from CSR",
            ),
            # Prune specific value from CSC
            (
                [1.0, 5.0, 2.0, 5.0],
                [0, 1, 0, 1],
                [0, 2, 4],
                (1,),
                (2, 2),
                5.0,
                2,
                "Remove 5.0 values from CSC",
            ),
        ],
    )
    def test_prune_custom_value(
        self,
        data: list[float],
        indices: list[int],
        indptr: list[int],
        compressed_axes: tuple[int, ...],
        shape: tuple[int, ...],
        prune_value: float,
        expected_nnz_after: int,
        description: str,
    ) -> None:
        """Test pruning custom values from sparse arrays."""
        gcxs = GCXS(
            data=data,
            indices=indices,
            indptr=indptr,
            compressed_axes=compressed_axes,
            shape=shape,
            allow_materialize=True,
        )
        pruned = gcxs.prune(value=prune_value)

        assert pruned.nnz == expected_nnz_after, f"Failed for: {description}"

        # Verify the pruned value is not present
        if pruned.nnz > 0:
            assert jnp.all(pruned.data != prune_value), f"Pruned value still present: {description}"


# ==============================================================================
# Materialize Tests
# ==============================================================================


class TestGCXSMaterialize:
    """Test converting sparse GCXS arrays to dense arrays."""

    @pytest.mark.parametrize(
        "data,indices,indptr,compressed_axes,shape,expected_dense,description",
        [
            # Simple CSR case
            (
                [1.0, 2.0, 3.0],
                [0, 2, 1],
                [0, 2, 3],
                (0,),
                (2, 3),
                [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
                "CSR to dense",
            ),
            # CSC case
            # CSC: compressed_axes=(1,) means columns are compressed
            # Column 0: indices=[0], data=[1.0] → (0,0)=1.0
            # Column 1: indices=[1], data=[2.0] → (1,1)=2.0
            # Column 2: indices=[0], data=[3.0] → (0,2)=3.0
            (
                [1.0, 2.0, 3.0],
                [0, 1, 0],
                [0, 1, 2, 3],
                (1,),
                (2, 3),
                [[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]],
                "CSC to dense",
            ),
            # Empty sparse array (CSR)
            (
                [],
                [],
                [0, 0, 0],
                (0,),
                (2, 2),
                [[0.0, 0.0], [0.0, 0.0]],
                "Empty CSR to dense zeros",
            ),
            # Fully populated matrix (CSR)
            (
                [1.0, 2.0, 3.0, 4.0],
                [0, 1, 0, 1],
                [0, 2, 4],
                (0,),
                (2, 2),
                [[1.0, 2.0], [3.0, 4.0]],
                "Fully populated CSR matrix",
            ),
        ],
    )
    def test_materialize_to_dense(
        self,
        data: list[float],
        indices: list[int],
        indptr: list[int],
        compressed_axes: tuple[int, ...],
        shape: tuple[int, ...],
        expected_dense: list[Any],
        description: str,
    ) -> None:
        """Test materializing sparse arrays to dense format."""
        gcxs = GCXS(
            data=data,
            indices=indices,
            indptr=indptr,
            compressed_axes=compressed_axes,
            shape=shape,
            allow_materialize=True,
        )
        dense = gcxs.materialise()

        expected = jnp.array(expected_dense)
        assert jnp.allclose(dense, expected), f"Failed for: {description}"
        assert dense.shape == shape, f"Shape mismatch: {description}"

    def test_materialize_blocked_when_disabled(self) -> None:
        """Test that materialization is blocked when allow_materialize=False."""
        gcxs = GCXS(
            data=[1.0],
            indices=[0],
            indptr=[0, 1],
            compressed_axes=(0,),
            shape=(1, 2),
            allow_materialize=False,
        )

        with pytest.raises(ValueError, match="Refusing to materialize"):
            gcxs.materialise()


# ==============================================================================
# Transpose Tests
# ==============================================================================


class TestGCXSTranspose:
    """Test transpose operations on GCXS arrays."""

    def test_transpose_2d_csr_to_csc(self) -> None:
        """Test that transposing CSR creates CSC with swapped format."""
        import quax

        # Create CSR matrix
        data = [1.0, 2.0, 3.0]
        indices = [0, 2, 1]
        indptr = [0, 2, 3]
        csr = GCXS(
            data=data,
            indices=indices,
            indptr=indptr,
            compressed_axes=(0,),
            shape=(2, 3),
            allow_materialize=True,
        )

        # Transpose
        transpose_fn = quax.quaxify(jnp.transpose)
        csc = transpose_fn(csr)

        # Check format swap
        assert csc.shape == (3, 2), "Shape should be transposed"
        assert csc.compressed_axes == (1,), "CSR should become CSC"
        assert csc.nnz == csr.nnz, "nnz should not change"

        # Verify materialized form
        csr_dense = csr.materialise()
        csc_dense = csc.materialise()
        assert jnp.allclose(csc_dense, csr_dense.T)

    def test_transpose_2d_csc_to_csr(self) -> None:
        """Test that transposing CSC creates CSR with swapped format."""
        import quax

        # Create CSC matrix
        data = [1.0, 2.0, 3.0]
        indices = [0, 1, 0]
        indptr = [0, 1, 2, 3]
        csc = GCXS(
            data=data,
            indices=indices,
            indptr=indptr,
            compressed_axes=(1,),
            shape=(2, 3),
            allow_materialize=True,
        )

        # Transpose
        transpose_fn = quax.quaxify(jnp.transpose)
        csr = transpose_fn(csc)

        # Check format swap
        assert csr.shape == (3, 2), "Shape should be transposed"
        assert csr.compressed_axes == (0,), "CSC should become CSR"
        assert csr.nnz == csc.nnz, "nnz should not change"

        # Verify materialized form
        csc_dense = csc.materialise()
        csr_dense = csr.materialise()
        assert jnp.allclose(csr_dense, csc_dense.T)

    def test_transpose_identity(self) -> None:
        """Test identity transpose (no-op)."""
        import quax

        csr = GCXS(
            data=[1.0, 2.0],
            indices=[0, 1],
            indptr=[0, 1, 2],
            compressed_axes=(0,),
            shape=(2, 2),
            allow_materialize=True,
        )

        transpose_fn = quax.quaxify(jnp.transpose)
        result = transpose_fn(csr, axes=(0, 1))

        # Should be unchanged
        assert result.shape == csr.shape
        assert jnp.array_equal(result.data, csr.data)

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

        # For CSR (compressed_axes=(0,)), indptr length must be shape[0] + 1
        indptr_len = shape[0] + 1
        gcxs = GCXS(
            data=[1.0],
            indices=[0],
            indptr=[0, 1] + [1] * (indptr_len - 2),
            compressed_axes=(0,),
            shape=shape,
            allow_materialize=True,
        )

        transpose_fn = quax.quaxify(jnp.transpose)
        with pytest.raises(ValueError, match=error_pattern):
            transpose_fn(gcxs, axes=axes)


# ==============================================================================
# dot_general Tests
# ==============================================================================


class TestGCXSDotGeneral:
    """Test dot_general (generalized matrix/tensor multiplication)."""

    def test_csr_matmul_csr(self) -> None:
        """Test CSR @ CSR matrix multiplication."""
        import quax

        # A = [[1, 0], [0, 2]]
        A = GCXS(
            data=[1.0, 2.0],
            indices=[0, 1],
            indptr=[0, 1, 2],
            compressed_axes=(0,),
            shape=(2, 2),
            allow_materialize=True,
        )

        # B = [[3, 0], [0, 4]]
        B = GCXS(
            data=[3.0, 4.0],
            indices=[0, 1],
            indptr=[0, 1, 2],
            compressed_axes=(0,),
            shape=(2, 2),
            allow_materialize=True,
        )

        matmul_fn = quax.quaxify(jnp.matmul)
        C = matmul_fn(A, B)

        # Expected: [[3, 0], [0, 8]]
        assert C.shape == (2, 2)
        expected_dense = jnp.array([[3.0, 0.0], [0.0, 8.0]])
        assert jnp.allclose(C.materialise(), expected_dense)

    def test_csr_matmul_dense(self) -> None:
        """Test CSR @ Dense matrix multiplication."""
        import quax

        # A = [[1, 0], [0, 2]] (sparse CSR)
        A = GCXS(
            data=[1.0, 2.0],
            indices=[0, 1],
            indptr=[0, 1, 2],
            compressed_axes=(0,),
            shape=(2, 2),
            allow_materialize=True,
        )

        # B = [[3, 1], [2, 4]] (dense)
        B = jnp.array([[3.0, 1.0], [2.0, 4.0]])

        matmul_fn = quax.quaxify(jnp.matmul)
        C = matmul_fn(A, B)

        # Expected: [[3, 1], [4, 8]]
        expected = jnp.array([[3.0, 1.0], [4.0, 8.0]])
        assert jnp.allclose(C, expected)

    def test_dense_matmul_csr(self) -> None:
        """Test Dense @ CSR matrix multiplication."""
        import quax

        # A = [[1, 0], [0, 2]] (dense)
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])

        # B = [[3, 0], [0, 4]] (sparse CSR)
        B = GCXS(
            data=[3.0, 4.0],
            indices=[0, 1],
            indptr=[0, 1, 2],
            compressed_axes=(0,),
            shape=(2, 2),
            allow_materialize=True,
        )

        matmul_fn = quax.quaxify(jnp.matmul)
        C = matmul_fn(A, B)

        # Expected: [[3, 0], [0, 8]]
        expected = jnp.array([[3.0, 0.0], [0.0, 8.0]])
        assert jnp.allclose(C, expected)

    def test_csc_matmul_csc(self) -> None:
        """Test CSC @ CSC matrix multiplication."""
        import quax

        # Create two CSC matrices
        A = GCXS(
            data=[1.0, 2.0],
            indices=[0, 1],
            indptr=[0, 1, 2],
            compressed_axes=(1,),
            shape=(2, 2),
            allow_materialize=True,
        )

        B = GCXS(
            data=[3.0, 4.0],
            indices=[0, 1],
            indptr=[0, 1, 2],
            compressed_axes=(1,),
            shape=(2, 2),
            allow_materialize=True,
        )

        matmul_fn = quax.quaxify(jnp.matmul)
        C = matmul_fn(A, B)

        # Verify using dense computation
        A_dense = A.materialise()
        B_dense = B.materialise()
        expected = A_dense @ B_dense

        assert jnp.allclose(C.materialise(), expected)

    def test_empty_arrays_multiplication(self) -> None:
        """Test multiplication of empty sparse arrays."""
        import quax

        A = GCXS(
            data=[],
            indices=[],
            indptr=[0, 0],
            compressed_axes=(0,),
            shape=(1, 2),
            allow_materialize=True,
        )

        B = GCXS(
            data=[],
            indices=[],
            indptr=[0, 0, 0],
            compressed_axes=(0,),
            shape=(2, 3),
            allow_materialize=True,
        )

        matmul_fn = quax.quaxify(jnp.matmul)
        C = matmul_fn(A, B)

        assert C.shape == (1, 3)
        assert C.nnz == 0

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

        # Create minimal GCXS arrays for testing
        lhs = GCXS(
            data=[1.0],
            indices=[0],
            indptr=[0, 1] + [1] * (lhs_shape[0] - 1),
            compressed_axes=(0,),
            shape=lhs_shape,
            allow_materialize=True,
        )

        rhs = GCXS(
            data=[1.0],
            indices=[0],
            indptr=[0, 1] + [1] * (rhs_shape[0] - 1),
            compressed_axes=(0,),
            shape=rhs_shape,
            allow_materialize=True,
        )

        dot_general_fn = quax.quaxify(jax.lax.dot_general)
        with pytest.raises(ValueError, match=error_pattern):
            dot_general_fn(lhs, rhs, dimension_numbers=dimension_numbers)
