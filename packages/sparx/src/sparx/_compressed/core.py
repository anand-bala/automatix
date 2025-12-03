from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, final

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import quax
from jaxtyping import Array, Scalar, Shaped
from typing_extensions import override

from sparx._sparse_array import SparseArray
from sparx._utils import equivalent, normalize_axis

if TYPE_CHECKING:
    from sparx._coo.core import COO

type Shape = tuple[int, ...]


@final
class GCXS(SparseArray):
    """A sparse multidimensional array

    This is stored in GCXS format, a generalization of the GCRS/GCCS formats
    from [Efficient storage scheme for n-dimensional sparse array: GCRS/GCCS](
    https://ieeexplore.ieee.org/document/7237032). GCXS generalizes the CRS/CCS
    sparse matrix formats.

    For arrays with ndim == 2, GCXS is the same CSR/CSC.
    For arrays with ndim >2, any combination of axes can be compressed,
    significantly reducing storage.

    GCXS consists of 3 arrays. Let the 3 arrays be RO, CO and VL. The first element
    of array RO is the integer 0 and later elements are the number of
    cumulative non-zero elements in each row for GCRS, column for
    GCCS. CO stores column indexes of non-zero elements at each row for GCRS, column for GCCS.
    VL stores the values of the non-zero array elements.
    """

    data: Shaped[Array, " nnz"]
    """An array holding the values corresponding to the coordinates in `coords`"""
    indices: Shaped[Array, " nnz"]
    indptr: Array
    compressed_axes: tuple[int, ...] = eqx.field(static=True)
    _shape: tuple[int, ...] = eqx.field(static=True)
    allow_materialize: bool = eqx.field(static=True)
    """Flag to control if the quax'd array should be materialized into a dense array """

    def __init__(
        self,
        data: Shaped[Array, " nnz"],
        indices: Shaped[Array, " nnz"],
        indptr: Array,
        compressed_axes: int | tuple[int, ...],
        shape: int | tuple[int, ...],
        allow_materialize: bool = False,
    ) -> None:
        data = jnp.asarray(data)
        indices = jnp.asarray(indices)
        indptr = jnp.asarray(indptr)
        if not isinstance(compressed_axes, Iterable):
            compressed_axes = (compressed_axes,)
        if not isinstance(shape, Iterable):
            shape = (shape,)

        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.compressed_axes = compressed_axes
        self._shape = shape
        self.allow_materialize = allow_materialize

    def __check_init__(self) -> None:
        # Validate basic shape constraints from parent class
        if not all(isinstance(sh, int) and int(sh) >= 0 for sh in self._shape):
            raise ValueError("shape must be a non-negative integer or a tuple of non-negative integers.")

        # Validate GCXR-specific constraints
        if len(self.data) != len(self.indices):
            msg = f"data length ({len(self.data)}) must equal indices length ({len(self.indices)})"
            raise ValueError(msg)

        # Validate compressed_axes are within bounds
        if not all(0 <= ax < len(self._shape) for ax in self.compressed_axes):
            msg = f"compressed_axes {self.compressed_axes} invalid for shape {self._shape}"
            raise ValueError(msg)

        # Validate indptr structure
        if len(self.indptr) > 0:
            if self.indptr[0] != 0:
                raise ValueError(f"indptr must start at 0, got {self.indptr[0]}")
            if self.indptr[-1] != len(self.data):
                msg = f"indptr[-1] ({self.indptr[-1]}) must equal len(data) ({len(self.data)})"
                raise ValueError(msg)

            # For 2D case, validate indptr length
            if len(self._shape) == 2 and len(self.compressed_axes) == 1:
                expected_len = self._shape[self.compressed_axes[0]] + 1
                if len(self.indptr) != expected_len:
                    msg = f"indptr length ({len(self.indptr)}) doesn't match compressed dimension ({expected_len})"
                    raise ValueError(msg)

    @override
    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self._shape, self.data.dtype)

    @property
    @override
    def nnz(self) -> int:
        return len(self.data)

    def __len__(self) -> int:
        return self.shape[0]

    def to_coo(self) -> "COO":
        """Convert GCXR to COO format.

        Returns
        -------
        COO
            The array in COO (coordinate list) format
        """
        from sparx._coo.core import COO

        nnz = len(self.data)
        ndim = len(self.shape)
        coords = jnp.zeros((ndim, nnz), dtype=jnp.int32)

        # Handle 2D CSR case (compressed_axes=(0,))
        if ndim == 2 and self.compressed_axes == (0,):
            # Decompress row indices using indptr
            # For each row i, elements are in range [indptr[i], indptr[i+1])
            row_indices = jnp.repeat(
                jnp.arange(len(self.indptr) - 1, dtype=jnp.int32),
                jnp.diff(self.indptr).astype(jnp.int32),
                total_repeat_length=nnz,
            )
            coords = coords.at[0].set(row_indices)
            coords = coords.at[1].set(self.indices)  # Column indices

        # Handle 2D CSC case (compressed_axes=(1,))
        elif ndim == 2 and self.compressed_axes == (1,):
            # Decompress column indices using indptr
            col_indices = jnp.repeat(
                jnp.arange(len(self.indptr) - 1, dtype=jnp.int32),
                jnp.diff(self.indptr),
                total_repeat_length=nnz,
            )
            coords = coords.at[0].set(self.indices)  # Row indices
            coords = coords.at[1].set(col_indices)

        # General multi-dimensional case
        else:
            raise NotImplementedError("Only 2D CSR/CSC supported for now")

        return COO(data=self.data, coords=coords, shape=self.shape, allow_materialize=self.allow_materialize)

    @classmethod
    def from_coo(cls, coo: "COO", compressed_axes: int | tuple[int, ...] = (0,)) -> GCXS:
        """Convert COO format to GCXR.

        Parameters
        ----------
        coo : COO
            The sparse array in COO format
        compressed_axes : int or tuple of int
            Which axes to compress. Default is (0,) for CSR format.

        Returns
        -------
        GCXR
            The array in GCXR format
        """
        if not isinstance(compressed_axes, Iterable):
            compressed_axes = (compressed_axes,)

        # For 2D CSR
        if len(coo.shape) == 2 and compressed_axes == (0,):
            return cls._from_coo_csr_2d(coo)

        # For 2D CSC
        elif len(coo.shape) == 2 and compressed_axes == (1,):
            return cls._from_coo_csc_2d(coo)

        else:
            raise NotImplementedError("Only 2D CSR/CSC supported for now")

    @classmethod
    def _from_coo_csr_2d(cls, coo: "COO") -> GCXS:
        """Convert 2D COO to CSR format."""
        m, n = coo.shape
        nnz = coo.nnz

        if nnz == 0:
            return cls(
                data=jnp.array([], dtype=coo.data.dtype),
                indices=jnp.array([], dtype=jnp.int32),
                indptr=jnp.zeros(m + 1, dtype=jnp.int32),
                compressed_axes=(0,),
                shape=coo.shape,
                allow_materialize=coo.allow_materialize,
            )

        # Sort by row, then column
        row_coords = coo.coords[0]
        col_coords = coo.coords[1]

        # Lexicographic sort: row major
        sort_keys = row_coords * n + col_coords
        sort_order = jnp.argsort(sort_keys)

        sorted_rows = row_coords[sort_order]
        sorted_cols = col_coords[sort_order]
        sorted_data = coo.data[sort_order]

        # Build indptr - count elements per row
        row_counts = jnp.bincount(sorted_rows, length=m)
        indptr = jnp.zeros(m + 1, dtype=jnp.int32)
        indptr = indptr.at[1:].set(jnp.cumsum(row_counts))

        return cls(
            data=sorted_data,
            indices=sorted_cols,  # Column indices
            indptr=indptr,
            compressed_axes=(0,),
            shape=coo.shape,
            allow_materialize=coo.allow_materialize,
        )

    @classmethod
    def _from_coo_csc_2d(cls, coo: "COO") -> GCXS:
        """Convert 2D COO to CSC format."""
        m, n = coo.shape
        nnz = coo.nnz

        if nnz == 0:
            return cls(
                data=jnp.array([], dtype=coo.data.dtype),
                indices=jnp.array([], dtype=jnp.int32),
                indptr=jnp.zeros(n + 1, dtype=jnp.int32),
                compressed_axes=(1,),
                shape=coo.shape,
                allow_materialize=coo.allow_materialize,
            )

        # Sort by column, then row
        row_coords = coo.coords[0]
        col_coords = coo.coords[1]

        # Lexicographic sort: column major
        sort_keys = col_coords * m + row_coords
        sort_order = jnp.argsort(sort_keys)

        sorted_rows = row_coords[sort_order]
        sorted_cols = col_coords[sort_order]
        sorted_data = coo.data[sort_order]

        # Build indptr - count elements per column
        col_counts = jnp.bincount(sorted_cols, length=n)
        indptr = jnp.zeros(n + 1, dtype=jnp.int32)
        indptr = indptr.at[1:].set(jnp.cumsum(col_counts))

        return cls(
            data=sorted_data,
            indices=sorted_rows,  # Row indices
            indptr=indptr,
            compressed_axes=(1,),
            shape=coo.shape,
            allow_materialize=coo.allow_materialize,
        )

    @override
    def materialise(self) -> Shaped[Array, " {self._shape}"]:
        """Convert to dense array.

        Returns
        -------
        Array
            Dense array representation

        Raises
        ------
        ValueError
            If allow_materialize is False
        """
        if not self.allow_materialize:
            raise ValueError("Refusing to materialize GCXR sparse array. Make sure you `enable_materialize` first.")

        # Handle scalar case (shape = ())
        if self.shape == ():
            return self.data[0] if self.nnz > 0 else jnp.array(self.data.dtype.type(0))

        # Use COO conversion as intermediate step
        coo = self.to_coo()
        return coo.materialise()

    @override
    def prune(self, *, value: Scalar | None = None) -> GCXS:
        """Remove elements equal to the specified value.

        Parameters
        ----------
        value : scalar, optional
            Value to prune. Default is 0.

        Returns
        -------
        GCXR
            Pruned sparse array

        Examples
        --------
        >>> data = np.array([1, 0, 2, 0, 3])
        >>> indices = np.array([0, 1, 2, 3, 4])
        >>> indptr = np.array([0, 5])
        >>> s = GCXR(data, indices, indptr, compressed_axes=(0,), shape=(1, 5))
        >>> s.prune()
        >>> s.nnz
        3
        """
        value = value if value is not None else self.data.dtype.type(0)
        mask = ~equivalent(self.data, value)

        # Filter data and indices
        new_data = self.data[mask]
        new_indices = self.indices[mask]

        # Rebuild indptr by counting elements per compressed slice
        # For each slice i, count how many elements in [indptr[i], indptr[i+1]) pass the mask
        old_indptr = self.indptr
        new_indptr = jnp.zeros_like(old_indptr)

        # Compute counts for each slice
        for i in range(len(old_indptr) - 1):
            start = old_indptr[i]
            end = old_indptr[i + 1]
            slice_mask = mask[start:end]
            count = jnp.sum(slice_mask)
            new_indptr = new_indptr.at[i + 1].set(new_indptr[i] + count)

        return GCXS(
            data=new_data,
            indices=new_indices,
            indptr=new_indptr,
            compressed_axes=self.compressed_axes,
            shape=self._shape,
            allow_materialize=self.allow_materialize,
        )


# ==============================================================================
# Helper Functions for GCXR Operations
# ==============================================================================


def _compute_output_shape(
    lhs_shape: Shape, rhs_shape: Shape, lhs_contract: Shape, rhs_contract: Shape, lhs_batch: Shape, rhs_batch: Shape
) -> Shape:
    """
    Compute output shape for tensor contraction.

    The output keeps:
    - All batch dimensions (aligned between lhs and rhs)
    - All non-contracted, non-batch dimensions from lhs
    - All non-contracted, non-batch dimensions from rhs
    """
    result = []

    # Batch dimensions
    for i in lhs_batch:
        result.append(lhs_shape[i])

    # Non-contracted, non-batch dimensions from lhs
    for i in range(len(lhs_shape)):
        if i not in lhs_contract and i not in lhs_batch:
            result.append(lhs_shape[i])

    # Non-contracted, non-batch dimensions from rhs
    for i in range(len(rhs_shape)):
        if i not in rhs_contract and i not in rhs_batch:
            result.append(rhs_shape[i])

    return tuple(result)


def _dot_csr_csr_jax(
    out_shape: tuple[int, int],
    a_data: Array,
    b_data: Array,
    a_indices: Array,
    b_indices: Array,
    a_indptr: Array,
    b_indptr: Array,
) -> tuple[Array, Array, Array]:
    """
    Multiply two CSR matrices: C = A @ B.

    This uses scan to process rows sequentially to avoid vmap issues with
    dynamic slicing.

    Parameters
    ----------
    out_shape : tuple
        Shape of result matrix (n_row, n_col)
    a_data, a_indices, a_indptr : arrays
        CSR format for matrix A
    b_data, b_indices, b_indptr : arrays
        CSR format for matrix B

    Returns
    -------
    data : array
        Non-zero values of result
    indices : array
        Column indices of result
    indptr : array
        Row pointers of result
    """
    n_row, n_col = out_shape

    def process_row(carry, i):
        """Process row i using scan."""
        result_data_list, result_indices_list, result_nnz_list = carry

        # Get row i from A
        a_start = a_indptr[i]
        a_end = a_indptr[i + 1]

        # Build accumulator for row i of result
        sums = jnp.zeros(n_col, dtype=a_data.dtype)

        # For each k where A[i,k] != 0
        def inner_loop(j, sums):
            k = a_indices[a_start + j]
            a_val = a_data[a_start + j]

            # Get row k from B
            b_start = b_indptr[k]
            b_end = b_indptr[k + 1]

            # Accumulate products into sums
            def accumulate_b_element(jj, sums):
                b_col = b_indices[b_start + jj]
                b_val = b_data[b_start + jj]
                return sums.at[b_col].add(a_val * b_val)

            sums = lax.fori_loop(0, b_end - b_start, accumulate_b_element, sums)
            return sums

        # Accumulate over all non-zeros in row i of A
        sums = lax.fori_loop(0, a_end - a_start, inner_loop, sums)

        # Extract non-zeros from sums
        nonzero_mask = sums != 0
        row_nnz = jnp.sum(nonzero_mask)

        # Get non-zero data and indices (padded to n_col for consistent shape)
        row_data_padded = jnp.where(nonzero_mask, sums, 0)
        row_indices_padded = jnp.where(nonzero_mask, jnp.arange(n_col, dtype=jnp.int32), n_col)

        # Append to result lists
        result_data_list = result_data_list.at[i].set(row_data_padded)
        result_indices_list = result_indices_list.at[i].set(row_indices_padded)
        result_nnz_list = result_nnz_list.at[i].set(row_nnz)

        return (result_data_list, result_indices_list, result_nnz_list), None

    # Initialize carry
    init_data = jnp.zeros((n_row, n_col), dtype=a_data.dtype)
    init_indices = jnp.full((n_row, n_col), n_col, dtype=jnp.int32)
    init_nnz = jnp.zeros(n_row, dtype=jnp.int32)
    init_carry = (init_data, init_indices, init_nnz)

    # Process all rows using scan
    (result_data_padded, result_indices_padded, row_nnzs), _ = lax.scan(process_row, init_carry, jnp.arange(n_row))

    # Flatten and compress
    data_flat = result_data_padded.ravel()
    indices_flat = result_indices_padded.ravel()
    valid_mask = indices_flat < n_col

    data = data_flat[valid_mask]
    indices = indices_flat[valid_mask]

    # Build indptr
    indptr = jnp.zeros(n_row + 1, dtype=jnp.int32)
    indptr = indptr.at[1:].set(jnp.cumsum(row_nnzs))

    return data, indices, indptr


def _dot_csr_ndarray_jax(
    out_shape: tuple[int, int],
    a_data: Array,
    a_indices: Array,
    a_indptr: Array,
    b: Array,
) -> Array:
    """
    Multiply CSR matrix with dense array: C = A @ B.

    Returns dense result.

    Parameters
    ----------
    out_shape : tuple
        Output shape (m, n)
    a_data, a_indices, a_indptr : arrays
        CSR format for sparse matrix A
    b : array
        Dense matrix B

    Returns
    -------
    out : array
        Dense result matrix
    """
    m, n = out_shape

    def process_row(i: int) -> Array:
        """Compute row i of output."""
        a_start = a_indptr[i]
        a_end = a_indptr[i + 1]

        # Row i of result = sum over k of A[i,k] * B[k,:]
        def inner_loop(j, result):
            k = a_indices[a_start + j]
            return result.at[:].add(a_data[a_start + j] * b[k, :])

        # Use fori_loop to accumulate - handles zero iterations gracefully
        result = lax.fori_loop(0, a_end - a_start, inner_loop, jnp.zeros(n, dtype=jnp.result_type(a_data.dtype, b.dtype)))
        return result

    # Vectorize over rows
    out = jax.vmap(process_row)(jnp.arange(m))
    return out


# ==============================================================================
# Quax Registrations
# ==============================================================================


@quax.register(lax.transpose_p)
def _(a: GCXS, *, permutation: Sequence[int]) -> GCXS:
    """Transpose operation for GCXR arrays."""
    permutation = tuple(normalize_axis(permutation, a.ndim))

    # Validate permutation (JAX doesn't do this for custom types)
    if len(np.unique(permutation)) < len(permutation):
        raise ValueError("repeated axis in transpose")

    if permutation == tuple(range(a.ndim)):
        # axes don't change
        return a

    # Special case: 2D transpose is just a format swap (CSR ↔ CSC)
    if len(a.shape) == 2 and permutation == (1, 0) and len(a.compressed_axes) == 1:
        # CSR (compressed_axes=(0,)) becomes CSC (compressed_axes=(1,))
        # CSC (compressed_axes=(1,)) becomes CSR (compressed_axes=(0,))
        new_compressed_axes = (1 - a.compressed_axes[0],)
        return GCXS(
            data=a.data,
            indices=a.indices,
            indptr=a.indptr,
            compressed_axes=new_compressed_axes,
            shape=(a.shape[1], a.shape[0]),
            allow_materialize=a.allow_materialize,
        )

    # General case: convert to COO, transpose, convert back
    coo = a.to_coo()
    transposed_coo: COO = quax.quaxify(jnp.transpose)(coo, axes=permutation)  # type: ignore[arg-type, assignment]
    assert isinstance(transposed_coo, COO)
    # Use same compression scheme after transpose
    return GCXS.from_coo(transposed_coo, compressed_axes=a.compressed_axes)


@quax.register(lax.dot_general_p)
def _(
    lhs: GCXS,
    rhs: GCXS,
    *,
    dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
    precision=None,  # noqa: ANN001
    preferred_element_type=None,  # noqa: ANN001
    out_sharding=None,  # noqa: ANN001
) -> GCXS:
    """dot_general for GCXR × GCXR."""
    # Unpack dimension specification
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    # Convert to tuples
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    # Validate dimension compatibility
    for lhs_dim, rhs_dim in zip(lhs_contract, rhs_contract):
        if lhs.shape[lhs_dim] != rhs.shape[rhs_dim]:
            raise ValueError(
                f"Incompatible shapes for dot_general: "
                f"lhs.shape[{lhs_dim}] = {lhs.shape[lhs_dim]} but "
                f"rhs.shape[{rhs_dim}] = {rhs.shape[rhs_dim]}. "
                f"Contracting dimensions must have matching sizes."
            )

    for lhs_dim, rhs_dim in zip(lhs_batch, rhs_batch):
        if lhs.shape[lhs_dim] != rhs.shape[rhs_dim]:
            raise ValueError(
                f"Incompatible shapes for dot_general: "
                f"lhs.shape[{lhs_dim}] = {lhs.shape[lhs_dim]} but "
                f"rhs.shape[{rhs_dim}] = {rhs.shape[rhs_dim]}. "
                f"Batch dimensions must have matching sizes."
            )

    # Compute output shape
    output_shape = _compute_output_shape(lhs.shape, rhs.shape, lhs_contract, rhs_contract, lhs_batch, rhs_batch)

    # Handle empty arrays
    if lhs.nnz == 0 or rhs.nnz == 0:
        if len(output_shape) == 2:
            compressed_axes = (0,)
            indptr = jnp.zeros(output_shape[0] + 1, dtype=jnp.int32)
        else:
            # For non-2D, use CSR-like compression on first axis
            compressed_axes = tuple(range(len(output_shape) - 1)) if len(output_shape) > 1 else (0,)
            indptr = jnp.zeros(2, dtype=jnp.int32)

        return GCXS(
            data=jnp.array([], dtype=lhs.data.dtype),
            indices=jnp.array([], dtype=jnp.int32),
            indptr=indptr,
            compressed_axes=compressed_axes,
            shape=output_shape,
            allow_materialize=lhs.allow_materialize or rhs.allow_materialize,
        )

    # Optimized path: 2D simple matmul
    if len(lhs.shape) == 2 and len(rhs.shape) == 2 and lhs_contract == (1,) and rhs_contract == (0,) and len(lhs_batch) == 0:
        # Both CSR - use optimized kernel
        if lhs.compressed_axes == (0,) and rhs.compressed_axes == (0,):
            data, indices, indptr = _dot_csr_csr_jax(
                output_shape, lhs.data, rhs.data, lhs.indices, rhs.indices, lhs.indptr, rhs.indptr
            )

            return GCXS(
                data=data,
                indices=indices,
                indptr=indptr,
                compressed_axes=(0,),
                shape=output_shape,
                allow_materialize=lhs.allow_materialize or rhs.allow_materialize,
            )

        # Both CSC - transpose to CSR, multiply, transpose back
        elif lhs.compressed_axes == (1,) and rhs.compressed_axes == (1,):
            # Use the identity: (A @ B)^T = B^T @ A^T
            # For CSC matrices, this means we can multiply as if they were CSR
            data, indices, indptr = _dot_csr_csr_jax(
                output_shape[::-1], rhs.data, lhs.data, rhs.indices, lhs.indices, rhs.indptr, lhs.indptr
            )

            return GCXS(
                data=data,
                indices=indices,
                indptr=indptr,
                compressed_axes=(1,),
                shape=output_shape,
                allow_materialize=lhs.allow_materialize or rhs.allow_materialize,
            )

        # Mixed formats - convert to same format
        else:
            # Convert both to CSR for simplicity
            if lhs.compressed_axes != (0,):
                lhs = GCXS.from_coo(lhs.to_coo(), compressed_axes=(0,))
            if rhs.compressed_axes != (0,):
                rhs = GCXS.from_coo(rhs.to_coo(), compressed_axes=(0,))

            data, indices, indptr = _dot_csr_csr_jax(
                output_shape, lhs.data, rhs.data, lhs.indices, rhs.indices, lhs.indptr, rhs.indptr
            )

            return GCXS(
                data=data,
                indices=indices,
                indptr=indptr,
                compressed_axes=(0,),
                shape=output_shape,
                allow_materialize=lhs.allow_materialize or rhs.allow_materialize,
            )

    # General path: arbitrary dimensions, batch dims, etc.
    # Convert to COO, use COO dot_general, convert back

    lhs_coo = lhs.to_coo()
    rhs_coo = rhs.to_coo()

    result_coo = quax.quaxify(lax.dot_general)(
        lhs_coo,
        rhs_coo,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    # Convert result back to GCXR (use CSR format for 2D, or first axis compression for higher-D)
    compressed_axes = (0,) if len(output_shape) >= 1 else (0,)
    return GCXS.from_coo(result_coo, compressed_axes=compressed_axes)


@quax.register(lax.dot_general_p)
def _(
    lhs: GCXS,
    rhs: Array,
    *,
    dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
    precision=None,  # noqa: ANN001
    preferred_element_type=None,  # noqa: ANN001
    out_sharding=None,  # noqa: ANN001
) -> Array:
    """dot_general for GCXR × dense Array - returns dense."""
    # Unpack dimension specification
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    # For simple 2D matmul with CSR
    if (
        len(lhs.shape) == 2
        and len(rhs.shape) == 2
        and tuple(lhs_contract) == (1,)
        and tuple(rhs_contract) == (0,)
        and len(lhs_batch) == 0
        and lhs.compressed_axes == (0,)
    ):
        output_shape = (lhs.shape[0], rhs.shape[1])
        return _dot_csr_ndarray_jax(output_shape, lhs.data, lhs.indices, lhs.indptr, rhs)

    # General case: convert GCXR to COO, then use COO × Array

    lhs_coo = lhs.to_coo()
    return quax.quaxify(lax.dot_general)(
        lhs_coo,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


@quax.register(lax.dot_general_p)
def _(
    lhs: Array,
    rhs: GCXS,
    *,
    dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
    precision=None,  # noqa: ANN001
    preferred_element_type=None,  # noqa: ANN001
    out_sharding=None,  # noqa: ANN001
) -> Array:
    """dot_general for dense Array × GCXR - returns dense."""
    # For most cases, convert GCXR to COO and use COO × Array

    rhs_coo = rhs.to_coo()
    return quax.quaxify(lax.dot_general)(
        lhs,
        rhs_coo,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


@quax.register(lax.concatenate_p)
def _(operands: tuple, *, dimension: int) -> GCXS:
    """Concatenate GCXS arrays along a given dimension.

    Parameters
    ----------
    operands : tuple of GCXS
        Arrays to concatenate
    dimension : int
        Dimension along which to concatenate

    Returns
    -------
    GCXS
        Concatenated array
    """
    if not operands:
        raise ValueError("Need at least one array to concatenate")

    first = operands[0]

    # Validate all operands have same format
    for op in operands[1:]:
        if op.compressed_axes != first.compressed_axes:
            # Mixed formats - convert all to COO
            from sparx._coo.core import COO

            coo_ops = tuple(op.to_coo() for op in operands)
            result_coo: COO = quax.quaxify(jnp.concatenate)(coo_ops, axis=dimension)  # type: ignore[assignment]
            assert isinstance(result_coo, COO)
            return GCXS.from_coo(result_coo, compressed_axes=first.compressed_axes)

    # All have same compressed_axes
    compressed_axes = first.compressed_axes

    # Fast path: concatenating along the compressed axis for 2D arrays
    if len(first.shape) == 2 and len(compressed_axes) == 1:
        compressed_axis = compressed_axes[0]

        # CSR: concatenating along axis 0 (rows)
        if compressed_axis == 0 and dimension == 0:
            # Concatenate data and indices
            result_data = jnp.concatenate([op.data for op in operands])
            result_indices = jnp.concatenate([op.indices for op in operands])

            # Concatenate indptr arrays (adjust offsets)
            indptr_parts = []
            offset = 0
            for i, op in enumerate(operands):
                if i == 0:
                    indptr_parts.append(op.indptr)
                else:
                    # Skip first element (0) and add offset to rest
                    indptr_parts.append(op.indptr[1:] + offset)
                offset = indptr_parts[-1][-1]

            result_indptr = jnp.concatenate(indptr_parts)

            # New shape: concatenate dimension grows
            new_shape = list(first.shape)
            new_shape[0] = sum(op.shape[0] for op in operands)

            return GCXS(
                data=result_data,
                indices=result_indices,
                indptr=result_indptr,
                compressed_axes=compressed_axes,
                shape=tuple(new_shape),
                allow_materialize=any(op.allow_materialize for op in operands),
            )

        # CSC: concatenating along axis 1 (columns)
        elif compressed_axis == 1 and dimension == 1:
            # Same logic as CSR axis-0 case
            result_data = jnp.concatenate([op.data for op in operands])
            result_indices = jnp.concatenate([op.indices for op in operands])

            indptr_parts = []
            offset = 0
            for i, op in enumerate(operands):
                if i == 0:
                    indptr_parts.append(op.indptr)
                else:
                    indptr_parts.append(op.indptr[1:] + offset)
                offset = indptr_parts[-1][-1]

            result_indptr = jnp.concatenate(indptr_parts)

            new_shape = list(first.shape)
            new_shape[1] = sum(op.shape[1] for op in operands)

            return GCXS(
                data=result_data,
                indices=result_indices,
                indptr=result_indptr,
                compressed_axes=compressed_axes,
                shape=tuple(new_shape),
                allow_materialize=any(op.allow_materialize for op in operands),
            )

    # General case: convert to COO, concatenate, convert back
    from sparx._coo.core import COO

    coo_ops = tuple(op.to_coo() for op in operands)
    result_coo: COO = quax.quaxify(jnp.concatenate)(coo_ops, axis=dimension)  # type: ignore[assignment]
    assert isinstance(result_coo, COO)
    return GCXS.from_coo(result_coo, compressed_axes=compressed_axes)


@quax.register(lax.reshape_p)
def _(operand: GCXS, *, new_sizes: tuple[int, ...], dimensions: tuple[int, ...] | None) -> GCXS:
    """Reshape GCXS array.

    Parameters
    ----------
    operand : GCXS
        Array to reshape
    new_sizes : tuple of int
        New shape
    dimensions : tuple of int or None
        Permutation to apply before reshape (transpose)

    Returns
    -------
    GCXS
        Reshaped array
    """
    old_shape = operand.shape

    # Handle transpose via dimensions parameter
    if dimensions is not None:
        operand = quax.quaxify(jnp.transpose)(operand, axes=dimensions)
        old_shape = operand.shape

    # If shape unchanged, return as-is
    if old_shape == new_sizes:
        return operand

    # Fast path: only size-1 dimensions are added or removed (expand_dims/squeeze)
    # In this case, we can adjust compressed_axes without changing data structure
    old_shape_no_ones = tuple(s for s in old_shape if s != 1)
    new_shape_no_ones = tuple(s for s in new_sizes if s != 1)

    if old_shape_no_ones == new_shape_no_ones:
        # Map old compressed axes to new positions
        # Find mapping from old non-1 positions to new non-1 positions
        old_non_one_positions = [i for i, s in enumerate(old_shape) if s != 1]
        new_non_one_positions = [i for i, s in enumerate(new_sizes) if s != 1]

        # Map old compressed axes to new
        new_compressed_axes = []
        for old_ax in operand.compressed_axes:
            if old_ax in old_non_one_positions:
                old_idx = old_non_one_positions.index(old_ax)
                new_ax = new_non_one_positions[old_idx]
                new_compressed_axes.append(new_ax)

        return GCXS(
            data=operand.data,
            indices=operand.indices,
            indptr=operand.indptr,
            compressed_axes=tuple(new_compressed_axes),
            shape=new_sizes,
            allow_materialize=operand.allow_materialize,
        )

    # General case: convert to COO, reshape, convert back
    from sparx._coo.core import COO

    coo = operand.to_coo()
    reshaped_coo: COO = quax.quaxify(jnp.reshape)(coo, new_sizes)  # type: ignore[assignment]
    assert isinstance(reshaped_coo, COO)

    # After general reshape, use default compression (first axis for 2D, etc.)
    if len(new_sizes) == 2:
        new_compressed_axes = (0,)
    else:
        # For higher dimensions, compress first axis
        new_compressed_axes = (0,)

    return GCXS.from_coo(reshaped_coo, compressed_axes=new_compressed_axes)
