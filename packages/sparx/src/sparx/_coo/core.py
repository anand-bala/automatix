from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Self, final

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import quax
from jaxtyping import Array, Int, Scalar, Shaped
from typing_extensions import override

from sparx._sparse_array import SparseArray
from sparx._utils import equivalent, normalize_axis

type Shape = tuple[int, ...]


@final
class COO(SparseArray):
    """A sparse multidimensional array using the coordinate list format."""

    data: Shaped[Array, " nnz"]
    """An array holding the values corresponding to the coordinates in `coords`"""
    coords: Int[Array, " ndim nnz"]
    """An array holding the coordinates of every non-zero element"""
    _shape: tuple[int, ...] = eqx.field(static=True)
    allow_materialize: bool = eqx.field(static=True)
    """Flag to control if the quax'd array should be materialized into a dense array """

    def __init__(
        self,
        data: Shaped[Array, " nnz"],
        coords: Int[Array, "ndim nnz"],
        shape: int | tuple[int, ...],
        allow_materialize: bool = False,
    ) -> None:
        data = jnp.asarray(data)
        coords = jnp.asarray(coords)
        if not isinstance(shape, Iterable):
            shape = (shape,)

        self.data = data
        self.coords = coords
        self._shape = shape
        self.allow_materialize = allow_materialize

    def __check_init__(self) -> None:
        if len(self.data) != self.coords.shape[1]:
            msg = "The data length does not match the coordinates given.\nlen(data) = {}, but {} coords specified."
            raise ValueError(msg.format(len(self.data), self.coords.shape[1]))
        if len(self._shape) != self.coords.shape[0]:
            msg = (
                "Shape specified by `shape` doesn't match the "
                "shape of `coords`; len(shape)={} != coords.shape[0]={}"
                "(and coords.shape={})"
            )
            raise ValueError(msg.format(len(self._shape), self.coords.shape[0], self.coords.shape))

    @override
    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self._shape, self.data.dtype)

    @override
    def materialise(self) -> Shaped[Array, " {self._shape}"]:
        if not self.allow_materialize:
            raise ValueError("Refusing to materialize COO sparse array. Make sure you `enable_materialize` first.")

        # Handle scalar case (shape = ())
        if self.shape == ():
            # For a scalar, we should have exactly one data element with no coordinates
            return self.data[0] if self.nnz > 0 else jnp.array(self.data.dtype.type(0))

        def add_at_idx(
            arr: Shaped[Array, " {self._shape}"], idx: tuple[int, ...], val: Scalar
        ) -> Shaped[Array, " {self._shape}"]:
            return arr.at[idx].add(val)

        x = jnp.zeros(self.shape, self.dtype.type)
        # convert coords to a tuple of Arrays.
        # Len of tuple should be ndim
        coords = tuple(self.coords)
        assert len(coords) == x.ndim
        # use jax's ability to automatically pair up indices to values in .at
        x = x.at[coords].set(self.data)
        return x

    @property
    @override
    def nnz(self) -> int:
        return self.coords.shape[1]

    @override
    def prune(self, *, value: Scalar | None = None) -> Self:
        """
        Examples
        --------
        >>> coords = np.array([[0, 1, 2, 3]])
        >>> data = np.array([1, 0, 1, 2])
        >>> s = COO(data, coords, shape=(4,))
        >>> s.prune()
        >>> s.nnz
        3
        """
        value = value if value is not None else self.data.dtype.type(0)
        mask = ~equivalent(self.data, value)
        return COO(
            data=self.data[mask],
            coords=self.coords[:, mask],
            shape=self.shape,
            allow_materialize=self.allow_materialize,
        )

    def __len__(self) -> int:
        return self.shape[0]


@quax.register(lax.transpose_p)
def _(a: COO, *, permutation: Sequence[int]) -> COO:
    permutation = tuple(normalize_axis(permutation, a.ndim))

    # Validate permutation (JAX doesn't do this for custom types)
    if len(np.unique(permutation)) < len(permutation):
        raise ValueError("repeated axis in transpose")

    if permutation == tuple(range(a.ndim)):
        # axes don't change
        return a

    shape = tuple(a.shape[ax] for ax in permutation)
    return COO(
        data=a.data,
        coords=a.coords[permutation, :],
        shape=shape,
        allow_materialize=a.allow_materialize,
    )


@quax.register(lax.dot_general_p)
def _(
    lhs: COO,
    rhs: COO,
    *,
    dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
    precision=None,  # noqa: ANN001
    preferred_element_type=None,  # noqa: ANN001
    out_sharding=None,  # noqa: ANN001
) -> COO:
    # Unpack the dimension specification (always provided by the primitive)
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    # Convert to tuples for easier handling
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    # Validate dimension compatibility
    # Note: JAX's dot_general doesn't validate these for custom types,
    # so we need to do it ourselves to provide meaningful error messages
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

    # First, let's figure out the output shape
    output_shape = _compute_output_shape(lhs.shape, rhs.shape, lhs_contract, rhs_contract, lhs_batch, rhs_batch)

    # Handle the empty case early
    if lhs.data.shape[0] == 0 or rhs.data.shape[0] == 0:
        return COO(
            data=jnp.array([], dtype=lhs.data.dtype),
            coords=jnp.zeros((len(output_shape), 0), dtype=jnp.int32),
            shape=output_shape,
            allow_materialize=lhs.allow_materialize or rhs.allow_materialize,
        )

    # The core operation: find all valid pairs and compute their contributions
    # This is doing something conceptually similar to a nested loop:
    # for each non-zero in lhs:
    #     for each non-zero in rhs:
    #         if their contracting coordinates match:
    #             multiply them and add to the appropriate output location
    #
    # But we need to vectorize this for JAX/XLA efficiency.

    # We'll examine all pairs, so we need to broadcast
    # Shape will be (n_lhs, n_rhs) for intermediate computations
    lhs_coords_expanded = lhs.coords[:, :, None]  # (ndim, n_lhs, 1)
    rhs_coords_expanded = rhs.coords[:, None, :]  # (ndim, 1, n_rhs)

    # Check which pairs have matching coordinates on contracting dimensions
    matches = jnp.ones((lhs.nnz, rhs.nnz), dtype=bool)
    for lhs_dim, rhs_dim in zip(lhs_contract, rhs_contract):
        matches = matches & (lhs_coords_expanded[lhs_dim] == rhs_coords_expanded[rhs_dim])

    # Also check batch dimensions must match
    for lhs_dim, rhs_dim in zip(lhs_batch, rhs_batch):
        matches = matches & (lhs_coords_expanded[lhs_dim] == rhs_coords_expanded[rhs_dim])

    # Find which pairs actually match
    lhs_indices, rhs_indices = jnp.where(matches)
    # Compute the products for matching pairs
    products = lhs.data[lhs_indices] * rhs.data[rhs_indices]

    # Now compute the output coordinates for each product
    # This involves mapping from input coordinates to output coordinates
    output_coords = _compute_output_coordinates(
        lhs.coords,
        rhs.coords,
        lhs_indices,
        rhs_indices,
        lhs.shape,
        rhs.shape,
        lhs_contract,
        rhs_contract,
        lhs_batch,
        rhs_batch,
        output_shape,
    )

    # Multiple products might map to the same output coordinate
    # We need to sum them using a scatter-add operation
    output_coords, output_data = _sum_duplicates(output_coords, products, output_shape)

    return COO(
        data=output_data,
        coords=output_coords,
        shape=output_shape,
        allow_materialize=lhs.allow_materialize or rhs.allow_materialize,
    )


def _compute_output_shape(
    lhs_shape: Shape, rhs_shape: Shape, lhs_contract: Shape, rhs_contract: Shape, lhs_batch: Shape, rhs_batch: Shape
) -> Shape:
    """
    Figures out what the output shape should be after the contraction.

    The output keeps:
    - All batch dimensions (they're aligned between lhs and rhs)
    - All non-contracted, non-batch dimensions from lhs
    - All non-contracted, non-batch dimensions from rhs
    """
    # Start with batch dimensions
    result = []
    for i in lhs_batch:
        result.append(lhs_shape[i])

    # Add non-contracted, non-batch dimensions from lhs
    for i in range(len(lhs_shape)):
        if i not in lhs_contract and i not in lhs_batch:
            result.append(lhs_shape[i])

    # Add non-contracted, non-batch dimensions from rhs
    for i in range(len(rhs_shape)):
        if i not in rhs_contract and i not in rhs_batch:
            result.append(rhs_shape[i])

    return tuple(result)


def _compute_output_coordinates(
    lhs_coords,
    rhs_coords,
    lhs_indices,
    rhs_indices,
    lhs_shape,
    rhs_shape,
    lhs_contract,
    rhs_contract,
    lhs_batch,
    rhs_batch,
    output_shape,
):
    """
    Maps input coordinates to output coordinates.

    The output coordinate is built from:
    - Batch dimensions (taken from either lhs or rhs, they're the same)
    - Non-contracted lhs dimensions
    - Non-contracted rhs dimensions
    """
    n_pairs = lhs_indices.shape[0]
    output_ndim = len(output_shape)
    output_coords = jnp.zeros((output_ndim, n_pairs), dtype=jnp.int32)

    out_idx = 0

    # Add batch dimensions
    for lhs_dim in lhs_batch:
        output_coords = output_coords.at[out_idx].set(lhs_coords[lhs_dim, lhs_indices])
        out_idx += 1

    # Add non-contracted, non-batch dimensions from lhs
    for lhs_dim in range(len(lhs_shape)):
        if lhs_dim not in lhs_contract and lhs_dim not in lhs_batch:
            output_coords = output_coords.at[out_idx].set(lhs_coords[lhs_dim, lhs_indices])
            out_idx += 1

    # Add non-contracted, non-batch dimensions from rhs
    for rhs_dim in range(len(rhs_shape)):
        if rhs_dim not in rhs_contract and rhs_dim not in rhs_batch:
            output_coords = output_coords.at[out_idx].set(rhs_coords[rhs_dim, rhs_indices])
            out_idx += 1

    return output_coords


def _sum_duplicates(coords, data, shape):
    """
    Sums values that map to the same coordinate.

    This is necessary because multiple input pairs can contribute
    to the same output location (that's the whole point of summation
    in tensor contraction!).
    """
    if data.shape[0] == 0:
        return coords, data

    # Convert coordinates to flat indices for easier grouping
    flat_indices = _coords_to_flat_index(coords, shape)

    # Sort by flat index to group duplicates together
    sort_order = jnp.argsort(flat_indices)
    flat_indices_sorted = flat_indices[sort_order]
    data_sorted = data[sort_order]
    coords_sorted = coords[:, sort_order]

    # Use segment_sum to add up values with the same index
    # First, find where the flat index changes
    unique_mask = jnp.concatenate([jnp.array([True]), flat_indices_sorted[1:] != flat_indices_sorted[:-1]])

    # Get segment IDs (which group each element belongs to)
    segment_ids = jnp.cumsum(unique_mask) - 1
    num_segments = jnp.max(segment_ids).item() + 1

    # Sum within each segment
    summed_data = jax.ops.segment_sum(data_sorted, segment_ids, num_segments)

    # Get unique coordinates
    unique_coords = coords_sorted[:, unique_mask]

    return unique_coords, summed_data


def _coords_to_flat_index(coords, shape):
    """
    Converts multi-dimensional coordinates to flat (linear) indices.

    This is like numpy's ravel_multi_index - it maps an n-dimensional
    coordinate to a single integer.
    """
    ndim, nnz = coords.shape
    flat_idx = jnp.zeros(nnz, dtype=jnp.int32)

    stride = 1
    for dim in range(ndim - 1, -1, -1):
        flat_idx += coords[dim] * stride
        stride *= shape[dim]

    return flat_idx


def _flat_index_to_coords(flat_idx, shape):
    """
    Converts flat (linear) indices to multi-dimensional coordinates.

    This is like numpy's unravel_index - maps a single integer to
    an n-dimensional coordinate.
    """
    ndim = len(shape)
    nnz = flat_idx.shape[0]
    coords = jnp.zeros((ndim, nnz), dtype=jnp.int32)

    remaining = flat_idx
    for dim in range(ndim - 1, -1, -1):
        coords = coords.at[dim].set(remaining % shape[dim])
        remaining = remaining // shape[dim]

    return coords


@quax.register(lax.concatenate_p)
def _(operands: tuple, *, dimension: int) -> COO:
    """
    Concatenate COO arrays along a given dimension.

    Parameters
    ----------
    operands : tuple of COO
        Arrays to concatenate
    dimension : int
        Axis along which to concatenate

    Returns
    -------
    COO
        Concatenated sparse array
    """
    # Handle empty operands
    if len(operands) == 0:
        raise ValueError("Need at least one array to concatenate")

    if len(operands) == 1:
        return operands[0]

    # Check all operands are COO
    if not all(isinstance(op, COO) for op in operands):
        raise NotImplementedError("Cannot concatenate mixed COO and non-COO arrays")

    # Validate shapes (all dimensions except concat dimension must match)
    first_shape = operands[0].shape
    ndim = len(first_shape)

    # Normalize dimension
    dimension = dimension if dimension >= 0 else ndim + dimension

    for i, op in enumerate(operands[1:], 1):
        if len(op.shape) != ndim:
            raise ValueError(f"All arrays must have same ndim: {ndim} != {len(op.shape)}")
        for d in range(ndim):
            if d != dimension and op.shape[d] != first_shape[d]:
                raise ValueError(
                    f"All arrays must have same shape except along concat axis: "
                    f"dimension {d}: {first_shape[d]} != {op.shape[d]}"
                )

    # Concatenate data
    all_data = [op.data for op in operands]
    result_data = jnp.concatenate(all_data)

    # Concatenate and adjust coords
    all_coords = []
    offset = 0
    for op in operands:
        coords = op.coords
        # Add offset to the concatenation dimension
        adjusted_coords = coords.at[dimension].add(offset)
        all_coords.append(adjusted_coords)
        offset += op.shape[dimension]

    result_coords = jnp.concatenate(all_coords, axis=1)

    # Compute new shape
    new_shape = list(first_shape)
    new_shape[dimension] = sum(op.shape[dimension] for op in operands)

    return COO(
        data=result_data,
        coords=result_coords,
        shape=tuple(new_shape),
        allow_materialize=any(op.allow_materialize for op in operands),
    )


@quax.register(lax.reshape_p)
def _(operand: COO, *, new_sizes: tuple[int, ...], dimensions: tuple[int, ...] | None) -> COO:
    """
    Reshape COO array to new shape.

    Parameters
    ----------
    operand : COO
        Array to reshape
    new_sizes : tuple of int
        New shape
    dimensions : tuple of int or None
        Permutation of axes (None means no permutation)

    Returns
    -------
    COO
        Reshaped sparse array
    """
    old_shape = operand.shape

    # Handle transpose if dimensions is provided
    if dimensions is not None:
        # Transpose first, then reshape
        transposed_coords = operand.coords[list(dimensions), :]
        transposed_shape = tuple(old_shape[d] for d in dimensions)
        operand = COO(
            data=operand.data,
            coords=transposed_coords,
            shape=transposed_shape,
            allow_materialize=operand.allow_materialize,
        )
        old_shape = transposed_shape

    # Check if shapes are compatible
    if np.prod(old_shape) != np.prod(new_sizes):
        raise ValueError(f"Cannot reshape array of size {np.prod(old_shape)} into shape {new_sizes}")

    # Special case: no actual reshape needed
    if old_shape == new_sizes:
        return operand

    # Special case: adding/removing dimensions of size 1
    # This is common for expand_dims and squeeze operations
    old_shape_no_ones = tuple(s for s in old_shape if s != 1)
    new_shape_no_ones = tuple(s for s in new_sizes if s != 1)

    if old_shape_no_ones == new_shape_no_ones:
        # Only difference is size-1 dimensions
        # Need to adjust coordinates by inserting/removing entries
        return _reshape_with_ones(operand, old_shape, new_sizes)

    # General case: convert via flat indices
    # This is the fallback for arbitrary reshapes
    return _reshape_general(operand, old_shape, new_sizes)


def _reshape_with_ones(operand: COO, old_shape: tuple[int, ...], new_shape: tuple[int, ...]) -> COO:
    """
    Reshape when only size-1 dimensions change.

    This handles expand_dims and squeeze efficiently.
    """
    old_ndim = len(old_shape)
    new_ndim = len(new_shape)
    old_coords = operand.coords
    nnz = operand.nnz

    if new_ndim > old_ndim:
        # Adding dimensions - find where
        new_coords = jnp.zeros((new_ndim, nnz), dtype=jnp.int32)
        old_idx = 0
        for new_idx in range(new_ndim):
            if new_shape[new_idx] == 1:
                # New size-1 dimension - fill with zeros
                pass  # already zeros
            else:
                # Copy from old coords
                new_coords = new_coords.at[new_idx].set(old_coords[old_idx])
                old_idx += 1
    else:
        # Removing dimensions - filter out size-1 dims
        new_coords_list = []
        for old_idx in range(old_ndim):
            if old_shape[old_idx] != 1:
                new_coords_list.append(old_coords[old_idx])
        new_coords = jnp.stack(new_coords_list, axis=0)

    return COO(
        data=operand.data,
        coords=new_coords,
        shape=new_shape,
        allow_materialize=operand.allow_materialize,
    )


def _reshape_general(operand: COO, old_shape: tuple[int, ...], new_shape: tuple[int, ...]) -> COO:
    """
    General reshape using flat index conversion.

    This handles arbitrary reshapes by:
    1. Converting old coordinates to flat indices
    2. Converting flat indices to new coordinates
    """
    # Convert old coordinates to flat indices
    flat_indices = _coords_to_flat_index(operand.coords, old_shape)

    # Convert flat indices to new coordinates
    new_coords = _flat_index_to_coords(flat_indices, new_shape)

    return COO(
        data=operand.data,
        coords=new_coords,
        shape=new_shape,
        allow_materialize=operand.allow_materialize,
    )
