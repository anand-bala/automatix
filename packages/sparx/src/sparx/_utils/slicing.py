# Most of this file is taken from https://github.com/pydata/sparse/blob/31411bcdad99c1feebd1d5a5fbba9620be074fd6/sparse/numba_backend/_slicing.py#L11

import typing
from collections.abc import Iterable
from types import EllipsisType

import numpy as np
from jaxtyping import ArrayLike, Bool, Int, UInt

type AllowedIndex = int | EllipsisType | slice | None | Int[ArrayLike, "..."] | Bool[ArrayLike, "..."]
type NormalizedIndex = int | slice | None | UInt[ArrayLike, "..."]
type Shape = int | tuple[int, ...]


def normalize_index(idx: AllowedIndex | tuple[AllowedIndex, ...], shape: tuple[int, ...]) -> tuple[NormalizedIndex, ...]:
    """Normalize slicing indexes
    1.  Replaces ellipses with many full slices
    2.  Adds full slices to end of index
    3.  Checks bounding conditions
    4.  Replaces numpy arrays with lists
    5.  Posify's slices integers and lists
    6.  Normalizes slices to canonical form
    Examples
    --------
    >>> normalize_index(1, (10,))
    (1,)
    >>> normalize_index(-1, (10,))
    (9,)
    >>> normalize_index([-1], (10,))
    (array([9]),)
    >>> normalize_index(slice(-3, 10, 1), (10,))
    (slice(7, 10, 1),)
    >>> normalize_index((Ellipsis, None), (10,))
    (slice(0, 10, 1), None)
    """
    if not isinstance(idx, tuple):
        idx = (idx,)

    # replace_ellipsis already validates that there's at most one Ellipsis
    idx = replace_ellipsis(len(shape), idx)

    # The number of `None`s in `idx`
    num_newaxis = sum(int(i is None) for i in idx)
    # The number of explicitly sliced dimensions (i.e., `idx` is some prefix of the possible index)
    num_explicit_idx = sum(
        int(i.ndim)
        if (hasattr(i, "ndim") and i.ndim >= 1)  # If indexing with arrays, add the dimensions of the array
        else 1  # Else, just add 1
        for i in idx
        if i is not None
    )
    if num_explicit_idx > len(shape):
        raise IndexError(f"Too many indices ({num_explicit_idx}) for array of shape ({shape})")

    # The number of implicitly sliced dimensions at the end
    num_implicit_idx = len(shape) - num_explicit_idx

    # Expand the idx to reflect implicitly ':' sliced dimensions at the end
    idx += (slice(None),) * num_implicit_idx
    assert len(idx) == num_newaxis + len(shape)

    # Compute the final shape with the newaxis (None) dimensions
    shape_iter = iter(shape)
    final_shape = tuple(None if i is None else next(shape_iter) for i in idx)
    assert next(shape_iter) is None, "We somehow didn't exhaust the shape"

    def _compose_normalizing(idx: AllowedIndex, dim: int | None) -> NormalizedIndex:
        check_index(idx, dim)
        idx_ = sanitize_index(idx)
        idx_ = replace_none(idx, dim)
        idx_ = posify_index(idx, dim)
        idx_ = clip_slice(idx, dim)
        return typing.cast(NormalizedIndex, idx_)

    return tuple(_compose_normalizing(i, d) for i, d in zip(idx, final_shape, strict=True))


def replace_ellipsis(
    expected_ndim: int, idx: tuple[AllowedIndex, ...]
) -> tuple[int | slice | None | Int[ArrayLike, "..."] | Bool[ArrayLike, "..."], ...]:
    """Replace ... with slices, :, : ,:
    >>> replace_ellipsis(4, (3, Ellipsis, 2))
    (3, slice(None, None, None), slice(None, None, None), 2)
    >>> replace_ellipsis(2, (Ellipsis, None))
    (slice(None, None, None), slice(None, None, None), None)
    """
    # Careful about using in or index because index may contain arrays
    ellipsis_loc = tuple(i for i, ind in enumerate(idx) if ind is Ellipsis)
    if not ellipsis_loc:
        return idx  # type: ignore[return-value]
    if (n_ellipsis := len(ellipsis_loc)) > 1:
        raise IndexError(f"Index can have at most 1 Ellipsis, got {n_ellipsis} in {idx}")

    loc = ellipsis_loc[0]
    extra_dimensions = expected_ndim - (len(idx) - sum(i is None for i in idx) - 1)
    return idx[:loc] + (slice(None, None, None),) * extra_dimensions + idx[loc + 1 :]  # type: ignore[return-value]


def check_index(ind: AllowedIndex, ndim: int | None) -> None:
    """Check validity of index for a given dimension
    Examples
    --------
    >>> check_index(3, 5)
    >>> check_index(5, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index is not smaller than dimension 5 >= 5
    >>> check_index(6, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index is not smaller than dimension 6 >= 5
    >>> check_index(-1, 5)
    >>> check_index(-6, 5)
    Traceback (most recent call last):
    ...
    IndexError: Negative index is not greater than negative dimension -6 <= -5
    >>> check_index([1, 2], 5)
    >>> check_index([6, 3], 5)
    Traceback (most recent call last):
    ...
    IndexError: Index out of bounds for dimension 5
    >>> check_index(slice(0, 3), 5)
    """
    if ind is None or isinstance(ind, slice):
        return
    assert ndim is not None
    # unknown dimension, assumed to be in bounds
    if isinstance(ind, Iterable):
        x = np.asanyarray(ind)
        if np.issubdtype(x.dtype, np.integer) and ((x >= ndim) | (x < -ndim)).any():
            raise IndexError(f"Index out of bounds for dimension {ndim:d}")
        if x.dtype == np.bool_ and len(x) != ndim:
            raise IndexError(
                f"boolean index did not match indexed array; dimension is {ndim:d} "
                f"but corresponding boolean dimension is {len(x):d}"
            )
    elif not isinstance(ind, int):
        raise IndexError("only integers, slices (`:`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices")

    elif ind >= ndim:
        raise IndexError(f"Index is not smaller than dimension {ind:d} >= {ndim:d}")

    elif ind < -ndim:
        msg = "Negative index is not greater than negative dimension {:d} <= -{:d}"
        raise IndexError(msg.format(ind, ndim))


def sanitize_index(ind: AllowedIndex) -> AllowedIndex:
    """Sanitize the elements for indexing along one axis
    >>> sanitize_index([2, 3, 5])
    array([2, 3, 5])
    >>> sanitize_index([True, False, True, False])
    array([0, 2])
    >>> sanitize_index(np.array([1, 2, 3]))
    array([1, 2, 3])
    >>> sanitize_index(np.array([False, True, True]))
    array([1, 2])
    >>> type(sanitize_index(np.int32(0)))  # doctest: +SKIP
    <type 'int'>
    >>> sanitize_index(0.5)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    IndexError: only integers, slices (`:`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices; got {float}
    """

    def _sanitize_index_element(ind: AllowedIndex) -> int | None:
        return int(ind) if ind is not None else None

    if ind is None:
        return None
    if isinstance(ind, int):
        return _sanitize_index_element(ind)
    if isinstance(ind, slice):
        return slice(
            _sanitize_index_element(ind.start),
            _sanitize_index_element(ind.stop),
            _sanitize_index_element(ind.step),
        )
    ind = np.asarray(ind)
    if ind.dtype == np.bool_:
        nonzero = np.nonzero(ind)
        if len(nonzero) == 1:
            # If a 1-element tuple, unwrap the element
            nonzero = nonzero[0]
        return np.asanyarray(nonzero)
    if np.issubdtype(ind.dtype, np.integer):
        return ind

    raise IndexError(
        f"only integers, slices (`:`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices; got {ind.dtype}"
    )


def posify_index(ind: AllowedIndex, dim: int | None) -> AllowedIndex:
    """Flip negative indices around to positive ones"""
    if dim is None:
        assert ind is None
        return ind
    if isinstance(ind, int):
        if ind < 0:
            return ind + dim
        return ind
    if isinstance(ind, np.ndarray | list):
        ind = np.asanyarray(ind)
        return np.where(ind < 0, ind + dim, ind)
    if isinstance(ind, slice):
        start, stop, step = ind.start, ind.stop, ind.step
        assert isinstance(start, int)
        assert isinstance(stop, int)
        assert isinstance(step, int)

        if start < 0:
            start += dim

        if not (0 > stop >= step) and stop < 0:
            stop += dim

        return slice(start, stop, ind.step)

    return ind


def clip_slice(idx: AllowedIndex, dim: int | None) -> AllowedIndex:
    """
    Clip slice to its effective size given the shape.


    Examples
    --------
    >>> clip_slice(slice(0, 20, 1), 10)
    slice(0, 10, 1)
    """
    if not isinstance(idx, slice):
        return idx
    assert dim is not None

    start, stop, step = idx.start, idx.stop, idx.step
    assert isinstance(start, int)
    assert isinstance(stop, int)
    assert isinstance(step, int)

    if step > 0:
        start = max(start, 0)
        stop = min(stop, dim)
        if start > stop:
            start = stop
    else:
        start = min(start, dim - 1)
        stop = max(stop, -1)
        if start < stop:
            start = stop
    return slice(start, stop, step)


def replace_none(idx: AllowedIndex, dim: int | None) -> AllowedIndex:
    """
    Normalize slices to canonical form, i.e.
    replace ``None`` with the appropriate integers.

    Examples
    --------
    >>> replace_none(slice(None, None, None), 10)
    slice(0, 10, 1)
    """
    if not isinstance(idx, slice):
        return idx
    assert dim is not None

    start, stop, step = idx.start, idx.stop, idx.step
    if step is None:
        step = 1
    if step > 0:
        if start is None:
            start = 0
        if stop is None:
            stop = dim
    else:
        if start is None:
            start = dim - 1
        if stop is None:
            stop = -1
    return slice(start, stop, step)
