from __future__ import annotations

import typing
from collections.abc import Sequence

import array_api_compat

from algebraic.types import (
    Array,
    Number,
    is_a_device,
    is_array,
    is_cpu_device,
    is_jax_array,
    is_jax_device,
    is_numpy_array,
    is_torch_device,
)

if typing.TYPE_CHECKING:
    from algebraic import AlgebraicArray
    from algebraic.types import Device


def normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    """Return a sorted tuple of non-negative axis indices.

    Parameters
    ----------
    axis : int or sequence of int or None
        ``None`` (all axes), a single ``int``, or a sequence of ``int``\\s.
        Negative values are resolved modulo *ndim*.
    ndim : int
        Number of dimensions of the array being operated on.

    Returns
    -------
    tuple of int
        Sorted tuple of non-negative axis indices.
    """
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(sorted(a % ndim for a in axis))


def validate_semiring(*arrays: "AlgebraicArray") -> None:
    """Raise ``ValueError`` if any two inputs have different semiring instances.

    Parameters
    ----------
    *arrays : AlgebraicArray
        One or more ``AlgebraicArray`` instances.

    Raises
    ------
    ValueError
        If two arrays carry different semirings (compared with ``==``).
    """
    if len(arrays) < 2:
        return
    first = arrays[0].semiring
    for arr in arrays[1:]:
        if arr.semiring != first:
            raise ValueError(f"All AlgebraicArray inputs must share the same semiring; got {first!r} and {arr.semiring!r}.")


def asanyarray(x: "AlgebraicArray | Array | Number") -> Array:
    """Convert an object to an array.

    If scalar or unsupported type, will convert to NumPy array.

    Parameters
    ----------
    x : AlgebraicArray or Array or Number
        Object to convert.

    Returns
    -------
    Array
        The underlying array data.
    """
    from algebraic import AlgebraicArray

    if isinstance(x, AlgebraicArray):
        return x.data
    if is_array(x):
        return x
    # default to numpy array
    import numpy as np

    return np.asanyarray(x)


def maybe_unwrap(x: "AlgebraicArray | Array | Number") -> Array | Number:
    from algebraic import AlgebraicArray

    if isinstance(x, AlgebraicArray):
        return x.data
    if is_array(x):
        return x

    return x


def _get_device(x: "AlgebraicArray | Array") -> "Device":
    """Return the device of an array or AlgebraicArray."""
    from algebraic import AlgebraicArray

    if isinstance(x, AlgebraicArray):
        return x.device
    return array_api_compat.device(x)


def _resolve_device(dev_a: "Device | None", dev_b: "Device | None") -> "Device | None":
    """Return the higher-precedence device.

    GPU devices take precedence over ``'cpu'`` and ``None``.
    """
    if dev_a is None:
        return dev_b
    if dev_b is None:
        return dev_a
    if str(dev_a).lower() == "cpu":
        return dev_b
    if str(dev_b).lower() == "cpu":
        return dev_a
    return dev_a


def _check_device_types_compatible(dev_a: "Device", dev_b: "Device") -> None:
    """Raise ``TypeError`` if *dev_a* and *dev_b* come from different frameworks."""
    # None and str ("cpu") are compatible with everything.
    if dev_a is None or dev_b is None:
        return
    if isinstance(dev_a, str) or isinstance(dev_b, str):
        return
    # Both are concrete device objects - compare framework types.
    if type(dev_a).__module__.split(".")[0] != type(dev_b).__module__.split(".")[0]:
        raise TypeError(
            f"Cannot mix devices from different frameworks: "
            f"{type(dev_a).__name__} ({dev_a}) vs {type(dev_b).__name__} ({dev_b})"
        )


def common_device(*xs: "AlgebraicArray | Array | Number") -> "Device":
    """Return the common target device for a collection of arrays and scalars.

    GPU devices take precedence over ``'cpu'`` and ``None``. If only scalars
    are given, returns ``'cpu'``.

    Raises
    ------
    TypeError
        If arrays belong to different frameworks (e.g. a ``torch.device`` and
        a ``jax.Device``).
    """
    from algebraic import AlgebraicArray

    target: Device | None = None
    for x in xs:
        if isinstance(x, Number):
            continue
        if isinstance(x, AlgebraicArray) or is_array(x):
            dev = _get_device(x)
            if target is not None:
                _check_device_types_compatible(target, dev)
            target = _resolve_device(target, dev)

    if target is None:
        # All inputs were scalars.
        return "cpu"
    return target


def _scalar_to_array(x: Number, *, reference: "AlgebraicArray | Array") -> Array:
    """Convert a scalar to an array matching the backend of *reference*."""
    from algebraic import AlgebraicArray

    ref = reference.data if isinstance(reference, AlgebraicArray) else reference
    xp = array_api_compat.array_namespace(ref)
    return typing.cast(Array, xp.asarray(x))


def to_device(x: Array, target: "Device") -> Array:
    """Move *x* to *target*, handling cross-backend transfers (e.g. numpy -> JAX GPU)."""
    assert is_a_device(target)
    src_device = array_api_compat.device(x)

    if is_cpu_device(target) and is_cpu_device(src_device):
        return x

    # array_api_compat.to_device cannot move numpy arrays to non-CPU devices.
    # the next two guards should handle that case
    if is_jax_device(target) and is_numpy_array(x):
        import jax.numpy as jnp

        x = jnp.asarray(x, device=target)

    if is_torch_device(target) and is_numpy_array(x):
        import torch

        x = torch.as_tensor(x, device=target)

    # jax arrays cannot handle string device types
    # hope that the target string is a supported backend
    if is_jax_array(x) and isinstance(target, str):
        import jax

        target = jax.devices(target)[0]

    return typing.cast(Array, array_api_compat.to_device(x, target))


def to_common_device(*xs: "AlgebraicArray | Array | Number") -> tuple["AlgebraicArray | Array", ...]:
    """Move all inputs to a common device, converting scalars to arrays.

    Scalars are converted to arrays matching the backend of a sibling array
    input.  If every input is a scalar, each is wrapped in a NumPy array and
    the device is ``'cpu'``.

    Returns
    -------
    tuple of AlgebraicArray or Array
        The inputs, each transferred to the common device.
    """
    import numpy as np

    from algebraic import AlgebraicArray

    target = common_device(*xs)

    # Find a reference array for scalar conversion (first non-scalar).
    reference: AlgebraicArray | Array | None = None
    for x in xs:
        if isinstance(x, AlgebraicArray) or is_array(x):
            reference = x
            break

    results: list[AlgebraicArray | Array] = []
    for x in xs:
        if isinstance(x, Number):
            if reference is not None:
                arr = _scalar_to_array(x, reference=reference)
                arr = to_device(arr, target)
            else:
                arr = typing.cast(Array, np.asarray(x))
            results.append(arr)
        elif isinstance(x, AlgebraicArray):
            if str(_get_device(x)) != str(target):
                new_data = to_device(x.data, target)
                results.append(AlgebraicArray(new_data, x.semiring, x._vdot, x._matmul))
            else:
                results.append(x)
        else:
            # Raw array
            results.append(to_device(x, target))

    return tuple(results)


import optree  # noqa: E402

pytree = optree.pytree.reexport(namespace="algebraic", module="algebraic.utils.pytree")
pytree.__doc__ = """Re-export of ``optree`` bound to the ``"algebraic"`` namespace.

Import this module instead of ``optree`` directly to get pytree utilities
that default to the ``"algebraic"`` namespace::

    from algebraic.utils import pytree

    pytree.flatten(obj)                 # uses namespace="algebraic"
    pytree.tree_map(f, obj)             # uses namespace="algebraic"

"""

del optree
