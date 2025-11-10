"""Compatibility layer for semiring class to kernel conversion.

This module provides adapters to support both the legacy class-based API
and the new kernel-based GPU API, enabling gradual migration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union

from automatix.algebra.spec import AbstractSemiring

if TYPE_CHECKING:
    from automatix.algebra.kernels import AlgebraicStructure


def normalize_semiring(
    semiring: Union[Type[AbstractSemiring], "AlgebraicStructure"],
) -> "AlgebraicStructure":
    """Convert a semiring class or kernel to a kernel instance.

    This adapter ensures both old (class-based) and new (kernel-based)
    APIs work seamlessly. It is the primary normalization point for
    all semiring-taking functions.

    Parameters
    ----------
    semiring : Type[AbstractSemiring] | AlgebraicStructure
        Either a semiring class (will be converted via .to_kernel())
        or an AlgebraicStructure kernel instance (returned as-is).

    Returns
    -------
    AlgebraicStructure
        The kernel representation of the semiring.

    Raises
    ------
    TypeError
        If semiring is neither a valid class nor AlgebraicStructure.

    Examples
    --------
    >>> from automatix.algebra.backends.jax_ import MinPlusSemiring
    >>> kernel = normalize_semiring(MinPlusSemiring)
    >>> # kernel is now an AlgebraicStructure instance
    """
    if isinstance(semiring, type) and issubclass(semiring, AbstractSemiring):
        # Class-based: convert to kernel
        return semiring.to_kernel()
    elif hasattr(semiring, "add") and hasattr(semiring, "mul"):
        # Duck-typing: assume it is already an AlgebraicStructure
        return semiring
    else:
        raise TypeError(f"Expected Type[AbstractSemiring] or AlgebraicStructure, got {type(semiring).__name__}")
