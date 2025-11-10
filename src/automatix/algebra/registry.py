"""Semiring registry and factory for discovering and instantiating semirings.

This module provides a centralized registry for managing semiring implementations
across different backends (JAX, PyTorch, NumPy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type

from automatix.algebra.spec import AbstractSemiring

if TYPE_CHECKING:
    from automatix.algebra.kernels import AlgebraicStructure

# Type for semiring classes
SemiringType = Type[AbstractSemiring]

# Registry mapping: (name, backend) -> semiring_class
_REGISTRY: dict[tuple[str, str], SemiringType] = {}

# Kernel registry for GPU-optimized implementations
_KERNEL_REGISTRY: dict[tuple[str, str], "AlgebraicStructure"] = {}


def register(name: str, backend: str = "jax") -> Callable[[SemiringType], SemiringType]:
    """Decorator to register a semiring class in the registry.

    Parameters
    ----------
    name : str
        Canonical name for the semiring (e.g., "MaxPlus", "Counting").
        Names are case-sensitive.
    backend : str, optional
        Backend identifier (default: "jax"). Common values: "jax", "torch", "numpy".

    Returns
    -------
    Callable[[SemiringType], SemiringType]
        Decorator that registers the class and returns it unchanged.

    Examples
    --------
    >>> @register("MaxPlus", backend="jax")
    ... class MaxPlusSemiring(AbstractSemiring):
    ...     ...
    """

    def decorator(cls: SemiringType) -> SemiringType:
        key = (name, backend)
        if key in _REGISTRY:
            raise ValueError(
                f"Semiring '{name}' with backend '{backend}' is already registered. "
                f"Use a different name or call unregister() first."
            )
        _REGISTRY[key] = cls
        return cls

    return decorator


def unregister(name: str, backend: str = "jax") -> None:
    """Unregister a semiring from the registry.

    Parameters
    ----------
    name : str
        Canonical name for the semiring.
    backend : str, optional
        Backend identifier (default: "jax").

    Raises
    ------
    KeyError
        If the semiring is not registered.
    """
    key = (name, backend)
    if key not in _REGISTRY:
        raise KeyError(f"Semiring '{name}' with backend '{backend}' is not registered. Available: {list_semirings(backend)}")
    del _REGISTRY[key]


def get_semiring(name: str, backend: str = "jax") -> SemiringType:
    """Get a semiring class from the registry.

    Parameters
    ----------
    name : str
        Canonical name for the semiring (e.g., "MaxPlus", "Counting").
    backend : str, optional
        Backend identifier (default: "jax").

    Returns
    -------
    SemiringType
        The semiring class.

    Raises
    ------
    KeyError
        If the semiring is not registered.

    Examples
    --------
    >>> MaxPlus = get_semiring("MaxPlus", backend="jax")
    >>> weights = MaxPlus.zeros((3, 3))
    """
    key = (name, backend)
    if key not in _REGISTRY:
        available = list_semirings(backend)
        raise KeyError(f"Semiring '{name}' with backend '{backend}' not found. Available for '{backend}': {available}")
    return _REGISTRY[key]


def list_semirings(backend: str | None = None) -> list[str]:
    """List all registered semirings, optionally filtered by backend.

    Parameters
    ----------
    backend : str, optional
        If provided, only return semirings for this backend.
        If None, return all registered semirings grouped by backend.

    Returns
    -------
    list[str]
        List of semiring names. If backend is None, names are formatted as
        "SemiringName (backend)".

    Examples
    --------
    >>> list_semirings("jax")
    ['Counting', 'MaxPlus', 'MinPlus', ...]

    >>> list_semirings()
    ['Counting (jax)', 'MaxPlus (jax)', 'MaxPlus (torch)', ...]
    """
    if backend is None:
        return sorted([f"{name} ({b})" for name, b in _REGISTRY.keys()])
    else:
        return sorted([name for name, b in _REGISTRY.keys() if b == backend])


def get_available_backends() -> list[str]:
    """Get list of all backends with registered semirings.

    Returns
    -------
    list[str]
        Sorted list of unique backend identifiers.

    Examples
    --------
    >>> get_available_backends()
    ['jax', 'torch', 'numpy']
    """
    backends = {backend for _, backend in _REGISTRY.keys()}
    return sorted(backends)


def register_kernel(
    name: str,
    kernel: "AlgebraicStructure",
    backend: str = "jax",
) -> None:
    """Register a GPU-optimized kernel for a semiring.

    Parameters
    ----------
    name : str
        Canonical name for the kernel (should match semiring name).
    kernel : AlgebraicStructure
        The kernel to register.
    backend : str, optional
        Backend identifier (default: "jax").

    Raises
    ------
    ValueError
        If kernel is already registered.
    """
    key = (name, backend)
    if key in _KERNEL_REGISTRY:
        raise ValueError(
            f"Kernel '{name}' with backend '{backend}' is already registered. Use a different name or unregister first."
        )
    _KERNEL_REGISTRY[key] = kernel


def unregister_kernel(name: str, backend: str = "jax") -> None:
    """Unregister a kernel from the registry."""
    key = (name, backend)
    if key not in _KERNEL_REGISTRY:
        raise KeyError(f"Kernel '{name}' with backend '{backend}' not found.")
    del _KERNEL_REGISTRY[key]


def get_kernel(name: str, backend: str = "jax") -> "AlgebraicStructure":
    """Get a kernel from the registry.

    Parameters
    ----------
    name : str
        Canonical name for the kernel.
    backend : str, optional
        Backend identifier (default: "jax").

    Returns
    -------
    AlgebraicStructure
        The kernel.

    Raises
    ------
    KeyError
        If kernel is not found.
    """
    key = (name, backend)
    if key not in _KERNEL_REGISTRY:
        available = list_kernels(backend)
        raise KeyError(f"Kernel '{name}' with backend '{backend}' not found. Available: {available}")
    return _KERNEL_REGISTRY[key]


def list_kernels(backend: str | None = None) -> list[str]:
    """List all registered kernels, optionally filtered by backend."""
    if backend is None:
        return sorted([f"{name} ({b})" for name, b in _KERNEL_REGISTRY.keys()])
    else:
        return sorted([name for name, b in _KERNEL_REGISTRY.keys() if b == backend])


# Auto-register JAX semirings on import
def _register_jax_semirings() -> None:
    """Register all built-in JAX semirings and their kernels."""
    # Import here to avoid circular imports
    from automatix.algebra.backends.jax_ import (
        CountingSemiring,
        LatticeAlgebra,
        LeftLSEMaxMinSemiring,
        LeftMaxMinSemiring,
        LogSemiring,
        LSEMaxMinSemiring,
        MaxMinAlgebra,
        MaxMinSemiring,
        MaxPlusSemiring,
        MinPlusSemiring,
        RightLSEMaxMinSemiring,
        RightMaxMinSemiring,
    )

    semirings = {
        "Counting": CountingSemiring,
        "MaxMin": MaxMinSemiring,
        "LeftMaxMin": LeftMaxMinSemiring,
        "RightMaxMin": RightMaxMinSemiring,
        "MaxMinAlgebra": MaxMinAlgebra,
        "LSEMaxMin": LSEMaxMinSemiring,
        "LeftLSEMaxMin": LeftLSEMaxMinSemiring,
        "RightLSEMaxMin": RightLSEMaxMinSemiring,
        "MaxPlus": MaxPlusSemiring,
        "MinPlus": MinPlusSemiring,
        "Log": LogSemiring,
        "Lattice": LatticeAlgebra,
    }

    for name, semiring_class in semirings.items():
        # Register semiring classes
        register(name, "jax")(semiring_class)  # type: ignore[type-abstract]
        # Register kernels
        kernel = semiring_class.to_kernel()
        register_kernel(name, kernel, "jax")


def _register_boolean_kernels() -> None:
    """Register differentiable Boolean kernels."""
    from automatix.algebra.backends.boolean_kernels import create_boolean_kernel

    # Register soft Boolean (default for learning)
    soft_kernel = create_boolean_kernel(mode="soft")
    register_kernel("BooleanSoft", soft_kernel, "jax")

    # Register smooth Boolean with different temperatures
    smooth_kernel_t1 = create_boolean_kernel(mode="smooth", temperature=1.0)
    register_kernel("BooleanSmooth", smooth_kernel_t1, "jax")

    smooth_kernel_t10 = create_boolean_kernel(mode="smooth", temperature=10.0)
    register_kernel("BooleanSmoothSharp", smooth_kernel_t10, "jax")

    # Register straight-through estimator
    ste_kernel = create_boolean_kernel(mode="ste")
    register_kernel("BooleanSTE", ste_kernel, "jax")


# Register on import
_register_jax_semirings()
_register_boolean_kernels()

# Export public API
__all__ = [
    "register",
    "unregister",
    "get_semiring",
    "list_semirings",
    "get_available_backends",
    "register_kernel",
    "unregister_kernel",
    "get_kernel",
    "list_kernels",
]
