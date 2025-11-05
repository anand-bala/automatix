"""Semiring registry and factory for discovering and instantiating semirings.

This module provides a centralized registry for managing semiring implementations
across different backends (JAX, PyTorch, NumPy).
"""

from typing import Callable, Type

from automatix.algebra.spec import AbstractSemiring

# Type for semiring classes
SemiringType = Type[AbstractSemiring]

# Registry mapping: (name, backend) -> semiring_class
_REGISTRY: dict[tuple[str, str], SemiringType] = {}


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


# Auto-register JAX semirings on import
def _register_jax_semirings() -> None:
    """Register all built-in JAX semirings."""
    # Import here to avoid circular imports
    from automatix.algebra.semiring.jax_backend import (
        CountingSemiring,
        LatticeAlgebra,
        LeftLSEMaxMinSemiring,
        LeftMaxMinSemiring,
        LogSemiring,
        LSEMaxMinSemiring,
        MaxMinAlgebra,
        MaxMinSemiring,
        MaxPlusSemiring,
        RightLSEMaxMinSemiring,
        RightMaxMinSemiring,
    )

    register("Counting", "jax")(CountingSemiring)
    register("MaxMin", "jax")(MaxMinSemiring)
    register("LeftMaxMin", "jax")(LeftMaxMinSemiring)
    register("RightMaxMin", "jax")(RightMaxMinSemiring)
    register("MaxMinAlgebra", "jax")(MaxMinAlgebra)
    register("LSEMaxMin", "jax")(LSEMaxMinSemiring)
    register("LeftLSEMaxMin", "jax")(LeftLSEMaxMinSemiring)
    register("RightLSEMaxMin", "jax")(RightLSEMaxMinSemiring)
    register("MaxPlus", "jax")(MaxPlusSemiring)
    register("Log", "jax")(LogSemiring)
    register("Lattice", "jax")(LatticeAlgebra)


# Register on import
_register_jax_semirings()

# Export public API
__all__ = [
    "register",
    "unregister",
    "get_semiring",
    "list_semirings",
    "get_available_backends",
]
