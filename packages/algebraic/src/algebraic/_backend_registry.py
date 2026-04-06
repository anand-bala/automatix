"""Lazy-import registry for backend-specific class dispatch."""

from __future__ import annotations

from collections.abc import Callable

from algebraic.types import Backend

# A provider is a no-arg callable that returns a class (lazy import).
type _Provider = Callable[[], type]


class BackendClassRegistry:
    """Maps ``Backend`` enum values to lazily-imported concrete classes.

    Parameters
    ----------
    name : str
        Human-readable name for error messages (e.g. ``"RankDecomposition"``).

    Examples
    --------
    >>> registry = BackendClassRegistry("RankDecomposition")
    >>> registry.register(Backend.NUMPY, lambda: NumpyRankDecomposition)
    >>> cls = registry[Backend.NUMPY]
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._providers: dict[Backend, _Provider] = {}

    def register(self, backend: Backend, provider: _Provider) -> None:
        """Register a lazy provider for a backend."""
        self._providers[backend] = provider

    def __getitem__(self, backend: str | Backend) -> type:
        """Look up and return the class for *backend*, importing lazily."""
        backend = Backend(backend)
        provider = self._providers.get(backend)
        if provider is None:
            raise ValueError(
                f"No {self._name} implementation registered for backend {backend!r}. "
                f"Registered: {sorted(b.value for b in self._providers)}"
            )
        return provider()
