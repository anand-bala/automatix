"""Backend-specific mixins for object reconstruction.

Each mixin provides `_replace_attr(name, value)` which creates a new instance
with one attribute changed, using the appropriate backend mechanism.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing

from typing_extensions import Self


class EqxReplaceMixin:
    """Mixin for JAX/equinox classes using ``eqx.tree_at``."""

    def _replace_attr(self, name: str, value: object) -> Self:
        import equinox as eqx

        return typing.cast(Self, eqx.tree_at(lambda t: getattr(t, name), self, value))


class DataclassReplaceMixin:
    """Mixin for frozen-dataclass (NumPy) classes using ``dataclasses.replace``."""

    def _replace_attr(self, name: str, value: object) -> Self:
        return dataclasses.replace(self, **{name: value})  # type: ignore[type-var]


class TorchReplaceMixin:
    """Mixin for ``torch.nn.Module`` classes that reconstruct via ``__init__``."""

    def _replace_attr(self, name: str, value: object) -> Self:
        cls = type(self)
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters if p != "self"]
        kwargs = {p: (value if p == name else getattr(self, p)) for p in params}
        return typing.cast(Self, cls(**kwargs))
