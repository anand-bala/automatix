import typing
from collections.abc import Hashable

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree

from algebraic._better_abc import BetterABCMeta
from algebraic.array.base import AlgebraicArray
from algebraic.types import AlgebraicPyTree, AnyPyTree


class EqxMeta(type(eqx.Module), BetterABCMeta):  # type: ignore[misc]
    """Combined metaclass resolving the conflict between equinox's ``_ModuleMeta`` and ``BetterABCMeta``.

    Both are subclasses of ``abc.ABCMeta`` but neither is a subclass of the other,
    so a combined metaclass is required.
    """

    pass


jtu.register_dataclass(AlgebraicArray, data_fields=("data",), meta_fields=("semiring", "_vdot", "_matmul"))


def jaxify(obj: AnyPyTree) -> PyTree:
    cls = typing.cast(Hashable, type(obj))
    if issubclass(cls, AlgebraicPyTree):
        jtu.register_pytree_node_class(cls)
    return obj
