import typing
from collections.abc import Hashable

import jax.tree_util as jtu
from jaxtyping import PyTree

from algebraic.array.base import AlgebraicArray
from algebraic.polynomials.dok import PolyDict
from algebraic.polynomials.monomial_basis import MonomialBasis
from algebraic.polynomials.rank_decomp import RankDecomposition
from algebraic.types import AlgebraicPyTree, AnyPyTree

jtu.register_dataclass(AlgebraicArray, data_fields=("data",), meta_fields=("semiring", "_vdot", "_matmul"))


jtu.register_pytree_node_class(PolyDict)
jtu.register_pytree_node_class(MonomialBasis)
jtu.register_pytree_node_class(RankDecomposition)


def jaxify(obj: AnyPyTree) -> PyTree:
    cls = typing.cast(Hashable, type(obj))
    if issubclass(cls, AlgebraicPyTree):
        jtu.register_pytree_node_class(cls)
    return obj
