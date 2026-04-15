import jax.tree_util as jtu
from jaxtyping import PyTree

from algebraic.array.base import AlgebraicArray
from algebraic.polynomials.dok import PolyDict
from algebraic.polynomials.monomial_basis import MonomialBasis
from algebraic.polynomials.rank_decomp import LowRankFactors, RankDecomposition
from algebraic.types import AlgebraicPyTree, AnyPyTree

jtu.register_pytree_node_class(AlgebraicArray)
jtu.register_pytree_node_class(PolyDict)
jtu.register_pytree_node_class(MonomialBasis)
jtu.register_pytree_node_class(RankDecomposition)
jtu.register_pytree_node_class(LowRankFactors)


def jaxify(obj: AnyPyTree) -> PyTree:
    cls = type(obj)
    if issubclass(cls, AlgebraicPyTree):
        jtu.register_pytree_node_class(cls)
    return obj
