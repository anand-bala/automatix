"""Multi-backend semiring algebra library.

The :mod:`algebraic` module is the main entrypoint for the package. It
re-exports all array operations, semiring specifications, and polynomial
types.

Supported backends: NumPy, JAX, and PyTorch.

Examples
--------
>>> import algebraic
>>> sr = algebraic.semirings.tropical_semiring(minplus=True)
>>> a = algebraic.array([1.0, 2.0, 3.0], semiring=sr, backend="numpy")
>>> b = algebraic.array([4.0, 5.0, 6.0], semiring=sr, backend="numpy")
>>> c = a + b  # tropical add: [min(1,4), min(2,5), min(3,6)]
"""

# Re-export the submodules
from algebraic import polynomials as polynomials
from algebraic import semirings as semirings

# Re-export some base datastructures and helper functions
from algebraic.array import AlgebraicArray as AlgebraicArray

# Re-export all the Array API-like operations
from algebraic.ops import add as add
from algebraic.ops import allclose as allclose
from algebraic.ops import array as array
from algebraic.ops import broadcast_arrays as broadcast_arrays
from algebraic.ops import broadcast_to as broadcast_to
from algebraic.ops import concat as concat
from algebraic.ops import cumulative_prod as cumulative_prod
from algebraic.ops import cumulative_sum as cumulative_sum
from algebraic.ops import diagonal as diagonal
from algebraic.ops import diff as diff
from algebraic.ops import einsum as einsum
from algebraic.ops import equal as equal
from algebraic.ops import expand_dims as expand_dims
from algebraic.ops import eye as eye
from algebraic.ops import flip as flip
from algebraic.ops import full as full
from algebraic.ops import isclose as isclose
from algebraic.ops import matmul as matmul
from algebraic.ops import matrix_power as matrix_power
from algebraic.ops import matrix_transpose as matrix_transpose
from algebraic.ops import moveaxis as moveaxis
from algebraic.ops import multiply as multiply
from algebraic.ops import negative as negative
from algebraic.ops import not_equal as not_equal
from algebraic.ops import ones as ones
from algebraic.ops import ones_like as ones_like
from algebraic.ops import outer as outer
from algebraic.ops import permute_dims as permute_dims
from algebraic.ops import positive as positive
from algebraic.ops import prod as prod
from algebraic.ops import repeat as repeat
from algebraic.ops import reshape as reshape
from algebraic.ops import roll as roll
from algebraic.ops import square as square
from algebraic.ops import squeeze as squeeze
from algebraic.ops import stack as stack
from algebraic.ops import subtract as subtract
from algebraic.ops import sum as sum
from algebraic.ops import take as take
from algebraic.ops import take_along_axis as take_along_axis
from algebraic.ops import tensordot as tensordot
from algebraic.ops import tile as tile
from algebraic.ops import trace as trace
from algebraic.ops import unique_all as unique_all
from algebraic.ops import unique_counts as unique_counts
from algebraic.ops import unique_inverse as unique_inverse
from algebraic.ops import unique_values as unique_values
from algebraic.ops import unstack as unstack
from algebraic.ops import vecdot as vecdot
from algebraic.ops import where as where
from algebraic.ops import zeros as zeros
from algebraic.ops import zeros_like as zeros_like

# Re-export specifications and helper functions
from algebraic.spec import AlgebraicStructure as AlgebraicStructure
from algebraic.spec import BooleanAlgebra as BooleanAlgebra
from algebraic.spec import BoundedDistributiveLattice as BoundedDistributiveLattice
from algebraic.spec import DeMorganAlgebra as DeMorganAlgebra
from algebraic.spec import HeytingAlgebra as HeytingAlgebra
from algebraic.spec import Ring as Ring
from algebraic.spec import Semiring as Semiring
from algebraic.spec import StoneAlgebra as StoneAlgebra
from algebraic.spec import has_complement as has_complement
from algebraic.spec import is_demorgan_algebra as is_demorgan_algebra
from algebraic.spec import is_heyting_algebra as is_heyting_algebra
from algebraic.spec import is_ring as is_ring
from algebraic.spec import is_stone_algebra as is_stone_algebra

# Reexport functional transforms
from algebraic.transforms import vmap as vmap
