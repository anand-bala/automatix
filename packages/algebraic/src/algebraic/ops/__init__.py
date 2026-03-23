"""Backend-agnostic array operations for `AlgebraicArray`.

This package provides two categories of operations:

* **Semiring ops** (`algebraic.ops._semiring_ops`): operations that use
  the semiring's `add` / `mul` in place of standard arithmetic
  (`sum`, `prod`, `matmul`, `cumulative_sum`, etc.).

* **Pass-through ops** (`algebraic.ops._passthrough`): operations that
  delegate to the underlying data array via `array_api_compat` and
  re-wrap the result (`reshape`, `stack`, `where`, etc.).
"""

from algebraic.ops._creation import array as array
from algebraic.ops._creation import full as full
from algebraic.ops._creation import ones as ones
from algebraic.ops._creation import ones_like as ones_like
from algebraic.ops._creation import zeros as zeros
from algebraic.ops._creation import zeros_like as zeros_like
from algebraic.ops._passthrough import UniqueAllResult as UniqueAllResult
from algebraic.ops._passthrough import UniqueCountsResult as UniqueCountsResult
from algebraic.ops._passthrough import UniqueInverseResult as UniqueInverseResult
from algebraic.ops._passthrough import broadcast_arrays as broadcast_arrays
from algebraic.ops._passthrough import broadcast_to as broadcast_to
from algebraic.ops._passthrough import concat as concat
from algebraic.ops._passthrough import diagonal as diagonal
from algebraic.ops._passthrough import equal as equal
from algebraic.ops._passthrough import expand_dims as expand_dims
from algebraic.ops._passthrough import flip as flip
from algebraic.ops._passthrough import matrix_transpose as matrix_transpose
from algebraic.ops._passthrough import moveaxis as moveaxis
from algebraic.ops._passthrough import not_equal as not_equal
from algebraic.ops._passthrough import permute_dims as permute_dims
from algebraic.ops._passthrough import positive as positive
from algebraic.ops._passthrough import repeat as repeat
from algebraic.ops._passthrough import reshape as reshape
from algebraic.ops._passthrough import roll as roll
from algebraic.ops._passthrough import squeeze as squeeze
from algebraic.ops._passthrough import stack as stack
from algebraic.ops._passthrough import take as take
from algebraic.ops._passthrough import take_along_axis as take_along_axis
from algebraic.ops._passthrough import tile as tile
from algebraic.ops._passthrough import unique_all as unique_all
from algebraic.ops._passthrough import unique_counts as unique_counts
from algebraic.ops._passthrough import unique_inverse as unique_inverse
from algebraic.ops._passthrough import unique_values as unique_values
from algebraic.ops._passthrough import unstack as unstack
from algebraic.ops._passthrough import where as where
from algebraic.ops._semiring_ops import add as add
from algebraic.ops._semiring_ops import cumulative_prod as cumulative_prod
from algebraic.ops._semiring_ops import cumulative_sum as cumulative_sum
from algebraic.ops._semiring_ops import diff as diff
from algebraic.ops._semiring_ops import matmul as matmul
from algebraic.ops._semiring_ops import matrix_power as matrix_power
from algebraic.ops._semiring_ops import multiply as multiply
from algebraic.ops._semiring_ops import negative as negative
from algebraic.ops._semiring_ops import outer as outer
from algebraic.ops._semiring_ops import prod as prod
from algebraic.ops._semiring_ops import square as square
from algebraic.ops._semiring_ops import subtract as subtract
from algebraic.ops._semiring_ops import sum as sum
from algebraic.ops._semiring_ops import tensordot as tensordot
from algebraic.ops._semiring_ops import trace as trace
from algebraic.ops._semiring_ops import vecdot as vecdot
