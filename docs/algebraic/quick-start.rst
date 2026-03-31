Quick Start
===========

Recommended Import
------------------

.. code-block:: python

   import algebraic

The top-level ``algebraic`` module re-exports all array operations, semiring
specifications, and polynomial types.

Basic Semiring Operations
-------------------------

.. code-block:: python

   from algebraic.semirings import tropical_semiring, max_min_algebra, boolean_algebra

   # Tropical semiring (MaxPlus: max is addition, + is multiplication)
   maxplus = tropical_semiring(minplus=False)
   a = maxplus.add(2.0, 3.0)  # max(2, 3) = 3
   b = maxplus.mul(2.0, 3.0)  # 2 + 3 = 5

   # Tropical semiring (MinPlus: min is addition, + is multiplication)
   minplus = tropical_semiring(minplus=True)  # or just tropical_semiring()
   c = minplus.add(2.0, 3.0)  # min(2, 3) = 2
   d = minplus.mul(2.0, 3.0)  # 2 + 3 = 5

   # Max-Min algebra (for robustness/STL semantics)
   maxmin = max_min_algebra()
   e = maxmin.add(-0.5, 0.2)  # max(-0.5, 0.2) = 0.2
   f = maxmin.mul(-0.5, 0.2)  # min(-0.5, 0.2) = -0.5

   # Boolean algebra
   bool_alg = boolean_algebra(mode="logic")
   true = bool_alg.one
   false = bool_alg.zero
   result = bool_alg.add(true, false)  # True OR False = True

AlgebraicArray: Arrays with Semiring Semantics
-----------------------------------------------

The ``AlgebraicArray`` class wraps backend arrays and overrides arithmetic operations to
use semiring semantics.

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring

   # Create algebraic arrays with tropical semiring
   tropical = tropical_semiring(minplus=True)
   a = algebraic.array([1.0, 2.0, 3.0], semiring=tropical, backend="numpy")
   b = algebraic.array([4.0, 5.0, 6.0], semiring=tropical, backend="numpy")

   # Element-wise operations use semiring semantics
   c = a + b  # Tropical addition: [min(1,4), min(2,5), min(3,6)] = [1, 2, 3]
   d = a * b  # Tropical multiplication: [1+4, 2+5, 3+6] = [5, 7, 9]

   # Reductions use semiring operations
   total = algebraic.sum(a)  # min(1, 2, 3) = 1
   product = algebraic.prod(a)  # 1 + 2 + 3 = 6

   # Matrix multiplication with @ operator
   A = algebraic.array([[1.0, 2.0], [3.0, 4.0]], semiring=tropical, backend="numpy")
   B = algebraic.array([[5.0, 6.0], [7.0, 8.0]], semiring=tropical, backend="numpy")
   C = A @ B  # Tropical matmul: C[i,j] = min_k(A[i,k] + B[k,j])
   # Result: [[6, 7], [8, 9]]

Boolean Algebra for Graph and Logic Operations
-----------------------------------------------

.. code-block:: python

   import algebraic
   from algebraic.semirings import boolean_algebra

   # Boolean algebra for reachability
   bool_alg = boolean_algebra(mode="logic")

   # Adjacency matrix: edge from i to j
   adj = algebraic.array([
       [False, True,  False],
       [False, False, True],
       [True,  False, False]
   ], semiring=bool_alg, backend="numpy")

   # Matrix multiplication computes 2-step reachability
   reach_2 = adj @ adj
   # reach_2[i,j] = True if there's a path of length 2 from i to j

   # Transitive closure: adj + adj^2 + adj^3 + ...
   reach = adj
   for _ in range(3):
       reach = reach + (reach @ adj)
   # reach[i,j] = True if there's any path from i to j

Smooth Boolean Operations for Learning
---------------------------------------

.. code-block:: python

   import algebraic
   from algebraic.semirings import boolean_algebra

   # Differentiable boolean operations for neural networks
   smooth_bool = boolean_algebra(mode="smooth", temperature=10.0)
   soft_bool = boolean_algebra(mode="soft")

   # Example: Soft logical operations on continuous values
   x = algebraic.array([0.9, 0.8, 0.1], semiring=soft_bool, backend="numpy")
   y = algebraic.array([0.7, 0.3, 0.2], semiring=soft_bool, backend="numpy")

   # Soft AND: element-wise multiplication
   z_and = x * y  # [0.63, 0.24, 0.02]

   # Soft OR: probabilistic OR formula
   z_or = x + y  # [0.97, 0.86, 0.28]

