Functional Transforms
=====================

When ``algebraic`` computations are used inside a training loop or a
tight numerical simulation, two transforms matter most: *JIT compilation*
(trace and compile a function for fast repeated execution) and *batching*
(apply a function independently over a leading batch dimension).

JIT compilation uses each backend's native API directly (``jax.jit``,
``torch.compile``). Batching is available through ``algebraic.vmap``,
which delegates to the appropriate backend.

JIT Compilation
---------------

``AlgebraicArray`` is registered as a JAX PyTree (via
``algebraic.utils.jax``), so ``jax.jit`` works out of the box. Import
the utility module once for its registration side-effect, then use
``jax.jit`` as usual:

.. code-block:: python

   import jax
   import algebraic
   import algebraic.utils.jax  # registers AlgebraicArray as a JAX PyTree
   from algebraic.semirings import tropical_semiring

   tropical = tropical_semiring(minplus=True)

   @jax.jit
   def all_pairs_shortest_paths(dist):
       """Floyd-Warshall via repeated tropical matrix multiplication."""
       n = dist.shape[0]
       result = dist
       for _ in range(n - 1):
           result = result + (result @ dist)
       return result

   # First call traces and compiles; subsequent calls are fast.
   dist = algebraic.zeros((8, 8), semiring=tropical, backend="jax")
   paths = all_pairs_shortest_paths(dist)

For PyTorch, use ``torch.compile`` directly.

.. important::
   JIT-compiled functions must not contain Python control flow that depends
   on array *values* (shapes and dtypes are fine). This is a JAX constraint
   that also applies here. Loops over a fixed number of steps (as in the
   example above) are safe; conditionals on ``arr.data > 0`` are not.

Batching with vmap
-------------------

``vmap`` lifts a function that operates on a single example into one that
operates on a batch, without writing explicit loops or reshaping. Under JAX
this compiles to vectorised hardware instructions.

.. code-block:: python

   from algebraic import vmap
   import algebraic
   from algebraic.semirings import boolean_algebra

   bool_alg = boolean_algebra(mode="soft")

   def single_matmul(a, b):
       """Multiply two matrices."""
       return a @ b

   # Lift to batch dimension 0 for both arguments.
   batched_matmul = vmap(single_matmul, backend="jax")

   batch_size = 16
   A = algebraic.zeros((batch_size, 4, 4), semiring=bool_alg, backend="jax")
   B = algebraic.ones( (batch_size, 4, 4), semiring=bool_alg, backend="jax")

   C = batched_matmul(A, B)   # shape: (batch_size, 4, 4)

For PyTorch, ``vmap`` wraps ``torch.vmap``, automatically handling the
conversion between ``AlgebraicArray`` and raw tensors.

Combining jit and vmap
-----------------------

The two transforms compose: ``jit(vmap(f))`` first vectorises ``f`` over the
batch dimension, then compiles the vectorised version. This is the standard
pattern for training loops:

.. code-block:: python

   import jax
   import algebraic.utils.jax
   from algebraic import vmap

   fast_batched = jax.jit(vmap(my_semiring_fn, backend="jax"))

Backend Support Summary
-----------------------

+----------------+--------------------+--------------------+
| Backend        | JIT                | vmap               |
+================+====================+====================+
| ``"jax"``      | ``jax.jit``        | ``jax.vmap``       |
+----------------+--------------------+--------------------+
| ``"torch"``    | ``torch.compile``  | ``torch.vmap``     |
+----------------+--------------------+--------------------+
| ``"numpy"``    | not applicable     | not supported      |
+----------------+--------------------+--------------------+

If your semiring computation is prototype-stage and you are iterating
quickly, start with ``backend="numpy"`` for easy debugging. Once the logic
is correct, switch to ``backend="jax"`` and add ``@jax.jit`` to recover
performance.
