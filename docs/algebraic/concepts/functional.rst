Functional Transforms
=====================

When ``algebraic`` computations are used inside a training loop or a
tight numerical simulation, two transforms matter most: *JIT compilation*
(trace and compile a function for fast repeated execution) and *batching*
(apply a function independently over a leading batch dimension).  Both are
available for the JAX and PyTorch backends.

These transforms live in ``algebraic._jax_wrappers`` and are imported
directly from there.  The module name reflects its origins but the wrappers
support both JAX and PyTorch; the ``backend`` argument selects which.

.. note::
   ``algebraic._jax_wrappers`` is a semi-internal module -- the leading
   underscore signals that its API may change.  If you depend on it, pin
   the version.  A stable public path is planned for a future release.

JIT Compilation
---------------

Wrapping a function with ``jit`` traces it once on the first call and then
executes the compiled version on all subsequent calls.  This removes Python
overhead and enables backend-specific optimisations (XLA fusion for JAX,
``torch.compile`` for PyTorch).

.. code-block:: python

   from algebraic._jax_wrappers import jit
   import algebraic
   from algebraic.semirings import tropical_semiring

   tropical = tropical_semiring(minplus=True)

   @jit(backend="jax")
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

For JAX, ``jit`` wraps ``jax.jit``.  For PyTorch, it wraps
``torch.compile``.  The NumPy backend does not support JIT; passing
``backend="numpy"`` is a no-op (the function runs eagerly).

.. important::
   JIT-compiled functions must not contain Python control flow that depends
   on array *values* (shapes and dtypes are fine).  This is a JAX constraint
   that also applies here.  Loops over a fixed number of steps (as in the
   example above) are safe; conditionals on ``arr.data > 0`` are not.

Batching with vmap
-------------------

``vmap`` lifts a function that operates on a single example into one that
operates on a batch, without writing explicit loops or reshaping.  Under JAX
this compiles to vectorised hardware instructions.

.. code-block:: python

   from algebraic._jax_wrappers import vmap
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

For PyTorch, ``vmap`` wraps ``torch.vmap``.  Support is present but not
fully tested for all operation combinations; prefer JAX if batching is a
primary concern.

Combining jit and vmap
-----------------------

The two transforms compose: ``jit(vmap(f))`` first vectorises ``f`` over the
batch dimension, then compiles the vectorised version.  This is the standard
pattern for training loops:

.. code-block:: python

   from algebraic._jax_wrappers import jit, vmap

   fast_batched = jit(vmap(my_semiring_fn, backend="jax"), backend="jax")

Backend Support Summary
-----------------------

+----------------+------------------+--------------------+
| Backend        | JIT              | vmap               |
+================+==================+====================+
| ``"jax"``      | ``jax.jit``      | ``jax.vmap``       |
+----------------+------------------+--------------------+
| ``"torch"``    | ``torch.compile``| ``torch.vmap``     |
+----------------+------------------+--------------------+
| ``"numpy"``    | no-op (eager)    | not supported      |
+----------------+------------------+--------------------+

If your semiring computation is prototype-stage and you are iterating
quickly, start with ``backend="numpy"`` for easy debugging.  Once the logic
is correct, switch to ``backend="jax"`` and add ``@jit`` to recover
performance.
