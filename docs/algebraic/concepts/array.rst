Algebraic Array
===============

:py:class:`~algebraic.array.AlgebraicArray` is the primary data structure in
``algebraic``.  It is a thin wrapper around a backend array (NumPy, JAX, or
PyTorch) that carries a semiring and overrides the standard arithmetic
operators to dispatch to that semiring's ``add`` and ``mul`` instead of the
usual ``+`` and ``*``.

The result is that you write array code in the normal style -- element-wise
operations, matrix products, reductions -- and the semiring controls what
those operations actually compute, without any changes to the calling code.

Backends
--------

Every ``AlgebraicArray`` is backed by one of three array libraries, selected
at creation time via the ``backend`` parameter:

- ``"numpy"`` -- NumPy arrays.  The default.  Eager, CPU-only, no JIT.
- ``"jax"`` -- JAX arrays.  Supports JIT compilation and ``vmap``; required
  for gradient-based use (e.g. differentiable tropical or smooth Boolean).
- ``"torch"`` -- PyTorch tensors.  Supports ``torch.compile`` and GPU.

All three backends expose the same semiring operations through
``AlgebraicArray``.  Switching backends is a one-line change at the creation
call; the arithmetic code above it is identical.

.. seealso::

   :doc:`Functional Transforms <functional>`
      JIT and vmap support for JAX and PyTorch backends.

Array Creation
--------------

Because ``AlgebraicArray`` is abstract (each backend has its own concrete
subclass), you should always create arrays through the top-level factory
functions rather than instantiating backend classes directly:

.. autofunction:: algebraic.array
   :no-index:

.. autofunction:: algebraic.zeros
   :no-index:

.. autofunction:: algebraic.zeros_like
   :no-index:

.. autofunction:: algebraic.ones
   :no-index:

.. autofunction:: algebraic.ones_like
   :no-index:

.. autofunction:: algebraic.full
   :no-index:

``algebraic.zeros`` fills the array with the semiring's additive identity
(``semiring.zero``), and ``algebraic.ones`` fills it with the multiplicative
identity (``semiring.one``).  For the min-plus tropical semiring those are
``inf`` and ``0`` respectively -- which is the right initialisation for a
distance matrix.

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring

   tropical = tropical_semiring(minplus=True)

   # A 3x3 distance matrix initialised to "no edge" (inf).
   dist = algebraic.zeros((3, 3), semiring=tropical, backend="numpy")

   # An identity matrix for tropical matrix exponentiation (0 on diagonal, inf elsewhere).
   # zeros_like copies shape/semiring/backend from an existing array.
   identity = algebraic.zeros_like(dist)

Accessing the Underlying Array
--------------------------------

The raw backend array is always available as ``.data``:

.. code-block:: python

   import algebraic
   from algebraic.semirings import max_min_algebra

   mm = max_min_algebra()
   arr = algebraic.array([1.0, -2.0, 3.0], semiring=mm, backend="numpy")

   arr.data          # numpy.ndarray([1., -2., 3.])
   arr.shape         # (3,)
   arr.ndim          # 1
   arr.dtype         # dtype('float64')

Use ``.data`` whenever you need to pass the raw values to a backend function
that does not know about ``AlgebraicArray``, or when you want to inspect
values without any semiring indirection.

Arithmetic Operators
--------------------

The four Python arithmetic operators are overloaded to use semiring semantics:

+------------+-----------------------------------+------------------------------------------+
| Operator   | Semiring operation                | Example (min-plus)                       |
+============+===================================+==========================================+
| ``a + b``  | element-wise ``semiring.add``     | ``min(a[i], b[i])``                      |
+------------+-----------------------------------+------------------------------------------+
| ``a * b``  | element-wise ``semiring.mul``     | ``a[i] + b[i]``                          |
+------------+-----------------------------------+------------------------------------------+
| ``a @ b``  | matrix product with add/mul       | :math:`C_{ij} = \min_k(A_{ik} + B_{kj})` |
+------------+-----------------------------------+------------------------------------------+
| ``-a``     | element-wise negation             | requires Ring or complement              |
+------------+-----------------------------------+------------------------------------------+
| ``a - b``  | element-wise subtraction          | requires Ring                            |
+------------+-----------------------------------+------------------------------------------+

Subtraction and negation are only defined when the semiring is a
:class:`~algebraic.spec.Ring` (which has ``additive_inverse``) or a
:class:`~algebraic.spec.DeMorganAlgebra` (which has ``complement``).
Attempting them on a plain ``Semiring`` raises ``NotImplementedError``.

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring, boolean_algebra

   tropical = tropical_semiring(minplus=True)
   a = algebraic.array([1.0, 2.0, 3.0], semiring=tropical, backend="numpy")
   b = algebraic.array([4.0, 1.0, 0.0], semiring=tropical, backend="numpy")

   a + b   # min: [1, 1, 0]
   a * b   # +  : [5, 3, 3]

   bool_alg = boolean_algebra(mode="logic")
   p = algebraic.array([True, False, True], semiring=bool_alg, backend="numpy")
   q = algebraic.array([True, True, False], semiring=bool_alg, backend="numpy")

   p + q   # OR : [True, True, True]
   p * q   # AND: [True, False, False]
   -p      # NOT: [False, True, False]

Semiring Compatibility
-----------------------

Operations between two ``AlgebraicArray`` instances require that both carry
the *same* semiring object (by identity, not just by type).  Mixing semirings
raises a ``ValueError`` at runtime.  This is a deliberate design choice: there
is no implicit coercion or promotion between semirings.

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring

   tp1 = tropical_semiring(minplus=True)
   tp2 = tropical_semiring(minplus=True)  # different object, same semantics

   a = algebraic.array([1.0], semiring=tp1, backend="numpy")
   b = algebraic.array([2.0], semiring=tp2, backend="numpy")

   a + b  # raises ValueError: semirings do not match

To avoid this, create all arrays for a computation from a single shared
semiring instance rather than calling the factory function multiple times.

Functional Index Updates
-------------------------

``AlgebraicArray`` supports functional (copy-on-write) index updates via the
``.at`` interface, modelled after JAX's ``jnp.ndarray.at``.  Each update
returns a new array without modifying the original:

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring

   tropical = tropical_semiring(minplus=True)
   arr = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=tropical, backend="numpy")

   # Overwrite a single element.
   new_arr = arr.at[1].set(0.5)       # [1.0, 0.5, 3.0, 4.0]

   # Semiring addition at index (min for tropical).
   updated = arr.at[1].add(1.5)       # [1.0, min(2.0, 1.5), 3.0, 4.0] = [1.0, 1.5, 3.0, 4.0]

   # Semiring multiplication at index (+ for tropical).
   scaled = arr.at[2].multiply(2.0)   # [1.0, 2.0, 3.0+2.0, 4.0] = [1.0, 2.0, 5.0, 4.0]

.. important::
   While this interface is idiomatic for JAX arrays (which are immutable and
   always return copies), in the NumPy and PyTorch backends ``.at`` updates
   will also return a copy of the underlying array.  If this causes a
   performance issue, file a bug report.
