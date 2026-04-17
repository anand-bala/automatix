Array Operations
================

``algebraic`` provides a set of operations over
:py:class:`~algebraic.array.AlgebraicArray` that mirror the
`Python Array API <https://data-apis.org/array-api/latest/>`__ but replace
standard arithmetic with semiring operations wherever appropriate. The
operations fall into three groups:

1. *Semiring operations* -- reductions, contractions, and scans that use the
   semiring's ``add`` and ``mul``.
2. *Ring-only operations* -- operations that require subtraction and therefore
   need a :class:`~algebraic.spec.Ring` or :class:`~algebraic.spec.BooleanAlgebra`.
3. *Passthrough operations* -- shape manipulation (reshape, transpose, concat,
   etc.) that do not involve arithmetic and simply re-wrap the underlying
   backend array.

All operations are importable from the top-level ``algebraic`` module:

.. code-block:: python

   import algebraic

   algebraic.sum(arr)
   algebraic.matmul(a, b)
   algebraic.einsum("ij,jk->ik", a, b)

Reductions
----------

:func:`~algebraic.ops.sum` reduces an array along one or more axes using the
semiring's ``add``. :func:`~algebraic.ops.prod` does the same with ``mul``.

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring, max_min_algebra

   tropical = tropical_semiring(minplus=True)
   a = algebraic.array([3.0, 1.0, 4.0, 1.0, 5.0], semiring=tropical, backend="numpy")

   algebraic.sum(a)           # min(3, 1, 4, 1, 5) = 1.0
   algebraic.prod(a)          # 3 + 1 + 4 + 1 + 5 = 14.0

   mm = max_min_algebra()
   b = algebraic.array([[1.0, 3.0], [2.0, 0.5]], semiring=mm, backend="numpy")

   algebraic.sum(b, axis=0)   # max along rows: [max(1,2), max(3,0.5)] = [2.0, 3.0]
   algebraic.sum(b, axis=1)   # max along cols: [max(1,3), max(2,0.5)] = [3.0, 2.0]

Both functions accept ``axis`` (int or sequence of ints, or ``None`` to reduce
all axes) and ``keepdims``.

Matrix Operations
-----------------

Matrix multiply
^^^^^^^^^^^^^^^

:func:`~algebraic.ops.matmul` (also accessible as the ``@`` operator) computes
the generalised matrix product where the inner sum uses ``semiring.add`` and
the product uses ``semiring.mul``:

.. math::

   C_{ij} = \bigoplus_k A_{ik} \otimes B_{kj}

For the min-plus semiring this is the shortest-path step. For the boolean
semiring it is boolean reachability in one hop.

.. code-block:: python

   import algebraic
   from algebraic.semirings import boolean_algebra

   bool_alg = boolean_algebra(mode="logic")
   adj = algebraic.array([
       [False, True,  False],
       [False, False, True ],
       [False, False, False],
   ], semiring=bool_alg, backend="numpy")

   reach2 = adj @ adj     # two-hop reachability
   reach3 = reach2 @ adj  # three-hop reachability

:func:`~algebraic.ops.matrix_power` raises a square matrix to a non-negative
integer power using binary exponentiation (:math:`O(\log n)` multiplications):

.. code-block:: python

   reach_k = algebraic.matrix_power(adj, 5)  # up to 5-hop reachability

Tensor contractions
^^^^^^^^^^^^^^^^^^^

:func:`~algebraic.ops.tensordot` generalises matrix multiply to higher-rank
tensors. ``axes=2`` is equivalent to ``matmul`` for 2-D arrays. Explicit
axis lists ``(lhs_axes, rhs_axes)`` allow arbitrary contractions:

.. code-block:: python

   # Contract the last axis of A with the first axis of B.
   result = algebraic.tensordot(a, b, axes=1)

   # Explicit contraction: A[i,j,k] * B[k,j] summed over j and k.
   result = algebraic.tensordot(a, b, axes=([1, 2], [1, 0]))

:func:`~algebraic.ops.vecdot` computes the inner product of two arrays
contracted along a single axis (default: the last axis):

.. code-block:: python

   dot = algebraic.vecdot(a, b)          # semiring inner product
   dot = algebraic.vecdot(a, b, axis=0)  # contract along axis 0

:func:`~algebraic.ops.outer` computes the outer product of two 1-D arrays
using semiring multiplication. The result has shape
``(a.shape[0], b.shape[0])``:

.. code-block:: python

   mat = algebraic.outer(a, b)  # mat[i, j] = a[i] * b[j]

:func:`~algebraic.ops.trace` sums the diagonal using semiring addition:

.. code-block:: python

   t = algebraic.trace(square_matrix)

Einstein summation
^^^^^^^^^^^^^^^^^^

:func:`~algebraic.ops.einsum` evaluates arbitrary tensor contractions
specified by an einsum string, using ``opt_einsum`` to find an efficient
contraction path and executing each pairwise step with semiring operations.

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring

   tropical = tropical_semiring(minplus=True)
   A = algebraic.array([[0.0, 1.0], [2.0, 3.0]], semiring=tropical, backend="numpy")
   B = algebraic.array([[4.0, 5.0], [6.0, 7.0]], semiring=tropical, backend="numpy")

   # Tropical matrix multiply via einsum.
   C = algebraic.einsum("ij,jk->ik", A, B)

   # Batch matrix multiply.
   C_batch = algebraic.einsum("bij,bjk->bik", batch_A, batch_B)

   # Element-wise product then sum (dot product semantics).
   scalar = algebraic.einsum("i,i->", a, b)

``einsum`` is the most flexible contraction interface. Use it when
``matmul``, ``tensordot``, or ``vecdot`` would require awkward reshaping.

Prefix Scans
------------

:func:`~algebraic.ops.cumulative_sum` and
:func:`~algebraic.ops.cumulative_prod` compute inclusive prefix scans using
``semiring.add`` and ``semiring.mul`` respectively:

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring

   tropical = tropical_semiring(minplus=True)
   a = algebraic.array([3.0, 1.0, 4.0, 1.0], semiring=tropical, backend="numpy")

   # Cumulative min (running best so far).
   algebraic.cumulative_sum(a)
   # [3.0, min(3,1), min(3,1,4), min(3,1,4,1)] = [3.0, 1.0, 1.0, 1.0]

   # Cumulative + (running total cost).
   algebraic.cumulative_prod(a)
   # [3.0, 3+1, 3+1+4, 3+1+4+1] = [3.0, 4.0, 8.0, 9.0]

Both functions accept an ``axis`` argument (default 0) and an
``include_initial`` flag. When ``include_initial=True`` the output length
along the scan axis is ``n + 1``, with the semiring identity prepended.

Ring-Only Operations
--------------------

Some operations are only meaningful when the underlying semiring supports
subtraction. ``algebraic`` checks for this at call time using
:func:`~algebraic.spec.is_ring`.

:func:`~algebraic.ops.subtract` (also ``a - b``) requires a
:class:`~algebraic.spec.Ring` or :class:`~algebraic.spec.BooleanAlgebra`:

.. code-block:: python

   from algebraic.semirings import boolean_algebra

   bool_alg = boolean_algebra(mode="soft")
   p = algebraic.array([0.9, 0.2], semiring=bool_alg, backend="numpy")
   q = algebraic.array([0.4, 0.1], semiring=bool_alg, backend="numpy")

   diff = p - q   # soft Boolean subtraction (p + complement(q))

:func:`~algebraic.ops.diff` computes finite differences along an axis,
equivalent to ``x[1:] - x[:-1]``, with optional ``prepend``/``append``
values and an order parameter ``n``:

.. code-block:: python

   import algebraic
   from algebraic.semirings import boolean_algebra

   bool_alg = boolean_algebra(mode="soft")
   signal = algebraic.array([0.1, 0.5, 0.9, 0.6], semiring=bool_alg, backend="numpy")

   # First-order difference.
   d = algebraic.diff(signal)          # shape (3,)

   # Second-order difference.
   d2 = algebraic.diff(signal, n=2)    # shape (2,)

Passthrough Operations
----------------------

Shape manipulation operations do not touch semiring arithmetic; they re-wrap
the transformed backend array with the same semiring. These mirror the Array
API specification:

- ``algebraic.reshape(x, shape)``
- ``algebraic.permute_dims(x, axes)`` (transpose generalisation)
- ``algebraic.concat(arrays, axis=0)``
- ``algebraic.stack(arrays, axis=0)``
- ``algebraic.expand_dims(x, axis)``
- ``algebraic.squeeze(x, axis=None)``
- ``algebraic.flip(x, axis=None)``
- ``algebraic.roll(x, shift, axis=None)``
- ``algebraic.diagonal(x)`` / ``algebraic.moveaxis(x, source, destination)``
- ``algebraic.broadcast_to(x, shape)``
- ``algebraic.broadcast_arrays(*arrays)``
- ``algebraic.where(condition, x, y)``

For operations that take multiple ``AlgebraicArray`` arguments (e.g. ``concat``,
``stack``), all inputs must share the same semiring instance.

.. code-block:: python

   import algebraic
   from algebraic.semirings import tropical_semiring

   tropical = tropical_semiring(minplus=True)
   a = algebraic.array([[1.0, 2.0], [3.0, 4.0]], semiring=tropical, backend="numpy")

   b = algebraic.permute_dims(a, (1, 0))  # transpose: [[1,3],[2,4]]
   c = algebraic.reshape(a, (4,))         # flatten: [1, 2, 3, 4]
   d = algebraic.flip(a, axis=1)          # reverse cols: [[2,1],[4,3]]
