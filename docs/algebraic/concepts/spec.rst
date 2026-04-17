Algebraic Structures
====================

The central abstraction in ``algebraic`` is the *semiring*, a simple data
object that bundles two binary operations, their identities, and optional
properties that downstream code can query at runtime. Almost every function
in the library accepts a semiring (or one of its richer sub-types) as a
parameter, and that parameter controls how arithmetic is performed.

The Semiring
------------

A semiring is the pair :math:`(S, \oplus, \otimes, 0, 1)` where:

- :math:`\oplus` (``add``) is the semiring *addition*, which must be
  associative, commutative, and have :math:`0` as its identity.
- :math:`\otimes` (``mul``) is the semiring *multiplication*, which must be
  associative, distribute over :math:`\oplus`, and have :math:`1` as its
  identity.
- :math:`0` (``zero``) absorbs under multiplication: :math:`0 \otimes a = 0`.

In ``algebraic`` a semiring is an instance of :class:`~algebraic.spec.Semiring`:

.. code-block:: python

   from algebraic.spec import Semiring

   # Build a semiring from plain Python functions.
   import math

   minplus = Semiring(
       add=min,
       mul=lambda a, b: a + b,
       zero=math.inf,
       one=0.0,
   )

In practice you will rarely construct semirings by hand. The factory functions
in :mod:`algebraic.semirings` cover the most common cases and return properly
typed objects with all properties set.

Built-in Semirings
------------------

The library ships four semiring families. Choosing the right one is the first
decision when using ``algebraic``.

Counting semiring
^^^^^^^^^^^^^^^^^

The *counting semiring* :math:`(\mathbb{R}, +, \times, 0, 1)` uses standard
addition and multiplication.

.. code-block:: python

   from algebraic.semirings import counting_semiring

   sr = counting_semiring()
   sr.add(2.0, 3.0)   # 5.0
   sr.mul(2.0, 3.0)   # 6.0

Tropical semiring
^^^^^^^^^^^^^^^^^

The *tropical semiring* replaces addition with ``min`` (or ``max``) and
multiplication with ordinary ``+``. Tropical matrix multiplication therefore
computes shortest (or longest) paths in a single step.

.. code-block:: python

   from algebraic.semirings import tropical_semiring

   minplus = tropical_semiring(minplus=True)   # (min, +, inf, 0)
   maxplus = tropical_semiring(minplus=False)  # (max, +, -inf, 0)

   minplus.add(2.0, 3.0)  # min(2, 3) = 2
   minplus.mul(2.0, 3.0)  # 2 + 3 = 5

Both variants also accept a ``smooth=True`` flag, which replaces the sharp
``min``/``max`` with a soft log-sum-exp approximation controlled by
``temperature``. Smooth mode keeps the operations differentiable, which is
useful when tropical computations feed into a gradient-based optimizer.

Max-Min algebra
^^^^^^^^^^^^^^^

The *max-min algebra* uses ``max`` as addition and ``min`` as multiplication
over the extended reals. This is the natural semiring for robustness semantics
such as Signal Temporal Logic (STL): the "best" satisfaction value of a
conjunction of constraints is the minimum of the individual values, and the
satisfaction over a disjunction is the maximum.

.. code-block:: python

   from algebraic.semirings import max_min_algebra

   mm = max_min_algebra()
   mm.add(-0.5, 0.2)  # max(-0.5, 0.2) = 0.2
   mm.mul(-0.5, 0.2)  # min(-0.5, 0.2) = -0.5

By default ``max_min_algebra()`` returns a
:class:`~algebraic.spec.DeMorganAlgebra` where negation is ``-x``, so
:math:`\neg a` corresponds to flipping the sign of a robustness value. If you
only need one half of the real line (e.g., non-negative robustness), pass
``only="positive"`` or ``only="negative"`` to obtain a simpler
:class:`~algebraic.spec.BoundedDistributiveLattice`. Like the tropical
semiring, ``smooth=True`` is available for gradient-friendly approximations.

Boolean algebra
^^^^^^^^^^^^^^^

``boolean_algebra()`` returns a :class:`~algebraic.spec.BooleanAlgebra`
where addition is OR and multiplication is AND. The ``mode`` parameter
selects how the operations behave, which matters for differentiability:

- ``"logic"`` -- exact boolean operations (non-differentiable).
- ``"soft"`` -- probabilistic approximation using arithmetic
  (:math:`a \lor b \approx a + b - ab`); smooth and fast.
- ``"smooth"`` -- sigmoid-based; controlled by ``temperature``.
- ``"ste"`` / ``"std-fuzzy"`` -- standard fuzzy / Straight-Through Estimator,
  using ``max``/``min`` with :math:`1 - x` as complement.

.. code-block:: python

   from algebraic.semirings import boolean_algebra

   logic  = boolean_algebra(mode="logic")   # exact
   soft   = boolean_algebra(mode="soft")    # smooth, differentiable
   smooth = boolean_algebra(mode="smooth", temperature=10.0)

The Algebraic Structure Hierarchy
----------------------------------

:class:`~algebraic.spec.Semiring` is the base. The hierarchy extends it
toward richer structures, each adding operations or enforcing laws:

.. code-block:: text

   Semiring
   |-- Ring                          (adds additive_inverse)
   |-- BoundedDistributiveLattice    (idempotent add and mul; has top/bottom)
       |-- DeMorganAlgebra           (adds complement; De Morgan's laws)
       |   |-- BooleanAlgebra        (also a Ring; law of excluded middle)
       |-- HeytingAlgebra            (adds implication; pseudo-complement)
       |-- StoneAlgebra              (adds pseudo-complement; Stone's law)

A :class:`~algebraic.spec.BoundedDistributiveLattice` exposes ``join``/``meet``
aliases for ``add``/``mul`` and ``top``/``bottom`` aliases for ``one``/``zero``.
These are purely for readability; the underlying objects are identical.

:class:`~algebraic.spec.BooleanAlgebra` satisfies the contracts of Ring,
HeytingAlgebra, and StoneAlgebra simultaneously. Passing a ``BooleanAlgebra``
wherever any of those are expected is always safe.

Runtime Type Guards
-------------------

When writing code that must behave differently depending on whether a semiring
has subtraction or complementation, use the type guards in
:mod:`algebraic.spec`:

.. code-block:: python

   from algebraic.spec import is_ring, has_complement, is_boolean_algebra

   def negate_if_possible(algebra, value):
       if has_complement(algebra):
           return algebra.complement(value)
       if is_ring(algebra):
           return algebra.additive_inverse(value)
       raise TypeError("semiring does not support negation")

Available guards: :func:`~algebraic.spec.is_ring`,
:func:`~algebraic.spec.is_demorgan_algebra`,
:func:`~algebraic.spec.is_heyting_algebra`,
:func:`~algebraic.spec.is_stone_algebra`,
:func:`~algebraic.spec.is_boolean_algebra`,
:func:`~algebraic.spec.has_complement`.

Algebraic Properties
--------------------

Semiring instances carry a ``properties`` set that downstream code can query:

.. code-block:: python

   minplus = tropical_semiring(minplus=True)
   minplus.is_idempotent_add()   # True  (min(a, a) = a)
   minplus.is_commutative()      # True
   minplus.is_idempotent_mul()   # False (+  is not idempotent)

   counting = counting_semiring()
   counting.is_idempotent_add()  # False

Properties do not enforce invariants; they are hints. When you build a custom
semiring, populate ``properties`` honestly so that any code that branches on
these flags gets the right result.

Defining a Custom Semiring
--------------------------

Any plain Python callable satisfying the semiring axioms can be wrapped. The
cleanest way to build a new semiring is usually as a *product* of two existing
ones, where addition and multiplication each apply component-wise.

The example below combines the Boolean semiring (OR, AND) with the min-plus
tropical semiring (min, +) into a single structure that simultaneously tracks
reachability (bool) and shortest-path cost (float):

.. code-block:: python

   import math
   from algebraic.spec import Semiring

   def reach_cost_add(a, b):
       # OR for reachability; min for cost.
       return (a[0] or b[0], min(a[1], b[1]))

   def reach_cost_mul(a, b):
       # AND for reachability; + for cost (tropical multiplication).
       return (a[0] and b[0], a[1] + b[1])

   reach_cost = Semiring(
       add=reach_cost_add,
       mul=reach_cost_mul,
       zero=(False, math.inf),   # OR identity = False; min identity = inf
       one=(True, 0.0),          # AND identity = True; + identity = 0
   )

   # zero absorbs under multiplication:
   # reach_cost.mul((False, inf), (True, 3.0)) == (False, inf)  [zero]
   # one is a multiplicative identity:
   # reach_cost.mul((True, 0.0), (True, 3.0)) == (True, 3.0)   [a]

The identities and absorption law hold because each component obeys its own
semiring laws independently.

Custom semirings work with scalar operations (``reach_cost.add``,
``reach_cost.mul``) but may not integrate with
:class:`~algebraic.array.AlgebraicArray` unless the ``add`` and ``mul``
callables are vectorizable over the chosen backend. For array-backed use,
the operations must accept and return backend arrays rather than Python tuples.
