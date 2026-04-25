Polynomial Representations
==========================

``algebraic`` provides three representations of *multilinear polynomials* over
a semiring. These are polynomials in Boolean variables
:math:`x_0, x_1, \ldots, x_{n-1}` where every variable appears at most once
per monomial (no powers), and where arithmetic is performed using the semiring's
``add`` and ``mul`` in place of the usual ``+`` and ``*``.

Multilinear polynomials appear naturally when evaluating automata runs or
computing reach-avoid conditions: each variable represents whether a particular
state is active, and the polynomial encodes the set of satisfying state
combinations as a semiring-weighted sum of products.

All three representations live under ``algebraic.polynomials`` and expose a
common interface: variable constructors, arithmetic operators (``+`` and
``*``), evaluation, and conversion to other representations.

Choosing a Representation
--------------------------

The three representations differ in how they store coefficients and in their
computational trade-offs:

+---------------------+-------------------------------------+---------------------------------------------+
| Class               | Storage                             | Best when                                   |
+=====================+=====================================+=============================================+
| ``PolyDict``        | Sparse dict: monomial -> coefficient| Few non-zero monomials; symbolic work       |
+---------------------+-------------------------------------+---------------------------------------------+
| ``MonomialBasis``   | Dense tensor of shape ``(2,)*n``    | Full enumeration of all :math:`2^n` terms   |
+---------------------+-------------------------------------+---------------------------------------------+
| ``RankDecomposition``| CP factors of shape ``(R, d, n+1)``| Structured low-rank problems; JIT-friendly  |
+---------------------+-------------------------------------+---------------------------------------------+
| ``LowRankFactors``  | Weights ``(R, d, n)`` + bias        | Training pipelines needing separate params  |
|                     | ``(R, d)``                          |                                             |
+---------------------+-------------------------------------+---------------------------------------------+

For small ``n`` (up to roughly 10--12 variables), ``MonomialBasis`` or
``PolyDict`` are straightforward. For larger problems where the polynomial
is expected to have low rank (e.g. automata with many states but sparse
transitions), ``RankDecomposition`` is more efficient and is the representation
used internally by ``automatix``'s ``PolynomialOperator``.

All three representations require a :class:`~algebraic.spec.BoundedDistributiveLattice`
as their algebra. Plain :class:`~algebraic.spec.Semiring` instances are not
accepted because multilinear arithmetic requires idempotent operations.

PolyDict: Sparse Dictionary
----------------------------

:class:`~algebraic.polynomials.PolyDict` stores the polynomial as a mapping
from frozen bit vectors (monomials) to ``AlgebraicArray`` coefficients. It
is the most transparent representation and behaves like a dictionary.

.. code-block:: python

   import algebraic
   from algebraic.polynomials import PolyDict
   from algebraic.semirings import boolean_algebra

   bool_alg = boolean_algebra(mode="logic")

   # Create variables x0, x1, x2 for a 3-variable polynomial.
   x0 = PolyDict.variable(0, num_vars=3, algebra=bool_alg, backend="numpy")
   x1 = PolyDict.variable(1, num_vars=3, algebra=bool_alg, backend="numpy")
   x2 = PolyDict.variable(2, num_vars=3, algebra=bool_alg, backend="numpy")

   # Arithmetic uses semiring operations.
   # p = (x0 AND x1) OR x2
   p = (x0 * x1) + x2

   # Evaluate at a concrete point.
   result = p.evaluate({0: True, 1: False, 2: True})   # True

   # Inspect monomials: keys are frozenbitarray instances.
   for monomial, coeff in p.items():
       print(monomial, coeff)

   # Constant polynomial.
   const = PolyDict.constant(True, num_vars=3, algebra=bool_alg, backend="numpy")

   # Zero and one.
   zero = PolyDict.zero(num_vars=3, algebra=bool_alg, backend="numpy")
   one  = PolyDict.one(num_vars=3, algebra=bool_alg, backend="numpy")

``PolyDict`` is the easiest representation to inspect manually and is well
suited to symbolic manipulation where you iterate over monomials. However,
its dictionary overhead makes it slower than tensor-backed representations for
large-scale numerical computations.

MonomialBasis: Dense Tensor
-----------------------------

:class:`~algebraic.polynomials.MonomialBasis` stores all :math:`2^n`
coefficients as a dense tensor of shape ``(2, 2, ..., 2)`` (one axis per
variable). The entry at index ``(i_0, i_1, ..., i_{n-1})`` is the coefficient
of the monomial :math:`x_0^{i_0} x_1^{i_1} \cdots x_{n-1}^{i_{n-1}}`.

.. code-block:: python

   import algebraic
   from algebraic.polynomials import MonomialBasis
   from algebraic.semirings import max_min_algebra

   mm = max_min_algebra()

   x0 = MonomialBasis.variable(0, num_vars=2, algebra=mm, backend="numpy")
   x1 = MonomialBasis.variable(1, num_vars=2, algebra=mm, backend="numpy")

   # Build: max(min(x0, x1), 0.5)
   half = MonomialBasis.constant(0.5, num_vars=2, algebra=mm, backend="numpy")
   p = (x0 * x1) + half

   # The coefficient tensor has shape (2, 2).
   print(p.coeffs.shape)   # (2, 2)

   # Evaluate.
   result = p.evaluate({0: 0.8, 1: 0.3})

``MonomialBasis`` is memory-intensive for large ``n`` (exponential growth) but
offers fast elementwise tensor arithmetic. It is best for problems with a
small, fixed number of variables where you need the full coefficient structure.

RankDecomposition: Structured CP Factorization
-----------------------------------------------

:class:`~algebraic.polynomials.RankDecomposition` represents a multivariate
polynomial using a CP (CANDECOMP/PARAFAC) decomposition over degree-``d``
factorizations:

.. math::

   p(x) = \sum_{r=1}^{R} \prod_{k=1}^{d}
   \left\langle f_{r,k}, \phi(x) \right\rangle

where :math:`f \in \mathcal{R}^{R \times d \times (n+1)}` is a tensor of
coefficients, and :math:`\phi(x) \in \mathcal{R}^{n+1}` is the feature map

.. math::

   \phi(x) = (1, x_0, x_1, \dots, x_{n-1}).

Equivalently, each :math:`f_{r,k}` defines a linear form in the inputs
(including a bias term), and each rank-:math:`r` component is a product of
``d`` such linear forms. The full polynomial is obtained by summing over
``R`` components. This corresponds to a CP decomposition of the order-``d``
coefficient tensor of the polynomial.


.. code-block:: python

   import algebraic
   from algebraic.polynomials import RankDecomposition
   from algebraic.semirings import boolean_algebra

   bool_alg = boolean_algebra(mode="soft")

   x0 = RankDecomposition.variable(0, num_vars=3, algebra=bool_alg, backend="numpy")
   x1 = RankDecomposition.variable(1, num_vars=3, algebra=bool_alg, backend="numpy")
   x2 = RankDecomposition.variable(2, num_vars=3, algebra=bool_alg, backend="numpy")

   # Each multiplication may increase rank; addition does not.
   p = (x0 * x1) + x2

   print(p.rank)       # number of rank-1 components
   print(p.degree)     # maximum monomial degree
   print(p.num_vars)   # 3

   # Evaluate.
   result = p.evaluate({0: 0.9, 1: 0.4, 2: 0.1})

The factors tensor is a regular ``AlgebraicArray``, so batch evaluation and
JIT compilation work naturally. This representation is the most
JIT-friendly of the three and is what ``automatix``'s ``PolynomialOperator``
uses internally for AFA runs.

.. note::
   Polynomial multiplication can increase the rank. The ``max_rank``
   parameter (set at construction time or via subclassing) controls when the
   representation truncates low-weight components. A higher rank is more
   accurate but uses more memory; the right setting depends on the problem.

Composition
^^^^^^^^^^^

``RankDecomposition`` supports *variable substitution* (polynomial
composition), where each variable :math:`x_i` is replaced by another
polynomial :math:`g_i(x)`.

Formally, given a polynomial :math:`p(x_0, \dots, x_{n-1})` and a set of
polynomials :math:`\{g_i(x)\}_{i=0}^{n-1}`, the composition is

.. math::

   (p \circ g)(x) = p\big(g_0(x), g_1(x), \dots, g_{n-1}(x)\big).

Under the rank-decomposition parameterization

.. math::

   p(x) = \sum_{r=1}^{R} \prod_{k=1}^{d}
   \left\langle f_{r,k}, \phi(x) \right\rangle,

composition is implemented by substituting each feature
:math:`\phi_i(x)` with

.. math::

   \phi_0(x) = 1, \quad
   \phi_i(x) \mapsto g_{i-1}(x) \;\; \text{for } i > 0,

yielding

.. math::

   (p \circ g)(x)
   = \sum_{r=1}^{R} \prod_{k=1}^{d}
   \left(
      f_{r,k,0}
      \;+\;
      \sum_{i=1}^{n} f_{r,k,i} \, g_{i-1}(x)
   \right).

That is, each linear form :math:`\langle f_{r,k}, \phi(x) \rangle` is
transformed into a linear combination of the substituted polynomials,
and each rank-1 term becomes a product of such polynomials. Since each
:math:`g_i(x)` is itself represented as a rank decomposition, this
operation is closed within the representation (up to an increase in
rank).

This operation is used, for example, to step an algebraic finite
automaton (AFA) forward by one symbol.

.. code-block:: python

   # Replace each variable with a new polynomial over the next time step.
   substitutions = [new_poly_for_var_i for i in range(num_vars)]
   stepped = p.compose(substitutions)

LowRankFactors: Split-Storage CP Factorization
------------------------------------------------

:class:`~algebraic.polynomials.LowRankFactors` is a variant of
``RankDecomposition`` that stores the constant (bias) and variable factors
separately, analogous to how a neural network layer separates
:math:`W \mathbf{x}` from :math:`\mathbf{b}`:

- ``weights``: shape ``(R, d, n)`` -- one entry per variable per degree per rank component.
- ``bias``: shape ``(R, d)`` -- the constant factor for each degree and rank component.

This split is useful for training pipelines where you need independent
parameter groups (e.g. separate learning rates, freezing the bias, or applying
different regularization to weights and bias).

Mathematically, ``LowRankFactors`` represents the same polynomial as
``RankDecomposition``:

.. math::

   p(x) = \sum_{r=1}^{R} \prod_{k=1}^{d}
   \left( b_{r,k} + \sum_{i=0}^{n-1} w_{r,k,i} \, x_i \right)

where :math:`b_{r,k}` is the bias and :math:`w_{r,k,i}` are the weights.
The merged form ``(R, d, n+1)`` used by ``RankDecomposition`` simply
concatenates :math:`b` as the zeroth column.

.. code-block:: python

   import algebraic
   from algebraic.polynomials import LowRankFactors
   from algebraic.semirings import boolean_algebra

   bool_alg = boolean_algebra(mode="soft")

   x0 = LowRankFactors.variable(0, num_vars=3, algebra=bool_alg, backend="numpy")
   x1 = LowRankFactors.variable(1, num_vars=3, algebra=bool_alg, backend="numpy")

   p = x0 * x1

   # Separate parameter tensors for training.
   print(p.weights.shape)  # (R, d, 3)
   print(p.bias.shape)     # (R, d)

   # Convert to/from RankDecomposition for interop.
   rd = p.to_rank_decomposition()
   p_back = LowRankFactors.from_rank_decomposition(rd)

All arithmetic operations (``+``, ``*``, ``evaluate``, ``compose``) produce
``LowRankFactors`` instances, and the class is registered as a JAX pytree
so ``jit``, ``vmap``, and ``grad`` work out of the box. Because ``weights``
and ``bias`` are separate leaves in the pytree, JAX and PyTorch autodiff
naturally computes independent gradients for each.

Conversion Between Representations
------------------------------------

``PolyDict`` is the common intermediate format. Both ``MonomialBasis`` and
``RankDecomposition`` can convert to and from a ``PolyDict``:

.. code-block:: python

   # MonomialBasis -> PolyDict
   pd = mb.to_poly_dict()

   # PolyDict -> MonomialBasis
   mb2 = MonomialBasis.from_poly_dict(pd, algebra=bool_alg, backend="numpy")

   # RankDecomposition -> PolyDict
   pd2 = rd.to_poly_dict()

   # PolyDict -> RankDecomposition
   rd2 = RankDecomposition.from_poly_dict(pd2, algebra=bool_alg, backend="numpy")

Conversion is useful when you need to inspect individual monomial coefficients
(switch to ``PolyDict``) or when migrating between computation styles.

Shared Interface
-----------------

All four representations support the following common operations:

- ``p + q`` -- semiring addition (OR / max / min depending on algebra).
- ``p * q`` -- semiring multiplication (AND / min / + depending on algebra).
- ``p.evaluate(assignment)`` -- substitute concrete values for each variable
  and return a scalar or ``AlgebraicArray``.
- ``p.num_vars`` -- number of Boolean variables.
- ``.algebra`` -- the ``BoundedDistributiveLattice`` used for arithmetic.
- ``.backend`` -- the array backend (``"numpy"``, ``"jax"``, or ``"torch"``).
- ``cls.variable(i, ...)`` -- constructor for the :math:`i`-th variable.
- ``cls.constant(value, ...)`` -- constructor for a constant polynomial.
- ``cls.zero(...)`` / ``cls.one(...)`` -- additive / multiplicative identities.


Composition: ``shortcircuit`` and ``atol`` tradeoffs
-----------------------------------------------------

``RankDecomposition.compose`` and ``LowRankFactors.compose`` accept two
keyword arguments that control the prune step at every beam iteration of
batched composition:

.. code-block:: python

   out = state.compose(replacements, atol=1e-6, shortcircuit=True)

``shortcircuit`` (default ``True`` for batched compose)
   When ``True``, each beam step uses the **vectorized fast prune**:
   ``strip_identity_slots`` (vectorized over the whole batch) followed by
   hard rank/degree truncation.  No ``O(R^2)`` per-element passes.

   When ``False``, each beam step runs the **per-element smart prune**
   (``deduplicate_rank_dim``, ``idempotence_pruning``,
   ``merge_compatible_components``, and -- if degree exceeds ``max_degree``
   -- ``reduce_degree`` via monomial expansion).  This is exact for
   idempotent algebras when followed by hard truncation, but each batch
   element is pruned in a Python loop with ``O(R^2)`` work per step,
   making it dramatically slower for large batches.

``atol`` (default ``1e-6``)
   Tolerance for equality checks inside the smart-prune passes.  ``0``
   uses exact equality (``algebraic.equal``); a positive value uses
   ``algebraic.isclose`` with that absolute tolerance.  A non-zero
   default is appropriate for soft / smooth Boolean algebras whose
   values cluster near 0 or 1 but rarely match exactly due to
   floating-point arithmetic.

When the fast path is appropriate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fast path (``shortcircuit=True``) is the right default for
**non-idempotent / float-valued algebras** such as the soft and smooth
Boolean algebras.  In those settings the smart passes almost never fire
because exact (or even tolerance-based) equality between independently
computed floating point values is rare, so the ``O(R^2)`` work is pure
overhead.

When to use the smart prune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``shortcircuit=False`` when:

- the algebra is genuinely idempotent (e.g. logic Boolean, max-min)
  *and* you want the smart passes to find real duplicates / dominated
  components before hard truncation,
- approximation quality matters more than throughput, or
- the batch size is small enough that the per-element Python loop is
  not the bottleneck.

What the fast path drops (lossiness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fast prune is **lossy by design**.  Per beam step, after multiplying
the beam by the next replacement candidate:

1. ``strip_identity_slots`` removes only **trailing** identity slots.
   The leading identity slot from the beam's prior state is **not**
   removed, so degree typically grows by ``D_replacement`` per step.
2. **Hard degree truncation** to ``max_degree`` drops the trailing
   degree slots from every component.  Each dropped slot was a factor
   in the rank-1 component's product, so dropping it changes the
   component's value.  For values in :math:`[0, 1]` (soft Boolean),
   product of fewer factors is **larger**, so degree truncation tends
   to produce an **over-approximation**.
3. **Hard rank truncation** to ``max_rank`` drops the trailing rank
   components from the sum.  Dropping summands tends to produce an
   **under-approximation**.

The two effects partially cancel but the result is **not a sound
one-sided bound** in general.  Validate empirically by sampling a few
points and comparing fast vs.\ smart output, e.g.:

.. code-block:: python

   out_fast = state.compose(replacements, atol=ATOL, shortcircuit=True)
   out_slow = state.compose(replacements, atol=ATOL, shortcircuit=False)
   diff = (out_fast.evaluate(pts).data - out_slow.evaluate(pts).data).abs()
   print("max drift", float(diff.max()), "mean drift", float(diff.mean()))

If the empirical drift is acceptable for your use case, the fast path
gives a major speedup (single fused operation per beam step instead of
``B`` Python iterations × four ``O(R^2)`` passes).  If not, prefer the
smart path or implement a more refined local compressor (see notes in
``utils/poly.py`` on potential strategies: slot-merge by join, slot
absorption, envelope clustering).

``pack`` (default ``True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A third keyword, ``pack``, controls how identity slots are removed
during the fast prune step.

When ``pack=True`` (default), the prune step uses
``pack_non_identity_slots`` instead of ``strip_identity_slots``: it
moves identity slots that appear *anywhere* in the degree axis to the
back via a stable sort, then slices off the trailing identity region.
Because rank-1 components are commutative products over the degree
axis, this rearrangement is mathematically exact (modulo ``atol``).

When ``pack=False`` the prune step falls back to
``strip_identity_slots``, which only removes *trailing* identity slots.

Pack makes the fast path **strictly less lossy**: identity slots
introduced by ``pad_upto`` padding or beam initialization are now
compacted away for free, instead of getting dropped by the subsequent
hard degree truncation.  The kwarg is threaded through
``RankDecomposition.compose``, ``LowRankFactors.compose``,
``batched_compose_factors``, ``batched_contraction_compression``,
``batched_prune_fast``, and ``prune_factors``.
