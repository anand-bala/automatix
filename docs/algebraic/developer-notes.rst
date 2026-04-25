Developer Notes
===============

This page collects implementation-level details that are useful when
extending ``algebraic`` itself (in particular the polynomial backend)
or when tuning performance of code that depends on it. End users of
the public polynomial API can usually skip this page; see
:doc:`concepts/polynomials` for the user-facing overview.

.. contents::
   :local:
   :depth: 2


Polynomial Composition: ``shortcircuit``, ``atol``, and ``pack``
-----------------------------------------------------------------

:meth:`~algebraic.polynomials.RankDecomposition.compose` and
:meth:`~algebraic.polynomials.LowRankFactors.compose` accept three
keyword arguments that control the prune step at every beam iteration of
batched composition:

.. code-block:: python

   out = state.compose(replacements, atol=1e-6, shortcircuit=True, pack=True)

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

.. warning::
   The fast prune is **lossy by design**.

Per beam step, after multiplying the beam by the next replacement
candidate:

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
:meth:`~algebraic.polynomials.RankDecomposition.compose`,
:meth:`~algebraic.polynomials.LowRankFactors.compose`,
``batched_compose_factors``, ``batched_contraction_compression``,
``batched_prune_fast``, and ``prune_factors``.


Batched Polynomial Arithmetic: Vectorized vs. Per-Batch Loops
--------------------------------------------------------------

The internal helpers in ``algebraic.utils.poly`` distinguish two
classes of operation based on whether the output shape is
**data-independent**:

Fully vectorized over batch (single tensor op, no Python loop)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These work on any leading batch shape ``(*batch, R, D, N+1)`` via
``...`` indexing.  They are used by every batched operator and form
the core of the fast path:

- ``_add_factors`` -- pad + concat along the rank axis.
- ``_multiply_factors`` -- broadcast + concat + reshape.
- ``pad_upto`` -- identity-padding along rank/degree.
- ``strip_identity_slots`` -- trim trailing identity slots.
- ``pack_non_identity_slots`` -- sort identity slots to the back via a
  stable argsort, then slice.
- ``batched_prune_fast`` -- pack/strip + hard rank/degree truncation.
- ``batched_contraction_compression`` -- the beam-search loop in
  composition; vectorized when ``shortcircuit=True``.

Per-batch Python loop (unavoidable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The smart prune passes produce a *different* output rank for each
batch element:

- ``deduplicate_rank_dim`` indexes ``factors[keep]`` along axis 0
  with a per-element boolean mask.
- ``idempotence_pruning`` drops lattice-dominated rows the same way.
- ``merge_compatible_components`` mutates rank by replacing two rows
  with one in a search loop.
- ``reduce_degree`` runs a per-component subset-DP and reconstructs
  a different number of monomials per component.

Because each batch element produces a ragged rank, these passes
cannot share a single tensor output.  They are wrapped in
``_prune_per_batch`` which loops over the batch axis, applies
``prune_factors`` per element, and pads the results back to a uniform
``(B, R', D', N+1)`` tensor.  This helper is the *only* place a
Python-level batch loop appears; ``__add__`` / ``__mul__`` /
``batched_contraction_compression`` all funnel through it for the
smart path.

Why not vectorize the smart passes too
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A vectorized smart prune would have to keep the rank fixed (say at the
worst-case ``R_max``) and replace pruned components with an
**additive-identity sentinel row** -- a row containing
``algebra.zero`` in at least one slot, so its product evaluates to 0
and it contributes nothing to the rank-summed polynomial.  Every
downstream pass (identity-slot detection, dedup, merge, evaluate)
would then need to recognise and skip these sentinels.  This is a
significant rewrite that entangles the passes, and is only worth
doing if ``_prune_per_batch`` is measured as a real bottleneck.

JAX ``vmap`` of ``prune_factors`` does **not** sidestep this problem:
the boolean masking inside the smart passes still requires
data-dependent shapes that JAX tracing cannot represent without the
sentinel-row rewrite.

For the common case (``shortcircuit=True``, the default for batched
``compose``), no Python loop runs -- every step uses
``batched_prune_fast`` which is a single fused tensor operation.
