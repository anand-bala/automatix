Sparse Factors for RankDecomposition: Investigation Plan
=========================================================

**Status**: planning / investigation
**Scope**: ``algebraic.polynomials`` -- ``RankDecomposition`` hot paths


Overview
--------

This document lays out the investigation and implementation plan for
exploiting the structural sparsity inside ``RankDecomposition.factors``
to improve performance of ``compose`` and ``evaluate``.

The central question is not "should we add a sparse backend to
``AlgebraicArray``?" but rather "can we exploit the known sparsity
pattern of the CP factors tensor to make ``compose`` and ``evaluate``
strictly faster, with an acceptable maintenance cost?"


Observed Sparsity
-----------------

``RankDecomposition`` stores polynomials as CP (CANDECOMP/PARAFAC)
factors of shape ``(R, D, N+1)`` (unbatched) or ``(B, R, D, N+1)``
(batched).  Each degree slot ``factors[r, d, :]`` is a vector over the
``N+1`` selector axis: index 0 is the constant and index ``i+1``
selects variable ``x_i``.  A full inner product is taken at evaluation
time (``einsum "rdk,k->rd"``), so a slot is free to hold any number of
non-zero entries.

**Construction sparsity (1-sparse per slot).**  In all canonical
construction paths each slot has exactly one non-zero entry:

* ``variable(i, ...)``: non-zero at ``(0, 0, i+1)`` only.
* ``constant(v, ...)``: non-zero at ``(0, 0, 0)`` only.
* ``from_sparse()``: one variable (or constant) per slot by construction.
* ``pad_upto()``: identity rows ``[one, zero, ..., zero]``.
* ``_add_factors``, ``_multiply_factors``: concatenate along rank/degree
  axes without touching individual slot vectors, so 1-sparsity is
  inherited from the inputs.

**Multi-non-zero slots arise from two independent sources.**

*Source 1 -- the slow-path prune.*
``merge_compatible_components`` (``shortcircuit=False``) finds two
rank-1 components that agree on every degree slot except one and merges
the differing slots by adding them.  If component A has slot
``[0, 1, 0, ...]`` (selects ``x_0``) and component B has slot
``[0, 0, 1, ...]`` (selects ``x_1``), the merged slot becomes
``[0, 1, 1, ...]`` -- two non-zeros encoding ``x_0 + x_1`` in a single
slot.  This is intentional: it reduces rank by 1 at no degree cost.

*Source 2 -- composition with multi-non-zero replacements (any path).*
The einsum ``"pdk,kqev->pdqev"`` evaluates
``contracted[p,d,q,e,v] = sum_k factors[p,d,k] * q_factors[k,q,e,v]``.
When ``factors[p,d,:]`` is 1-sparse (one non-zero at ``k=k0``), the
sum collapses to a single term
``contracted[p,d,q,e,:] = factors[p,d,k0] * q_factors[k0,q,e,:]``.
The output slot *inherits the slot vector of the replacement polynomial
for variable* ``k0``.  If that replacement already carries a merged
slot with multiple non-zeros, the result does too -- with no involvement
of ``merge_compatible_components`` and through the fast path.

Concretely: composing ``x_1 * x_2`` (1-sparse) with replacements
``x_1 = x_3 + x_4`` and ``x_2 = x_5 + x_6`` where each replacement is
stored as a rank-1 degree-1 polynomial with a merged slot produces
output slots ``[0,0,0,1,1,0,0]`` and ``[0,0,0,0,0,1,1]`` entirely via
the fast path.

**Summary**: the fast path does not introduce new multi-non-zero slots
itself (it does not add slot vectors together), but it fully propagates
those it receives from replacement inputs.  1-sparse slots are only
guaranteed when both the state polynomial and every replacement are
1-sparse -- a property that holds at initial construction time but is
not maintained across composition chains once merging or multi-non-zero
replacements enter the picture.

**Density estimate (at construction time)** (scratch bench defaults
``N=20, R=4, D=4``): each factor tensor of shape ``(4, 4, 21)`` has at
most ``D=4`` non-zero entries across 336 total -- under 5% dense.
In practice the fill fraction rises as composition chains progress and
merged slots propagate.


Scope Recommendation: Specialize, Do Not Generalize
----------------------------------------------------

Two paths are on the table:

1. **General sparse ``AlgebraicArray``** -- replace or augment the
   ``data: Array`` field with a sparse backend tensor.
2. **Specialized sparse factors inside ``RankDecomposition``** -- keep
   ``AlgebraicArray`` dense, but change *how factors are stored and
   processed* internally.

**Recommendation: path 2.**

Reasons:

* JAX ``jax.experimental.sparse`` and PyTorch ``torch.sparse`` both
  support only standard ``+``/``*``, not user-supplied semiring ops.
  We would have to re-implement every ``AlgebraicArray`` kernel in a
  sparse-aware form -- large surface area, high maintenance cost.
* The only tensors that are structurally sparse are the CP factor
  arrays.  Every other ``AlgebraicArray`` in the system (weight
  matrices in ``MatrixOperator``, evaluation results, etc.) is dense.
  A general sparse array would be overkill for what is essentially a
  fixed 1-sparse-per-slot pattern.
* Specializing inside ``RankDecomposition`` is a localized change
  confined to ``rank_decomp.py`` and ``utils/poly.py`` -- the public
  API is untouched.

**Invariant that the sparse representation must respect**:
missing (implicitly zero) indices must correspond to the *semiring
zero*, not the arithmetic zero.  The representation must be
parameterized on ``algebra.zero`` and must not assume it is the
number 0.  This is automatically satisfied if we represent each slot
as a single ``(index, value)`` pair and treat all other positions as
``algebra.zero`` by convention.


Concrete Sparse Representation
-------------------------------

The "1-sparse-per-slot" structure only holds reliably on the fast path.
Rather than a fixed-index-per-slot encoding, the investigation should
consider a **k-sparse** representation that supports the general case
while still being cheaper than dense for small fill fractions.

Two candidate formats:

**Option A -- COO-style per slot (fixed k=1, fast path only)**

.. code-block::

   slot_idx:  Array[int]    shape (*batch, R, D)   -- single selected index in {0..N}
   slot_val:  AlgebraicArray shape (*batch, R, D)   -- value at that index

Applicable only while ``shortcircuit=True`` and no merging has
occurred.  Any transition to the slow path requires converting back to
dense first.  A ``to_dense()`` / ``from_dense()`` pair would guard
both directions.

Storage cost: ``2 * R * D`` vs ``R * D * (N+1)`` dense.

**Option B -- Compressed sparse rows along the variable axis**

Store ``slot_indices`` and ``slot_values`` as variable-length lists per
``(r, d)`` pair.  This supports merged slots with up to ``N+1``
non-zeros without falling back to dense.

More general but harder to vectorize: Python loops over ``(r, d)``
pairs are necessary unless the sparsity pattern is uniform across the
batch.

**Recommendation for the investigation**: start with Option A (simpler
to benchmark, covers the dominant batched fast-path use case) and
measure whether the gains justify the added conversion overhead before
considering Option B.


Hot Paths and Expected Gains
-----------------------------

evaluate_factors
~~~~~~~~~~~~~~~~

Current::

   selector = concat([one, points])          # (N+1,)
   contracted = einsum("rdk,k->rd", factors, selector)  # O(R * D * N)
   degree_prod = prod(contracted, axis=1)    # (R,)
   result = sum(degree_prod, axis=0)         # ()

With Option A sparse factors (1-sparse slots)::

   selected_vals = selector[slot_idx]        # (R, D) -- gather, O(R * D)
   contracted    = slot_val * selected_vals  # (R, D) -- semiring mul, O(R * D)
   degree_prod   = prod(contracted, axis=1)  # (R,)
   result        = sum(degree_prod, axis=0)  # ()

Reduction: O(R*D*N) -> O(R*D).  Expected speedup: ~N on the contraction
step (e.g. 20x for N=20).  The ``prod`` and ``sum`` steps are unchanged.

With multi-non-zero slots (Option B or after slow-path merge), the
gather becomes a sparse dot-product per slot, which is harder to
vectorize; whether this is still faster than the dense einsum at
typical fill fractions (2-5 non-zeros per slot) needs to be benchmarked.

compose_factors (unbatched)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current einsum::

   # (N+1, R2, D2, N+1) x (R1, D1, N+1) -> (R1, D1, R2, D2, N+1)
   contracted = einsum("pdk,kqev->pdqev", factors, q_factors)

The ``k`` axis (size ``N+1``) is the contraction axis.  With 1-sparse
slots, ``factors[p, d, :]`` has one non-zero at ``k = slot_idx[p, d]``,
so the contraction reduces to a single slice::

   contracted[p, d, :, :, :] = slot_val[p, d] * q_factors[slot_idx[p,d], :, :, :]

This is ``O(R1 * D1 * R2 * D2 * N)`` instead of
``O(R1 * D1 * (N+1) * R2 * D2 * N)`` -- a factor of ``N+1`` reduction
on the most expensive step of ``compose``.

With k-sparse slots (k > 1), the slice becomes a sum over k non-zero
entries, giving a ``(N+1)/k`` reduction -- still a win when ``k << N``.

batched_compose_factors
~~~~~~~~~~~~~~~~~~~~~~~

Same analysis; the batched einsum contracts over ``bpdk,bkqev->bpdqev``.
With 1-sparse slots the ``k`` loop collapses to a single gather per
``(b, p, d)`` slot::

   contracted[b, p, d, :, :, :] = slot_val[b, p, d] * q_factors[b, slot_idx[b,p,d], :, :, :]

This is expressible as a batched ``gather + mul`` without any explicit
loop when all batch elements have the same ``slot_idx`` structure
(which they do in the current polynomial operator workflow).

_multiply_factors
~~~~~~~~~~~~~~~~~

This just concatenates rank and degree axes without touching the
variable axis -- it maps directly to the sparse format with a concat
of ``slot_idx`` and ``slot_val`` along the appropriate axes.
No algorithmic change needed; the tensor shapes shrink because we no
longer carry the ``N+1`` axis.

_add_factors
~~~~~~~~~~~~

Same: concat along rank axis after degree-padding.  In the sparse
format the "identity pad" slots are ``(slot_idx=0, slot_val=one)``
which is cheap.

Pruning passes
~~~~~~~~~~~~~~

* ``strip_identity_slots`` / ``pack_non_identity_slots``: identity slot
  is ``(slot_idx=0, slot_val=one)`` in sparse form.  The equality check
  simplifies to ``slot_idx == 0 and slot_val == one``.
* ``deduplicate_rank_dim``, ``idempotence_pruning``: compare ``(R, D, 2)``
  tensors instead of ``(R, D, N+1)`` -- memory bandwidth savings.
* ``reduce_degree`` / ``merge_compatible_components``: these work on the
  dense reconstructed form; call ``to_dense()`` before entering and
  return a new sparse form after.  These are already slow-path only.


Go / No-Go Criteria
--------------------

Before implementing, the following must be verified empirically.
The default verdict is NO unless each criterion is met.

1. **Strict improvement on compose**:
   Benchmark ``compose`` (batched, ``shortcircuit=True``) with the
   sparse representation on the reference parameters
   ``B=256, R=4, D=4, N=20`` (the scratch bench defaults).
   The sparse path must be at least **1.5x faster** end-to-end
   (not just the einsum step) on both CPU (NumPy) and GPU (Torch/CUDA).

2. **Strict improvement on evaluate**:
   Same requirement for the ``evaluate`` hot path.

3. **No correctness regression**:
   The existing polynomial test suite must pass unchanged.
   Add property-based tests that compare sparse-path outputs to dense
   outputs on randomly generated polynomials for all supported semirings.

4. **No new public API surface**:
   The sparse representation must be an internal implementation detail.
   ``RankDecomposition.factors`` may remain a property that returns the
   reconstructed dense ``AlgebraicArray`` for compatibility, or the
   property is deprecated and existing tests are updated.

5. **Invariant maintenance check**:
   Audit every code path that writes to ``factors`` to confirm the
   1-sparse invariant is maintained on the fast path, and that the slow
   path converts to dense before writing and converts back afterwards.

If criterion 1 or 2 fails (i.e., the gather overhead outweighs the
contraction savings on the target backend), the investigation closes
with a note that the dense representation is already near-optimal
given backend vectorization, and no implementation is done.


Implementation Phases
---------------------

Phase 0: Benchmarking and Sparsity Audit (prerequisite)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Run the scratch bench (``scratch_compose_bench.py``) with profiling
  to confirm that the einsum contraction (``batched_compose_factors``
  and ``evaluate_factors``) is the dominant cost and not the pruning
  passes.
* Instrument ``RankDecomposition.factors`` to log the fraction of
  non-zero entries in practice across several ``compose`` chains.
* Prototype a minimal Python-level sparse evaluate (using NumPy fancy
  indexing) and compare wall-clock time against the dense einsum for
  ``N=20, 50, 100``.

**Checkpoint**: if the contraction is not the bottleneck, stop.

Phase 1: Sparse Factors Dataclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Introduce ``_SparseFactor`` (internal, not exported)::

      @dataclass
      class _SparseFactor:
          slot_idx: Array       # shape (*batch, R, D), dtype int
          slot_val: AlgebraicArray  # shape (*batch, R, D)

* Implement ``_SparseFactor.to_dense() -> AlgebraicArray`` and
  ``_SparseFactor.from_dense(factors: AlgebraicArray) -> _SparseFactor``.
* Replace ``self.factors`` storage in ``RankDecomposition`` with
  ``self._sparse_factors: _SparseFactor``, keeping ``factors`` as a
  property.

Phase 2: Sparse Hot-Path Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Rewrite ``evaluate_factors`` and ``batched_evaluate_factors`` in
  ``utils/poly.py`` to accept the sparse form and use gather + mul.
* Rewrite ``compose_factors`` and ``batched_compose_factors`` to avoid
  the ``k``-axis summation (replace einsum with gather + elementwise
  scale of ``q_factors``).
* Rewrite ``_add_factors`` and ``_multiply_factors`` to operate on the
  sparse form directly.
* Add a ``to_dense()`` escape hatch for the slow-path pruning passes
  that require the full slot vector.

Phase 3: Pruning Adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Adapt ``strip_identity_slots`` / ``pack_non_identity_slots`` to
  operate on ``(slot_idx, slot_val)`` pairs without expanding to dense.
* Adapt ``deduplicate_rank_dim`` and ``idempotence_pruning``:
  comparisons over ``(R, D, 2)`` instead of ``(R, D, N+1)``.
* Keep ``reduce_degree``, ``merge_compatible_components``, and
  ``contraction_compression`` operating in dense form; wrap with
  ``from_dense`` / ``to_dense`` calls.

Phase 4: Validation and Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Verify correctness against the existing test suite.
* Run the scratch bench and confirm go/no-go criteria are satisfied.
* Update ``scratch_compose_bench.py`` (or a dedicated benchmark) to
  compare dense vs sparse paths side-by-side.
* Profile memory usage to confirm the reduction in allocation size.

Phase 5: Cleanup (if phases 1-4 pass)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Remove ``LowRankFactors.to_merged`` / ``from_merged`` round-trips
  that exist today to share the dense-factor arithmetic -- if the
  sparse kernels handle both ``RankDecomposition`` and
  ``LowRankFactors`` directly, the merge/split overhead disappears.
* Update AGENTS.md / developer notes to document the sparse invariant
  and the conversion contract.


Out of Scope
------------

* A general-purpose sparse ``AlgebraicArray``: see scope recommendation
  above.
* Sparse representations for ``MonomialBasis``: the coefficients tensor
  of shape ``(2,)^n`` does not have the same 1-sparse-per-slot
  structure; ``PolyDict`` already covers the sparse monomial-basis use
  case.
* Sparse ``MatrixOperator`` weight tensors: these are dense by design
  (transition matrices over NFA states).


Open Questions
--------------

1. In production workloads (AFA-based LTL monitoring), what is the
   steady-state slot fill-fraction after a sequence of compose steps?
   Multi-non-zero slots arise from two sources -- slow-path merges AND
   inheritance from replacement inputs -- so the fill fraction can grow
   even when ``shortcircuit=True``.  Measuring actual fill at runtime is
   the prerequisite for deciding whether any sparse format is worth it,
   and whether Option A (single-index) or Option B (variable-length) is
   the right target.

2. What is the GPU gather latency for ``slot_idx``-based indexing vs
   a dense ``einsum`` at the sizes we care about?  PyTorch's
   ``torch.gather`` and JAX's ``jnp.take`` have non-trivial overheads
   at small batch sizes.

3. Can ``batched_compose_factors`` exploit the sparse structure while
   remaining JIT-compilable under JAX ``jit`` / ``vmap``?  The gather
   indices are data-dependent, which interacts with JAX's static-shape
   requirement for ``jit``.  May need ``jnp.take`` rather than boolean
   masking.

4. Should the sparse form be exposed at the ``RankDecomposition`` API
   level (e.g. ``RankDecomposition.from_sparse_factors(...)``) or
   remain purely internal?  The former would let external code skip the
   ``to_dense()`` construction overhead for hand-crafted polynomials.
