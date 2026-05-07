==============================================================
Gradient-Based Training with ``RankDecomposition`` Factors
==============================================================

This document is a self-contained reference for an agent implementing a
gradient-based training pipeline (in JAX or PyTorch) on top of the CP /
rank-decomposition polynomial representations defined in
``algebraic.polynomials.rank_decomp``.  It covers exactly what the
training-loop author needs to know to keep the factor tensors in the
gradient path through composition (``compose``) and evaluation
(``evaluate``), and to run those operations under ``jax.jit`` /
``jax.grad`` / ``jax.vmap`` (or their Torch analogues, including
``torch.compile``).

It assumes the implementer should *not* re-read the source of
``algebraic`` at training time -- the rules below are sufficient.

.. contents::
   :local:
   :depth: 2


Data model in one paragraph
===========================

A polynomial is represented as a CP / rank decomposition

.. math::

   p(x_1, \dots, x_N) \;=\; \sum_{r=1}^{R} \prod_{k=1}^{D}
       \Big( f_{r,k,0} \;+\; \sum_{i=1}^{N} f_{r,k,i}\, x_i \Big),

where the trainable tensor is the factor array ``f`` of shape
``(R, D, N+1)``.  Column ``0`` is the bias / constant term; columns
``1..N`` weight variables ``x_1..x_N``.  Batched polynomials carry an
arbitrary leading shape: ``(*batch, R, D, N+1)``.

Two dataclasses wrap this tensor:

* ``algebraic.polynomials.RankDecomposition`` -- single ``factors``
  array of shape ``(*batch, R, D, N+1)``.
* ``algebraic.polynomials.LowRankFactors`` -- the same tensor *split*
  into ``weights`` of shape ``(*batch, R, D, N)`` and ``bias`` of
  shape ``(*batch, R, D)``.  Use this when you want separate optimizer
  parameter groups, separate learning rates, or to freeze the bias.
  ``LowRankFactors.to_merged()`` reassembles the merged
  ``(R, D, N+1)`` form internally for every operation.

The ``factors`` (or ``weights``/``bias``) live inside an
``algebraic.AlgebraicArray``; the raw backend tensor is reachable as
``factors.data``.  ``AlgebraicArray`` overloads ``+``, ``*``, ``@`` to
the algebra's semiring operations.

.. note::

   The dataclasses are registered as pytrees (via
   ``algebraic.utils.pytree.register_node_class``).  ``AlgebraicArray``
   is itself a pytree.  This means JAX transforms (``jit``, ``grad``,
   ``vmap``, ``scan``) traverse them out of the box: leaves are the
   raw backend arrays (``factors.data`` / ``weights.data`` /
   ``bias.data``); semiring, ``max_rank``, ``max_degree``,
   ``max_replacement_degree``, and the backend live in the static
   ``aux_data``.  Do not put trainable scalars in ``aux_data``.


Pick a differentiable semiring
==============================

Standard Boolean algebra is non-differentiable.  For training, use a
soft / continuous lattice algebra so that the semiring ``add`` / ``mul``
have well-defined gradients almost everywhere:

* ``algebraic.semirings.boolean_algebra(mode="logic")`` -- crisp
  Boolean, **not** suitable for gradient flow.
* ``algebraic.semirings.boolean_algebra(mode="...")`` -- continuous
  relaxations (e.g. probabilistic / Goedel / product-style).  Pick the
  variant whose ``add`` / ``mul`` are smooth enough for your loss.
* ``algebraic.semirings.max_min_algebra()`` -- piecewise-linear; OK with
  subgradient methods.
* ``algebraic.semirings.tropical_semiring(minplus=False)`` -- max-plus,
  also piecewise-linear.

Whichever you pick, the training loop's only constraint is that
``add`` / ``mul`` from that algebra produce gradients you are happy to
descend on.  ``RankDecomposition`` itself is algebra-agnostic.

The ``BoundedDistributiveLattice`` constraint at construction time (see
``RankDecomposition.__init__``) is structural -- any lattice algebra
satisfies it, including the continuous relaxations.


JIT-safe operations: the rules
==============================

Every mutating operation on ``RankDecomposition`` /
``LowRankFactors`` accepts the same four pruning knobs:

* ``atol: float = 1e-6``
* ``shortcircuit: bool``
* ``pack: bool = True``
* ``static_shape: bool = False``

For training inside ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` (or with
``torch.compile`` in the analogous PyTorch case), the **only** safe
combination is::

    shortcircuit=True, pack=True, static_shape=True

This is enforced -- any other combination with ``static_shape=True``
raises ``ValueError``.  The shortcuts are summarized below.

+--------------------+-----------+-------------------------------+--------------+
| Operation          | JIT-safe? | Defaults                      | Notes        |
+====================+===========+===============================+==============+
| ``evaluate``       | yes       | n/a                           | pure einsum  |
|                    |           |                               | over factors |
+--------------------+-----------+-------------------------------+--------------+
| ``add``            | with      | ``shortcircuit=False``        | concat along |
|                    | flags     |                               | rank axis    |
+--------------------+-----------+-------------------------------+--------------+
| ``mul``            | with      | ``shortcircuit=False``        | rank-product |
|                    | flags     |                               | concat       |
+--------------------+-----------+-------------------------------+--------------+
| ``compose``        | with      | ``shortcircuit=True``,        | beam search  |
|                    | flags     | ``static_shape=False``        | over degree  |
+--------------------+-----------+-------------------------------+--------------+
| ``prune``          | with      | ``shortcircuit=False``        | manual       |
|                    | flags     |                               | compression  |
+--------------------+-----------+-------------------------------+--------------+
| ``normalize``      | **no**    | n/a                           | round-trips  |
|                    |           |                               | through      |
|                    |           |                               | ``PolyDict`` |
+--------------------+-----------+-------------------------------+--------------+
| ``to_sparse``      | **no**    | n/a                           | Python dict  |
+--------------------+-----------+-------------------------------+--------------+
| ``from_sparse``    | **no**    | n/a                           | Python loop  |
+--------------------+-----------+-------------------------------+--------------+

Why ``normalize`` is not JIT-safe (and how ``compose`` already handles
it).  ``RankDecomposition.normalize()`` round-trips through
``PolyDict`` via ``to_sparse()``; that path uses Python ``dict``,
``bool(allclose(...))``, and value-dependent control flow.  None of it
traces.  When you call ``compose(..., static_shape=True)`` the method
*skips* the trailing ``normalize()`` and instead bounds the degree
statically through the prune knobs (``pack_non_identity_slots`` +
``pack_non_zero_components`` + a static slice at ``max_degree``).  That
is the path you want during training.

Why ``shortcircuit=True`` and ``pack=True`` are required.  The "smart"
prune passes (``deduplicate_rank_dim``, ``idempotence_pruning``,
``merge_compatible_components``, ``reduce_degree``) all branch on
boolean array contents and produce **value-dependent shapes** -- they
will not trace.  ``shortcircuit=True`` switches to the fast path that
only does shape-preserving rearrangements (``pack_non_identity_slots``,
``pack_non_zero_components``) followed by static slices at
``max_rank`` / ``max_degree``.  ``static_shape=True`` additionally
disables the value-dependent trailing slice inside those packs, leaving
the input rank/degree dimensions intact until the final static slice.

Why ``pack=True`` matters numerically.  Without packing, the fast path
falls back to ``strip_identity_slots`` (which only trims trailing
identity slots) and to a hard ``factors[:max_rank]`` truncation that
can drop meaningful rank-1 components if zero rank-1 components were
shuffled to the front.  ``pack=True`` first sorts non-identity slots
to the *front* of the degree axis and non-zero rank-1 components to
the *front* of the rank axis (both via ``argsort`` with ``stable=True``,
which is differentiable -- gradients flow through ``take_along_axis``),
so the subsequent static slice is strictly less lossy.


Gradient flow: what is and isn't differentiable
===============================================

* All factor arithmetic in ``add``, ``mul``, ``compose``, ``evaluate``
  is plain semiring ``+`` / ``*`` on backend tensors -- the gradient
  path is intact.
* The pack/sort steps used in the JIT-safe prune path
  (``pack_non_identity_slots``, ``pack_non_zero_components``) are
  ``argsort`` + ``take_along_axis``: gather operations that pass
  gradients through to the original factor tensor positions.  Treat
  them like a permutation.
* The static slices ``factors[..., :max_rank, :, :]`` and
  ``factors[..., :, :max_degree, :]`` are simple slicing: gradients
  flow to the kept entries, dropped entries get zero gradient.  This
  *is* a (sub)gradient of a thresholding operation -- if a meaningful
  rank-1 component is permanently sorted past ``max_rank`` it will
  receive zero updates.  Size ``max_rank`` / ``max_degree``
  generously, especially early in training.
* The equality check ``_eq_or_close`` (used to decide what is an
  identity slot or a zero rank-1) is a boolean-valued comparator with
  zero gradient.  Its output is only used as a sort key, so this is
  fine.
* ``normalize`` / ``to_sparse`` / ``from_sparse`` are not in the
  gradient path under any setting -- they are training-time-only
  utilities, useful for evaluation/diagnostics outside of ``jit``.
* The ``algebra``, ``max_rank``, ``max_degree``,
  ``max_replacement_degree``, and ``backend`` fields are
  static/non-trainable (they live in the pytree ``aux_data``).


Recommended pattern for JAX
===========================

Treat the factor tensor (or, with ``LowRankFactors``, the
(``weights``, ``bias``) pair) as the trainable parameter.  Wrap it in
an ``AlgebraicArray`` only when you call into ``algebraic``; the
wrapper is cheap and pytree-friendly.

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import optax

    import algebraic
    from algebraic.polynomials import RankDecomposition

    algebra = algebraic.semirings.boolean_algebra(mode="...")  # pick a soft mode
    NUM_VARS, MAX_RANK, MAX_DEGREE = 8, 16, 6

    def make_poly(factors_data: jax.Array) -> RankDecomposition:
        # `factors_data` has shape (R, D, N+1) and lives in JAX.
        factors = algebraic.array(factors_data, semiring=algebra, backend="jax")
        return RankDecomposition(
            factors,
            max_rank=MAX_RANK,
            max_degree=MAX_DEGREE,
            backend="jax",
        )

    @jax.jit
    def loss_fn(factors_data: jax.Array,
                replacement_factors: list[jax.Array],
                points: jax.Array,
                targets: jax.Array) -> jax.Array:
        p = make_poly(factors_data)
        replacements = [make_poly(rf) for rf in replacement_factors]

        # JIT-safe compose: shortcircuit=True, pack=True, static_shape=True.
        composed = p.compose(replacements, static_shape=True)

        # Evaluate is naturally JIT-safe.
        preds = composed.evaluate(points)            # AlgebraicArray
        return jnp.mean((preds.data - targets) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(1e-3)
    opt_state = opt.init(factors_data)

    # Training step
    def step(factors_data, replacement_factors, points, targets, opt_state):
        loss, grads = grad_fn(factors_data, replacement_factors, points, targets)
        updates, opt_state = opt.update(grads, opt_state)
        factors_data = optax.apply_updates(factors_data, updates)
        return factors_data, opt_state, loss

    step = jax.jit(step)

Notes on this pattern:

* Keep the *raw* ``jax.Array`` (``factors_data``) as the optimizer
  parameter, not the ``RankDecomposition``.  Reconstruct the dataclass
  inside the jitted function.  This sidesteps any subtlety around
  ``register_node_class`` / ``aux_data`` while still exercising the
  full operation graph under ``jit`` / ``grad``.
* If you prefer to keep the dataclass as the parameter, that also works
  -- ``RankDecomposition`` is a registered pytree, so
  ``jax.tree_util.tree_leaves(p)`` returns ``[p.factors.data]`` and
  the dataclass / ``AlgebraicArray`` aux data are reconstructed by JAX.
  ``optax`` will then update only the leaf.
* For batched training (one polynomial per example), ``vmap`` over the
  *batch* axis of ``factors_data`` and use the (already vectorized)
  batched code paths.  Equivalently, give the ``RankDecomposition`` a
  non-empty ``batch_shape`` (e.g. ``(B, R, D, N+1)``) and call the same
  methods -- they branch on ``self.batch_shape`` internally and use
  ``batched_compose_factors`` / ``batched_evaluate_factors``.  Both
  branches are JIT-safe with the same ``static_shape=True`` flag.
* Do **not** call ``normalize()`` inside the jitted region.  If you
  must canonicalize for evaluation/inspection, do it *outside* ``jit``,
  on detached arrays.


Recommended pattern for PyTorch
===============================

Use the ``torch`` backend; the same flag combination
(``shortcircuit=True``, ``pack=True``, ``static_shape=True``) is also
the right choice under ``torch.compile`` for the same reason -- value-
dependent shapes break the dynamo graph.  Outside ``torch.compile``
PyTorch's eager mode tolerates value-dependent shapes, but the path is
slower and you get no speedup; standardize on the static-shape path.

.. code-block:: python

    import torch
    import algebraic
    from algebraic.polynomials import RankDecomposition

    algebra = algebraic.semirings.boolean_algebra(mode="...")
    NUM_VARS, MAX_RANK, MAX_DEGREE = 8, 16, 6

    factors_data = torch.randn(MAX_RANK, MAX_DEGREE, NUM_VARS + 1,
                               requires_grad=True)
    optim = torch.optim.Adam([factors_data], lr=1e-3)

    def make_poly(t: torch.Tensor) -> RankDecomposition:
        factors = algebraic.array(t, semiring=algebra, backend="torch")
        return RankDecomposition(
            factors, max_rank=MAX_RANK, max_degree=MAX_DEGREE, backend="torch"
        )

    def loss_fn(factors_data, replacements_data, points, targets):
        p = make_poly(factors_data)
        replacements = [make_poly(r) for r in replacements_data]
        composed = p.compose(replacements, static_shape=True)
        preds = composed.evaluate(points)
        return ((preds.data - targets) ** 2).mean()

    for batch in loader:
        optim.zero_grad()
        loss = loss_fn(factors_data, batch.replacements, batch.points, batch.targets)
        loss.backward()
        optim.step()

If you want separate parameter groups for variable weights and bias,
use ``LowRankFactors`` instead:

.. code-block:: python

    from algebraic.polynomials import LowRankFactors

    weights = torch.randn(MAX_RANK, MAX_DEGREE, NUM_VARS, requires_grad=True)
    bias    = torch.randn(MAX_RANK, MAX_DEGREE,            requires_grad=True)

    optim = torch.optim.Adam([
        {"params": [weights], "lr": 1e-3},
        {"params": [bias],    "lr": 1e-4},  # e.g. slower bias
    ])

    def make_poly(w, b):
        w_a = algebraic.array(w, semiring=algebra, backend="torch")
        b_a = algebraic.array(b, semiring=algebra, backend="torch")
        return LowRankFactors(w_a, b_a, max_rank=MAX_RANK,
                              max_degree=MAX_DEGREE, backend="torch")

``LowRankFactors`` exposes the same ``compose`` / ``add`` / ``mul`` /
``evaluate`` API as ``RankDecomposition``.  Internally each operation
calls ``to_merged()`` (a single concat along the last axis) to assemble
the ``(R, D, N+1)`` form, runs the same kernels, and splits the result
back via ``_replace_merged``.  Gradient flow through the
concat/split is exact.


Partitioning trainable leaves from static metadata
===================================================

Both pipelines above reconstruct the dataclass by hand inside the loss
function (``make_poly``).  A cleaner alternative -- and the one we
recommend for non-trivial pipelines -- is to use
``algebraic.utils.pytree`` to *flatten* the dataclass into (leaves,
treespec) once, hand the leaves to the optimizer, and *unflatten* it
back into a live ``RankDecomposition`` / ``LowRankFactors`` inside the
training step.

``algebraic.utils.pytree`` is a re-export of ``optree`` bound to the
``"algebraic"`` namespace.  All algebraic dataclasses
(``AlgebraicArray``, ``RankDecomposition``, ``LowRankFactors``,
``PolyDict``, ``MonomialBasis``) are registered in that namespace, so
``pytree.flatten`` correctly partitions:

* **Leaves** -- the raw backend tensors (``factors.data``, or
  ``weights.data`` / ``bias.data``).  These are the things you want
  gradients on and that the optimizer should update.
* **Treespec / aux data** -- algebra, ``max_rank``, ``max_degree``,
  ``max_replacement_degree``, backend, plus the structural shape of
  the dataclass.  These are static; the treespec round-trips them
  through ``pytree.unflatten`` without any tracing.

The split is exactly what JAX transforms and ``torch.compile`` need:
trainable arrays are pure tensors, everything else is closed-over
Python state.

JAX
---

.. code-block:: python

    import jax
    import optax
    from algebraic.polynomials import RankDecomposition
    from algebraic.utils import pytree

    p0 = RankDecomposition(...)               # built once, on host
    leaves, spec = pytree.flatten(p0)          # leaves: list[jax.Array]

    opt = optax.adam(1e-3)
    opt_state = opt.init(leaves)

    def loss_fn(leaves, replacement_leaf_lists, points, targets):
        # Rebuild the live dataclass(es) -- no Python branching on tensors.
        p = pytree.unflatten(spec, leaves)
        replacements = [pytree.unflatten(spec, ls) for ls in replacement_leaf_lists]
        composed = p.compose(replacements, static_shape=True)
        preds = composed.evaluate(points)
        return ((preds.data - targets) ** 2).mean()

    @jax.jit
    def step(leaves, replacement_leaf_lists, points, targets, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(
            leaves, replacement_leaf_lists, points, targets
        )
        updates, opt_state = opt.update(grads, opt_state)
        leaves = optax.apply_updates(leaves, updates)
        return leaves, opt_state, loss

The ``spec`` is closed over by ``loss_fn`` and ``step`` -- it is
hashable Python data, so it is fine as a captured constant under
``jax.jit``.  Only ``leaves`` (and the per-step inputs) flow through
the gradient.

Because the algebraic types are also registered with ``jax.tree_util``,
you can equivalently let JAX do the partitioning for you and pass the
``RankDecomposition`` itself as a parameter pytree -- ``optax`` will
update only its leaves.  The explicit ``pytree.flatten`` / ``unflatten``
form is preferred when you want to be sure of *which* fields are
static (and to keep the optimizer state shape independent of the
dataclass layout).

PyTorch
-------

For PyTorch, ``algebraic.utils.torch`` provides the same idea wrapped
in an ``nn.Module``:

.. code-block:: python

    import torch
    from algebraic.polynomials import LowRankFactors
    from algebraic.utils.torch import torchify

    p0 = LowRankFactors(...)                  # built once, on host
    module = torchify(p0)                     # PyTreeModule[LowRankFactors]

    optim = torch.optim.Adam(module.parameters(), lr=1e-3)

    for batch in loader:
        optim.zero_grad()
        p = module()                          # reconstructs LowRankFactors
        composed = p.compose(batch.replacements, static_shape=True)
        preds = composed.evaluate(batch.points)
        loss = ((preds.data - batch.targets) ** 2).mean()
        loss.backward()
        optim.step()

``torchify(obj)`` calls ``pytree.flatten(obj)`` internally, registers
every leaf tensor as an ``nn.Parameter``, and stashes the treespec as
attribute state.  ``module()`` calls ``pytree.unflatten(spec,
list(self.parameters()))`` to give you back a fully live
``LowRankFactors`` (or ``RankDecomposition``, ``AlgebraicArray``,
``PolyDict``, ``MonomialBasis``) with autograd hooked up through the
parameters.  This composes cleanly with ``model.to(device)``,
``state_dict`` checkpointing, DDP, and ``torch.compile``.

If you prefer to skip the ``nn.Module`` wrapper, use
``pytree.flatten`` / ``pytree.unflatten`` directly as in the JAX
example -- the API is symmetric.

Either way, the rule of thumb is: **flatten once, outside the hot
path; unflatten inside the loss function**.  That keeps the static
treespec out of the gradient graph and the trainable leaves
unambiguously typed as backend tensors.


Shape contract for ``compose`` and ``evaluate``
================================================

``compose(replacements, ...)``

* ``self.factors`` shape: ``(*batch, R, D, N+1)``.
* ``replacements`` is a sequence of length ``self.num_vars`` (i.e. ``N``).
* Each ``replacements[i].factors`` shape: ``(*batch, R_i, D_i, N+1)``
  *or* ``(R_i, D_i, N+1)`` (the unbatched form is broadcast across the
  batch by ``prepare_replacement_factors``).
* All replacements must share the same ``num_vars`` ``N`` as ``self``.
* Output ``factors`` shape:

  * with ``static_shape=True``: ``(*batch, R', D', N+1)`` where
    ``R' <= max_rank`` and ``D' <= max_degree`` (both reached via
    static slices, so the shape is statically inferable from
    ``max_rank`` / ``max_degree`` and the input shapes -- safe under
    ``jit`` / ``vmap``).
  * with ``static_shape=False``: same upper bounds, but the actual
    dimensions can be smaller (value-dependent) due to the dynamic
    trailing slices inside the packs.

``evaluate(points)``

* Unbatched: ``points`` of shape ``(N,)`` -> result is a scalar
  ``AlgebraicArray``.
* Batched: ``points`` of shape ``(B, N)`` -> result of shape ``(B,)``.
* Internally an ``einsum`` over the variable axis followed by a
  ``prod`` over the degree axis and a ``sum`` over the rank axis.
  Fully differentiable.


Practical guidance
==================

1. Start with ``LowRankFactors`` if you want clean parameter groups,
   otherwise ``RankDecomposition`` is fine.
2. Pick a continuous lattice algebra (``boolean_algebra(mode=...)``
   relaxation, max-min, or tropical) -- crisp Boolean has no gradient.
3. Initialize ``factors_data`` (or ``weights`` / ``bias``) as a normal
   backend tensor with ``requires_grad=True`` (Torch) or as the
   optimizer state (JAX).  Do not initialize at the algebra's identity
   element -- a small random perturbation makes the sort keys in
   ``pack_non_identity_slots`` / ``pack_non_zero_components`` strictly
   ordered and the gradient cleaner.
4. Always pass ``static_shape=True`` (which forces ``shortcircuit=True``
   and ``pack=True``) inside the jitted/compiled region.  Outside it,
   you can call the smarter pruning passes (``shortcircuit=False``)
   between training iterations to actively compress the
   representation -- this is the analogue of pruning a sparse model.
5. Choose ``max_rank`` and ``max_degree`` generously.  These are *hard*
   static bounds; rank-1 components past ``max_rank`` (after the
   non-zero pack) and degree slots past ``max_degree`` (after the
   non-identity pack) get a zero gradient.  Increasing them later is
   trivial (re-pad), shrinking them is lossy.
6. ``compose`` is the most expensive op (beam search over the degree
   axis); inside ``jit`` it produces a fully unrolled graph of length
   ``D``.  Avoid recomposing more depths than you actually need per
   step.
7. Never call ``normalize()``, ``to_sparse()``, or ``from_sparse()``
   inside the gradient path.  Use them only in evaluation /
   diagnostics outside of ``jit`` / ``compile``.
8. ``add`` / ``mul`` default to ``shortcircuit=False``; pass
   ``static_shape=True`` (and the implied flags) explicitly when you
   call them from a jitted region.

A minimal "training step" checklist:

* Trainable leaves: raw backend arrays (``factors.data`` /
  ``weights.data`` / ``bias.data``).  Static aux: algebra, ranks,
  degrees, backend.
* Inside ``jit`` / ``compile``: only ``add``, ``mul``, ``compose``,
  ``evaluate`` -- all with ``static_shape=True``.
* Outside ``jit`` / ``compile`` (optional, between steps):
  ``prune(shortcircuit=False)`` to compact the representation.
* Loss: any scalar function of ``composed.evaluate(points).data``.
* Gradients: standard ``jax.grad`` / ``loss.backward()``.

If the agent follows this contract, the factor tensors stay in the
gradient path through composition and evaluation, the operations
trace cleanly under ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` (or
``torch.compile``), and no further inspection of the ``algebraic``
source is required.
