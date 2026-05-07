"""Polynomial factor manipulation utilities for CP/rank decompositions.

All free-standing helpers for operating on CP factor arrays (shape ``(R, D, N+1)``)
live here, keeping the dataclass files (``RankDecomposition``, ``LowRankFactors``)
focused on the data types.
"""

from __future__ import annotations

from collections.abc import Sequence

import array_api_compat
from bitarray import frozenbitarray

import algebraic.ops as algebraic
from algebraic.array import AlgebraicArray
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Array, Backend, Scalar, is_array


def _eq_or_close(a: AlgebraicArray, b: AlgebraicArray, atol: float) -> Array:
    """Elementwise equality with optional tolerance.

    ``atol == 0`` -> exact equality; ``atol > 0`` -> ``isclose`` with that tol.
    """
    if atol > 0.0:
        return algebraic.isclose(a, b, rtol=0.0, atol=atol)
    return algebraic.equal(a, b)


# -- Core evaluation & composition -------------------------------------------------


def evaluate_factors(factors: AlgebraicArray, points: Array | AlgebraicArray, backend: str | Backend) -> AlgebraicArray:
    """Evaluate CP factors at a given point.

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.
    points : Array or AlgebraicArray
        An array of shape ``(num_vars,)`` to replace each variable with.
    backend : str or Backend
        Backend to use.

    Returns
    -------
    AlgebraicArray
        The evaluated value with scalar (in the algebra) value.
    """
    device = factors.device
    algebra = factors.semiring
    one_array = algebraic.ones((1,), semiring=algebra, backend=backend, device=device)
    if is_array(points):
        points_array = algebraic.array(points, semiring=algebra, backend=backend, device=device)
    else:
        points_array = points
    selector = algebraic.concat([one_array, points_array])

    # (R, D, N+1) x (N+1,) -> (R, D): inner sum over variables
    contracted = algebraic.einsum("rdk,k->rd", factors, selector)
    # (R, D) -> (R,): product over degree axis
    degree_prod = algebraic.prod(contracted, axis=1)
    # (R,) -> (): sum over rank axis
    result = algebraic.sum(degree_prod, axis=0)

    return result


def batched_evaluate_factors(
    factors: AlgebraicArray,
    points: Array | AlgebraicArray,
    backend: str | Backend,
) -> AlgebraicArray:
    """Evaluate a batch of CP factors at corresponding points.

    Parameters
    ----------
    factors : AlgebraicArray
        Batch of CP factors of shape ``(B, R, D, N+1)``.
    points : Array or AlgebraicArray
        An array of shape ``(B, num_vars)`` to replace each variable with.
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.
    backend : str or Backend
        Backend to use.

    Returns
    -------
    AlgebraicArray
        The evaluated array of shape ``(B,)``.
    """
    batch = factors.shape[0]
    device = factors.device
    algebra = factors.semiring
    one_col = algebraic.ones((batch, 1), semiring=algebra, backend=backend, device=device)
    if is_array(points):
        points_array = algebraic.array(points, semiring=algebra, backend=backend, device=device)
    else:
        points_array = points
    selector = algebraic.concat([one_col, points_array], axis=1)

    # (B, R, D, N+1) x (B, N+1) -> (B, R, D): inner sum over variables
    contracted = algebraic.einsum("brdk,bk->brd", factors, selector)
    # (B, R, D) -> (B, R): product over degree axis
    degree_prod = algebraic.prod(contracted, axis=2)
    # (B, R) -> (B,): sum over rank axis
    result = algebraic.sum(degree_prod, axis=1)

    return result


def compose_factors(
    factors: AlgebraicArray,
    replacement_factors: Sequence[AlgebraicArray],
    max_rank: int,
    max_degree: int | None,
    *,
    atol: float = 1e-6,
    shortcircuit: bool = False,
    pack: bool = True,
    static_shape: bool = False,
) -> AlgebraicArray:
    """Compose CP factors with replacement factor arrays.

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.
    replacement_factors : Sequence[AlgebraicArray]
        Sequence of N replacement factor arrays, each of shape ``(R_i, D_i, N+1)``.
    max_rank : int
        Maximum rank for pruning.
    max_degree : int or None
        Maximum degree for pruning (``None`` disables degree reduction).
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.

    Returns
    -------
    AlgebraicArray
        New CP factors of shape ``(R', D', N+1)``.
    """
    algebra = factors.semiring
    assert isinstance(algebra, Lattice)
    # shape: (N+1, R2, D2, N+1)
    q_factors = prepare_replacement_factors(replacement_factors, algebra)

    # n-mode contraction over the variable axis ``k``.
    #
    # An obvious-but-wrong implementation is
    #     einsum("pdk,kqev->pdqev", factors, q_factors)
    # which sums over ``k``.  For monomial source slots (a single non-zero in
    # ``factors[p, d, :]``) this is correct -- the einsum simply selects the
    # corresponding ``q_factors[k]`` column.  For multi-non-zero "non-monomial"
    # slots (produced by :func:`merge_compatible_components` via the
    # distributive law) the einsum's ``OR``-sum collapses two CP-factor
    # tensors element-wise, which is **not** the CP representation of their
    # polynomial sum (CP sum is rank-axis concatenation, not elementwise OR).
    #
    # The correct contraction broadcasts ``factors`` and ``q_factors`` and
    # reshapes the (k, q) axes into a single rank axis: each (p, d, k, q)
    # stripe is ``factors[p, d, k] * q_factors[k, q]`` (a weighted rank-1
    # component of ``q_factors[k]``).  Stripes where ``factors[p, d, k] == 0``
    # become all-zero rank-1 components, which contribute 0 to the polynomial
    # sum -- exactly the behaviour we want.
    p, d, k = factors.shape
    _, q, e, v = q_factors.shape
    expanded = factors[:, :, :, None, None, None] * q_factors[None, None, :, :, :, :]
    # (p, d, k, q, e, v) -> (p, d, k*q, e, v)
    contracted = algebraic.reshape(expanded, (p, d, k * q, e, v))

    return contraction_compression(
        contracted, max_rank, max_degree, atol=atol, shortcircuit=shortcircuit, pack=pack, static_shape=static_shape
    )


def prepare_replacement_factors(
    replacement_factors: Sequence[AlgebraicArray],
    algebra: Lattice,
    batch_shape: tuple[int, ...] = (),
) -> AlgebraicArray:
    """Prepare padded array of replacement factor arrays.

    Each replacement may be either:
    - unbatched ``(R_i, D_i, N+1)``, or
    - batched ``(*batch_shape, R_i, D_i, N+1)`` matching ``batch_shape``.

    Unbatched replacements are broadcast across the batch.  ``batch_shape=()``
    yields the original unbatched output shape.

    Parameters
    ----------
    replacement_factors : Sequence[AlgebraicArray]
        Sequence of N replacement factor arrays.
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.
    batch_shape : tuple[int, ...], optional
        Leading batch shape; ``()`` for unbatched output.

    Returns
    -------
    AlgebraicArray
        Padded array of shape ``(*batch_shape, N+1, R_max, D_max, N+1)``.
        Index 0 along the variable axis: constant (identity: always 1).
        Index i+1: replacement for variable ``x_i``.
    """
    bs_len = len(batch_shape)

    def _rdn(q: AlgebraicArray) -> tuple[int, int, int]:
        # Strip leading batch dims if present; unbatched replacements (ndim == 3)
        # are broadcast across the batch.
        offset = q.ndim - 3
        return q.shape[offset], q.shape[offset + 1], q.shape[offset + 2]

    target_rank, target_degree, n_plus_1 = tuple(
        map(  # pyrefly: ignore[bad-specialization]
            max,
            zip(*(_rdn(q) for q in replacement_factors)),
        )
    )
    backend = Backend.from_array(replacement_factors[0].data)
    device = replacement_factors[0].device

    # Identity factors: (1, 1, N+1) -> (target_rank, target_degree, N+1)
    one_factors_base = algebraic.zeros((1, 1, n_plus_1), semiring=algebra, backend=backend, device=device)
    one_factors_base = one_factors_base.at[(0, 0, 0)].set(algebra.one)
    one_factors = pad_upto(one_factors_base, max_rank=target_rank, max_degree=target_degree)
    if batch_shape:
        one_factors = algebraic.broadcast_to(one_factors, (*batch_shape, target_rank, target_degree, n_plus_1))

    padded: list[AlgebraicArray] = [one_factors]
    for q in replacement_factors:
        if batch_shape and q.ndim == 3:
            # Unbatched replacement: pad then broadcast across batch
            q_padded = pad_upto(q, max_rank=target_rank, max_degree=target_degree)
            q_padded = algebraic.broadcast_to(q_padded, (*batch_shape, target_rank, target_degree, n_plus_1))
        else:
            q_padded = pad_upto(q, max_rank=target_rank, max_degree=target_degree)
        padded.append(q_padded)

    new_replacements = algebraic.stack(padded, axis=bs_len)

    assert new_replacements.shape == (*batch_shape, n_plus_1, target_rank, target_degree, n_plus_1)
    return new_replacements


def pad_upto(
    factors: AlgebraicArray,
    *,
    max_rank: int,
    max_degree: int,
) -> AlgebraicArray:
    """Pad rank/degree axes with identity elements up to the given maximum.

    Supports any leading batch shape: ``factors`` may be ``(R, D, N+1)`` or
    ``(*batch, R, D, N+1)``.
    """
    *batch_shape, rank, degree, n_plus_1 = factors.shape

    if max_rank <= rank and max_degree <= degree:
        return factors

    backend = Backend.from_array(factors.data)
    device = factors.device
    algebra = factors.semiring

    new_rank = max(rank, max_rank)
    new_degree = max(degree, max_degree)
    return_shape = (*batch_shape, new_rank, new_degree, n_plus_1)

    # Base: all identity, then overwrite existing slots
    one_terms = algebraic.broadcast_to(
        algebraic.eye(1, n_plus_1, semiring=algebra, backend=backend, device=device),
        (*batch_shape, rank, new_degree, n_plus_1),
    )
    degree_padded = one_terms.at[..., :, :degree, :].set(factors)

    if new_rank > rank:
        zero_terms = algebraic.zeros(
            (*batch_shape, new_rank - rank, new_degree, n_plus_1), semiring=algebra, backend=backend, device=device
        )
        rank_padded = algebraic.concat((degree_padded, zero_terms), axis=-3)
    else:
        rank_padded = degree_padded

    assert rank_padded.shape == return_shape
    return rank_padded


def contraction_compression(
    contracted: AlgebraicArray,
    max_rank: int,
    max_degree: int | None,
    *,
    atol: float = 1e-6,
    shortcircuit: bool = False,
    pack: bool = True,
    static_shape: bool = False,
) -> AlgebraicArray:
    """Beam search over tensor contractions.

    Parameters
    ----------
    contracted : AlgebraicArray
        Shape ``[R, D, R_max, D_max, N+1]``.
    max_rank : int
        Maximum rank for pruning at each beam step.
    max_degree : int or None
        Maximum degree for pruning (``None`` disables degree reduction).
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.

    Returns
    -------
    AlgebraicArray
        Shape ``[R_out, D_out, N+1]``.
    """
    rank1, degree1, rank2, degree2, n_plus_1 = contracted.shape
    backend = Backend.from_array(contracted.data)
    algebra = contracted.semiring

    # (R, D, R_max, D_max, N+1) -> (D, R*R_max, D_max, N+1)
    candidates = algebraic.permute_dims(contracted, (1, 0, 2, 3, 4))
    candidates = algebraic.reshape(candidates, (degree1, rank1 * rank2, degree2, n_plus_1))

    device = contracted.device

    if degree1 == 0:
        return algebraic.broadcast_to(
            algebraic.eye(1, n_plus_1, semiring=algebra, backend=backend, device=device), (1, 1, n_plus_1)
        )

    # Initialize beam with the first candidate directly (skips an identity-multiply
    # that would otherwise leave a leading identity slot the trailing-only
    # ``strip_identity_slots`` could not remove).
    beam = candidates[0]
    beam = prune_factors(
        beam, max_rank, max_degree, atol=atol, shortcircuit=shortcircuit, pack=pack, static_shape=static_shape
    )

    for d in range(1, degree1):
        candidate_d = candidates[d]  # (rank1 * rank2, degree2, n+1)
        beam = _multiply_factors(beam, candidate_d)
        beam = prune_factors(
            beam, max_rank, max_degree, atol=atol, shortcircuit=shortcircuit, pack=pack, static_shape=static_shape
        )

    return beam


# -- Pruning utilities ----------------------------------------------------------


def deduplicate_rank_dim(factors: AlgebraicArray, atol: float = 1e-6) -> AlgebraicArray:
    """Remove duplicate rank components (keep first occurrence of each)."""
    a = factors[:, None, :, :]  # (rank, 1, d, n+1)
    b = factors[None, :, :, :]  # (1, rank, d, n+1)

    # eq[i, j] = True iff row i equals row j
    eq = _eq_or_close(a, b, atol).all((2, 3))  # (rank, rank) raw bool array

    backend = Backend.from_array(factors.data)
    xp = backend.get_array_namespace()
    rank = factors.shape[0]
    arange = xp.arange(rank)
    earlier = arange[:, None] < arange[None, :]  # (rank, rank)
    earlier = array_api_compat.to_device(earlier, factors.device)

    # is_dup[j] = True if some earlier row i (i < j) is identical to row j
    is_dup = (eq & earlier).any(0)
    keep = ~is_dup

    return factors[keep]


def idempotence_pruning(factors: AlgebraicArray, atol: float = 1e-6) -> AlgebraicArray:
    """Remove terms dominated by lattice idempotence laws."""
    # p <= q if p + q == q  (p is dominated by q)
    a = factors[:, None, :, :]  # (rank, 1, d, n+1)
    b = factors[None, :, :, :]  # (1, rank, d, n+1)

    added = a + b
    check = _eq_or_close(added, b, atol).all((2, 3))  # check[i,j] = (a[i] <= a[j])

    backend = Backend.from_array(factors.data)
    xp = backend.get_array_namespace()
    rank = factors.shape[0]
    arange = xp.arange(rank)
    off_diag = arange[:, None] != arange[None, :]  # True where i != j
    off_diag = array_api_compat.to_device(off_diag, factors.device)

    check = check & off_diag

    # keep[i] = True iff no other j dominates i
    keep = ~check.any(1)

    return factors[keep]


# -- New pruning strategies --------------------------------------------------


def strip_identity_slots(factors: AlgebraicArray, atol: float = 1e-6) -> AlgebraicArray:
    """Strip trailing degree slots that are multiplicative identity across all rank components.

    Works on any leading batch shape: ``factors`` may be ``(R, D, N+1)`` or
    ``(*batch, R, D, N+1)``.  A slot ``factors[..., r, k, :]`` is identity when
    it equals ``[algebra.one, algebra.zero, ..., algebra.zero]`` (contributes
    factor 1 to the product for every input).  If the *last* slot is identity
    for **all** batch + rank components it can be removed without changing the
    polynomial.  The function trims all such trailing slots, keeping at least
    one.

    .. note::
       This helper is **not** JIT-safe: the trailing-identity loop branches on
       array values via ``bool(...)`` and produces value-dependent output
       shapes.  For JIT/grad/vmap callers, use the default ``pack=True`` path
       (which routes through :func:`pack_non_identity_slots`, a shape-preserving
       variant) instead of ``pack=False``.

    Complexity: O(B * R * D * N).

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(*batch, R, D, N+1)``.

    Returns
    -------
    AlgebraicArray
        CP factors of shape ``(*batch, R, D', N+1)`` with ``1 <= D' <= D``.
    """
    degree = factors.shape[-2]
    n_plus_1 = factors.shape[-1]
    if degree <= 1:
        return factors

    backend = Backend.from_array(factors.data)
    device = factors.device
    algebra = factors.semiring

    identity_row = algebraic.zeros((n_plus_1,), semiring=algebra, backend=backend, device=device)
    identity_row = identity_row.at[0].set(algebra.one)

    # Walk degree axis from the end; trim while all (batch, rank) are identity at that slot
    new_degree = degree
    for k in range(degree - 1, 0, -1):  # stop at 1 to keep at least degree 1
        slot = factors[..., k, :]  # (*batch, R, N+1)
        if bool(_eq_or_close(slot, identity_row, atol).all()):
            new_degree -= 1
        else:
            break

    if new_degree == degree:
        return factors
    return factors[..., :, :new_degree, :]


def pack_non_identity_slots(
    factors: AlgebraicArray,
    atol: float = 1e-6,
    *,
    static_shape: bool = False,
) -> AlgebraicArray:
    """Pack non-identity degree slots to the front for every rank component.

    Works on any leading batch shape: ``factors`` may be ``(R, D, N+1)`` or
    ``(*batch, R, D, N+1)``.  Identity slots ``[one, zero, ..., zero]`` are
    pushed to the back of the degree axis via a stable sort on a 0/1 key.
    With ``static_shape=False`` (default) the trailing identity region is
    then sliced off based on the global max non-identity count -- a free
    degree reduction.  Because rank-1 components are commutative products
    over the degree axis, this rearrangement is mathematically exact
    (modulo the equality tolerance ``atol``).

    With ``static_shape=True`` the output shape matches the input shape;
    the trailing identity region is left in place (each identity slot
    multiplies the rank-1 product by 1, so it is mathematically inert).
    This mode is JIT-safe -- no value-dependent shapes -- and is required
    for use inside ``jax.jit`` / ``jax.vmap`` / ``jax.grad``.  Any
    compaction below ``D`` is then the caller's responsibility (typically
    via a static slice ``factors[..., :max_degree, :]``).

    Compared to :func:`strip_identity_slots` this also handles identity slots
    that appear in the *middle* of the degree axis (e.g. introduced by
    ``pad_upto`` padding or beam initialisation).

    Complexity: O(B * R * D * log D + B * R * D * N).

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(*batch, R, D, N+1)``.
    atol : float, optional
        Tolerance for the identity-slot equality check.  ``0`` = exact equality.
    static_shape : bool, optional
        When ``True``, skip the value-dependent trailing slice; the output
        keeps the input degree dimension intact.  Required for JIT.  Default
        ``False`` preserves the historical free-compaction behaviour.

    Returns
    -------
    AlgebraicArray
        CP factors with non-identity slots packed to the front along the
        degree axis.  Shape ``(*batch, R, D', N+1)`` with ``1 <= D' <= D``
        (``static_shape=False``) or exactly ``(*batch, R, D, N+1)``
        (``static_shape=True``).
    """
    *batch_shape, rank, degree, n_plus_1 = factors.shape
    if degree <= 1:
        return factors

    backend = Backend.from_array(factors.data)
    device = factors.device
    algebra = factors.semiring
    xp = backend.get_array_namespace()

    identity_row = algebraic.zeros((n_plus_1,), semiring=algebra, backend=backend, device=device)
    identity_row = identity_row.at[0].set(algebra.one)

    # is_identity[..., r, d] = True iff slot (r, d) is the identity row
    is_identity = _eq_or_close(factors, identity_row, atol).all(-1)  # (*batch, R, D)

    # Stable sort: identity slots (key=1) move to the back, non-identity (key=0) stay in order
    sort_key = xp.astype(is_identity, xp.int32)  # (*batch, R, D)
    indices = xp.argsort(sort_key, axis=-1, stable=True)  # (*batch, R, D)
    indices_b = xp.broadcast_to(indices[..., None], (*batch_shape, rank, degree, n_plus_1))

    packed = algebraic.take_along_axis(factors, indices_b, axis=-2)
    if static_shape:
        return packed

    # Output degree is the global max non-identity count, at least 1
    non_identity_count = xp.sum(xp.astype(~is_identity, xp.int32), axis=-1)  # (*batch, R)
    new_degree = int(xp.max(non_identity_count))
    if new_degree < 1:
        new_degree = 1
    if new_degree >= degree:
        return packed
    return packed[..., :, :new_degree, :]


def pack_non_zero_components(
    factors: AlgebraicArray,
    atol: float = 1e-6,
    *,
    static_shape: bool = False,
) -> AlgebraicArray:
    """Sort rank-1 components so non-zero components come first.

    A rank-1 component is the zero polynomial iff *any* of its degree slots is
    the all-zero coefficient row (since the rank-1 product picks up a zero
    factor and the absorbing law ``0 * x = 0`` kills the component).  Such
    components contribute nothing to the additive sum-over-rank, so they are
    safe to push to the back of the rank axis.  Doing so before a hard
    ``factors[:max_rank]`` truncation guarantees we keep meaningful components
    rather than padded zero ones.

    Works on any leading batch shape.  With ``static_shape=False`` (default)
    the trailing all-zero-rank region is sliced off based on the per-element
    max non-zero count (free rank reduction); the output has dynamic rank.
    With ``static_shape=True`` the output shape matches the input shape (no
    truncation, JIT-safe).

    Complexity: O(B * R * D * N).

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(*batch, R, D, N+1)``.
    atol : float, optional
        Tolerance for the all-zero check.  ``0`` = exact equality with the
        algebra's zero element.
    static_shape : bool, optional
        When ``True``, skip the value-dependent trailing slice; the output
        keeps the input rank dimension intact.  Required for JIT.

    Returns
    -------
    AlgebraicArray
        CP factors with non-zero rank-1 components at the front of the rank
        axis.  Shape ``(*batch, R', D, N+1)`` with ``1 <= R' <= R``
        (``static_shape=False``) or exactly ``(*batch, R, D, N+1)``
        (``static_shape=True``).
    """
    *batch_shape, rank, degree, n_plus_1 = factors.shape
    if rank <= 1:
        return factors

    backend = Backend.from_array(factors.data)
    device = factors.device
    algebra = factors.semiring
    xp = backend.get_array_namespace()

    zero_row = algebraic.zeros((n_plus_1,), semiring=algebra, backend=backend, device=device)

    # is_zero_slot[..., r, d] = True iff slot (r, d) is the all-zero row
    is_zero_slot = _eq_or_close(factors, zero_row, atol).all(-1)  # (*batch, R, D)
    # is_zero_rank[..., r] = True iff *any* slot of rank-1 r is all-zero
    # (which makes the whole rank-1 the zero polynomial via the absorbing law)
    is_zero_rank = is_zero_slot.any(-1)  # (*batch, R)

    # Stable sort: zero rank-1s (key=1) move to the back, non-zero (key=0) stay in order
    sort_key = xp.astype(is_zero_rank, xp.int32)  # (*batch, R)
    indices = xp.argsort(sort_key, axis=-1, stable=True)  # (*batch, R)
    indices_b = xp.broadcast_to(indices[..., None, None], (*batch_shape, rank, degree, n_plus_1))

    packed = algebraic.take_along_axis(factors, indices_b, axis=-3)
    if static_shape:
        return packed

    # Dynamic slice: keep only as many ranks as the global max non-zero count
    # (across all batch elements; ensures uniform output shape).
    non_zero_count = xp.sum(xp.astype(~is_zero_rank, xp.int32), axis=-1)  # (*batch,)
    new_rank = int(xp.max(non_zero_count))
    if new_rank < 1:
        new_rank = 1
    if new_rank >= rank:
        return packed
    return packed[..., :new_rank, :, :]


def merge_compatible_components(factors: AlgebraicArray, atol: float = 1e-6) -> AlgebraicArray:
    """Merge rank-1 component pairs that differ at exactly one slot.

    Uses distributivity: ``(common * f_j) + (common * g_j) = common * (f_j + g_j)``.
    Each successful merge reduces rank by 1 with no change to degree.

    Complexity: O(R^2 * D * N) per pass.

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.

    Returns
    -------
    AlgebraicArray
        CP factors of shape ``(R', D, N+1)`` with ``R' <= R``.
    """
    backend = Backend.from_array(factors.data)
    xp = backend.get_array_namespace()

    # At most rank-1 merges possible; one iteration per merge
    for _ in range(factors.shape[0]):
        rank, degree, n_plus_1 = factors.shape
        if rank < 2:
            break

        # slot_eq[i, j, k] = True iff components i and j are identical at slot k
        a = factors[:, None, :, :]  # (R, 1, D, N+1)
        b = factors[None, :, :, :]  # (1, R, D, N+1)
        slot_eq = _eq_or_close(a, b, atol).all(3)  # (R, R, D) raw bool array

        # match_count[i, j] = number of matching slots
        match_count = xp.sum(slot_eq, axis=2)  # (R, R) int array

        # Search for the first upper-triangular pair with degree - 1 matching slots
        found = False
        for i in range(rank):
            for j in range(i + 1, rank):
                if int(match_count[i, j]) == degree - 1:
                    # Find the single differing slot
                    slot_eq_ij = slot_eq[i, j, :]  # type: ignore[index]  # (D,) bool
                    diff_k = -1
                    for k in range(degree):
                        if not bool(slot_eq_ij[k]):
                            diff_k = k
                            break
                    if diff_k < 0:
                        continue

                    # Build merged component: copy of component i with slot diff_k joined
                    merged_slot = (
                        factors[i : i + 1, diff_k : diff_k + 1, :] + factors[j : j + 1, diff_k : diff_k + 1, :]
                    )  # (1, 1, N+1)
                    comp_i = factors[i : i + 1, :, :]  # (1, D, N+1)
                    parts: list[AlgebraicArray] = []
                    if diff_k > 0:
                        parts.append(comp_i[:, :diff_k, :])
                    parts.append(merged_slot)
                    if diff_k < degree - 1:
                        parts.append(comp_i[:, diff_k + 1 :, :])
                    merged_component = algebraic.concat(parts, axis=1)  # (1, D, N+1)

                    # Remove i and j from factors, append merged component
                    arange = xp.arange(rank)
                    arange = array_api_compat.to_device(arange, factors.device)
                    keep = (arange != i) & (arange != j)
                    remaining = factors[keep]  # (R-2, D, N+1)
                    factors = algebraic.concat([remaining, merged_component], axis=0)

                    found = True
                    break
            if found:
                break

        if not found:
            break

    return factors


def _component_to_monomials(
    component: AlgebraicArray,
    num_vars: int,
    backend: str | Backend,
) -> dict[frozenbitarray, AlgebraicArray]:
    """Run subset-DP on a single rank-1 CP component to get its monomial expansion.

    Parameters
    ----------
    component : AlgebraicArray
        Single-component factors of shape ``(D, N+1)``.
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.
    num_vars : int
        Number of polynomial variables (``N``).
    backend : str or Backend
        Backend to use.

    Returns
    -------
    dict
        Mapping from ``frozenbitarray`` monomial masks to coefficient ``AlgebraicArray``.
    """
    device = component.device
    algebra = component.semiring
    zero = algebraic.zeros((), semiring=algebra, backend=backend, device=device)
    one = algebraic.ones((), semiring=algebra, backend=backend, device=device)
    degree = component.shape[0]

    dp: dict[int, AlgebraicArray] = {0: one}

    for k in range(degree):
        new_dp: dict[int, AlgebraicArray] = {}
        for mask, c in dp.items():
            const_factor: AlgebraicArray = component[k, 0]
            if not bool(algebraic.allclose(const_factor, zero)):
                contribution = const_factor * c
                new_dp[mask] = (new_dp[mask] + contribution) if mask in new_dp else contribution
            for i in range(num_vars):
                var_factor: AlgebraicArray = component[k, i + 1]
                if bool(algebraic.allclose(var_factor, zero)):
                    continue
                new_mask = mask | (1 << i)
                contribution = var_factor * c
                new_dp[new_mask] = (new_dp[new_mask] + contribution) if new_mask in new_dp else contribution
        dp = new_dp

    result: dict[frozenbitarray, AlgebraicArray] = {}
    for mask, coeff in dp.items():
        if not bool(algebraic.allclose(coeff, zero)):
            bits = frozenbitarray([bool(mask & (1 << i)) for i in range(num_vars)])
            result[bits] = (result[bits] + coeff) if bits in result else coeff

    return result


def _monomials_to_factors(
    monomials: dict[frozenbitarray, AlgebraicArray],
    max_degree: int,
    num_vars: int,
    algebra: Lattice,
    backend: str | Backend,
    device: object | None,
) -> AlgebraicArray:
    """Convert a monomial dict to CP factors of shape ``(R, max_degree, N+1)``.

    Parameters
    ----------
    monomials : dict
        Mapping from monomial bitmask to coefficient.
    max_degree : int
        Target degree for the output factors.
    num_vars : int
        Number of polynomial variables (``N``).
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.
    backend : str or Backend
        Backend to use.
    device : object or None
        Target device.

    Returns
    -------
    AlgebraicArray
        Factors of shape ``(len(monomials), max_degree, N+1)``.
    """
    rank = len(monomials)
    factors = algebraic.zeros((rank, max_degree, num_vars + 1), semiring=algebra, backend=backend, device=device)

    for r, (monomial, coeff) in enumerate(monomials.items()):
        vars_in_monomial = [i for i, bit in enumerate(monomial) if bit]

        if len(vars_in_monomial) == 0:
            factors = factors.at[(r, 0, 0)].set(coeff)
            for k in range(1, max_degree):
                factors = factors.at[(r, k, 0)].set(algebra.one)
        else:
            for k, var_idx in enumerate(vars_in_monomial):
                if k < max_degree:
                    val: AlgebraicArray | Scalar = coeff if k == 0 else algebra.one
                    factors = factors.at[(r, k, var_idx + 1)].set(val)
            for k in range(min(len(vars_in_monomial), max_degree), max_degree):
                factors = factors.at[(r, k, 0)].set(algebra.one)

    return factors


def _pack_component(
    component: AlgebraicArray,
    max_degree: int,
    atol: float = 1e-6,
) -> AlgebraicArray:
    """Pack a single rank-1 component's non-identity slots to the front.

    Collects all non-identity slots (at most ``max_degree``) and returns a new
    component of shape ``(max_degree, N+1)`` with non-identity slots first,
    followed by identity padding.

    Parameters
    ----------
    component : AlgebraicArray
        Single-component factors of shape ``(D, N+1)``.
    algebra : BoundedDistributiveLattice
        Lattice algebra used to determine the identity element.
    max_degree : int
        Target degree of the output.

    Returns
    -------
    AlgebraicArray
        Packed factors of shape ``(max_degree, N+1)``.
    """
    degree, n_plus_1 = component.shape
    backend = Backend.from_array(component.data)
    device = component.device
    algebra = component.semiring

    identity_row = algebraic.zeros((1, n_plus_1), semiring=algebra, backend=backend, device=device)
    identity_row = identity_row.at[(0, 0)].set(algebra.one)

    non_identity: list[AlgebraicArray] = []
    for k in range(degree):
        slot = component[k : k + 1, :]  # (1, N+1)
        if not bool(_eq_or_close(slot, identity_row, atol).all()):
            non_identity.append(slot)
            if len(non_identity) >= max_degree:
                break  # Reached max; any extras are dropped (lossy)

    # Pad with identity slots to reach max_degree
    padding_needed = max_degree - len(non_identity)
    parts = non_identity + [identity_row] * padding_needed
    return algebraic.concat(parts, axis=0)  # (max_degree, N+1)


def reduce_degree(factors: AlgebraicArray, max_degree: int, max_rank: int, atol: float = 1e-6) -> AlgebraicArray:
    """Reduce degree by decomposing over-degree components via monomial expansion.

    For each component whose effective degree (number of non-identity slots)
    exceeds ``max_degree``, runs the subset-DP to get monomials and reconstructs
    at ``max_degree`` (lossy for monomials with degree > ``max_degree``).
    Good components (effective degree <= ``max_degree``) are compacted by packing
    their non-identity slots to the front and padding with identity.

    Complexity: O(R * D * N) for good components + O(K * D * 2^n * n) for bad,
    where K = number of over-bound components.

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.
    max_degree : int
        Target maximum degree.
    max_rank : int
        Maximum rank for dedup/idempotence pruning after reconstruction.

    Returns
    -------
    AlgebraicArray
        CP factors of shape ``(R', max_degree, N+1)`` with ``R' <= R + num_bad_monomials``.
    """
    rank, degree, n_plus_1 = factors.shape
    num_vars = n_plus_1 - 1
    backend = Backend.from_array(factors.data)
    device = factors.device
    algebra = factors.semiring
    assert isinstance(algebra, Lattice)

    if degree <= max_degree:
        return factors

    identity_row = algebraic.zeros((1, n_plus_1), semiring=algebra, backend=backend, device=device)
    identity_row = identity_row.at[(0, 0)].set(algebra.one)

    good_packed: list[AlgebraicArray] = []
    bad_monomials: dict[frozenbitarray, AlgebraicArray] = {}

    for r in range(rank):
        component = factors[r]  # (D, N+1)

        # Count non-identity slots
        eff_degree = 0
        for k in range(degree):
            slot = component[k : k + 1, :]  # (1, N+1)
            if not bool(_eq_or_close(slot, identity_row, atol).all()):
                eff_degree += 1

        if eff_degree <= max_degree:
            # Good component: pack non-identity slots to front, shape (max_degree, N+1)
            packed = _pack_component(component, max_degree, atol)
            good_packed.append(packed[None, :, :])  # (1, max_degree, N+1)
        else:
            # Bad component: run subset-DP and merge monomials
            comp_monomials = _component_to_monomials(component, num_vars, backend)
            for bits, coeff in comp_monomials.items():
                bad_monomials[bits] = (bad_monomials[bits] + coeff) if bits in bad_monomials else coeff

    parts: list[AlgebraicArray] = []

    if good_packed:
        parts.append(algebraic.concat(good_packed, axis=0))  # (G, max_degree, N+1)

    if bad_monomials:
        parts.append(_monomials_to_factors(bad_monomials, max_degree, num_vars, algebra, backend, device))

    if not parts:
        return algebraic.zeros((0, max_degree, n_plus_1), semiring=algebra, backend=backend, device=device)

    combined = parts[0] if len(parts) == 1 else algebraic.concat(parts, axis=0)

    # Apply dedup + idempotence pruning on combined result
    if combined.shape[0] > 1:
        combined = deduplicate_rank_dim(combined, atol)
    if combined.shape[0] > 1:
        combined = idempotence_pruning(combined, atol)

    return combined


def prune_factors(
    factors: AlgebraicArray,
    max_rank: int,
    max_degree: int | None = None,
    *,
    atol: float = 1e-6,
    shortcircuit: bool = False,
    pack: bool = True,
    static_shape: bool = False,
) -> AlgebraicArray:
    """Reduce rank/degree of CP factors via a sequence of cheap-to-expensive strategies.

    When ``shortcircuit=True`` the expensive O(R^2) smart passes (dedup /
    idempotence / merge / monomial-expansion ``reduce_degree``) are skipped in
    favour of a fast path: ``strip_identity_slots`` followed by hard
    rank/degree truncation.  This is the right setting for non-idempotent /
    float-valued algebras (e.g. soft Boolean) where the equality-based passes
    almost never fire and just waste cycles.

    When ``shortcircuit=False`` (the default) the full pipeline runs:
        1. ``strip_identity_slots`` -- free degree reduction
        2. ``deduplicate_rank_dim`` -- free rank reduction
        3. ``idempotence_pruning`` -- free rank reduction
        4. ``merge_compatible_components`` -- free rank reduction
        5. ``reduce_degree`` (if ``max_degree`` provided and still exceeded)
        6. Re-apply dedup + idempotence after degree reduction
        7. Hard rank truncation ``factors[:max_rank]`` (lossy fallback)

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.
    max_rank : int
        Maximum rank for the output.
    max_degree : int or None, optional
        Maximum degree for the output (``None`` disables degree reduction).
    atol : float, optional
        Tolerance for equality checks in the smart passes (only used when
        ``shortcircuit=False``).  ``0`` = exact equality.
    shortcircuit : bool, optional
        When ``True`` (default), skip O(R^2) smart passes and only do
        ``strip_identity_slots`` + hard truncation.  When ``False`` run the
        full smart pipeline using ``atol`` for equality.
    pack : bool, optional
        When ``True`` (default), use :func:`pack_non_identity_slots` instead of
        :func:`strip_identity_slots`.  Packing also moves identity slots from
        the *middle* of the degree axis to the back before they are sliced off,
        which makes the fast path strictly less lossy (free degree compaction
        vs. a lossy hard truncation drop).

    Returns
    -------
    AlgebraicArray
        Pruned factors of shape ``(R', D', N+1)`` with ``R' <= max_rank``.
    """
    if shortcircuit:
        # Fast path: pack identity slots + push zero rank-1s to the back, then
        # hard truncate.  Both packs are O(R*D*N), shape-preserving, and JIT-safe.
        if pack:
            factors = pack_non_identity_slots(factors, atol, static_shape=static_shape)
        else:
            factors = strip_identity_slots(factors, atol)
        # Sort zero rank-1 components to the back so the rank truncation below
        # keeps meaningful components.  Without this, compose's broadcast-style
        # contraction (which intentionally leaves zero rank-1 stripes from
        # zeroed slot[k]) can lose the non-zero component to truncation.
        factors = pack_non_zero_components(factors, atol, static_shape=static_shape)
        if max_degree is not None and factors.shape[1] > max_degree:
            factors = factors[:, :max_degree, :]
        if factors.shape[0] > max_rank:
            factors = factors[:max_rank]
        return factors

    # 1. Strip / pack identity slots (free degree reduction)
    if pack:
        factors = pack_non_identity_slots(factors, atol)
    else:
        factors = strip_identity_slots(factors, atol)

    # 2. Remove duplicate rank components
    factors = deduplicate_rank_dim(factors, atol)

    # 3. Remove lattice-dominated components
    factors = idempotence_pruning(factors, atol)

    # 4. Merge components that differ in exactly one slot (free rank reduction).
    # Note: this can produce multi-non-zero ("non-monomial") slots; compose's
    # contraction handles these correctly via a broadcast-reshape (rather than
    # the obvious-but-wrong einsum sum that mis-contracts multi-non-zero slots).
    factors = merge_compatible_components(factors, atol)

    # 5. Selective degree reduction (degree reduction at rank cost)
    if max_degree is not None and factors.shape[1] > max_degree:
        factors = reduce_degree(factors, max_degree, max_rank, atol)
        # 6. Re-apply dedup + idempotence after degree reduction
        if factors.shape[0] > 1:
            factors = deduplicate_rank_dim(factors, atol)
        if factors.shape[0] > 1:
            factors = idempotence_pruning(factors, atol)

    # 7. Hard rank truncation (lossy fallback)
    factors = factors[:max_rank]
    return factors


# -- Raw array arithmetic -------------------------------------------------------


def _multiply_factors(p: AlgebraicArray, q: AlgebraicArray) -> AlgebraicArray:
    """Core multiplication on raw arrays -- no simplification/compression.

    Works on any leading batch shape: ``p`` and ``q`` may be ``(R, D, N+1)`` or
    ``(*batch, R, D, N+1)`` (with matching ``*batch``).

    Parameters
    ----------
    p : AlgebraicArray
        Shape ``(*batch, R_p, d_p, n+1)``.
    q : AlgebraicArray
        Shape ``(*batch, R_q, d_q, n+1)``.

    Returns
    -------
    AlgebraicArray
        Shape ``(*batch, R_p * R_q, d_p + d_q, n+1)``.
    """
    *batch_shape, rank_p, degree_p, n_plus_1 = p.shape
    rank_q, degree_q = q.shape[-3], q.shape[-2]

    p_expanded = algebraic.broadcast_to(p[..., :, None, :, :], (*batch_shape, rank_p, rank_q, degree_p, n_plus_1))
    q_expanded = algebraic.broadcast_to(q[..., None, :, :, :], (*batch_shape, rank_p, rank_q, degree_q, n_plus_1))

    result = algebraic.concat([p_expanded, q_expanded], axis=-2)
    result = algebraic.reshape(result, (*batch_shape, rank_p * rank_q, degree_p + degree_q, n_plus_1))
    return result


def _add_factors(p: AlgebraicArray, q: AlgebraicArray) -> AlgebraicArray:
    """Add by concatenating rank-1 components.

    Works on any leading batch shape.  No pruning -- callers handle that.

    Parameters
    ----------
    p : AlgebraicArray
        Shape ``(*batch, R_p, d_p, n+1)``.
    q : AlgebraicArray
        Shape ``(*batch, R_q, d_q, n+1)``.

    Returns
    -------
    AlgebraicArray
        Shape ``(*batch, R_p + R_q, max(d_p, d_q), n+1)``.
    """
    p_rank = p.shape[-3]
    q_rank = q.shape[-3]
    p_degree = p.shape[-2]
    q_degree = q.shape[-2]
    d = max(p_degree, q_degree)

    a_padded = pad_upto(p, max_rank=p_rank, max_degree=d)
    b_padded = pad_upto(q, max_rank=q_rank, max_degree=d)

    return algebraic.concat([a_padded, b_padded], axis=-3)


def _prune_per_batch(
    factors: AlgebraicArray,
    max_rank: int,
    max_degree: int | None,
    *,
    atol: float = 1e-6,
    shortcircuit: bool = False,
    pack: bool = True,
) -> AlgebraicArray:
    """Apply :func:`prune_factors` to each batch element of ``(B, R, D, N+1)``.

    The smart prune passes (dedup / idempotence / merge / reduce_degree) all
    produce ragged ranks per batch element, so the loop is unavoidable.  Each
    pruned result is padded back to a common ``(B, R', D', N+1)`` shape.
    """
    batch = factors.shape[0]
    pruned = [
        prune_factors(factors[b], max_rank, max_degree, atol=atol, shortcircuit=shortcircuit, pack=pack) for b in range(batch)
    ]
    max_r = max(pf.shape[0] for pf in pruned)
    max_d = max(pf.shape[1] for pf in pruned)
    padded = [pad_upto(pf, max_rank=max_r, max_degree=max_d) for pf in pruned]
    return algebraic.stack(padded)


def batched_prune_fast(
    factors: AlgebraicArray,
    max_rank: int,
    max_degree: int | None = None,
    *,
    atol: float = 1e-6,
    pack: bool = True,
    static_shape: bool = False,
) -> AlgebraicArray:
    """Vectorized fast prune over ``(*batch, R, D, N+1)``: strip + hard truncate.

    Only :func:`strip_identity_slots` (or, when ``pack=True`` (default),
    :func:`pack_non_identity_slots` which also handles non-trailing identity
    slots) plus hard rank/degree truncation.  Both helpers are batch-aware via
    ``...`` indexing.  Skips all O(R^2) per-element passes; lossy when the
    input has non-identity slots beyond ``max_degree`` or rank components
    beyond ``max_rank``.

    Pass ``static_shape=True`` (in combination with ``pack=True``) to make this
    function fully JIT-safe: no value-dependent shapes are produced.
    """
    if pack:
        factors = pack_non_identity_slots(factors, atol, static_shape=static_shape)
    else:
        factors = strip_identity_slots(factors, atol)
    # Push zero rank-1 components to the back of the rank axis so the hard
    # truncation below preserves meaningful (non-zero) components.
    factors = pack_non_zero_components(factors, atol, static_shape=static_shape)
    if max_degree is not None and factors.shape[-2] > max_degree:
        factors = factors[..., :, :max_degree, :]
    if factors.shape[-3] > max_rank:
        factors = factors[..., :max_rank, :, :]
    return factors


def batched_contraction_compression(
    contracted: AlgebraicArray,
    max_rank: int,
    max_degree: int | None,
    *,
    atol: float = 1e-6,
    shortcircuit: bool = True,
    pack: bool = True,
    static_shape: bool = False,
) -> AlgebraicArray:
    """Batched beam search over tensor contractions.

    When ``shortcircuit=True`` (default), uses :func:`batched_prune_fast`
    (vectorized strip + hard truncation) at each beam step.  When ``False``,
    falls back to per-batch :func:`prune_factors` (slower but smarter).

    Parameters
    ----------
    contracted : AlgebraicArray
        Shape ``[B, R, D, R_max, D_max, N+1]``.
    max_rank : int
        Maximum rank for pruning at each beam step.
    max_degree : int or None
        Maximum degree for pruning (``None`` disables degree reduction).
    atol : float, optional
        Tolerance for equality checks; ``0`` = exact.
    shortcircuit : bool, optional
        When ``True`` (default), use the vectorized fast prune at each step.

    Returns
    -------
    AlgebraicArray
        Shape ``[B, R_out, D_out, N+1]``.
    """
    batch, rank1, degree1, rank2, degree2, n_plus_1 = contracted.shape
    backend = Backend.from_array(contracted.data)
    algebra = contracted.semiring

    # (B, R, D, R_max, D_max, N+1) -> (B, D, R*R_max, D_max, N+1)
    candidates = algebraic.permute_dims(contracted, (0, 2, 1, 3, 4, 5))
    candidates = algebraic.reshape(candidates, (batch, degree1, rank1 * rank2, degree2, n_plus_1))

    device = contracted.device

    if degree1 == 0:
        return algebraic.broadcast_to(
            algebraic.eye(1, n_plus_1, semiring=algebra, backend=backend, device=device),
            (batch, 1, 1, n_plus_1),
        )

    # Initialize beam with the first candidate (skips identity-multiply; see
    # unbatched ``contraction_compression`` for the rationale).
    beam = candidates[:, 0]
    if shortcircuit:
        beam = batched_prune_fast(beam, max_rank, max_degree, atol=atol, pack=pack, static_shape=static_shape)
    else:
        beam = _prune_per_batch(beam, max_rank, max_degree, atol=atol, shortcircuit=False, pack=pack)

    for d in range(1, degree1):
        candidate_d = candidates[:, d]  # (B, R*R_max, D_max, N+1)
        beam = _multiply_factors(beam, candidate_d)

        if shortcircuit:
            beam = batched_prune_fast(beam, max_rank, max_degree, atol=atol, pack=pack, static_shape=static_shape)
        else:
            beam = _prune_per_batch(beam, max_rank, max_degree, atol=atol, shortcircuit=False, pack=pack)

    return beam


def batched_compose_factors(
    factors: AlgebraicArray,
    replacement_factors: AlgebraicArray,
    max_rank: int,
    max_degree: int | None,
    *,
    atol: float = 1e-6,
    shortcircuit: bool = True,
    pack: bool = True,
    static_shape: bool = False,
) -> AlgebraicArray:
    """Batched composition of CP factors with prepared replacement factor arrays.

    Composes B polynomials with B replacement sets in parallel.

    Parameters
    ----------
    factors : AlgebraicArray
        Batch of CP factors, shape ``[B, R, D, N+1]``.
    replacement_factors : AlgebraicArray
        Batch of prepared replacement arrays, shape ``[B, N+1, R_max, D_max, N+1]``.
        Each element should be the output of :func:`prepare_replacement_factors`.
    max_rank : int
        Maximum rank for pruning at each beam step.
    max_degree : int or None
        Maximum degree for pruning (``None`` disables degree reduction).
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.

    Returns
    -------
    AlgebraicArray
        Batch of composed CP factors, shape ``[B, R_out, D_out, N+1]``.
    """
    # n-mode contraction over the variable axis ``k`` (batched form).
    #
    # See :func:`compose_factors` for the rationale: an einsum sum over ``k``
    # mis-contracts multi-non-zero source slots (produced by
    # ``merge_compatible_components``).  We instead broadcast and reshape the
    # (k, q) axes into a combined rank axis of size ``K*Q``.
    b, p, d, k = factors.shape
    _, _, q, e, v = replacement_factors.shape
    expanded = factors[:, :, :, :, None, None, None] * replacement_factors[:, None, None, :, :, :, :]
    # (b, p, d, k, q, e, v) -> (b, p, d, k*q, e, v)
    contracted = algebraic.reshape(expanded, (b, p, d, k * q, e, v))
    return batched_contraction_compression(
        contracted, max_rank, max_degree, atol=atol, shortcircuit=shortcircuit, pack=pack, static_shape=static_shape
    )


# -- Weight/bias helpers -------------------------------------------------------


def _merge_weights_bias(weights: AlgebraicArray, bias: AlgebraicArray) -> AlgebraicArray:
    """Merge split weights and bias into merged factors of shape ``(*batch, R, D, N+1)``.

    Parameters
    ----------
    weights : AlgebraicArray
        Variable factors of shape ``(*batch, R, D, N)``.
    bias : AlgebraicArray
        Constant factors of shape ``(*batch, R, D)``.

    Returns
    -------
    AlgebraicArray
        Merged factors of shape ``(*batch, R, D, N+1)`` with bias as the last column.
    """
    bias_col = algebraic.reshape(bias, (*bias.shape, 1))
    return algebraic.concat([bias_col, weights], axis=-1)


def _split_merged_factors(factors: AlgebraicArray) -> tuple[AlgebraicArray, AlgebraicArray]:
    """Split merged factors of shape ``(*batch, R, D, N+1)`` into weights and bias.

    Parameters
    ----------
    factors : AlgebraicArray
        Merged factors of shape ``(*batch, R, D, N+1)``.

    Returns
    -------
    tuple[AlgebraicArray, AlgebraicArray]
        ``(weights, bias)`` where weights has shape ``(*batch, R, D, N)``
        and bias has shape ``(*batch, R, D)``.
    """
    bias_col = factors[..., :1]  # (*batch, R, D, 1)
    weights = factors[..., 1:]  # (*batch, R, D, N)
    bias = algebraic.reshape(bias_col, bias_col.shape[:-1])  # (*batch, R, D)
    return weights, bias
