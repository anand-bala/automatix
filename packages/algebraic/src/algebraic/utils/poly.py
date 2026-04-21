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

# -- Core evaluation & composition -------------------------------------------------


def evaluate_factors(
    factors: AlgebraicArray,
    points: Array | AlgebraicArray,
    algebra: Lattice,
    backend: str | Backend,
) -> Scalar:
    """Evaluate CP factors at a given point.

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.
    points : Array or AlgebraicArray
        An array of shape ``(num_vars,)`` to replace each variable with.
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.
    backend : str or Backend
        Backend to use.

    Returns
    -------
    Scalar
        The raw evaluated scalar value.
    """
    rank, d, n_plus_1 = factors.shape
    num_vars = n_plus_1 - 1

    device = factors.device
    one_array = algebraic.ones((1,), semiring=algebra, backend=backend, device=device)
    if is_array(points):
        points_array = algebraic.array(points, semiring=algebra, backend=backend, device=device)
    else:
        points_array = points
    selector = algebraic.concat([one_array, points_array])

    result = algebraic.zeros((), semiring=algebra, backend=backend, device=device)
    for r in range(rank):
        component_value = algebraic.ones((), semiring=algebra, backend=backend, device=device)
        for k in range(d):
            dim_value = algebraic.zeros((), semiring=algebra, backend=backend, device=device)
            for i in range(num_vars + 1):
                term = factors[r, k, i] * selector[i]
                dim_value = dim_value + term
            component_value = component_value * dim_value
        result = result + component_value

    return result.data


def compose_factors(
    factors: AlgebraicArray,
    replacement_factors: Sequence[AlgebraicArray],
    max_rank: int,
    max_degree: int | None,
    algebra: Lattice,
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
    # shape: (N+1, R2, D2, N+1)
    q_factors = prepare_replacement_factors(replacement_factors, algebra)

    # n-mode contraction over variable axis
    # result: (R, D, R2, D2, N+1)
    contracted = algebraic.einsum("pdk,kqev->pdqev", factors, q_factors)

    return contraction_compression(contracted, max_rank, max_degree, algebra)


def prepare_replacement_factors(replacement_factors: Sequence[AlgebraicArray], algebra: Lattice) -> AlgebraicArray:
    """Prepare padded array of replacement factor arrays.

    Parameters
    ----------
    replacement_factors : Sequence[AlgebraicArray]
        Sequence of N replacement factor arrays, each of shape ``(R_i, D_i, N_i+1)``.
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.

    Returns
    -------
    AlgebraicArray
        Padded array of shape ``[N+1, R_max, D_max, N+1]``.
        Index 0: constant (identity: always 1).
        Index i+1: replacement for variable ``x_i``.
    """
    target_rank, target_degree, n_plus_1 = tuple(
        map(max, zip(*((q.shape[0], q.shape[1], q.shape[2]) for q in replacement_factors)))
    )
    backend = Backend.from_array(replacement_factors[0].data)
    device = replacement_factors[0].device

    # Create one-polynomial factors inline (avoids importing RankDecomposition)
    one_factors_base = algebraic.zeros((1, 1, n_plus_1), semiring=algebra, backend=backend, device=device)
    one_factors_base = one_factors_base.at[(0, 0, 0)].set(algebra.one)
    one_factors = pad_upto(
        one_factors_base,
        max_rank=target_rank,
        max_degree=target_degree,
        algebra=algebra,
    )
    new_replacements = algebraic.stack(
        [one_factors]
        + [pad_upto(q, max_rank=target_rank, max_degree=target_degree, algebra=algebra) for q in replacement_factors]
    )

    assert new_replacements.shape == (n_plus_1, target_rank, target_degree, n_plus_1)
    return new_replacements


def pad_upto(
    factors: AlgebraicArray,
    *,
    max_rank: int,
    max_degree: int,
    algebra: Lattice,
) -> AlgebraicArray:
    """Pad rank/degree axes with identity elements up to the given maximum."""
    rank, degree, n_plus_1 = factors.shape

    if max_rank <= rank and max_degree <= degree:
        return factors

    backend = Backend.from_array(factors.data)
    device = factors.device

    new_rank = max(rank, max_rank)
    new_degree = max(degree, max_degree)
    return_shape = (new_rank, new_degree, n_plus_1)

    # Base: all identity, then overwrite existing slots
    one_terms = algebraic.broadcast_to(
        algebraic.eye(1, n_plus_1, semiring=algebra, backend=backend, device=device),
        (rank, new_degree, n_plus_1),
    )
    degree_padded = one_terms.at[:, :degree, :].set(factors)

    zero_terms = algebraic.zeros((new_rank - rank, new_degree, n_plus_1), semiring=algebra, backend=backend, device=device)
    rank_padded = algebraic.concat((degree_padded, zero_terms), axis=0)

    assert rank_padded.shape == return_shape
    return rank_padded


def contraction_compression(
    contracted: AlgebraicArray,
    max_rank: int,
    max_degree: int | None,
    algebra: Lattice,
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

    # (R, D, R_max, D_max, N+1) -> (D, R*R_max, D_max, N+1)
    candidates = algebraic.permute_dims(contracted, (1, 0, 2, 3, 4))
    candidates = algebraic.reshape(candidates, (degree1, rank1 * rank2, degree2, n_plus_1))

    device = contracted.device
    identity = algebraic.broadcast_to(
        algebraic.eye(1, n_plus_1, semiring=algebra, backend=backend, device=device), (1, 1, n_plus_1)
    )
    beam = identity

    for d in range(degree1):
        candidate_d = candidates[d]  # (rank1 * rank2, degree2, n+1)
        beam = _multiply_factors(beam, candidate_d)
        beam = prune_factors(beam, max_rank, max_degree, algebra)

    return beam


# -- Pruning utilities ----------------------------------------------------------


def deduplicate_rank_dim(factors: AlgebraicArray) -> AlgebraicArray:
    """Remove duplicate rank components (keep first occurrence of each)."""
    a = factors[:, None, :, :]  # (rank, 1, d, n+1)
    b = factors[None, :, :, :]  # (1, rank, d, n+1)

    # eq[i, j] = True iff row i equals row j
    eq = algebraic.equal(a, b).all((2, 3))  # (rank, rank) raw bool array

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


def idempotence_pruning(factors: AlgebraicArray) -> AlgebraicArray:
    """Remove terms dominated by lattice idempotence laws."""
    # p <= q if p + q == q  (p is dominated by q)
    a = factors[:, None, :, :]  # (rank, 1, d, n+1)
    b = factors[None, :, :, :]  # (1, rank, d, n+1)

    added = a + b
    check = algebraic.equal(added, b).all((2, 3))  # check[i,j] = (a[i] <= a[j])

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


def strip_identity_slots(factors: AlgebraicArray, algebra: Lattice) -> AlgebraicArray:
    """Strip trailing degree slots that are multiplicative identity across all rank components.

    A slot ``factors[r, k, :]`` is identity when it equals
    ``[algebra.one, algebra.zero, ..., algebra.zero]`` (contributes factor 1
    to the product for every input).  If the *last* slot is identity for
    **all** rank components it can be removed without changing the polynomial.
    The function trims all such trailing slots, keeping at least one.

    Complexity: O(R * D * N).

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.
    algebra : BoundedDistributiveLattice
        Lattice algebra used to determine the identity element pattern.

    Returns
    -------
    AlgebraicArray
        CP factors of shape ``(R, D', N+1)`` with ``1 <= D' <= D``.
    """
    rank, degree, n_plus_1 = factors.shape
    if degree <= 1:
        return factors

    backend = Backend.from_array(factors.data)
    device = factors.device

    # Identity row pattern: [one, zero, ..., zero] of shape (1, 1, N+1) for broadcasting
    identity_row = algebraic.zeros((1, 1, n_plus_1), semiring=algebra, backend=backend, device=device)
    identity_row = identity_row.at[(0, 0, 0)].set(algebra.one)

    # Walk degree axis from the end; trim while all ranks are identity at that slot
    new_degree = degree
    for k in range(degree - 1, 0, -1):  # stop at 1 to keep at least degree 1
        slot = factors[:, k : k + 1, :]  # (R, 1, N+1)
        if bool(algebraic.equal(slot, identity_row).all()):
            new_degree -= 1
        else:
            break

    if new_degree == degree:
        return factors
    return factors[:, :new_degree, :]


def merge_compatible_components(factors: AlgebraicArray, algebra: Lattice) -> AlgebraicArray:
    """Merge rank-1 component pairs that differ at exactly one slot.

    Uses distributivity: ``(common * f_j) + (common * g_j) = common * (f_j + g_j)``.
    Each successful merge reduces rank by 1 with no change to degree.

    Complexity: O(R^2 * D * N) per pass.

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.

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
        slot_eq = algebraic.equal(a, b).all(3)  # (R, R, D) raw bool array

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
    algebra: Lattice,
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
    algebra: Lattice,
    max_degree: int,
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

    identity_row = algebraic.zeros((1, n_plus_1), semiring=algebra, backend=backend, device=device)
    identity_row = identity_row.at[(0, 0)].set(algebra.one)

    non_identity: list[AlgebraicArray] = []
    for k in range(degree):
        slot = component[k : k + 1, :]  # (1, N+1)
        if not bool(algebraic.equal(slot, identity_row).all()):
            non_identity.append(slot)
            if len(non_identity) >= max_degree:
                break  # Reached max; any extras are dropped (lossy)

    # Pad with identity slots to reach max_degree
    padding_needed = max_degree - len(non_identity)
    parts = non_identity + [identity_row] * padding_needed
    return algebraic.concat(parts, axis=0)  # (max_degree, N+1)


def reduce_degree(
    factors: AlgebraicArray,
    max_degree: int,
    max_rank: int,
    algebra: Lattice,
) -> AlgebraicArray:
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
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.

    Returns
    -------
    AlgebraicArray
        CP factors of shape ``(R', max_degree, N+1)`` with ``R' <= R + num_bad_monomials``.
    """
    rank, degree, n_plus_1 = factors.shape
    num_vars = n_plus_1 - 1
    backend = Backend.from_array(factors.data)
    device = factors.device

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
            if not bool(algebraic.equal(slot, identity_row).all()):
                eff_degree += 1

        if eff_degree <= max_degree:
            # Good component: pack non-identity slots to front, shape (max_degree, N+1)
            packed = _pack_component(component, algebra, max_degree)
            good_packed.append(packed[None, :, :])  # (1, max_degree, N+1)
        else:
            # Bad component: run subset-DP and merge monomials
            comp_monomials = _component_to_monomials(component, algebra, num_vars, backend)
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
        combined = deduplicate_rank_dim(combined)
    if combined.shape[0] > 1:
        combined = idempotence_pruning(combined)

    return combined


def prune_factors(
    factors: AlgebraicArray,
    max_rank: int,
    max_degree: int | None = None,
    algebra: Lattice | None = None,
) -> AlgebraicArray:
    """Reduce rank/degree of CP factors via a sequence of cheap-to-expensive strategies.

    Execution order:
    1. ``strip_identity_slots`` (if *algebra* provided) -- free degree reduction
    2. ``deduplicate_rank_dim`` -- free rank reduction
    3. ``idempotence_pruning`` -- free rank reduction
    4. ``merge_compatible_components`` (if *algebra* provided) -- free rank reduction
    5. ``reduce_degree`` (if *max_degree* provided and still exceeded) -- degree reduction
    6. Re-apply dedup + idempotence after degree reduction
    7. Hard rank truncation ``factors[:max_rank]`` (lossy fallback)

    Backward-compatible: ``max_degree=None`` and ``algebra=None`` preserve the
    original three-step behaviour (dedup, idempotence, truncation).

    Parameters
    ----------
    factors : AlgebraicArray
        CP factors of shape ``(R, D, N+1)``.
    max_rank : int
        Maximum rank for the output.
    max_degree : int or None, optional
        Maximum degree for the output (``None`` disables degree reduction).
    algebra : BoundedDistributiveLattice or None, optional
        Lattice algebra; required for the identity-based strategies.

    Returns
    -------
    AlgebraicArray
        Pruned factors of shape ``(R', D', N+1)`` with ``R' <= max_rank``.
    """
    # 1. Strip trailing identity slots (free degree reduction)
    if algebra is not None:
        factors = strip_identity_slots(factors, algebra)

    # 2. Remove duplicate rank components
    factors = deduplicate_rank_dim(factors)

    # 3. Remove lattice-dominated components
    factors = idempotence_pruning(factors)

    # 4. Merge components that differ in exactly one slot (free rank reduction)
    if algebra is not None:
        factors = merge_compatible_components(factors, algebra)

    # 5. Selective degree reduction (degree reduction at rank cost)
    if max_degree is not None and algebra is not None and factors.shape[1] > max_degree:
        factors = reduce_degree(factors, max_degree, max_rank, algebra)
        # 6. Re-apply dedup + idempotence after degree reduction
        if factors.shape[0] > 1:
            factors = deduplicate_rank_dim(factors)
        if factors.shape[0] > 1:
            factors = idempotence_pruning(factors)

    # 7. Hard rank truncation (lossy fallback)
    factors = factors[:max_rank]
    return factors


# -- Raw array arithmetic -------------------------------------------------------


def _multiply_factors(p: AlgebraicArray, q: AlgebraicArray) -> AlgebraicArray:
    """Core multiplication on raw arrays -- no simplification/compression.

    Parameters
    ----------
    p : AlgebraicArray
        Shape ``[R_p, d_p, n+1]``.
    q : AlgebraicArray
        Shape ``[R_q, d_q, n+1]``.

    Returns
    -------
    AlgebraicArray
        Shape ``[R_p * R_q, d_p + d_q, n+1]``.
    """
    rank_p, degree_p, n_plus_1 = p.shape
    rank_q, degree_q, _ = q.shape

    p_expanded = algebraic.broadcast_to(p[:, None, :, :], (rank_p, rank_q, degree_p, n_plus_1))
    q_expanded = algebraic.broadcast_to(q[None, :, :, :], (rank_p, rank_q, degree_q, n_plus_1))

    result = algebraic.concat([p_expanded, q_expanded], axis=2)
    result = algebraic.reshape(result, (rank_p * rank_q, degree_p + degree_q, n_plus_1))
    return result


def _add_factors(p: AlgebraicArray, q: AlgebraicArray, algebra: Lattice) -> AlgebraicArray:
    """Add by concatenating rank-1 components.

    For CP decomposition: ``p + q`` = sum of all components from both.
    """
    p_rank, p_degree, n_plus_1 = p.shape
    q_rank, q_degree, _ = q.shape
    d = max(p_degree, q_degree)

    a_padded = pad_upto(p, max_rank=p_rank, max_degree=d, algebra=algebra)
    b_padded = pad_upto(q, max_rank=q_rank, max_degree=d, algebra=algebra)

    return algebraic.concat([a_padded, b_padded], axis=0)


def _batched_multiply_factors(p: AlgebraicArray, q: AlgebraicArray) -> AlgebraicArray:
    """Batched core multiplication on raw arrays -- no simplification/compression.

    Parameters
    ----------
    p : AlgebraicArray
        Shape ``[B, R_p, d_p, n+1]``.
    q : AlgebraicArray
        Shape ``[B, R_q, d_q, n+1]``.

    Returns
    -------
    AlgebraicArray
        Shape ``[B, R_p * R_q, d_p + d_q, n+1]``.
    """
    batch, rank_p, degree_p, n_plus_1 = p.shape
    _, rank_q, degree_q, _ = q.shape

    p_expanded = algebraic.broadcast_to(p[:, :, None, :, :], (batch, rank_p, rank_q, degree_p, n_plus_1))
    q_expanded = algebraic.broadcast_to(q[:, None, :, :, :], (batch, rank_p, rank_q, degree_q, n_plus_1))

    result = algebraic.concat([p_expanded, q_expanded], axis=3)
    result = algebraic.reshape(result, (batch, rank_p * rank_q, degree_p + degree_q, n_plus_1))
    return result


def batched_contraction_compression(
    contracted: AlgebraicArray,
    max_rank: int,
    max_degree: int | None,
    algebra: Lattice,
) -> AlgebraicArray:
    """Batched beam search over tensor contractions.

    Uses batched multiplication for efficiency, then per-element pruning
    for correctness (pruning changes rank per element via boolean indexing,
    results are padded back to uniform shape for the next step).

    Parameters
    ----------
    contracted : AlgebraicArray
        Shape ``[B, R, D, R_max, D_max, N+1]``.
    max_rank : int
        Maximum rank for pruning at each beam step.
    max_degree : int or None
        Maximum degree for pruning (``None`` disables degree reduction).
    algebra : BoundedDistributiveLattice
        Lattice algebra governing operations.

    Returns
    -------
    AlgebraicArray
        Shape ``[B, R_out, D_out, N+1]``.
    """
    batch, rank1, degree1, rank2, degree2, n_plus_1 = contracted.shape
    backend = Backend.from_array(contracted.data)

    # (B, R, D, R_max, D_max, N+1) -> (B, D, R*R_max, D_max, N+1)
    candidates = algebraic.permute_dims(contracted, (0, 2, 1, 3, 4, 5))
    candidates = algebraic.reshape(candidates, (batch, degree1, rank1 * rank2, degree2, n_plus_1))

    device = contracted.device
    identity = algebraic.broadcast_to(
        algebraic.eye(1, n_plus_1, semiring=algebra, backend=backend, device=device),
        (batch, 1, 1, n_plus_1),
    )
    beam = identity

    for d in range(degree1):
        candidate_d = candidates[:, d]  # (B, R*R_max, D_max, N+1)
        beam = _batched_multiply_factors(beam, candidate_d)

        pruned = [prune_factors(beam[b], max_rank, max_degree, algebra) for b in range(batch)]
        max_pruned_rank = max(p.shape[0] for p in pruned)
        max_pruned_degree = max(p.shape[1] for p in pruned)
        padded = [pad_upto(p, max_rank=max_pruned_rank, max_degree=max_pruned_degree, algebra=algebra) for p in pruned]
        beam = algebraic.stack(padded)

    return beam


def batched_compose_factors(
    factors: AlgebraicArray,
    replacement_factors: AlgebraicArray,
    max_rank: int,
    max_degree: int | None,
    algebra: Lattice,
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
    # Batched einsum: contract over variable axis
    # (B, R, D, N+1) x (B, N+1, R_max, D_max, N+1) -> (B, R, D, R_max, D_max, N+1)
    contracted = algebraic.einsum("bpdk,bkqev->bpdqev", factors, replacement_factors)
    return batched_contraction_compression(contracted, max_rank, max_degree, algebra)


# -- Weight/bias helpers -------------------------------------------------------


def _merge_weights_bias(weights: AlgebraicArray, bias: AlgebraicArray) -> AlgebraicArray:
    """Merge split weights and bias into merged factors of shape ``(R, D, N+1)``.

    Parameters
    ----------
    weights : AlgebraicArray
        Variable factors of shape ``(R, D, N)``.
    bias : AlgebraicArray
        Constant factors of shape ``(R, D)``.

    Returns
    -------
    AlgebraicArray
        Merged factors of shape ``(R, D, N+1)`` with bias as column 0.
    """
    bias_col = algebraic.reshape(bias, (*bias.shape, 1))
    return algebraic.concat([bias_col, weights], axis=2)


def _split_merged_factors(factors: AlgebraicArray) -> tuple[AlgebraicArray, AlgebraicArray]:
    """Split merged factors of shape ``(R, D, N+1)`` into weights and bias.

    Parameters
    ----------
    factors : AlgebraicArray
        Merged factors of shape ``(R, D, N+1)``.

    Returns
    -------
    tuple[AlgebraicArray, AlgebraicArray]
        ``(weights, bias)`` where weights has shape ``(R, D, N)`` and bias has shape ``(R, D)``.
    """
    bias_col = factors[:, :, :1]  # (R, D, 1)
    weights = factors[:, :, 1:]  # (R, D, N)
    bias = algebraic.reshape(bias_col, bias_col.shape[:2])  # (R, D)
    return weights, bias
