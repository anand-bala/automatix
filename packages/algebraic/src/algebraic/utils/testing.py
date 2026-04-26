"""Shared testing utils for algebraic"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algebraic import AlgebraicArray
from algebraic.types import Array, Backend

if TYPE_CHECKING:
    from algebraic.polynomials.rank_decomp import LowRankFactors, RankDecomposition


def maybe_unwrap(x: AlgebraicArray | Array | Any) -> np.ndarray:  # noqa: ANN401
    if isinstance(x, AlgebraicArray):
        x = x.data
    return np.asanyarray(x)


def assert_allclose(actual: AlgebraicArray, desired: AlgebraicArray, *, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    assert actual.semiring == desired.semiring, f"Semirings differ: {actual.semiring} != {desired.semiring}"
    assert actual._vdot == desired._vdot and actual._matmul == desired._matmul

    np.testing.assert_allclose(actual.data, desired.data, rtol=rtol, atol=atol)


def assert_close(a: Any, b: Any, *, rtol: float = 1e-5, atol: float = 1e-8) -> None:  # noqa: ANN401
    """Assert two arrays are element-wise close after converting to NumPy."""
    np.testing.assert_allclose(maybe_unwrap(a), maybe_unwrap(b), rtol=rtol, atol=atol)


def assert_equal(a: Any, b: Any) -> None:  # noqa: ANN401
    """Assert two arrays are element-wise equal after converting to NumPy."""
    np.testing.assert_array_equal(maybe_unwrap(a), maybe_unwrap(b))


def make_array(value: Any, backend: str | Backend | None = None) -> Array:  # noqa: ANN401
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(value)
    if backend == "torch":
        import torch

        return torch.as_tensor(value)
    # Default fallback
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Sparsity profiling
# ---------------------------------------------------------------------------


def slot_nnz_stats(
    rd: RankDecomposition | LowRankFactors,
    atol: float = 0.0,
    percentiles: tuple[int, ...] = (50, 75, 90, 95, 99, 100),
    print_report: bool = False,
) -> dict[str, Any]:
    """Measure per-slot non-zero counts along the variable axis of CP factors.

    For a :class:`~algebraic.polynomials.RankDecomposition` or
    :class:`~algebraic.polynomials.LowRankFactors` with factors of shape
    ``(*batch, R, D, N+1)``, each ``(*batch, r, d)`` slot is a vector of
    length ``N+1``.  This function counts how many entries in each slot differ
    from ``algebra.zero`` and returns summary statistics.

    ``LowRankFactors`` stores weights and bias separately; they are merged into
    the ``(*batch, R, D, N+1)`` view via :meth:`~LowRankFactors.to_merged`
    before inspection.

    This is the primary diagnostic for deciding whether a sparse representation
    of the variable axis would be worthwhile: if ``max_nnz`` is well below
    ``N+1``, there is headroom for memory/compute savings.

    Parameters
    ----------
    rd : RankDecomposition or LowRankFactors
        The polynomial to inspect.
    atol : float, optional
        Entries within ``atol`` of ``algebra.zero`` are treated as zero.
        ``0.0`` (default) requires exact equality.
    percentiles : tuple of int, optional
        Percentiles of the nnz distribution to report.
    print_report : bool, optional
        If ``True``, print a human-readable summary to stdout.

    Returns
    -------
    dict
        Keys:

        ``"factors_shape"``
            Shape of the factors tensor.
        ``"n_plus_1"``
            Length of the variable axis (``N+1``).
        ``"nnz"``
            NumPy array of shape ``(*batch, R, D)`` with per-slot non-zero counts.
        ``"mean_nnz"``
            Mean NNZ over all slots.
        ``"max_nnz"``
            Maximum NNZ across all slots.
        ``"min_nnz"``
            Minimum NNZ across all slots.
        ``"percentiles"``
            Dict mapping each requested percentile to its NNZ value.
        ``"histogram"``
            Dict mapping each observed NNZ count (0..N+1) to the number of
            slots with that count.
        ``"density"``
            Mean NNZ / (N+1): fraction of each slot that is non-zero on average.
        ``"all_slots"``
            Total number of slots inspected.
        ``"zero_slots"``
            Number of slots where every entry is zero (the slot is semiring-zero).
    """
    from algebraic.polynomials.rank_decomp import LowRankFactors

    if isinstance(rd, LowRankFactors):
        factors_arr = rd.to_merged()
    else:
        factors_arr = rd.factors
    factors_np = np.asarray(factors_arr.data)  # (*batch, R, D, N+1)
    zero_val = float(np.asarray(rd.algebra.zero))
    n_plus_1 = factors_np.shape[-1]

    if atol > 0.0:
        is_nonzero = np.abs(factors_np - zero_val) > atol  # (*batch, R, D, N+1)
    else:
        is_nonzero = factors_np != zero_val  # (*batch, R, D, N+1)

    nnz = is_nonzero.sum(axis=-1).astype(np.int32)  # (*batch, R, D)
    flat = nnz.ravel()

    pct_values = {p: int(np.percentile(flat, p)) for p in percentiles}
    hist: dict[int, int] = {}
    for v in flat:
        hist[int(v)] = hist.get(int(v), 0) + 1

    result: dict[str, Any] = {
        "factors_shape": tuple(factors_np.shape),
        "n_plus_1": n_plus_1,
        "nnz": nnz,
        "mean_nnz": float(flat.mean()),
        "max_nnz": int(flat.max()),
        "min_nnz": int(flat.min()),
        "percentiles": pct_values,
        "histogram": dict(sorted(hist.items())),
        "density": float(flat.mean()) / n_plus_1,
        "all_slots": int(flat.size),
        "zero_slots": int((flat == 0).sum()),
    }

    if print_report:
        shape = result["factors_shape"]
        print(f"slot_nnz_stats  factors={shape}  N+1={n_plus_1}  atol={atol}")
        print(f"  slots total : {result['all_slots']}  (zero slots: {result['zero_slots']})")
        print(f"  nnz mean    : {result['mean_nnz']:.2f}  ({result['density'] * 100:.1f}% of N+1)")
        print(f"  nnz min/max : {result['min_nnz']} / {result['max_nnz']}")
        pct_str = "  ".join(f"p{p}={v}" for p, v in result["percentiles"].items())
        print(f"  percentiles : {pct_str}")
        max_bar = 40
        print("  histogram (nnz -> slot count):")
        total = result["all_slots"]
        for k, count in result["histogram"].items():
            bar = "#" * int(count / total * max_bar)
            print(f"    {k:3d}: {count:6d}  {bar}")

    return result
