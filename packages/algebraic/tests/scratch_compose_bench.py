"""Throwaway: time batched compose with logging.

Honors the new shortcircuit / atol / pack kwargs.  Set FAST=False to bench
the slow per-batch smart prune for comparison; set PACK=False to fall back
to ``strip_identity_slots`` only (no middle-identity pack).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence

import algebraic
import algebraic.ops as algebraic_ops
import torch
from algebraic.array import AlgebraicArray
from algebraic.polynomials import RankDecomposition
from algebraic.polynomials import rank_decomp as rd
from algebraic.semirings import boolean_algebra
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.spec import Semiring
from algebraic.types import Backend, is_torch_array
from algebraic.utils import poly as poly_utils

BACKEND = "torch"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256
RANK = 4
DEGREE = 4
NUM_VARS = 20
MAX_RANK = 4
MAX_DEGREE = 4
ATOL = 1e-6
FAST = True  # set False to compare with smart per-batch prune
PACK = True  # set False to fall back to strip_identity_slots only


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] {msg}", flush=True)


# ---- tracing wrappers ----

_orig_prepare = poly_utils.prepare_replacement_factors
_orig_batched_compose = poly_utils.batched_compose_factors
_orig_batched_compress = poly_utils.batched_contraction_compression


def trace_prepare(
    replacement_factors: Sequence[AlgebraicArray],
    algebra: Lattice,
    batch_shape: tuple[int, ...] = (),
) -> AlgebraicArray:
    log("  >>> prepare_replacement_factors")
    out = _orig_prepare(replacement_factors, algebra, batch_shape)
    sync()
    log(f"  <<< prepare_replacement_factors  shape={tuple(out.shape)}")
    return out


def trace_batched_compose(
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
    kw = {"atol": atol, "shortcircuit": shortcircuit, "pack": pack, "static_shape": static_shape}
    log(f"  >>> batched_compose_factors  state={tuple(factors.shape)}  q={tuple(replacement_factors.shape)}  kw={kw}")
    out = _orig_batched_compose(
        factors,
        replacement_factors,
        max_rank,
        max_degree,
        atol=atol,
        shortcircuit=shortcircuit,
        pack=pack,
        static_shape=static_shape,
    )
    sync()
    log(f"  <<< batched_compose_factors  out={tuple(out.shape)}")
    return out


def trace_batched_compress(
    contracted: AlgebraicArray,
    max_rank: int,
    max_degree: int | None,
    *,
    atol: float = 1e-6,
    shortcircuit: bool = True,
    pack: bool = True,
    static_shape: bool = False,
) -> AlgebraicArray:
    log(
        f"    >>> batched_contraction_compression  in={tuple(contracted.shape)}  "
        f"shortcircuit={shortcircuit} pack={pack} atol={atol}"
    )

    batch, rank1, degree1, rank2, degree2, n_plus_1 = contracted.shape
    backend = Backend.from_array(contracted.data)
    algebra = contracted.semiring

    candidates = algebraic_ops.permute_dims(contracted, (0, 2, 1, 3, 4, 5))
    candidates = algebraic_ops.reshape(candidates, (batch, degree1, rank1 * rank2, degree2, n_plus_1))

    device = contracted.device

    if degree1 == 0:
        beam = algebraic_ops.broadcast_to(
            algebraic_ops.eye(1, n_plus_1, semiring=algebra, backend=backend, device=device),
            (batch, 1, 1, n_plus_1),
        )
        log(f"    <<< batched_contraction_compression (degree1==0)  out={tuple(beam.shape)}")
        return beam

    # Initialize beam from the first candidate (skips identity-multiply).
    beam = candidates[:, 0]
    log(f"      beam init from candidates[:, 0]  beam.shape={tuple(beam.shape)}")
    sync()
    t0 = time.perf_counter()
    if shortcircuit:
        beam = poly_utils.batched_prune_fast(beam, max_rank, max_degree, atol=atol, pack=pack, static_shape=static_shape)
    else:
        beam = poly_utils._prune_per_batch(beam, max_rank, max_degree, atol=atol, shortcircuit=False, pack=pack)
    sync()
    log(f"        init prune done ({time.perf_counter() - t0:.3f}s)  beam.shape={tuple(beam.shape)}")

    for d in range(1, degree1):
        log(f"      beam iter d={d}  beam.shape={tuple(beam.shape)}")
        candidate_d = candidates[:, d]
        sync()
        t0 = time.perf_counter()
        beam = poly_utils._multiply_factors(beam, candidate_d)
        sync()
        log(f"        multiply done ({time.perf_counter() - t0:.3f}s)  beam.shape={tuple(beam.shape)}")

        sync()
        t0 = time.perf_counter()
        if shortcircuit:
            beam = poly_utils.batched_prune_fast(beam, max_rank, max_degree, atol=atol, pack=pack, static_shape=static_shape)
            sync()
            log(f"        FAST batched_prune_fast done ({time.perf_counter() - t0:.3f}s)  beam.shape={tuple(beam.shape)}")
        else:
            log(f"        SLOW per-batch prune (B={batch}, R={beam.shape[1]}, D={beam.shape[2]})")
            beam = poly_utils._prune_per_batch(beam, max_rank, max_degree, atol=atol, shortcircuit=False, pack=pack)
            sync()
            log(f"        slow prune done ({time.perf_counter() - t0:.3f}s)  beam.shape={tuple(beam.shape)}")

    log(f"    <<< batched_contraction_compression  out={tuple(beam.shape)}")
    return beam


poly_utils.prepare_replacement_factors = trace_prepare
poly_utils.batched_compose_factors = trace_batched_compose
poly_utils.batched_contraction_compression = trace_batched_compress

rd.prepare_replacement_factors = trace_prepare  # type: ignore[attr-defined]
rd.batched_compose_factors = trace_batched_compose  # type: ignore[attr-defined]


def random_factors(
    batch: int,
    rank: int,
    degree: int,
    num_vars: int,
    algebra: Semiring,
) -> RankDecomposition:
    raw = torch.rand((batch, rank, degree, num_vars + 1), device=DEVICE)
    arr = algebraic.array(raw, semiring=algebra, backend=BACKEND, device=DEVICE)
    return RankDecomposition(arr, max_rank=MAX_RANK, max_degree=MAX_DEGREE, backend=BACKEND)


def main() -> None:
    log(f"backend={BACKEND}  device={DEVICE}")
    log(f"state B={BATCH} R={RANK} d={DEGREE} N={NUM_VARS}")
    log(f"max_rank={MAX_RANK} max_degree={MAX_DEGREE}  shortcircuit={FAST} pack={PACK} atol={ATOL}")

    algebra = boolean_algebra(mode="soft")

    log("building state + 20 replacements")
    state = random_factors(BATCH, RANK, DEGREE, NUM_VARS, algebra)
    replacements = [random_factors(BATCH, RANK, DEGREE, NUM_VARS, algebra) for _ in range(NUM_VARS)]

    log("=== warm-up compose ===")
    sync()
    t0 = time.perf_counter()
    out = state.compose(replacements, atol=ATOL, shortcircuit=FAST, pack=PACK)
    sync()
    log(f"=== warm-up done in {time.perf_counter() - t0:.3f}s  out R={out.rank} D={out.degree} ===")

    n = 5
    log(f"=== timing {n} iterations ===")
    times: list[float] = []
    for i in range(n):
        sync()
        t0 = time.perf_counter()
        out = state.compose(replacements, atol=ATOL, shortcircuit=FAST, pack=PACK)
        sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        log(f"iter {i}: {dt:.3f}s")
    log(f"mean: {sum(times) / n:.3f}s  min: {min(times):.3f}s  max: {max(times):.3f}s")

    log("=== correctness check (fast vs slow on small case) ===")
    correctness_check(algebra)


def correctness_check(algebra: Semiring) -> None:
    """Compare fast vs slow path on a small batched case.

    Inputs are sampled from ``sigmoid(randn(...))`` so values cluster near 0
    and 1 (more realistic for soft Boolean than uniform-in-[0,1]).  Soft
    Boolean is approximate, but if the fast path is wildly off this will
    show it.
    """
    small_batch, small_n = 4, 6

    def sig_factors(num_vars: int) -> RankDecomposition:
        raw = torch.sigmoid(torch.randn((small_batch, RANK, DEGREE, num_vars + 1), device=DEVICE))
        arr = algebraic.array(raw, semiring=algebra, backend=BACKEND, device=DEVICE)
        return RankDecomposition(arr, max_rank=MAX_RANK, max_degree=MAX_DEGREE, backend=BACKEND)

    state = sig_factors(small_n)
    repls = [sig_factors(small_n) for _ in range(small_n)]

    out_fast = state.compose(repls, atol=ATOL, shortcircuit=True, pack=PACK)
    out_slow = state.compose(repls, atol=ATOL, shortcircuit=False, pack=PACK)

    pts = torch.sigmoid(torch.randn((small_batch, small_n), device=DEVICE))
    val_fast = out_fast.evaluate(pts).data
    assert is_torch_array(val_fast)
    val_slow = out_slow.evaluate(pts).data
    assert is_torch_array(val_slow)

    diff = (val_fast - val_slow).abs()
    log(f"  fast={[round(float(v), 4) for v in val_fast.tolist()]}")
    log(f"  slow={[round(float(v), 4) for v in val_slow.tolist()]}")
    log(f"  |diff|: max={float(diff.max()):.4f} mean={float(diff.mean()):.4f}")
    log(f"  mean(fast-slow) = {float((val_fast - val_slow).mean()):+.4f}  (positive => fast biased high)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("INTERRUPTED")
        sys.exit(1)
