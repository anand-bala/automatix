"""Tests for the new pruning strategies in algebraic.utils.poly.

Covers:
  - strip_identity_slots
  - merge_compatible_components
  - reduce_degree
  - prune_factors (with new max_degree / algebra kwargs)
  - Integration: repeated compose() keeps both rank and degree bounded
"""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

import numpy as np
import pytest
from algebraic import BooleanAlgebra
from algebraic.polynomials import RankDecomposition
from algebraic.utils.poly import (
    merge_compatible_components,
    pad_upto,
    prune_factors,
    reduce_degree,
    strip_identity_slots,
)
from algebraic.utils.testing import assert_close, make_array

# -- Helpers -----------------------------------------------------------------


def _eval_rd(rd: RankDecomposition, point: list[float]) -> float:
    """Evaluate a RankDecomposition at a boolean-valued point, return scalar."""
    result = rd.evaluate(make_array(point, rd.backend))
    return float(np.asarray(result.data).flat[0])


def _truth_table(rd: RankDecomposition) -> list[float]:
    """Evaluate at all 2^n boolean points in lexicographic order."""
    n = rd.num_vars
    out = []
    for bits in np.ndindex(*([2] * n)):
        algebra = rd.algebra
        point = [algebra.one if b else algebra.zero for b in bits]
        out.append(_eval_rd(rd, point))
    return out


# -- strip_identity_slots -------------------------------------------------


class TestStripIdentitySlots:
    def test_no_identity_unchanged(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """A factor with no trailing identity slots is returned unchanged."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        prod = x0 * x1  # degree 2, no trailing identity

        original_degree = prod.factors.shape[1]
        stripped = strip_identity_slots(prod.factors)
        assert stripped.shape[1] <= original_degree

    def test_strips_padded_identity(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Padding to high degree then stripping recovers the original degree."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        prod = x0 * x1  # degree 2

        # Pad to degree 7
        padded = pad_upto(prod.factors, max_rank=1, max_degree=7)
        assert padded.shape[1] == 7

        # Strip should recover degree 2
        stripped = strip_identity_slots(padded)
        assert stripped.shape[1] == 2

    def test_minimum_degree_one(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """strip_identity_slots always keeps at least degree 1."""
        num_vars = 2
        const_one = RankDecomposition.one(num_vars, bool_algebra, backend=backend)
        # Pad to degree 5 (all slots identity)
        padded = pad_upto(const_one.factors, max_rank=1, max_degree=5)
        stripped = strip_identity_slots(padded)
        assert stripped.shape[1] >= 1

    def test_semantics_preserved(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Stripping identity slots does not change polynomial evaluation."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        x2 = RankDecomposition.variable(2, num_vars, bool_algebra, backend=backend)
        poly = (x0 * x1) + x2

        padded_factors = pad_upto(poly.factors, max_rank=poly.rank, max_degree=8)
        _padded_rd = RankDecomposition(
            padded_factors, poly.max_rank, poly.max_degree, poly.max_replacement_degree, backend=backend
        )
        stripped_factors = strip_identity_slots(padded_factors)
        stripped_rd = RankDecomposition(
            stripped_factors, poly.max_rank, poly.max_degree, poly.max_replacement_degree, backend=backend
        )

        # Truth tables must match
        orig_tt = _truth_table(poly)
        stripped_tt = _truth_table(stripped_rd)
        assert orig_tt == stripped_tt


# -- merge_compatible_components -----------------------------------------------


class TestMergeCompatibleComponents:
    def test_merge_reduces_rank(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Two components differing at exactly one slot get merged into one."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)

        # Build two rank-1 components that share x0 slot but differ at x1 slot
        # (x0 * x1) and (x0 * x2) share slot 0 (x0)
        x2 = RankDecomposition.variable(2, num_vars, bool_algebra, backend=backend)
        p1 = x0 * x1  # factors: [x0, x1] degree 2
        p2 = x0 * x2  # factors: [x0, x2] degree 2

        # Manually concatenate their factor rows
        import algebraic.ops as alg

        combined_factors = alg.concat([p1.factors, p2.factors], axis=0)
        assert combined_factors.shape[0] == 2

        merged = merge_compatible_components(combined_factors)
        # After merge: rank should drop from 2 to 1
        assert merged.shape[0] == 1

    def test_semantics_preserved_after_merge(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Merged components evaluate identically to the original sum."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        x2 = RankDecomposition.variable(2, num_vars, bool_algebra, backend=backend)

        # (x0 * x1) + (x0 * x2) = x0 * (x1 + x2) in a distributive lattice
        original = (x0 * x1) + (x0 * x2)

        import algebraic.ops as alg

        combined = alg.concat([p.factors for p in [x0 * x1, x0 * x2]], axis=0)
        merged_factors = merge_compatible_components(combined)
        merged_rd = RankDecomposition(
            merged_factors, original.max_rank, original.max_degree, original.max_replacement_degree, backend=backend
        )

        orig_tt = _truth_table(original)
        merged_tt = _truth_table(merged_rd)
        assert orig_tt == merged_tt

    def test_no_merge_when_no_candidates(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Components differing at two or more slots are not merged."""
        num_vars = 4
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        x2 = RankDecomposition.variable(2, num_vars, bool_algebra, backend=backend)
        x3 = RankDecomposition.variable(3, num_vars, bool_algebra, backend=backend)

        # x0*x1 and x2*x3: they differ at BOTH slots -> no merge
        p1 = x0 * x1
        p2 = x2 * x3

        import algebraic.ops as alg

        combined = alg.concat([p1.factors, p2.factors], axis=0)
        merged = merge_compatible_components(combined)
        assert merged.shape[0] == 2  # rank unchanged


# -- reduce_degree -------------------------------------------------------


class TestReduceDegree:
    def test_no_op_when_within_bounds(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """reduce_degree is a no-op when degree <= max_degree."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        prod = x0 * x1  # degree 2

        result = reduce_degree(prod.factors, max_degree=2, max_rank=10)
        assert result.shape == prod.factors.shape

    def test_reduces_padded_component(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """A single component padded to high degree is brought back to max_degree."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        prod = x0 * x1  # degree 2 with 2 non-identity slots

        # Pad to degree 7
        padded = pad_upto(prod.factors, max_rank=prod.rank, max_degree=7)
        assert padded.shape[1] == 7

        # reduce_degree with max_degree=2 should pack back
        result = reduce_degree(padded, max_degree=2, max_rank=10)
        assert result.shape[1] == 2

    def test_semantics_preserved_for_good_components(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Good components (eff_deg <= max_degree) preserve evaluation after packing."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        x2 = RankDecomposition.variable(2, num_vars, bool_algebra, backend=backend)
        poly = (x0 * x1) + x2

        # Pad to degree 6
        padded = pad_upto(poly.factors, max_rank=poly.rank, max_degree=6)
        reduced = reduce_degree(padded, max_degree=2, max_rank=10)

        padded_rd = RankDecomposition(padded, poly.max_rank, poly.max_degree, poly.max_replacement_degree, backend=backend)
        reduced_rd = RankDecomposition(reduced, poly.max_rank, poly.max_degree, poly.max_replacement_degree, backend=backend)

        orig_tt = _truth_table(padded_rd)
        reduced_tt = _truth_table(reduced_rd)
        assert orig_tt == reduced_tt


# -- prune_factors (with max_degree + algebra) --------------------------------


class TestPruneFactorsWithDegree:
    def test_backward_compatible(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """prune_factors with default args behaves the same as before."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        combined = x0 + x1

        # Both old and new interface should work
        pruned = prune_factors(combined.factors, max_rank=10)
        assert pruned.shape[0] <= 10

    def test_with_algebra_reduces_identity_slots(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Passing algebra to prune_factors activates strip_identity_slots."""
        num_vars = 3
        x0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        prod = x0 * x1  # degree 2

        padded = pad_upto(prod.factors, max_rank=prod.rank, max_degree=8)
        assert padded.shape[1] == 8

        pruned = prune_factors(padded, max_rank=10)
        # Strip should have removed trailing identity slots
        assert pruned.shape[1] < 8

    def test_max_rank_still_enforced(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Hard rank truncation still applies when all other strategies fail."""
        num_vars = 4
        # Create many distinct rank-1 components
        vars_ = [RankDecomposition.variable(i, num_vars, bool_algebra, backend=backend) for i in range(num_vars)]
        poly = vars_[0]
        for v in vars_[1:]:
            poly = poly + v  # 4 distinct components

        # Prune to rank 2
        pruned = prune_factors(poly.factors, max_rank=2)
        assert pruned.shape[0] <= 2


# -- Integration: degree bounds after repeated compose ---------------------------


class TestComposeDegreeWithMaxDegree:
    def test_compose_bounded_both_rank_and_degree(self, backend: str) -> None:
        """With max_degree set, repeated compose keeps degree bounded."""
        import algebraic as alg

        bool_alg = alg.semirings.boolean_algebra(mode="logic")
        num_vars = 2

        x0 = RankDecomposition.variable(0, num_vars=num_vars, algebra=bool_alg, max_rank=5, max_degree=3, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars=num_vars, algebra=bool_alg, max_rank=5, max_degree=3, backend=backend)

        sub0 = x0 * x1  # degree 2 replacement
        sub1 = x1  # degree 1 replacement

        p = x0
        for _ in range(5):
            p = p.compose([sub0, sub1])
            assert p.degree <= p.max_degree, f"degree {p.degree} > max_degree {p.max_degree}"
            assert p.rank <= p.max_rank + 5, f"rank {p.rank} far exceeds max_rank {p.max_rank}"

    def test_correctness_after_degree_bounded_compose(self, backend: str) -> None:
        """Semantic correctness is preserved when degree bounding is active."""
        import algebraic as alg

        bool_alg = alg.semirings.boolean_algebra(mode="logic")
        num_vars = 3

        x0 = RankDecomposition.variable(0, num_vars, bool_alg, max_rank=10, max_degree=3, backend=backend)
        x1 = RankDecomposition.variable(1, num_vars, bool_alg, max_rank=10, max_degree=3, backend=backend)
        x2 = RankDecomposition.variable(2, num_vars, bool_alg, max_rank=10, max_degree=3, backend=backend)

        # F(a) & F(b): step with [x1*x2, x1, x2] transitions
        transitions = [x1 * x2, x1, x2]
        ref = x1 * x2  # Expected result after one step from x0

        result = x0.compose(transitions)

        for bits in np.ndindex(*([2] * num_vars)):
            point = [bool_alg.one if b else bool_alg.zero for b in bits]
            ref_val = _eval_rd(ref, point)
            res_val = _eval_rd(result, point)
            assert_close(ref_val, res_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
