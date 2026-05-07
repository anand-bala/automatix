"""Tests for JAX pytree registration of algebraic polynomial types."""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

from typing import Literal

import algebraic.ops as algebraic_ops
import jax
import jax.numpy as jnp
import pytest
from algebraic.polynomials import LowRankFactors, MonomialBasis, PolyDict, RankDecomposition
from algebraic.semirings import boolean_algebra, max_min_algebra
from algebraic.utils.poly import batched_evaluate_factors
from algebraic.utils.testing import assert_close


@pytest.fixture(autouse=True)
def activate_jax_pytrees() -> None:
    """Importing this module activates JAX pytree registration for all algebraic types."""
    import algebraic.utils.jax  # noqa: F401


class TestRankDecompositionJAX:
    """Test JAX jit and vmap through RankDecomposition."""

    def test_jit_evaluation(self, jax_backend: str) -> None:
        """jit-compiled evaluation should produce correct results."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend=jax_backend)

        result = jax.jit(lambda pt: x0.evaluate(pt))(jnp.array([2.0, 3.0]))
        assert_close(result, 2.0)

    def test_vmap_evaluation(self, jax_backend: str) -> None:
        """vmap should batch evaluation over a leading axis of points."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend=jax_backend)

        pts = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        results = jax.vmap(lambda pt: x0.evaluate(pt))(pts)

        assert_close(results, jnp.array([1.0, 2.0, 3.0]))

    def test_tree_flatten_unflatten_roundtrip(self, jax_backend: str) -> None:
        """Flattening and unflattening should reconstruct an equivalent polynomial."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend=jax_backend)
        p = x0 * x0

        leaves, treedef = jax.tree_util.tree_flatten(p)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        pt = jnp.array([3.0, 1.0])
        assert_close(
            p.evaluate(pt),
            reconstructed.evaluate(pt),
        )


class TestPolyDictJAX:
    """Test JAX pytree support for PolyDict."""

    def test_tree_flatten_unflatten_roundtrip(self, jax_backend: str) -> None:
        """Flattening and unflattening should reconstruct an equivalent PolyDict."""
        alg = max_min_algebra()
        x0 = PolyDict.variable(0, 2, algebra=alg, backend=jax_backend)
        x1 = PolyDict.variable(1, 2, algebra=alg, backend=jax_backend)
        p = x0 + x1

        leaves, treedef = jax.tree_util.tree_flatten(p)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        pt = jnp.array([2.0, 5.0])
        original_val = list(p.evaluate(pt).values())[0].data
        reconstructed_val = list(reconstructed.evaluate(pt).values())[0].data
        assert_close(original_val, reconstructed_val)


class TestRankDecompositionJITTraining:
    """End-to-end JIT/grad/vmap through compose, add, mul on RankDecomposition.

    These exercise the JIT-safe path enabled by ``static_shape=True`` (with
    ``shortcircuit=True`` and ``pack=True``).
    """

    def _soft_alg(self):
        # Continuous-valued lattice: gradients flow non-trivially through add/mul.
        return boolean_algebra(mode="soft")

    def test_jit_compose(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 3
        x = [
            RankDecomposition.variable(i, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=8)
            for i in range(num_vars)
        ]
        p = x[0].add(x[1], shortcircuit=True, pack=True, static_shape=True)

        @jax.jit
        def f(poly):
            composed = poly.compose([x[0], x[1], x[2]], shortcircuit=True, pack=True, static_shape=True)
            return composed.evaluate(jnp.array([0.7, 0.3, 0.9])).data

        eager = p.compose([x[0], x[1], x[2]], shortcircuit=True, pack=True, static_shape=True)
        eager_val = eager.evaluate(jnp.array([0.7, 0.3, 0.9])).data
        assert_close(f(p), eager_val)

    def test_jit_add_mul(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = RankDecomposition.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=8)
        b = RankDecomposition.variable(1, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=8)

        @jax.jit
        def f(p, q):
            r = p.mul(q, shortcircuit=True, pack=True, static_shape=True)
            r = r.add(p, shortcircuit=True, pack=True, static_shape=True)
            return r.evaluate(jnp.array([0.6, 0.8])).data

        # Run once to ensure trace succeeds; compare to eager.
        eager = a.mul(b, shortcircuit=True, pack=True, static_shape=True)
        eager = eager.add(a, shortcircuit=True, pack=True, static_shape=True)
        eager_val = eager.evaluate(jnp.array([0.6, 0.8])).data
        assert_close(f(a, b), eager_val)

    def test_grad_through_evaluate(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = RankDecomposition.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)

        def loss_fn(factors):
            poly = RankDecomposition(
                factors,
                a.max_rank,
                a.max_degree,
                a.max_replacement_degree,
                backend=jax_backend,
            )
            return poly.evaluate(jnp.array([0.7, 0.4])).data.sum()

        g = jax.grad(loss_fn)(a.factors)
        # Soft algebra over (0.7, 0.4) gives non-zero gradient w.r.t. factors.
        assert jnp.any(jnp.abs(g.data) > 0)

    def test_grad_through_compose(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = RankDecomposition.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        b = RankDecomposition.variable(1, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)

        def loss_fn(factors):
            poly = RankDecomposition(
                factors,
                a.max_rank,
                a.max_degree,
                a.max_replacement_degree,
                backend=jax_backend,
            )
            composed = poly.compose([a, b], shortcircuit=True, pack=True, static_shape=True)
            return composed.evaluate(jnp.array([0.7, 0.4])).data.sum()

        loss_val, g = jax.value_and_grad(loss_fn)(a.factors)
        assert jnp.isfinite(loss_val)
        assert jnp.any(jnp.abs(g.data) > 0)

    def test_vmap_compose(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = RankDecomposition.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        b = RankDecomposition.variable(1, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)

        # vmap over a batch of points.  Compose itself is shape-static so
        # vmap can broadcast through it via the pytree leaves.
        composed = a.compose([a, b], shortcircuit=True, pack=True, static_shape=True)
        pts = jnp.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.2]])
        results = jax.vmap(lambda pt: composed.evaluate(pt).data)(pts)
        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(jnp.array(results)))

    def test_static_dynamic_roundtrip(self, jax_backend: str) -> None:
        """static_shape=True and static_shape=False (eager) agree on a small case."""
        alg = self._soft_alg()
        num_vars = 2
        a = RankDecomposition.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        b = RankDecomposition.variable(1, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        p = a.add(b)  # default eager path

        composed_static = p.compose([a, b], shortcircuit=True, pack=True, static_shape=True)
        composed_eager = p.compose([a, b], shortcircuit=False, pack=True, static_shape=False)

        pt = jnp.array([0.6, 0.3])
        v_static = composed_static.evaluate(pt).data
        v_eager = composed_eager.evaluate(pt).data
        assert_close(v_static, v_eager, atol=1e-5)

    def test_prune_preserves_value(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = RankDecomposition.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        b = RankDecomposition.variable(1, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        p = a.add(b)
        pruned = p.prune(shortcircuit=True, pack=True, static_shape=True)

        pt = jnp.array([0.4, 0.7])
        assert_close(p.evaluate(pt).data, pruned.evaluate(pt).data, atol=1e-5)

    def test_static_shape_rejects_bad_combo(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = RankDecomposition.variable(0, num_vars, alg, backend=jax_backend)
        with pytest.raises(ValueError, match="static_shape=True requires"):
            a.compose([a, a], shortcircuit=False, static_shape=True)


class TestJITPathTruthTableFingerprint:
    """Compare polynomial paths via truth-table fingerprints over lattice extremals.

    For each expression we build the polynomial twice -- on the JIT-safe path
    (``shortcircuit=True, pack=True, static_shape=True``) and on the eager
    smart-prune path -- and compare the resulting truth tables at all ``2^n``
    bitvector inputs.

    For ``add`` / ``mul`` the two paths represent the *same* polynomial up to
    factorisation; their truth tables must agree **exactly** (no
    ``normalize()`` round-trip on either side, so factorisation differences
    cannot change polynomial values).

    For ``compose`` we test against a stricter **ground-truth** target: the
    composed polynomial evaluated at ``pt`` must equal ``source(q_0(pt), ...,
    q_{n-1}(pt))``.  The JIT-safe path is asserted to match the ground truth
    exactly.  The eager path is also tested but with a wider tolerance, since
    it appends ``normalize()`` (a ``to_sparse -> from_sparse`` round-trip) on
    the unbatched path that has been observed to disagree with the ground
    truth on some replacement patterns -- a separate latent bug worth
    surfacing rather than papering over with bounded Hamming.

    ``max_rank`` and ``max_degree`` are sized so neither path needs to
    hard-truncate -- truncation is a separate lossy step that this test does
    not exercise.
    """

    NUM_VARS = 3
    MAX_RANK = 64
    MAX_DEGREE = 8

    @staticmethod
    def _all_extremals(num_vars: int):
        # Shape: (2^n, n) of 0.0/1.0 floats.  Floats so soft Boolean accepts them.
        return jnp.array(
            [[(i >> k) & 1 for k in range(num_vars)] for i in range(1 << num_vars)],
            dtype=jnp.float32,
        )

    def _truth_table(self, poly, points):
        """Evaluate poly at every row of points; returns a (2^n,) array.

        Uses :func:`batched_evaluate_factors` directly (broadcasting the
        unbatched polynomial across the points axis) rather than ``jax.vmap``.
        For evaluation, the batched path is the same einsum/prod/sum chain as
        the unbatched one, just with one extra leading axis -- so we get the
        same JIT-friendliness without paying for vmap's broadcasting layer.
        Handles both :class:`RankDecomposition` (uses ``.factors``) and
        :class:`LowRankFactors` (uses ``.to_merged()``).
        """
        factors = poly.to_merged() if isinstance(poly, LowRankFactors) else poly.factors
        n_pts = points.shape[0]
        batched_factors = algebraic_ops.broadcast_to(factors[None, ...], (n_pts, *factors.shape))
        return batched_evaluate_factors(batched_factors, points, poly.backend).data

    def _ground_truth_compose_tt(self, source, reps, points):
        """Compute the ground-truth truth table of ``source.compose(reps)``.

        Evaluates each replacement at every extremal to get a substituted
        ``(2^n, n)`` point matrix, then evaluates ``source`` at those points.
        Bypasses the compose() machinery entirely.
        """
        rep_tts = jnp.stack([self._truth_table(rep, points) for rep in reps], axis=-1)  # (2^n, n)
        return self._truth_table(source, rep_tts)

    @pytest.mark.parametrize("mode", ["logic", "soft"])
    def test_compose_jit_path_matches_ground_truth(self, jax_backend: str, mode: Literal["logic", "soft"]) -> None:
        """JIT-safe compose result == direct substitute-then-evaluate."""
        alg = boolean_algebra(mode=mode)
        n = self.NUM_VARS
        x = [RankDecomposition.variable(i, n, alg, self.MAX_RANK, self.MAX_DEGREE, backend=jax_backend) for i in range(n)]
        jit_kw = dict(shortcircuit=True, pack=True, static_shape=True)

        # Source: (x_0 | x_1) & x_2.  Replacements: x_0 -> x_1 & x_2, x_1 -> x_2, x_2 -> x_0.
        x1_and_x2 = x[1].mul(x[2], **jit_kw)
        reps = [x1_and_x2, x[2], x[0]]
        source = x[0].add(x[1], **jit_kw).mul(x[2], **jit_kw)

        composed = source.compose(reps, **jit_kw)

        pts = self._all_extremals(n)
        tt_jit = self._truth_table(composed, pts)
        tt_truth = self._ground_truth_compose_tt(source, reps, pts)

        assert_close(tt_jit, tt_truth, atol=1e-6)

    @pytest.mark.parametrize("mode", ["logic", "soft"])
    def test_add_mul_truth_tables_match(self, jax_backend: str, mode: Literal["logic", "soft"]) -> None:
        """add/mul-only fingerprint: must match exactly on lattice extremals.

        No ``normalize()`` round-trip on either path, so factorisation
        differences alone cannot change polynomial values.
        """
        alg = boolean_algebra(mode=mode)
        n = self.NUM_VARS
        x = [RankDecomposition.variable(i, n, alg, self.MAX_RANK, self.MAX_DEGREE, backend=jax_backend) for i in range(n)]

        # (x0 | x1) & (x1 | x2) & (x0 | x2) -- non-trivial expression
        # whose CP factorisation is not unique.
        def build(use_jit_safe: bool):
            if use_jit_safe:
                kwargs = dict(shortcircuit=True, pack=True, static_shape=True)
                a = x[0].add(x[1], **kwargs)
                b = x[1].add(x[2], **kwargs)
                c = x[0].add(x[2], **kwargs)
                return a.mul(b, **kwargs).mul(c, **kwargs)
            return (x[0] + x[1]) * (x[1] + x[2]) * (x[0] + x[2])

        poly_static = build(True)
        poly_eager = build(False)

        pts = self._all_extremals(n)
        tt_static = self._truth_table(poly_static, pts)
        tt_eager = self._truth_table(poly_eager, pts)

        assert_close(tt_static, tt_eager, atol=1e-6)
        assert int(jnp.sum(jnp.round(tt_static) != jnp.round(tt_eager))) == 0

    def test_lowrank_compose_jit_path_matches_ground_truth(self, jax_backend: str) -> None:
        """LowRankFactors: JIT-safe compose matches ground truth on logic Boolean."""
        alg = boolean_algebra(mode="logic")
        n = self.NUM_VARS
        x = [LowRankFactors.variable(i, n, alg, self.MAX_RANK, self.MAX_DEGREE, backend=jax_backend) for i in range(n)]
        jit_kw = dict(shortcircuit=True, pack=True, static_shape=True)

        source = x[0].add(x[1], **jit_kw).mul(x[2], **jit_kw)
        reps = [x[1], x[2], x[0]]

        composed = source.compose(reps, **jit_kw)

        pts = self._all_extremals(n)
        tt_jit = self._truth_table(composed, pts)
        tt_truth = self._ground_truth_compose_tt(source, reps, pts)

        assert_close(tt_jit, tt_truth, atol=1e-6)

    def test_compose_eager_path_matches_ground_truth(self, jax_backend: str) -> None:
        """Counterpart to the JIT test for the eager path.

        Previously xfail due to a latent correctness bug: the smart-prune
        ``merge_compatible_components`` produced multi-non-zero source slots
        that ``compose_factors``'s einsum mis-contracted. Fixed by replacing
        the einsum with a broadcast-reshape that handles non-monomial slots
        correctly.
        """
        alg = boolean_algebra(mode="logic")
        n = self.NUM_VARS
        x = [RankDecomposition.variable(i, n, alg, self.MAX_RANK, self.MAX_DEGREE, backend=jax_backend) for i in range(n)]

        x1_and_x2 = x[1] * x[2]
        reps = [x1_and_x2, x[2], x[0]]
        source = (x[0] + x[1]) * x[2]
        composed = source.compose(reps)

        pts = self._all_extremals(n)
        tt_eager = self._truth_table(composed, pts)
        tt_truth = self._ground_truth_compose_tt(source, reps, pts)

        assert_close(tt_eager, tt_truth, atol=1e-6)

    @pytest.mark.parametrize(
        "source_idx,rep_idx_pattern",
        [
            # source[i] = (p_idx_a + p_idx_b) * x[c]; reps_idx = list of rep types
            ((0, 1, 2), (None, None, None)),  # identity reps
            ((0, 1, 2), (1, 2, 0)),  # cyclic permutation
            ((0, 1, 2), ("mul12", 2, 0)),  # rep[0] = x1 * x2 (non-trivial)
            ((1, 2, 0), ("mul01", 1, 2)),  # rotated source + nontrivial rep
        ],
    )
    def test_compose_distributive_patterns_match_ground_truth(
        self, jax_backend: str, source_idx: tuple[int, int, int], rep_idx_pattern: tuple[int | str | None, ...]
    ) -> None:
        """Several `(p+q)*r` patterns where merge_compatible_components fires.

        For each pattern, build the source via the eager path (which runs
        the full smart-prune incl. merge) and verify both the JIT-safe
        composed polynomial and the eager composed polynomial agree with the
        substitute-then-evaluate ground truth on every lattice extremal.
        """
        alg = boolean_algebra(mode="logic")
        n = self.NUM_VARS
        x = [RankDecomposition.variable(i, n, alg, self.MAX_RANK, self.MAX_DEGREE, backend=jax_backend) for i in range(n)]

        a, b, c = source_idx
        source = (x[a] + x[b]) * x[c]  # eager path: merge fires for the (x[a]+x[b]) part

        def resolve(token):
            if token is None:
                return x[0]  # placeholder, overridden by indexed list below
            if isinstance(token, int):
                return x[token]
            if token == "mul12":
                return x[1] * x[2]
            if token == "mul01":
                return x[0] * x[1]
            raise ValueError(token)

        if all(t is None for t in rep_idx_pattern):
            reps = [x[0], x[1], x[2]]
        else:
            reps = [resolve(t) for t in rep_idx_pattern]

        pts = self._all_extremals(n)
        tt_truth = self._ground_truth_compose_tt(source, reps, pts)

        # JIT path
        composed_jit = source.compose(reps, shortcircuit=True, pack=True, static_shape=True)
        tt_jit = self._truth_table(composed_jit, pts)
        assert_close(tt_jit, tt_truth, atol=1e-6)

        # Eager path
        composed_eager = source.compose(reps)
        tt_eager = self._truth_table(composed_eager, pts)
        assert_close(tt_eager, tt_truth, atol=1e-6)


class TestLowRankFactorsJITTraining:
    """Same JIT/grad coverage for LowRankFactors."""

    def _soft_alg(self):
        return boolean_algebra(mode="soft")

    def test_jit_compose(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = LowRankFactors.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        b = LowRankFactors.variable(1, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)

        @jax.jit
        def f(p):
            composed = p.compose([a, b], shortcircuit=True, pack=True, static_shape=True)
            return composed.evaluate(jnp.array([0.7, 0.3])).data

        eager = a.compose([a, b], shortcircuit=True, pack=True, static_shape=True)
        eager_val = eager.evaluate(jnp.array([0.7, 0.3])).data
        assert_close(f(a), eager_val)

    def test_grad_through_compose(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = LowRankFactors.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        b = LowRankFactors.variable(1, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)

        def loss_fn(weights, bias):
            poly = LowRankFactors(weights, bias, a.max_rank, a.max_degree, a.max_replacement_degree, backend=jax_backend)
            composed = poly.compose([a, b], shortcircuit=True, pack=True, static_shape=True)
            return composed.evaluate(jnp.array([0.7, 0.3])).data.sum()

        g = jax.grad(loss_fn, argnums=(0, 1))(a.weights, a.bias)
        assert jnp.any(jnp.abs(g[0].data) > 0) or jnp.any(jnp.abs(g[1].data) > 0)

    def test_prune_preserves_value(self, jax_backend: str) -> None:
        alg = self._soft_alg()
        num_vars = 2
        a = LowRankFactors.variable(0, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        b = LowRankFactors.variable(1, num_vars, alg, backend=jax_backend, max_degree=4, max_rank=4)
        p = a.add(b)
        pruned = p.prune(shortcircuit=True, pack=True, static_shape=True)

        pt = jnp.array([0.4, 0.7])
        assert_close(p.evaluate(pt).data, pruned.evaluate(pt).data, atol=1e-5)


class TestMonomialBasisJAX:
    """Test JAX pytree support for MonomialBasis."""

    def test_tree_flatten_unflatten_roundtrip(self, jax_backend: str) -> None:
        """Flattening and unflattening should reconstruct an equivalent MonomialBasis."""
        alg = boolean_algebra(mode="logic")
        x0 = MonomialBasis.variable(0, 2, algebra=alg, backend=jax_backend)
        x1 = MonomialBasis.variable(1, 2, algebra=alg, backend=jax_backend)
        p = x0 * x1

        leaves, treedef = jax.tree_util.tree_flatten(p)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        pt = jnp.array([True, True])
        original_val = p.evaluate(pt).coeffs.data
        reconstructed_val = reconstructed.evaluate(pt).coeffs.data
        assert_close(original_val, reconstructed_val)
