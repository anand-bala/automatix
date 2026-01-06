# mypy: disable-error-code="no-any-return,no-untyped-call"
from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import quax
from bitarray import frozenbitarray
from jaxtyping import Array, Scalar, Shaped

import algebraic.array.core as alge
from algebraic.array.core import AlgebraicArray
from algebraic.polynomials.sparse import SparsePolynomial
from algebraic.spec import BoundedDistributiveLattice as Lattice


class RankDecomposition[K: Lattice](eqx.Module):
    """CP (CANDECOMP/PARAFAC) decomposition of multilinear polynomial.

    Represents polynomial as sum of rank-1 components:
        p(x) = sum_{r=1}^R prod_{k=1}^d factors[r, k, index_k]

    where index_k in {0, 1, ..., n}:
        - 0 represents constant (always 1)
        - i (i>0) represents variable x_{i-1}
    """

    factors: Shaped[AlgebraicArray, "rank degree num_vars_p_1"]
    algebra: K
    max_rank: int = eqx.field(static=True)
    """Maximum rank for CP decomposition (controls memory usage)"""
    max_degree: int = eqx.field(static=True)
    """Maximum degree for polynomials (None = num_vars)"""
    max_replacement_degree: int = eqx.field(static=True)
    """Maximum degree for replacement polynomials in compose (None = max_degree)"""

    def __init__(
        self,
        factors: Shaped[AlgebraicArray, "rank degree num_vars_p_1"],
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.algebra = factors.semiring
        self.max_rank = max_rank if max_rank is not None else 100

        num_vars = factors.shape[0]
        self.max_degree = max_degree if max_degree is not None else num_vars
        self.max_replacement_degree = max_replacement_degree if max_replacement_degree is not None else self.max_degree

    def _replace_factors(self, factors: Shaped[AlgebraicArray, "rank degree num_vars_p_1"]) -> RankDecomposition[K]:
        return eqx.tree_at(lambda t: t.factors, self, factors)

    @property
    def rank(self) -> int:
        return self.factors.shape[0]

    @property
    def degree(self) -> int:
        return self.factors.shape[1]

    @property
    def num_vars(self) -> int:
        return self.factors.shape[2] - 1

    @staticmethod
    def variable(
        i: int,
        num_vars: int,
        algebra: K,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
    ) -> RankDecomposition[K]:
        """Create rank-1 polynomial representing variable x_i.

        Creates a CP decomposition with rank=1, degree=1.
        """
        # Single rank-1 component with degree 1
        # Shape: (1, 1, num_vars+1)
        factors = alge.zeros((1, 1, num_vars + 1), algebra)
        factors = factors.at[0, 0, i + 1].set(algebra.one)  # i+1 because 0 is constant

        return RankDecomposition(factors, max_rank, max_degree, max_replacement_degree)

    def _var_at(self, idx: int) -> RankDecomposition[K]:
        return RankDecomposition.variable(
            idx, self.num_vars, self.algebra, self.max_rank, self.max_degree, self.max_replacement_degree
        )

    @staticmethod
    def constant(
        value: Scalar,
        num_vars: int,
        algebra: K,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
    ) -> RankDecomposition[K]:
        """Create rank-1 polynomial representing constant.

        Creates a CP decomposition with rank=1, degree=1.
        """
        # Single rank-1 component with degree 1
        # Shape: (1, 1, num_vars+1)
        factors = alge.zeros((1, 1, num_vars + 1), algebra)
        factors = factors.at[0, 0, 0].set(value)  # index 0 is constant

        return RankDecomposition(factors, max_rank, max_degree, max_replacement_degree)

    @staticmethod
    def zero(
        num_vars: int,
        algebra: K,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
    ) -> RankDecomposition[K]:
        return RankDecomposition.constant(algebra.zero, num_vars, algebra, max_rank, max_degree, max_replacement_degree)

    @staticmethod
    def one(
        num_vars: int,
        algebra: K,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
    ) -> RankDecomposition[K]:
        return RankDecomposition.constant(algebra.one, num_vars, algebra, max_rank, max_degree, max_replacement_degree)

    def _make_const(self, val: Scalar) -> RankDecomposition[K]:
        return RankDecomposition.constant(
            val, self.num_vars, self.algebra, self.max_rank, self.max_degree, self.max_replacement_degree
        )

    def _pad_degree(self, target_degree: int) -> RankDecomposition[K]:
        """Pad polynomial to target degree by adding identity factors.

        For CP decomposition, padding adds dimensions with constant=1 (identity for multiplication).
        """
        if self.degree >= target_degree:
            return self

        rank, d, n_plus_1 = self.factors.shape
        extra_dims = target_degree - d

        # Add dimensions with constant=1 (identity for multiplication)
        padding = alge.zeros((rank, extra_dims, n_plus_1), self.algebra)
        padding = padding.at[:, :, 0].set(self.algebra.one)  # constant term = 1

        new_factors = quax.quaxify(jnp.concatenate)([self.factors, padding], axis=1)
        return self._replace_factors(new_factors)

    def __add__(self, other: RankDecomposition[K] | Scalar) -> RankDecomposition[K]:
        """Add by concatenating rank-1 components.

        For CP decomposition: p + q = sum of all components from both.
        """

        if jnp.isscalar(other):
            assert isinstance(other, Scalar)
            other = self._make_const(other)
        assert isinstance(other, RankDecomposition)

        assert other.num_vars == self.num_vars

        # Pad both to same degree
        d = max(self.degree, other.degree)
        a_padded = self._pad_degree(d)
        b_padded = other._pad_degree(d)

        # Concatenate along rank dimension
        new_factors = quax.quaxify(jnp.concatenate)([a_padded.factors, b_padded.factors], axis=0)
        result: RankDecomposition[K] = self._replace_factors(new_factors)
        result = result._compress_rank()

        return result

    @eqx.filter_jit
    @quax.quaxify
    def _multiply_arrays(self, p_arr: AlgebraicArray[K], q_arr: AlgebraicArray[K]) -> AlgebraicArray[K]:
        """Core multiplication logic on raw arrays (no simplification/compression).

        This is the pure computational kernel used by both mul() and compose().

        Performance:
        - No Python loops (fully JIT-compiled)
        - Broadcast operations are GPU-efficient
        - Memory-contiguous concatenation

        Args:
            p_arr: Shape [R_p, d_p, n+1]
            q_arr: Shape [R_q, d_q, n+1]

        Returns:
            Shape [R_p * R_q, d_p + d_q, n+1]
        """
        rank_p, degree_p, n_plus_1 = p_arr.shape
        rank_q, degree_q, _ = q_arr.shape

        # Broadcast to [R_p, R_q, d_p, n+1] and [R_p, R_q, d_q, n+1]
        p_expanded = jnp.broadcast_to(
            p_arr[:, None, :, :],  # [R_p, 1, d_p, n+1]
            (rank_p, rank_q, degree_p, n_plus_1),
        )
        q_expanded = jnp.broadcast_to(
            q_arr[None, :, :, :],  # [1, R_q, d_q, n+1]
            (rank_p, rank_q, degree_q, n_plus_1),
        )

        # Concatenate along degree dimension
        result = jnp.concatenate([p_expanded, q_expanded], axis=2)
        # Shape: [R_p, R_q, d_p + d_q, n+1]

        # Reshape to [R_p * R_q, d_p + d_q, n+1]
        result = result.reshape(rank_p * rank_q, degree_p + degree_q, n_plus_1)

        return result

    def __mul__(self, other: RankDecomposition[K]) -> RankDecomposition[K]:
        """Multiply two CP-decomposed polynomials.

        Delegates core multiplication to _multiply_arrays(), then applies
        simplification and compression.
        """
        # Core multiplication (Khatri-Rao product)
        new_factors = self._multiply_arrays(self.factors, other.factors)
        result = eqx.tree_at(lambda t: t.factors, self, new_factors)

        # Post-processing: Use fast heuristic first, falls back to exact if needed
        result = result._simplify_multilinear_fast()  # Apply x_i * x_i = x_i
        result = result._compress_rank()
        return result

    def _simplify_multilinear(self) -> RankDecomposition[K]:
        """Apply x_i * x_i = x_i to cap degree at num_vars.

        Uses sparse representation as intermediary for simplification.
        This handles both:
        1. Degree reduction (x_i * x_i -> x_i)
        2. Rank deduplication (merging identical monomials)
        """
        # Convert to sparse representation
        sparse = self.to_sparse()

        # Sparse already groups by monomial (automatic simplification)
        if len(sparse) == 0:
            return self._make_const(self.algebra.zero)

        # Determine max degree needed
        max_deg = max(sum(monomial) for monomial in sparse.keys())
        max_deg = min(max_deg, self.num_vars)  # Cap at num_vars

        # Convert back to CP form
        return self.from_sparse(sparse, max_degree=max_deg)

    def _simplify_multilinear_fast(self) -> RankDecomposition[K]:
        """Fast heuristic simplification using deduplication.

        This is an incomplete heuristic that catches common patterns:
        1. Duplicate variables within rank-1 components (x_i * x_i -> x_i)
        2. Duplicate rank-1 components (p + p -> p for idempotent addition)

        Falls back to exact sparse-based simplification if result is still complex.
        Uses Python loops for JIT compatibility (loops should be unrolled at compile time).

        Note
        ----
        This heuristic is faster than full sparse conversion O((n+1)^d) but incomplete.
        It won't detect absorption laws like x_0 + x_0*x_1 = x_0 (Boolean algebra).
        """
        # Apply fast heuristics
        poly = self._deduplicate_degrees_fast()
        poly = poly._deduplicate_ranks_fast()

        # Fallback thresholds: if still complex after heuristics, use exact method
        # These are conservative - adjust based on performance requirements
        threshold_rank = min(self.max_rank * 2, 200)
        threshold_degree = min(self.num_vars * 2, 20)

        if poly.rank > threshold_rank or poly.degree > threshold_degree:
            return poly._simplify_multilinear()

        return poly

    @eqx.filter_jit
    def _deduplicate_degrees_fast(self) -> RankDecomposition[K]:
        """Replace duplicate degree dimensions with identity within each rank.

        For multilinear polynomials: x_i * x_i = x_i
        If factors[r, k1, :] == factors[r, k2, :], replace one with identity (constant=1).

        Uses Python loops (should be unrolled by JIT) and jnp.where for compatibility.
        """
        new_factors = self.factors

        # Create identity vector once (constant = 1, all variables = 0)
        identity = alge.zeros(self.num_vars + 1, self.algebra)
        identity = identity.at[0].set(self.algebra.one)

        # Python loops over rank and degree dimensions
        # JAX will unroll these loops at compile time for fixed-size loops
        for r in range(self.rank):
            for k1 in range(self.degree):
                for k2 in range(k1 + 1, self.degree):
                    # Check if degree dimensions k1 and k2 select the same variable
                    is_duplicate = jnp.all(new_factors[r, k1, :] == new_factors[r, k2, :])

                    # Replace k2 with identity if duplicate (JIT-compatible with jnp.where)
                    new_factors_data = new_factors.data.at[r, k2, :].set(
                        jnp.where(is_duplicate, identity.data, new_factors[r, k2, :].data)
                    )
                    new_factors = eqx.tree_at(lambda t: t.data, new_factors, new_factors_data)
                    # new_factors = new_factors.at[r, k2, :].set(where(is_duplicate, identity, new_factors[r, k2, :]))

        return self._replace_factors(new_factors)

    @eqx.filter_jit
    def _deduplicate_ranks_fast(self) -> RankDecomposition[K]:
        """Mark duplicate rank-1 components as zero.

        For idempotent addition (Boolean, Tropical, MaxMin): p + p = p
        If two rank-1 components are identical, mark the later one as zero.

        Zero components don't affect evaluation and will be removed by compression.
        """
        # Track which ranks to keep (keep first occurrence of duplicates)
        keep_mask = jnp.ones(self.rank, dtype=bool)

        # Python loops - JAX should unroll these
        for r1 in range(self.rank):
            for r2 in range(r1 + 1, self.rank):
                # Check if entire rank-1 components are equal
                is_duplicate = jnp.all(self.factors[r1, :, :] == self.factors[r2, :, :])

                # Mark r2 for removal if it's a duplicate of r1
                keep_mask = keep_mask.at[r2].set(jnp.where(is_duplicate, False, keep_mask[r2]))

        # Replace duplicates with zero components
        # (Don't change array shape - keep JIT-compatible)
        new_factors = self.factors
        zero_component = alge.zeros((self.degree, self.num_vars + 1), self.algebra)

        for r in range(self.rank):
            should_zero = ~keep_mask[r]
            new_factors_data = new_factors.data.at[r, :, :].set(
                jnp.where(should_zero, zero_component.data, new_factors[r, :, :].data)
            )
            new_factors = eqx.tree_at(lambda t: t.data, new_factors, new_factors_data)
            # new_factors = new_factors.at[r, :, :].set(jnp.where(should_zero, zero_component, new_factors[r, :, :]))

        return self._replace_factors(new_factors)

    @eqx.filter_jit
    @quax.quaxify
    def _compress_rank(self) -> RankDecomposition[K]:
        """Compress to at most max_rank components using magnitude-based truncation.

        Keeps the top-max_rank components by L2 norm magnitude.
        """
        if self.rank <= self.max_rank:
            return self

        # Compute magnitude of each rank-1 component
        # magnitude[r] = prod_k norm(factors[r, k, :])
        # Note: Using jnp.zeros for numeric magnitudes (not algebraic values)
        magnitudes = jnp.zeros(self.rank)
        for r in range(self.rank):
            mag = 1.0  # Numeric computation
            for k in range(self.degree):
                # L2 norm of the k-th factor vector
                mag *= jnp.linalg.norm(self.factors[r, k, :])
            magnitudes = magnitudes.at[r].set(mag)

        # Keep top max_rank components
        top_indices = jnp.argsort(magnitudes)[-self.max_rank :]
        new_factors = self.factors[top_indices]

        return RankDecomposition(new_factors)

    def evaluate(self, points: Array | Mapping[int, Scalar]) -> RankDecomposition[K]:
        """Evaluate polynomial at given point.

        Args:
            poly: CP-decomposed polynomial
            points: Either Array of shape (num_vars,) for full evaluation,
                   or Mapping[int, Scalar] for partial evaluation

        Returns:
            Constant polynomial (RankDecomposition) after evaluation
        """
        # For simplicity, convert to full evaluation via compose
        # (partial evaluation is equivalent to composition with constants)
        if isinstance(points, Mapping):
            # Partial evaluation: only substitute specified variables
            replacements = {i: self._make_const(v) for i, v in points.items()}
            return self.compose(replacements)

        # Full evaluation: substitute all variables
        rank, d, _ = self.factors.shape

        # Build selector vector: [1, point[0], point[1], ..., point[n-1]]
        one_array = alge.ones((1,), self.algebra)
        selector = quax.quaxify(jnp.concatenate)([one_array, points])  # Shape: (n+1,)

        # For each rank-1 component
        result = alge.zeros((), self.algebra)
        for r in range(rank):
            # Evaluate component: prod_k sum_i factors[r,k,i] * selector[i]
            component_value = alge.ones((), self.algebra)
            for k in range(d):
                # Inner product of factors[r,k,:] with selector
                dim_value = alge.zeros((), self.algebra)
                for i in range(self.num_vars + 1):
                    term = self.factors[r, k, i] * selector[i]
                    dim_value = dim_value + term

                component_value = component_value * dim_value

            result = result + component_value

        # Return as constant polynomial
        return self._make_const(result.data)

    def _prepare_replacement_array(self, replacements: dict[int, RankDecomposition[K]]) -> AlgebraicArray[K]:
        """Prepare padded array of replacement polynomials.

        PERFORMANCE NOTE: Cache this result and reuse across multiple compose() calls!

        Returns:
            Array of shape [n+1, R_max, max_replacement_degree, m+1]
            Index 0: constant (identity: always 1)
            Index i+1: variable x_i (or its replacement)
        """
        # Build full replacement dict with identity for missing variables
        # q_array[0] = constant (identity)
        # q_array[i+1] = replacement for variable x_i
        full_replacements = (
            [self._make_const(self.algebra.one)]  # index 0: constant
            + [replacements.get(i, self._var_at(i)) for i in range(self.num_vars)]
        )

        # Calculate max shapes
        max_rank = max(q.rank for q in full_replacements)
        m_plus_1 = self.num_vars + 1

        # Pad each polynomial
        padded_list = []
        for q in full_replacements:
            rank, d, _ = q.factors.shape
            padded = alge.zeros((max_rank, self.max_replacement_degree, m_plus_1), self.algebra)
            # Copy existing factors
            padded = padded.at[:rank, :d, :].set(q.factors)
            # Pad extra degree dimensions with constant=1 (identity for multiplication)
            if d < self.max_replacement_degree:
                padded = padded.at[:rank, d:, 0].set(self.algebra.one)
            padded_list.append(padded)

        # Stack into contiguous array for fast indexing
        q_array = quax.quaxify(jnp.stack)(padded_list, axis=0)

        return q_array  # Shape: [n+1, R_max, max_replacement_degree, m+1]

    def compose(self, replacements: dict[int, RankDecomposition]) -> RankDecomposition[K]:
        """Compose p with replacement polynomials.

        Performance notes (CRITICAL for AFA hot path):
        - Fully JIT-compiled (no Python loops)
        - vmap over rank → GPU parallel across components
        - scan over degree → sequential but cache-friendly
        - Static max_replacement_degree for JIT optimization
        - CACHE the replacement array for reuse!

        Args:
            poly: Polynomial to compose
            replacements: Dict mapping variable indices to replacement polynomials

        Returns:
            Composed polynomial
        """
        result: AlgebraicArray[K]
        # Step 1: Prepare replacements (CACHE THIS in AFA code!)
        q_array = self._prepare_replacement_array(replacements)
        # Shape: [n+1, R_max, max_replacement_degree, m+1]

        # Step 2: Compose each rank-1 component
        # Note: Using Python loop over rank for now (degree is typically small)
        # Future optimization: Use vmap with fixed-size accumulator
        composed_list = []

        for r in range(self.rank):
            p_component = self.factors[r]  # [d_p, n+1]

            # Identify which index each factor selects (0=constant, 1=x_0, 2=x_1, ...)
            var_indices = jnp.argmax(p_component.data, axis=1)  # [d_p]

            # Gather the polynomials to multiply
            selected = q_array[var_indices]  # [d_p, R_max, max_replacement_degree, m+1]

            # Multiply sequentially
            result = selected[0]  # [R_max, max_replacement_degree, m+1]
            for k in range(1, self.degree):
                result = self._multiply_arrays(result, selected[k])

            composed_list.append(result)

        # Stack all composed components
        composed = quax.quaxify(jnp.stack)(composed_list, axis=0)
        assert isinstance(composed, AlgebraicArray)
        # Shape: [R_p, R_result, d_result, m+1]

        # Step 3: Flatten rank dimensions
        rank_p, rank_result, d_result, m_plus_1 = composed.shape
        result_factors = quax.quaxify(jnp.reshape)(composed, (rank_p * rank_result, d_result, m_plus_1))
        assert isinstance(result_factors, AlgebraicArray)

        result = self._replace_factors(result_factors)

        # Step 4: Simplify and compress
        # Use fast heuristic first (especially important for AFA hot path)
        # Falls back to exact simplification if needed
        result = result._simplify_multilinear_fast()
        result = result._compress_rank()
        return result

    def _index_to_bits(self, index: int) -> tuple[int, ...]:
        """Convert flat index to n-bit tuple."""
        from bitarray.util import int2ba

        return tuple(int2ba(index, length=self.num_vars))

    def to_sparse(self) -> SparsePolynomial:
        """Convert CP to sparse by enumerating all monomial evaluations.

        WARNING: This is expensive O((n+1)^d) where d is degree.
        """
        from itertools import product

        result: dict[frozenbitarray, Scalar] = {}

        # Enumerate all possible assignments (expensive: (n+1)^d possibilities)
        for assignment in product(range(self.num_vars + 1), repeat=self.degree):
            # Determine which variables are present in this assignment
            # (handles x_i * x_i = x_i automatically by mapping to same monomial)
            vars_present = frozenbitarray(
                [any(assignment[k] == i + 1 for k in range(self.degree)) for i in range(self.num_vars)]
            )

            # Evaluate coefficient for this assignment
            coeff = self.algebra.zero
            for r in range(self.rank):
                component = self.algebra.one
                for k in range(self.degree):
                    component = self.algebra.mul(component, self.factors[r, k, assignment[k]])
                coeff = self.algebra.add(coeff, component)

            # Add to sparse (accumulate if monomial already exists)
            if not jnp.allclose(coeff, self.algebra.zero):
                if vars_present in result:
                    result[vars_present] = self.algebra.add(result[vars_present], coeff)
                else:
                    result[vars_present] = coeff

        return SparsePolynomial(self.algebra, self.num_vars, result)

    @staticmethod
    def from_sparse(
        sparse: SparsePolynomial,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
    ) -> RankDecomposition[K]:
        """Convert sparse to CP form (each monomial becomes rank-1 component).

        Args:
            sparse: Sparse polynomial to convert
            max_degree: Maximum degree for result (default: self.max_degree)

        Returns:
            CP decomposition with one rank-1 component per monomial
        """
        if max_degree is None:
            max_degree = sparse.num_vars
        algebra = sparse.algebra
        num_vars = sparse.num_vars
        zero_poly = RankDecomposition.constant(
            algebra.zero, sparse.num_vars, algebra, max_rank, max_degree, max_replacement_degree
        )
        max_rank = zero_poly.max_rank
        max_degree = zero_poly.max_degree
        max_replacement_degree = zero_poly.max_degree

        if len(sparse) == 0:
            return zero_poly

        rank = len(sparse)
        factors = alge.zeros((rank, max_degree, num_vars + 1), algebra)

        for r, (monomial, coeff) in enumerate(sparse.items()):
            # monomial is a bitarray indicating which variables appear
            vars_in_monomial = [i for i, bit in enumerate(monomial) if bit]

            # Fill in factors for this rank-1 component
            for k, var_idx in enumerate(vars_in_monomial):
                if k < max_degree:
                    factors = factors.at[r, k, var_idx + 1].set(coeff if k == 0 else algebra.one)

            # Pad remaining dimensions with constant=1
            for k in range(len(vars_in_monomial), max_degree):
                factors = factors.at[r, k, 0].set(algebra.one)

        return RankDecomposition(factors)
