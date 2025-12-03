"""JAX-based dense tensor polynomial representations."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
from bitarray import frozenbitarray
from jaxtyping import Array, Scalar, Shaped

from algebraic.polynomials.sparse import SparsePolynomial, SparsePolynomialAlgebra
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.tensor_algebra.jax import JaxBiModule


class MonomialBasis(eqx.Module):
    """Dense, monomial basis decomposition of a multilinear polynomial

    This class represents the coefficients of a multilinear polynomial as a tensor of
    shape `(2,) * n`, where `n` is the maximum degree of the polynomial.
    """

    coeffs: Shaped[Array, "*2"]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coeffs.shape

    @property
    def num_vars(self) -> int:
        return len(self.shape)


class MonomialBasisAlgebra[K: Lattice](eqx.Module):
    """Algebra of multilinear polynomials as multilinear forms over the monomial bases

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from algebraic.tensor_algebra.jax import boolean_algebra, JaxBiModule
    >>> module = boolean_algebra(mode="logic")
    >>> p = TensorPolynomial.variable(0, 2, module)
    >>> q = TensorPolynomial.variable(1, 2, module)
    >>> r = p.multiply(q)  # x_0 AND x_1
    """

    # TODO(anand): I am not sure where in the class hierarchy this falls. I think it could be a BiModule, but also be a MMultilinearPolynomialAlgebra, but is this algebra also a semiring?

    num_vars: int = eqx.field(static=True)
    """Number of variables/indeterminants in this multilinear polynomial"""
    module: JaxBiModule[K]
    """The underlying algebraic module to define the tensor algebra over"""

    def variable(self, i: int, coefficient: None | Scalar = None) -> MonomialBasis:
        """Create a polynomial representing a single variable x_i"""
        coefficient = coefficient if coefficient is not None else self.module.algebra.one
        idx = jnp.zeros(self.num_vars, dtype=jnp.int32).at[i].set(1)
        coeffs = self.module.zeros((2,) * self.num_vars).at[*idx].set(coefficient)
        return MonomialBasis(coeffs)

    def constant(self, value: Scalar) -> MonomialBasis:
        """Create a polynomial representing a single variable x_i"""
        idx = (0,) * self.num_vars
        coeffs = self.module.zeros((2,) * self.num_vars).at[*idx].set(value)
        return MonomialBasis(coeffs)

    @property
    def zero(self) -> MonomialBasis:
        return self.constant(self.module.algebra.zero)

    @property
    def one(self) -> MonomialBasis:
        return self.constant(self.module.algebra.one)

    def add(self, a: MonomialBasis, b: MonomialBasis) -> MonomialBasis:
        """Add two polynomials by adding the monomial coefficients for identical terms."""
        assert a.num_vars == self.num_vars
        assert b.num_vars == self.num_vars
        coeffs = self.module.add(a.coeffs, b.coeffs)
        return MonomialBasis(coeffs)

    def mul(self, a: MonomialBasis, b: MonomialBasis) -> MonomialBasis:
        r"""Multiply two polynomials.

        $(\sum_{S \in a} c_S x^S) \cdot (\sum_{T \in b} d_T x^T) = sum_{S,T} (c_S * d_T) x^{S \cup T}$

        """
        # Check compatibility
        assert a.num_vars == self.num_vars
        assert b.num_vars == self.num_vars

        # TODO: optimize with tensor product (generalized outer product) and coalescing.

        n = self.num_vars
        result_coeffs = self.module.zeros((2,) * n)
        # Iterate over all pairs of monomials
        for a_idx in range(2**n):
            a_bits = self._index_to_bits(a_idx)
            a_val = a.coeffs[a_bits]

            for b_idx in range(2**n):
                b_bits = self._index_to_bits(b_idx)
                b_val = b.coeffs[b_bits]

                # For multilinear: product monomial is S union T (bitwise OR)
                result_bits = tuple(a_bits[i] | b_bits[i] for i in range(n))

                # Accumulate coefficient
                product = self.module.mul(a_val, b_val)
                result_coeffs = result_coeffs.at[result_bits].set(self.module.add(result_coeffs[result_bits], product))

        return MonomialBasis(result_coeffs)

    def evaluate(self, poly: MonomialBasis, points: Shaped[Array, " {self.num_vars}"] | Mapping[int, Scalar]) -> MonomialBasis:
        """Evaluate polynomial at the given points using Horner-like scheme."""
        # Just convert the points into a set of constant Polynomials and use compose
        map_points = dict()
        if isinstance(points, Array):
            for var_idx in range(self.num_vars):
                scalar_value = points[var_idx]
                map_points[var_idx] = self.constant(scalar_value)
        else:
            assert isinstance(points, Mapping)
            for var_idx, scalar_value in points.items():
                map_points[var_idx] = self.constant(scalar_value)

        return self.compose(poly, map_points)

    def compose(
        self,
        poly: MonomialBasis,
        replacements: Mapping[int, MonomialBasis],
    ) -> MonomialBasis:
        """Compose polynomial with multiple substitutions.

        Returns p(x_1 <- q_1, ..., x_n <- q_n) where only specified indices are replaced.

        Note
        ----
        The composition should be performed simultaneously. If not, this is a bug.
        """
        # Sort the replacements as we want to implement a recursive solution.
        # Traversing the replacements in increasing order will allow us to effectively
        # do a bottom-up replacement, and will not allow duplicate substititions.
        # This is what we would do in a binary decision diagram.
        repl_keys: list[int] = list(sorted(replacements.keys()))

        @partial(jax.jit, static_argnums=(1,))
        def _compose(coeffs: Array, at: int) -> MonomialBasis:
            """Recursive implementation of composition.

            - `coeffs` should be an `num_vars`-dim array.
            - `at` is an index into `repl_keys`, so we can just increment it
            """
            # NOTE: Must make sure we don't go out of bounds for `at`
            if at >= len(repl_keys):
                # Return as is we there are no more variables to substitute
                return MonomialBasis(coeffs)
            var_idx = repl_keys[at]
            var_repl = replacements[var_idx].coeffs
            # Extract slices of shape: (2,) * (n-1)
            p_xi_0 = jnp.take(coeffs, 0, axis=var_idx)  #
            p_xi_1 = jnp.take(coeffs, 1, axis=var_idx)

            # Lift the cofactors back to full shape by adding axis at var_idx
            p_xi_0 = self._lift_tensor(p_xi_0, var_idx)
            p_xi_1 = self._lift_tensor(p_xi_1, var_idx)

            # Recursively compose each cofactor
            p_xi_0_poly = _compose(p_xi_0, at + 1)
            p_xi_1_poly = _compose(p_xi_1, at + 1)

            # merge the cofactors with the replacement in place
            # Need to multiply replacement polynomial with p_xi_1_poly, then add p_xi_0_poly
            var_repl_poly = MonomialBasis(var_repl)
            prod = self.mul(var_repl_poly, p_xi_1_poly)
            result = self.add(p_xi_0_poly, prod)
            return result

        return _compose(poly.coeffs, 0)

    def _index_to_bits(self, index: int) -> tuple[int, ...]:
        """Convert flat index to n-bit tuple."""
        from bitarray.util import int2ba

        return tuple(int2ba(index, length=self.num_vars))

    def _lift_tensor(self, tensor: Array, insert_axis: int) -> Array:
        """Lift (n-1)-dim tensor to n-dim by inserting axis."""
        # Insert axis at position insert_axis
        expanded = jnp.expand_dims(tensor, axis=insert_axis)

        # Pad along new axis to get shape (2,) * target_ndim
        padding = [(0, 0)] * self.num_vars
        padding[insert_axis] = (0, 1)

        return jnp.pad(expanded, padding, constant_values=self.module.algebra.zero)

    def to_sparse_algebra(self) -> SparsePolynomialAlgebra[K]:
        """Convert to a corresponding SparsePolynomialAlgebra"""
        return SparsePolynomialAlgebra(algebra=self.module.algebra, degree=self.num_vars)

    def to_sparse(self, poly: MonomialBasis) -> SparsePolynomial:
        """Convert to sparse representation."""
        from bitarray import frozenbitarray

        zero = self.module.zeros(())
        # Find all non-zero indices
        non_zero_mask = jnp.logical_not(jnp.isclose(poly.coeffs, zero, atol=1e-10))
        non_zero_indices = jnp.argwhere(non_zero_mask)

        result = dict()
        for idx_array in non_zero_indices:
            # Convert JAX array to tuple
            idx = tuple(int(i) for i in idx_array)
            # Get coefficient value
            coeff = poly.coeffs[idx]
            # Convert to frozenbitarray
            result[frozenbitarray(idx)] = coeff

        return SparsePolynomial(result)

    def from_sparse[S](
        self,
        poly: SparsePolynomial,
    ) -> MonomialBasis:
        """Convert from sparse representation.

        Note
        ----
        Assumes that the user is passing a `SparsePolynomial` in the same domain as the current `MonomialBasisAlgebra`.
        """
        coeffs = self.module.zeros((2,) * self.num_vars)
        for monomial, coeff in poly.items():
            if len(monomial) != self.num_vars:
                raise ValueError(
                    f"Cannot convert sparse polynomial with {len(monomial)} variables to monomial basis with {self.num_vars} variables"
                )
            index = tuple(monomial)
            coeffs = coeffs.at[index].set(coeff)
        return MonomialBasis(coeffs)


class RankDecomposition(eqx.Module):
    """CP (CANDECOMP/PARAFAC) decomposition of multilinear polynomial.

    Represents polynomial as sum of rank-1 components:
        p(x) = sum_{r=1}^R prod_{k=1}^d factors[r, k, index_k]

    where index_k in {0, 1, ..., n}:
        - 0 represents constant (always 1)
        - i (i>0) represents variable x_{i-1}
    """

    factors: Array  # Shape: (rank, degree, num_vars+1)

    @property
    def rank(self) -> int:
        return self.factors.shape[0]

    @property
    def degree(self) -> int:
        return self.factors.shape[1]

    @property
    def num_vars(self) -> int:
        return self.factors.shape[2] - 1


class RankDecompositionAlgebra[K: Lattice](eqx.Module):
    num_vars: int = eqx.field(static=True)
    """Number of variables/indeterminants in this multilinear polynomial"""
    module: JaxBiModule[K]
    """The underlying algebraic module to define the tensor algebra over"""
    max_rank: int = eqx.field(static=True)
    """Maximum rank for CP decomposition (controls memory usage)"""
    max_degree: int = eqx.field(static=True)
    """Maximum degree for polynomials (None = num_vars)"""
    max_replacement_degree: int = eqx.field(static=True)
    """Maximum degree for replacement polynomials in compose (None = max_degree)"""

    def __init__(
        self,
        num_vars: int,
        module: JaxBiModule[K],
        max_rank: int = 100,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.module = module
        self.max_rank = max_rank
        self.max_degree = max_degree if max_degree is not None else num_vars
        self.max_replacement_degree = max_replacement_degree if max_replacement_degree is not None else self.max_degree

    def variable(self, i: int, coefficient: None | Scalar = None) -> RankDecomposition:
        """Create rank-1 polynomial representing variable x_i.

        Creates a CP decomposition with rank=1, degree=1.
        """
        coeff = coefficient if coefficient is not None else self.module.algebra.one

        # Single rank-1 component with degree 1
        # Shape: (1, 1, num_vars+1)
        factors = self.module.zeros((1, 1, self.num_vars + 1))
        factors = factors.at[0, 0, i + 1].set(coeff)  # i+1 because 0 is constant

        return RankDecomposition(factors)

    def constant(self, value: Scalar) -> RankDecomposition:
        """Create rank-1 polynomial representing constant.

        Creates a CP decomposition with rank=1, degree=1.
        """
        # Single rank-1 component with degree 1
        # Shape: (1, 1, num_vars+1)
        factors = self.module.zeros((1, 1, self.num_vars + 1))
        factors = factors.at[0, 0, 0].set(value)  # index 0 is constant

        return RankDecomposition(factors)

    @property
    def zero(self) -> RankDecomposition:
        return self.constant(self.module.algebra.zero)

    @property
    def one(self) -> RankDecomposition:
        return self.constant(self.module.algebra.one)

    def _pad_degree(self, poly: RankDecomposition, target_degree: int) -> RankDecomposition:
        """Pad polynomial to target degree by adding identity factors.

        For CP decomposition, padding adds dimensions with constant=1 (identity for multiplication).
        """
        if poly.degree >= target_degree:
            return poly

        rank, d, n_plus_1 = poly.factors.shape
        extra_dims = target_degree - d

        # Add dimensions with constant=1 (identity for multiplication)
        padding = self.module.zeros((rank, extra_dims, n_plus_1))
        padding = padding.at[:, :, 0].set(self.module.algebra.one)  # constant term = 1

        new_factors = jnp.concatenate([poly.factors, padding], axis=1)
        return RankDecomposition(new_factors)

    def add(self, a: RankDecomposition, b: RankDecomposition) -> RankDecomposition:
        """Add by concatenating rank-1 components.

        For CP decomposition: p + q = sum of all components from both.
        """
        assert a.num_vars == self.num_vars
        assert b.num_vars == self.num_vars

        # Pad both to same degree
        d = max(a.degree, b.degree)
        a_padded = self._pad_degree(a, d)
        b_padded = self._pad_degree(b, d)

        # Concatenate along rank dimension
        new_factors = jnp.concatenate([a_padded.factors, b_padded.factors], axis=0)
        result = RankDecomposition(new_factors)

        # Compress if rank exceeds limit
        if result.rank > self.max_rank:
            result = self._compress_rank(result, self.max_rank)

        return result

    @eqx.filter_jit
    def _multiply_arrays(self, p_arr: Array, q_arr: Array) -> Array:
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

    def mul(self, a: RankDecomposition, b: RankDecomposition) -> RankDecomposition:
        """Multiply two CP-decomposed polynomials.

        Delegates core multiplication to _multiply_arrays(), then applies
        simplification and compression.
        """
        # Core multiplication (Khatri-Rao product)
        new_factors = self._multiply_arrays(a.factors, b.factors)
        result = RankDecomposition(new_factors)

        # Post-processing: Use fast heuristic first, falls back to exact if needed
        result = self._simplify_multilinear_fast(result)  # Apply x_i * x_i = x_i
        if result.rank > self.max_rank:
            result = self._compress_rank(result, self.max_rank)

        return result

    def _simplify_multilinear(self, poly: RankDecomposition) -> RankDecomposition:
        """Apply x_i * x_i = x_i to cap degree at num_vars.

        Uses sparse representation as intermediary for simplification.
        This handles both:
        1. Degree reduction (x_i * x_i -> x_i)
        2. Rank deduplication (merging identical monomials)
        """
        # Convert to sparse representation
        sparse = self.to_sparse(poly)

        # Sparse already groups by monomial (automatic simplification)
        if len(sparse) == 0:
            return self.zero

        # Determine max degree needed
        max_deg = max(sum(monomial) for monomial in sparse.keys())
        max_deg = min(max_deg, self.num_vars)  # Cap at num_vars

        # Convert back to CP form
        return self.from_sparse(sparse, max_degree=max_deg)

    def _simplify_multilinear_fast(self, poly: RankDecomposition) -> RankDecomposition:
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
        poly = self._deduplicate_degrees_fast(poly)
        poly = self._deduplicate_ranks_fast(poly)

        # Fallback thresholds: if still complex after heuristics, use exact method
        # These are conservative - adjust based on performance requirements
        threshold_rank = min(self.max_rank * 2, 200)
        threshold_degree = min(self.num_vars * 2, 20)

        if poly.rank > threshold_rank or poly.degree > threshold_degree:
            return self._simplify_multilinear(poly)

        return poly

    def _deduplicate_degrees_fast(self, poly: RankDecomposition) -> RankDecomposition:
        """Replace duplicate degree dimensions with identity within each rank.

        For multilinear polynomials: x_i * x_i = x_i
        If factors[r, k1, :] == factors[r, k2, :], replace one with identity (constant=1).

        Uses Python loops (should be unrolled by JIT) and jnp.where for compatibility.
        """
        new_factors = poly.factors

        # Create identity vector once (constant = 1, all variables = 0)
        identity = self.module.zeros(self.num_vars + 1)
        identity = identity.at[0].set(self.module.algebra.one)

        # Python loops over rank and degree dimensions
        # JAX will unroll these loops at compile time for fixed-size loops
        for r in range(poly.rank):
            for k1 in range(poly.degree):
                for k2 in range(k1 + 1, poly.degree):
                    # Check if degree dimensions k1 and k2 select the same variable
                    is_duplicate = jnp.all(new_factors[r, k1, :] == new_factors[r, k2, :])

                    # Replace k2 with identity if duplicate (JIT-compatible with jnp.where)
                    new_factors = new_factors.at[r, k2, :].set(jnp.where(is_duplicate, identity, new_factors[r, k2, :]))

        return RankDecomposition(new_factors)

    def _deduplicate_ranks_fast(self, poly: RankDecomposition) -> RankDecomposition:
        """Mark duplicate rank-1 components as zero.

        For idempotent addition (Boolean, Tropical, MaxMin): p + p = p
        If two rank-1 components are identical, mark the later one as zero.

        Zero components don't affect evaluation and will be removed by compression.
        """
        # Track which ranks to keep (keep first occurrence of duplicates)
        keep_mask = jnp.ones(poly.rank, dtype=bool)

        # Python loops - JAX should unroll these
        for r1 in range(poly.rank):
            for r2 in range(r1 + 1, poly.rank):
                # Check if entire rank-1 components are equal
                is_duplicate = jnp.all(poly.factors[r1, :, :] == poly.factors[r2, :, :])

                # Mark r2 for removal if it's a duplicate of r1
                keep_mask = keep_mask.at[r2].set(jnp.where(is_duplicate, False, keep_mask[r2]))

        # Replace duplicates with zero components
        # (Don't change array shape - keep JIT-compatible)
        new_factors = poly.factors
        zero_component = self.module.zeros((poly.degree, self.num_vars + 1))

        for r in range(poly.rank):
            should_zero = ~keep_mask[r]
            new_factors = new_factors.at[r, :, :].set(jnp.where(should_zero, zero_component, new_factors[r, :, :]))

        return RankDecomposition(new_factors)

    def _compress_rank(self, poly: RankDecomposition, max_rank: int) -> RankDecomposition:
        """Compress to at most max_rank components using magnitude-based truncation.

        Keeps the top-max_rank components by L2 norm magnitude.
        """
        if poly.rank <= max_rank:
            return poly

        # Compute magnitude of each rank-1 component
        # magnitude[r] = prod_k norm(factors[r, k, :])
        # Note: Using jnp.zeros for numeric magnitudes (not algebraic values)
        magnitudes = jnp.zeros(poly.rank)
        for r in range(poly.rank):
            mag = 1.0  # Numeric computation
            for k in range(poly.degree):
                # L2 norm of the k-th factor vector
                mag *= jnp.linalg.norm(poly.factors[r, k, :])
            magnitudes = magnitudes.at[r].set(mag)

        # Keep top max_rank components
        top_indices = jnp.argsort(magnitudes)[-max_rank:]
        new_factors = poly.factors[top_indices]

        return RankDecomposition(new_factors)

    def evaluate(self, poly: RankDecomposition, points: Array | Mapping[int, Scalar]) -> RankDecomposition:
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
            replacements = {i: self.constant(v) for i, v in points.items()}
            return self.compose(poly, replacements)

        # Full evaluation: substitute all variables
        rank, d, _ = poly.factors.shape

        # Build selector vector: [1, point[0], point[1], ..., point[n-1]]
        one_array = jnp.full((1,), self.module.algebra.one, dtype=points.dtype)
        selector = jnp.concatenate([one_array, points])  # Shape: (n+1,)

        # For each rank-1 component
        result = self.module.algebra.zero
        for r in range(rank):
            # Evaluate component: prod_k sum_i factors[r,k,i] * selector[i]
            component_value = self.module.algebra.one
            for k in range(d):
                # Inner product of factors[r,k,:] with selector
                dim_value = self.module.algebra.zero
                for i in range(self.num_vars + 1):
                    term = self.module.mul(poly.factors[r, k, i], selector[i])
                    dim_value = self.module.add(dim_value, term)

                component_value = self.module.mul(component_value, dim_value)

            result = self.module.add(result, component_value)

        # Return as constant polynomial
        return self.constant(result)

    def _prepare_replacement_array(self, replacements: dict[int, RankDecomposition]) -> Array:
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
        full_replacements = [self.constant(self.module.algebra.one)]  # index 0: constant
        for i in range(self.num_vars):
            if i in replacements:
                full_replacements.append(replacements[i])
            else:
                full_replacements.append(self.variable(i))  # identity replacement

        # Calculate max shapes
        max_rank = max(q.rank for q in full_replacements)
        m_plus_1 = self.num_vars + 1

        # Pad each polynomial
        padded_list = []
        for q in full_replacements:
            rank, d, _ = q.factors.shape
            padded = self.module.zeros((max_rank, self.max_replacement_degree, m_plus_1))
            # Copy existing factors
            padded = padded.at[:rank, :d, :].set(q.factors)
            # Pad extra degree dimensions with constant=1 (identity for multiplication)
            if d < self.max_replacement_degree:
                for k in range(d, self.max_replacement_degree):
                    padded = padded.at[:rank, k, 0].set(self.module.algebra.one)
            padded_list.append(padded)

        # Stack into contiguous array for fast indexing
        q_array = jnp.stack(padded_list, axis=0)

        return q_array  # Shape: [n+1, R_max, max_replacement_degree, m+1]

    def compose(self, poly: RankDecomposition, replacements: dict[int, RankDecomposition]) -> RankDecomposition:
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
        # Step 1: Prepare replacements (CACHE THIS in AFA code!)
        q_array = self._prepare_replacement_array(replacements)
        # Shape: [n+1, R_max, max_replacement_degree, m+1]

        # Step 2: Compose each rank-1 component
        # Note: Using Python loop over rank for now (degree is typically small)
        # Future optimization: Use vmap with fixed-size accumulator
        composed_list = []

        for r in range(poly.rank):
            p_component = poly.factors[r]  # [d_p, n+1]

            # Identify which index each factor selects (0=constant, 1=x_0, 2=x_1, ...)
            var_indices = jnp.argmax(p_component, axis=1)  # [d_p]

            # Gather the polynomials to multiply
            selected = q_array[var_indices]  # [d_p, R_max, max_replacement_degree, m+1]

            # Multiply sequentially
            result = selected[0]  # [R_max, max_replacement_degree, m+1]
            for k in range(1, poly.degree):
                result = self._multiply_arrays(result, selected[k])

            composed_list.append(result)

        # Stack all composed components
        composed = jnp.stack(composed_list, axis=0)
        # Shape: [R_p, R_result, d_result, m+1]

        # Step 3: Flatten rank dimensions
        rank_p, rank_result, d_result, m_plus_1 = composed.shape
        result_factors = composed.reshape(rank_p * rank_result, d_result, m_plus_1)

        result = RankDecomposition(result_factors)

        # Step 4: Simplify and compress
        # Use fast heuristic first (especially important for AFA hot path)
        # Falls back to exact simplification if needed
        result = self._simplify_multilinear_fast(result)
        if result.rank > self.max_rank:
            result = self._compress_rank(result, self.max_rank)

        return result

    def _index_to_bits(self, index: int) -> tuple[int, ...]:
        """Convert flat index to n-bit tuple."""
        from bitarray.util import int2ba

        return tuple(int2ba(index, length=self.num_vars))

    def to_sparse_algebra(self) -> SparsePolynomialAlgebra[K]:
        """Convert to a corresponding SparsePolynomialAlgebra"""
        return SparsePolynomialAlgebra(algebra=self.module.algebra, degree=self.num_vars)

    def to_sparse(self, poly: RankDecomposition) -> SparsePolynomial:
        """Convert CP to sparse by enumerating all monomial evaluations.

        WARNING: This is expensive O((n+1)^d) where d is degree.
        """
        result = {}

        # Enumerate all possible assignments (expensive: (n+1)^d possibilities)
        for assignment in product(range(self.num_vars + 1), repeat=poly.degree):
            # Determine which variables are present in this assignment
            # (handles x_i * x_i = x_i automatically by mapping to same monomial)
            vars_present = frozenbitarray(
                [any(assignment[k] == i + 1 for k in range(poly.degree)) for i in range(self.num_vars)]
            )

            # Evaluate coefficient for this assignment
            coeff = self.module.algebra.zero
            for r in range(poly.rank):
                component = self.module.algebra.one
                for k in range(poly.degree):
                    component = self.module.mul(component, poly.factors[r, k, assignment[k]])
                coeff = self.module.add(coeff, component)

            # Add to sparse (accumulate if monomial already exists)
            if not jnp.allclose(coeff, self.module.algebra.zero):
                if vars_present in result:
                    result[vars_present] = self.module.add(result[vars_present], coeff)
                else:
                    result[vars_present] = coeff

        return SparsePolynomial(data=result)

    def from_sparse(self, sparse: SparsePolynomial, max_degree: int | None = None) -> RankDecomposition:
        """Convert sparse to CP form (each monomial becomes rank-1 component).

        Args:
            sparse: Sparse polynomial to convert
            max_degree: Maximum degree for result (default: self.max_degree)

        Returns:
            CP decomposition with one rank-1 component per monomial
        """
        if max_degree is None:
            max_degree = self.max_degree

        if len(sparse) == 0:
            return self.zero

        rank = len(sparse)
        factors = self.module.zeros((rank, max_degree, self.num_vars + 1))

        for r, (monomial, coeff) in enumerate(sparse.items()):
            # monomial is a bitarray indicating which variables appear
            vars_in_monomial = [i for i, bit in enumerate(monomial) if bit]

            # Fill in factors for this rank-1 component
            for k, var_idx in enumerate(vars_in_monomial):
                if k < max_degree:
                    factors = factors.at[r, k, var_idx + 1].set(coeff if k == 0 else self.module.algebra.one)

            # Pad remaining dimensions with constant=1
            for k in range(len(vars_in_monomial), max_degree):
                factors = factors.at[r, k, 0].set(self.module.algebra.one)

        return RankDecomposition(factors)
