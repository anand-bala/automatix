"""Backend-agnostic CP (CANDECOMP/PARAFAC) decomposition of multilinear polynomials."""

# NOTE: Do NOT add `from __future__ import annotations` to this module.
# The `AbstractVar` annotations from `algebraic._better_abc` are processed at class
# definition time and must not be stringified (PEP 563 lazy evaluation would break them).

# mypy: disable-error-code="no-any-return,no-untyped-call"

import copy
import typing
from collections.abc import Mapping
from itertools import product

import array_api_compat
from bitarray import frozenbitarray
from typing_extensions import Self

import algebraic.ops as alge
from algebraic._better_abc import AbstractClassVar, AbstractVar, BetterABCMeta
from algebraic.array import AlgebraicArray
from algebraic.polynomials.dok.base import PolyDict, _make_poly_dict
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Array, Backend, Number, Scalar, is_scalar


class RankDecomposition(metaclass=BetterABCMeta):
    """CP (CANDECOMP/PARAFAC) decomposition of multilinear polynomial.

    Represents polynomial as sum of rank-1 components:
        ``p(x) = sum_{r=1}^R prod_{k=1}^d factors[r, k, index_k]``

    where ``index_k`` in ``{0, 1, ..., n}``:
        - 0 represents constant (always 1)
        - ``i`` (``i>0``) represents variable ``x_{i-1}``
    """

    factors: AbstractVar[AlgebraicArray]
    algebra: AbstractVar[Lattice]
    max_rank: AbstractVar[int]
    """Maximum rank for CP decomposition (controls memory usage)"""
    max_degree: AbstractVar[int]
    """Maximum degree for polynomials (None = num_vars)"""
    max_replacement_degree: AbstractVar[int]
    """Maximum degree for replacement polynomials in compose (None = max_degree)"""
    backend: AbstractClassVar[str | Backend]
    """The specific backend from the derived class"""

    @classmethod
    def _get_backend(cls, backend: str | Backend | None) -> str | Backend:
        """Resolve backend: use explicit arg, class default, or fall back to NumPy."""
        if backend is not None:
            return backend
        try:
            return cls.backend
        except AttributeError:
            return Backend.NUMPY

    @property
    def rank(self) -> int:
        return self.factors.shape[0]

    @property
    def degree(self) -> int:
        return self.factors.shape[1]

    @property
    def num_vars(self) -> int:
        return self.factors.shape[2] - 1

    def _replace_factors(self, factors: AlgebraicArray) -> Self:
        """Return a new instance with the given factors, preserving other attrs."""
        clone = copy.copy(self)
        object.__setattr__(clone, "factors", factors)
        return clone

    # -- Factory methods -------------------------------------------------------

    @classmethod
    def variable(
        cls,
        i: int,
        num_vars: int,
        algebra: Lattice,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> Self:
        r"""Create rank-1 polynomial representing variable :math:`x_i`.

        Parameters
        ----------
        i : int
            Variable index (0-based).
        num_vars : int
            Total number of variables.
        algebra : BoundedDistributiveLattice
            Lattice algebra governing operations.
        max_rank : int or None, optional
            Maximum rank for the decomposition.
        max_degree : int or None, optional
            Maximum degree for the polynomial.
        max_replacement_degree : int or None, optional
            Maximum degree for replacement polynomials in :meth:`compose`.
        backend : str or Backend or None, optional
            Backend to use.

        Returns
        -------
        RankDecomposition
            A rank-1 polynomial with degree 1.

        Examples
        --------
        >>> from algebraic.polynomials import RankDecomposition
        >>> from algebraic.semirings import boolean_algebra
        >>> ba = boolean_algebra(mode="logic")
        >>> x0 = RankDecomposition.variable(0, num_vars=2, algebra=ba, backend="numpy")
        >>> x0.rank
        1
        """
        backend = cls._get_backend(backend)
        factors = alge.zeros((1, 1, num_vars + 1), semiring=algebra, backend=backend)
        xp = _get_xp(factors)
        factors = factors._wrap(_set_at_index(factors.data, (0, 0, i + 1), algebra.one, xp))

        return typing.cast(
            Self,
            _make_rank_decomposition(
                factors,
                algebra,
                max_rank,
                max_degree,
                max_replacement_degree,
                num_vars,
                backend,
            ),
        )

    @classmethod
    def constant(
        cls,
        value: Scalar,
        num_vars: int,
        algebra: Lattice,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> Self:
        """Create a rank-1 constant polynomial.

        Parameters
        ----------
        value : Scalar
            Constant value.
        num_vars : int
            Total number of variables.
        algebra : BoundedDistributiveLattice
            Lattice algebra governing operations.
        max_rank : int or None, optional
            Maximum rank for the decomposition.
        max_degree : int or None, optional
            Maximum degree for the polynomial.
        max_replacement_degree : int or None, optional
            Maximum degree for replacement polynomials in :meth:`compose`.
        backend : str or Backend or None, optional
            Backend to use.

        Returns
        -------
        RankDecomposition
            A rank-1 constant polynomial.

        Examples
        --------
        >>> from algebraic.polynomials import RankDecomposition
        >>> from algebraic.semirings import boolean_algebra
        >>> ba = boolean_algebra(mode="logic")
        >>> c = RankDecomposition.constant(True, num_vars=2, algebra=ba, backend="numpy")
        >>> c.rank
        1
        """
        backend = cls._get_backend(backend)
        factors = alge.zeros((1, 1, num_vars + 1), semiring=algebra, backend=backend)
        xp = _get_xp(factors)
        factors = factors._wrap(_set_at_index(factors.data, (0, 0, 0), value, xp))

        return _make_rank_decomposition(
            factors,
            algebra,
            max_rank,
            max_degree,
            max_replacement_degree,
            num_vars,
            backend,
        )

    @classmethod
    def zero(
        cls,
        num_vars: int,
        algebra: Lattice,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> Self:
        return cls.constant(algebra.zero, num_vars, algebra, max_rank, max_degree, max_replacement_degree, backend=backend)

    @classmethod
    def one(
        cls,
        num_vars: int,
        algebra: Lattice,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> Self:
        return cls.constant(algebra.one, num_vars, algebra, max_rank, max_degree, max_replacement_degree, backend=backend)

    def _var_at(self, idx: int) -> Self:
        cls = type(self)
        return cls.variable(
            idx,
            self.num_vars,
            self.algebra,
            self.max_rank,
            self.max_degree,
            self.max_replacement_degree,
            backend=self.backend,
        )

    def _make_const(self, val: Scalar) -> Self:
        cls = type(self)
        return cls.constant(
            val,
            self.num_vars,
            self.algebra,
            self.max_rank,
            self.max_degree,
            self.max_replacement_degree,
            backend=self.backend,
        )

    # -- Arithmetic ------------------------------------------------------------

    def _pad_degree(self, target_degree: int) -> Self:
        """Pad polynomial to target degree by adding identity factors.

        For CP decomposition, padding adds dimensions with constant=1 (identity
        for multiplication).
        """
        if self.degree >= target_degree:
            return self

        rank, d, n_plus_1 = self.factors.shape
        extra_dims = target_degree - d

        padding = alge.zeros((rank, extra_dims, n_plus_1), semiring=self.algebra, backend=self.backend)
        xp = _get_xp(padding)
        # Set constant term = 1 for all padded degree dims
        for r in range(rank):
            for k in range(extra_dims):
                padding = padding._wrap(_set_at_index(padding.data, (r, k, 0), self.algebra.one, xp))

        new_factors = alge.concat([self.factors, padding], axis=1)
        return self._replace_factors(new_factors)

    def __add__(self, other: "RankDecomposition | Scalar") -> Self:
        """Add by concatenating rank-1 components.

        For CP decomposition: ``p + q`` = sum of all components from both.
        """
        if is_scalar(other):
            other = self._make_const(other)
        assert isinstance(other, RankDecomposition)
        assert other.num_vars == self.num_vars

        d = max(self.degree, other.degree)
        a_padded = self._pad_degree(d)
        b_padded = other._pad_degree(d)

        new_factors = alge.concat([a_padded.factors, b_padded.factors], axis=0)
        result: Self = self._replace_factors(new_factors)
        result = result._compress_rank()
        return result

    def _multiply_arrays(self, p_arr: AlgebraicArray, q_arr: AlgebraicArray) -> AlgebraicArray:
        """Core multiplication logic on raw arrays (no simplification/compression).

        Parameters
        ----------
        p_arr : AlgebraicArray
            Shape ``[R_p, d_p, n+1]``.
        q_arr : AlgebraicArray
            Shape ``[R_q, d_q, n+1]``.

        Returns
        -------
        AlgebraicArray
            Shape ``[R_p * R_q, d_p + d_q, n+1]``.
        """
        rank_p, degree_p, n_plus_1 = p_arr.shape
        rank_q, degree_q, _ = q_arr.shape

        p_expanded = alge.broadcast_to(
            p_arr[:, None, :, :],
            (rank_p, rank_q, degree_p, n_plus_1),
        )
        q_expanded = alge.broadcast_to(
            q_arr[None, :, :, :],
            (rank_p, rank_q, degree_q, n_plus_1),
        )

        result = alge.concat([p_expanded, q_expanded], axis=2)
        result = alge.reshape(result, (rank_p * rank_q, degree_p + degree_q, n_plus_1))
        return result

    def __mul__(self, other: Self) -> Self:
        """Multiply two CP-decomposed polynomials.

        Delegates core multiplication to ``_multiply_arrays()``, then applies
        simplification and compression.
        """
        new_factors = self._multiply_arrays(self.factors, other.factors)
        result = self._replace_factors(new_factors)

        result = result._remove_zero_components()
        result = result._simplify_multilinear_fast()
        result = result._compress_rank()
        return result

    # -- Simplification --------------------------------------------------------

    def _simplify_multilinear(self) -> Self:
        """Apply ``x_i * x_i = x_i`` to cap degree at num_vars.

        Uses sparse representation as intermediary for simplification.
        """
        sparse = self.to_sparse()

        if len(sparse) == 0:
            return self._make_const(self.algebra.zero)  # ty:ignore[invalid-return-type]

        max_deg = max(sum(monomial) for monomial in sparse.keys())
        max_deg = max(1, min(max_deg, self.num_vars))

        return self.from_sparse(sparse, max_degree=max_deg)  # ty:ignore[invalid-return-type]

    def _simplify_multilinear_fast(self) -> Self:
        """Fast heuristic simplification using deduplication.

        This is an incomplete heuristic that catches common patterns:
        1. Duplicate variables within rank-1 components (``x_i * x_i -> x_i``)
        2. Duplicate rank-1 components (``p + p -> p`` for idempotent addition)

        Falls back to exact sparse-based simplification if result is still complex.

        Note
        ----
        This heuristic is faster than full sparse conversion ``O((n+1)^d)`` but
        incomplete.  It won't detect absorption laws like
        ``x_0 + x_0*x_1 = x_0`` (Boolean algebra).
        """
        poly = self._deduplicate_degrees_fast()
        poly = poly._deduplicate_ranks_fast()

        threshold_rank = min(self.max_rank * 2, 200)
        threshold_degree = min(self.num_vars * 2, 20)

        if poly.rank > threshold_rank or poly.degree > threshold_degree:
            return poly._simplify_multilinear()

        return poly

    def _deduplicate_degrees_fast(self) -> Self:
        """Replace duplicate degree dimensions with identity within each rank.

        For multilinear polynomials: ``x_i * x_i = x_i``.
        If ``factors[r, k1, :] == factors[r, k2, :]``, replace one with identity
        (constant=1).
        """
        xp = _get_xp(self.factors)

        identity_data = xp.zeros(self.num_vars + 1, dtype=self.factors.data.dtype)
        identity_data = _set_at_index(identity_data, (0,), self.algebra.one, xp)

        new_data = xp.asarray(self.factors.data)
        if hasattr(new_data, "copy"):
            new_data = new_data.copy()
        elif hasattr(new_data, "clone"):
            new_data = new_data.clone()

        for r in range(self.rank):
            for k1 in range(self.degree):
                for k2 in range(k1 + 1, self.degree):
                    is_duplicate = bool(xp.all(xp.equal(new_data[r, k1, :], new_data[r, k2, :])))
                    if is_duplicate:
                        new_data = _set_at_index(new_data, (r, k2), identity_data, xp)

        new_factors = self.factors._wrap(new_data)
        return self._replace_factors(new_factors)

    def _deduplicate_ranks_fast(self) -> Self:
        """Mark duplicate rank-1 components as zero.

        For idempotent addition (Boolean, Tropical, MaxMin): ``p + p = p``.
        If two rank-1 components are identical, mark the later one as zero.
        """
        xp = _get_xp(self.factors)

        keep_mask = [True] * self.rank

        for r1 in range(self.rank):
            if not keep_mask[r1]:
                continue
            for r2 in range(r1 + 1, self.rank):
                if not keep_mask[r2]:
                    continue
                is_duplicate = bool(
                    xp.all(
                        xp.equal(
                            self.factors.data[r1, :, :],
                            self.factors.data[r2, :, :],
                        )
                    )
                )
                if is_duplicate:
                    keep_mask[r2] = False

        new_data = xp.asarray(self.factors.data)
        if hasattr(new_data, "copy"):
            new_data = new_data.copy()
        elif hasattr(new_data, "clone"):
            new_data = new_data.clone()

        zero_component_data = xp.zeros(
            (self.degree, self.num_vars + 1),
            dtype=self.factors.data.dtype,
        )
        for r in range(self.rank):
            if not keep_mask[r]:
                new_data = _set_at_index(new_data, (r,), zero_component_data, xp)

        new_factors = self.factors._wrap(new_data)
        return self._replace_factors(new_factors)

    def _remove_zero_components(self) -> Self:
        """Remove rank-1 components that have any all-zero factors.

        A rank-1 component with an all-zero factor evaluates to zero everywhere,
        so it can be safely removed from the sum.
        """
        xp = _get_xp(self.factors)

        is_zero_coeff = alge.isclose(self.factors, self.algebra.zero)
        is_zero_factor = xp.all(is_zero_coeff, axis=2)
        has_zero_factor = xp.any(is_zero_factor, axis=1)
        keep_mask = ~has_zero_factor  # type: ignore[operator]

        if bool(xp.all(xp.asarray(~keep_mask))):
            return self._make_const(self.algebra.zero)

        keep_indices = [i for i in range(self.rank) if bool(keep_mask[i])]
        if len(keep_indices) == 0:
            return self._make_const(self.algebra.zero)

        kept_slices = [self.factors[i : i + 1] for i in keep_indices]
        new_factors = alge.concat(kept_slices, axis=0)
        return self._replace_factors(new_factors)

    def _compress_rank(self) -> Self:
        """Compress to at most max_rank components using magnitude-based truncation.

        Keeps the top-max_rank components by L2 norm magnitude.
        """
        if self.rank <= self.max_rank:
            return self

        xp = _get_xp(self.factors)
        magnitudes: list[float] = []
        for r in range(self.rank):
            mag = 1.0
            for k in range(self.degree):
                factor_data = self.factors.data[r, k, :]
                norm_val = float(_norm(factor_data, xp))
                mag *= norm_val
            magnitudes.append(mag)

        sorted_indices = sorted(range(self.rank), key=lambda i: magnitudes[i])
        top_indices = sorted_indices[-self.max_rank :]

        kept_slices = [self.factors[i : i + 1] for i in top_indices]
        new_factors = alge.concat(kept_slices, axis=0)
        return self._replace_factors(new_factors)

    # -- Evaluation / composition ----------------------------------------------

    def evaluate(self, points: "Array | Mapping[int, Scalar]") -> Self:
        """Evaluate polynomial at given point.

        Parameters
        ----------
        points : Array or Mapping[int, Scalar]
            Either an array of shape ``(num_vars,)`` for full evaluation,
            or a mapping from variable indices to scalar values for partial
            evaluation.

        Returns
        -------
        RankDecomposition
            Constant polynomial after evaluation.
        """
        if isinstance(points, Mapping):
            if set(points.keys()) >= set(range(self.num_vars)):
                point_list = [points[i] for i in range(self.num_vars)]
                xp = _get_xp(self.factors)
                return self.evaluate(xp.asarray(point_list))
            replacements: dict[int, Self] = {i: self._make_const(v) for i, v in points.items()}  # ty:ignore[invalid-assignment, invalid-argument-type]
            return self.compose(replacements)

        rank, d, _ = self.factors.shape

        one_array = alge.ones((1,), semiring=self.algebra, backend=self.backend)
        points_array = alge.array(points, semiring=self.algebra, backend=self.backend)
        selector = alge.concat([one_array, points_array])

        result = alge.zeros((), semiring=self.algebra, backend=self.backend)
        for r in range(rank):
            component_value = alge.ones((), semiring=self.algebra, backend=self.backend)
            for k in range(d):
                dim_value = alge.zeros((), semiring=self.algebra, backend=self.backend)
                for i in range(self.num_vars + 1):
                    term = self.factors[r, k, i] * selector[i]
                    dim_value = dim_value + term
                component_value = component_value * dim_value
            result = result + component_value

        return self._make_const(result.data)  # ty:ignore[invalid-return-type]

    def _prepare_replacement_array(self, replacements: dict[int, Self]) -> AlgebraicArray:
        """Prepare padded array of replacement polynomials.

        Returns:
            Array of shape [n+1, R_max, max_replacement_degree, m+1]
            Index 0: constant (identity: always 1)
            Index i+1: variable x_i (or its replacement)
        """
        full_replacements: list[Self] = [self._make_const(self.algebra.one)] + [
            replacements.get(i, self._var_at(i)) for i in range(self.num_vars)
        ]

        max_rep_rank = max(q.rank for q in full_replacements)
        m_plus_1 = self.num_vars + 1

        xp = _get_xp(self.factors)
        padded_list = []
        for q in full_replacements:
            q_rank, q_d, _ = q.factors.shape
            padded = alge.zeros(
                (max_rep_rank, self.max_replacement_degree, m_plus_1),
                semiring=self.algebra,
                backend=self.backend,
            )
            padded_data = xp.asarray(padded.data)
            if hasattr(padded_data, "copy"):
                padded_data = padded_data.copy()
            elif hasattr(padded_data, "clone"):
                padded_data = padded_data.clone()

            # Copy existing factors
            for r in range(q_rank):
                for k in range(q_d):
                    for v in range(m_plus_1):
                        padded_data = _set_at_index(
                            padded_data,
                            (r, k, v),
                            q.factors.data[r, k, v],
                            xp,
                        )

            # Pad extra degree dimensions with constant=1
            if q_d < self.max_replacement_degree:
                for r in range(q_rank):
                    for k in range(q_d, self.max_replacement_degree):
                        padded_data = _set_at_index(
                            padded_data,
                            (r, k, 0),
                            self.algebra.one,
                            xp,
                        )

            padded_list.append(padded._wrap(padded_data))

        q_array = alge.stack(padded_list, axis=0)
        return q_array

    def compose(self, replacements: dict[int, Self]) -> Self:
        """Compose polynomial with replacement polynomials.

        Parameters
        ----------
        replacements : dict[int, RankDecomposition]
            Dict mapping variable indices to replacement polynomials.

        Returns
        -------
        RankDecomposition
            The composed polynomial.
        """
        result_arr: AlgebraicArray
        q_array = self._prepare_replacement_array(replacements)

        xp = _get_xp(self.factors)
        composed_list = []

        for r in range(self.rank):
            p_component = self.factors[r]

            # Get the boolean array of all places where the array is not semiring 0
            # multiply by 1 to convert from bool to int if needed
            is_nonzero = 1 * ~alge.isclose(p_component.data, self.algebra.zero)

            var_indices: list[int] = []
            is_zero_factor_list: list[bool] = []
            for k in range(self.degree):
                row = is_nonzero[k]
                if bool(xp.any(row)):
                    idx = int(_argmax(row, xp))
                    var_indices.append(idx)
                    is_zero_factor_list.append(False)
                else:
                    var_indices.append(0)
                    is_zero_factor_list.append(True)

            has_any_zero = any(is_zero_factor_list)

            selected = [q_array[var_indices[k]] for k in range(self.degree)]

            result_arr = selected[0]

            if has_any_zero:
                temp_result = result_arr
                for k in range(1, self.degree):
                    temp_result = self._multiply_arrays(temp_result, selected[k])
                result_arr = alge.zeros(temp_result.shape, semiring=self.algebra, backend=self.backend)
            else:
                for k in range(1, self.degree):
                    result_arr = self._multiply_arrays(result_arr, selected[k])

            composed_list.append(result_arr)

        composed = alge.stack(composed_list, axis=0)
        assert isinstance(composed, AlgebraicArray)

        rank_p, rank_result, d_result, m_plus_1 = composed.shape
        result_factors = alge.reshape(composed, (rank_p * rank_result, d_result, m_plus_1))
        assert isinstance(result_factors, AlgebraicArray)

        result_poly = self._replace_factors(result_factors)

        result_poly = result_poly._remove_zero_components()
        result_poly = result_poly._simplify_multilinear_fast()
        result_poly = result_poly._compress_rank()
        return result_poly

    # -- Conversion ------------------------------------------------------------

    def _index_to_bits(self, index: int) -> tuple[int, ...]:
        """Convert flat index to n-bit tuple."""
        from bitarray.util import int2ba

        return tuple(int2ba(index, length=self.num_vars))

    def to_sparse(self) -> PolyDict:
        """Convert CP to sparse by enumerating all monomial evaluations.

        WARNING: This is expensive ``O((n+1)^d)`` where d is degree.
        """
        xp = _get_xp(self.factors)
        backend = Backend(self.backend)

        result: dict[frozenbitarray, AlgebraicArray] = {}

        for assignment in product(range(self.num_vars + 1), repeat=self.degree):
            vars_present = frozenbitarray(
                [any(assignment[k] == i + 1 for k in range(self.degree)) for i in range(self.num_vars)]
            )

            coeff: AlgebraicArray | Number = self.algebra.zero
            for r in range(self.rank):
                component: AlgebraicArray | Number = self.algebra.one
                for k in range(self.degree):
                    factor_value: AlgebraicArray = self.factors[r, k, assignment[k]]
                    component = factor_value * component
                coeff = component + coeff

            if not bool(alge.allclose(coeff, self.algebra.zero)):
                coeff_arr = (
                    coeff if isinstance(coeff, AlgebraicArray) else alge.array(coeff, semiring=self.algebra, backend=backend)
                )
                if vars_present in result:
                    result[vars_present] = result[vars_present] + coeff_arr
                else:
                    result[vars_present] = coeff_arr

        return _make_poly_dict(self.algebra, self.num_vars, result, backend)

    @classmethod
    def from_sparse(
        cls,
        sparse: PolyDict,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> Self:
        """Convert sparse to CP form (each monomial becomes rank-1 component).

        Parameters
        ----------
        sparse : PolyDict
            Sparse polynomial to convert.
        max_rank : int or None, optional
            Maximum rank for the result.
        max_degree : int or None, optional
            Maximum degree for the result (default: ``num_vars``).
        max_replacement_degree : int or None, optional
            Maximum degree for replacement polynomials in :meth:`compose`.
        backend : str or Backend or None, optional
            Backend to use.

        Returns
        -------
        RankDecomposition
            CP decomposition with one rank-1 component per monomial.
        """
        if max_degree is None:
            max_degree = sparse.num_vars
        algebra = sparse.algebra
        num_vars = sparse.num_vars
        backend = cls._get_backend(backend)

        zero_poly = cls.constant(
            algebra.zero,
            sparse.num_vars,
            algebra,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
        )
        max_rank = zero_poly.max_rank
        max_degree = zero_poly.max_degree
        max_replacement_degree = zero_poly.max_replacement_degree

        if len(sparse) == 0:
            return zero_poly

        rank = len(sparse)
        factors = alge.zeros((rank, max_degree, num_vars + 1), semiring=algebra, backend=backend)
        xp = _get_xp(factors)

        for r, (monomial, coeff) in enumerate(sparse.items()):
            coeff_raw = coeff.data if isinstance(coeff, AlgebraicArray) else coeff
            vars_in_monomial = [i for i, bit in enumerate(monomial) if bit]

            if len(vars_in_monomial) == 0:
                factors = factors._wrap(_set_at_index(factors.data, (r, 0, 0), coeff_raw, xp))
                for k in range(1, max_degree):
                    factors = factors._wrap(_set_at_index(factors.data, (r, k, 0), algebra.one, xp))
            else:
                for k, var_idx in enumerate(vars_in_monomial):
                    if k < max_degree:
                        val = coeff_raw if k == 0 else algebra.one
                        factors = factors._wrap(_set_at_index(factors.data, (r, k, var_idx + 1), val, xp))
                for k in range(len(vars_in_monomial), max_degree):
                    factors = factors._wrap(_set_at_index(factors.data, (r, k, 0), algebra.one, xp))

        return _make_rank_decomposition(
            factors,
            algebra,
            max_rank,
            max_degree,
            max_replacement_degree,
            num_vars,
            backend,
        )


# -- Module-level helpers ------------------------------------------------------


def _get_xp(arr: AlgebraicArray) -> typing.Any:  # noqa: ANN401
    """Get the array namespace for an AlgebraicArray."""
    return array_api_compat.array_namespace(arr.data)


def _set_at_index(
    data: typing.Any,  # noqa: ANN401
    idx: tuple[int, ...],
    value: typing.Any,  # noqa: ANN401
    xp: typing.Any,  # noqa: ANN401
) -> typing.Any:  # noqa: ANN401
    """Set a value at an index, handling JAX immutability and mutable backends."""
    from algebraic.array._index_update import _set_at_index as _set_at_index_impl

    if isinstance(value, AlgebraicArray):
        value = value.data
    return _set_at_index_impl(data, idx, value)


def _argmax(a: typing.Any, xp: typing.Any) -> typing.Any:  # noqa: ANN401
    """Backend-agnostic argmax (returns scalar index)."""
    import numpy as np

    if array_api_compat.is_jax_array(a):
        import jax.numpy as jnp

        return jnp.argmax(a)
    if array_api_compat.is_torch_array(a):
        import torch

        return torch.argmax(torch.as_tensor(a))
    return np.argmax(np.asarray(a))


def _norm(a: typing.Any, xp: typing.Any) -> float:  # noqa: ANN401
    """Backend-agnostic L2 norm."""
    import numpy as np

    if array_api_compat.is_jax_array(a):
        import jax.numpy as jnp

        return float(jnp.linalg.norm(a))
    if array_api_compat.is_torch_array(a):
        import torch

        return float(torch.linalg.norm(torch.as_tensor(a).float()))
    return float(np.linalg.norm(np.asarray(a)))


def _make_rank_decomposition(
    factors: AlgebraicArray,
    algebra: Lattice,
    max_rank: int | None,
    max_degree: int | None,
    max_replacement_degree: int | None,
    num_vars: int,
    backend: str | Backend,
) -> RankDecomposition:
    """Dispatch to the correct backend subclass."""
    backend = Backend(backend)
    _max_rank = max_rank if max_rank is not None else 100
    _max_degree = max_degree if max_degree is not None else num_vars
    _max_replacement_degree = max_replacement_degree if max_replacement_degree is not None else _max_degree

    if backend == Backend.JAX:
        from algebraic.polynomials.rank_decomp._jax import JaxRankDecomposition

        return JaxRankDecomposition(factors, algebra, _max_rank, _max_degree, _max_replacement_degree)
    if backend == Backend.TORCH:
        from algebraic.polynomials.rank_decomp._torch import TorchRankDecomposition

        return TorchRankDecomposition(factors, algebra, _max_rank, _max_degree, _max_replacement_degree)
    if backend == Backend.NUMPY:
        from algebraic.polynomials.rank_decomp._numpy import NumpyRankDecomposition

        return NumpyRankDecomposition(factors, algebra, _max_rank, _max_degree, _max_replacement_degree)
    raise ValueError(f"Unsupported backend: {backend!r}")
