"""Backend-agnostic CP (CANDECOMP/PARAFAC) decomposition of multilinear polynomials."""

import typing
from collections.abc import Sequence
from dataclasses import dataclass, field

from bitarray import frozenbitarray
from typing_extensions import Self

import algebraic.ops as algebraic
from algebraic.array import AlgebraicArray
from algebraic.polynomials.dok import PolyDict
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import AlgebraicPyTree, AnyPyTree, Array, Backend, Scalar, is_scalar
from algebraic.utils import pytree, validate_semiring
from algebraic.utils.poly import (
    _add_factors,
    _batched_add_factors,
    _batched_multiply_factors,
    _merge_weights_bias,
    _multiply_factors,
    _split_merged_factors,
    batched_compose_factors,
    batched_evaluate_factors,
    compose_factors,
    evaluate_factors,
    pad_upto,
    prepare_replacement_factors,
    prune_factors,
)


@pytree.register_node_class  # type: ignore[arg-type]
@dataclass
class RankDecomposition(AlgebraicPyTree):
    """CP (CANDECOMP/PARAFAC) decomposition of multilinear polynomial.

    Represents polynomial as sum of rank-1 components:
        ``p(x) = sum_{r=1}^R prod_{k=1}^d factors[r, k, index_k]``

    where ``index_k`` in ``{0, 1, ..., n}``:
        - 0 represents constant (always 1)
        - ``i`` (``i>0``) represents variable ``x_{i-1}``
    """

    factors: AlgebraicArray
    algebra: Lattice
    max_rank: int
    """Maximum rank for CP decomposition (controls memory usage)"""
    max_degree: int
    """Maximum degree for polynomials (None = num_vars)"""
    max_replacement_degree: int
    """Maximum degree for replacement polynomials in compose (None = max_degree)"""

    backend: Backend = field(default=Backend.NUMPY, kw_only=True)
    """The specific backend from the derived class"""

    def __init__(
        self,
        factors: AlgebraicArray,
        max_rank: int | None = 10,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> None:
        super().__init__()

        if backend is None:
            backend = Backend.from_array(factors.data)
        elif isinstance(backend, str):
            backend = Backend(backend)
        num_vars = factors.shape[-1] - 1

        if not isinstance(factors.semiring, Lattice):
            raise ValueError(
                f"Unsupported type for polynomial sub-algebra {type(factors.semiring)}; expected BoundedDistributiveLattice"
            )

        self.factors = factors
        self.algebra = factors.semiring
        self.max_rank = max_rank if max_rank is not None else 10
        self.max_degree = max_degree if max_degree is not None else num_vars
        self.max_replacement_degree = max_replacement_degree if max_replacement_degree is not None else self.max_degree
        self.backend = backend

    @property
    def rank(self) -> int:
        return self.factors.shape[-3]

    @property
    def degree(self) -> int:
        return self.factors.shape[-2]

    @property
    def num_vars(self) -> int:
        return self.factors.shape[-1] - 1

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch dimensions; ``()`` for an unbatched polynomial."""
        return self.factors.shape[:-3]

    def _replace_factors(self, factors: AlgebraicArray) -> "RankDecomposition":
        """Return a new instance with the given factors, preserving other attrs."""
        return RankDecomposition(factors, self.max_rank, self.max_degree, self.max_replacement_degree, backend=self.backend)

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
        backend: str | Backend,
        device: object | None = None,
        batch_shape: tuple[int, ...] = (),
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
        device : object or None, optional
            Target device for the underlying arrays.
        batch_shape : tuple[int, ...], optional
            Leading batch dimensions.  When non-empty the returned factors have
            shape ``(*batch_shape, 1, 1, num_vars+1)`` with identical values
            across the batch axis.

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
        factors = algebraic.zeros((1, 1, num_vars + 1), semiring=algebra, backend=backend, device=device)
        factors = factors.at[(0, 0, i + 1)].set(algebra.one)
        if batch_shape:
            factors = algebraic.broadcast_to(factors, (*batch_shape, 1, 1, num_vars + 1))

        return cls(
            factors,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
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
        backend: str | Backend,
        device: object | None = None,
        batch_shape: tuple[int, ...] = (),
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
        device : object or None, optional
            Target device for the underlying arrays.
        batch_shape : tuple[int, ...], optional
            Leading batch dimensions.  When non-empty the returned factors have
            shape ``(*batch_shape, 1, 1, num_vars+1)`` with identical values
            across the batch axis.

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
        factors = (
            algebraic.zeros((1, 1, num_vars + 1), semiring=algebra, backend=backend, device=device).at[(0, 0, 0)].set(value)
        )
        if batch_shape:
            factors = algebraic.broadcast_to(factors, (*batch_shape, 1, 1, num_vars + 1))

        return cls(
            factors,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
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
        backend: str | Backend,
        device: object | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Self:
        return cls.constant(
            algebra.zero,
            num_vars,
            algebra,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
            device=device,
            batch_shape=batch_shape,
        )

    @classmethod
    def one(
        cls,
        num_vars: int,
        algebra: Lattice,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend,
        device: object | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Self:
        return cls.constant(
            algebra.one,
            num_vars,
            algebra,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
            device=device,
            batch_shape=batch_shape,
        )

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
            device=self.factors.device,
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
            device=self.factors.device,
        )

    def normalize(self) -> "RankDecomposition":
        """Canonicalize via truth-table round-trip, bounding degree to max_degree.

        Converts to sparse form (enumerating all monomial evaluations) and back,
        producing a canonical CP decomposition with degree <= max_degree.
        This prevents unbounded degree growth across repeated :meth:`compose` calls.

        Raises
        ------
        ValueError
            If called on a batched polynomial (``batch_shape != ()``).
            Call element-wise or use :meth:`compose` which handles this internally.
        """
        if self.batch_shape:
            raise ValueError(
                "normalize() is not supported for batched RankDecomposition "
                "(batch_shape != ()). Compose handles normalization per element internally."
            )
        if self.degree <= self.max_degree:
            return self
        sparse = self.to_sparse()
        return RankDecomposition.from_sparse(
            sparse,
            self.max_rank,
            self.max_degree,
            self.max_replacement_degree,
            backend=self.backend,
        )

    def __add__(self, other: "RankDecomposition | Scalar") -> "RankDecomposition":
        """Add by concatenating rank-1 components.

        For CP decomposition: ``p + q`` = sum of all components from both.
        For batched polynomials each batch element is added independently.
        """
        if is_scalar(other):
            other = self._make_const(other)
        assert isinstance(other, RankDecomposition)
        assert other.num_vars == self.num_vars
        validate_semiring(self.factors, other.factors)

        if self.batch_shape:
            new_factors = _batched_add_factors(self.factors, other.factors, self.max_rank, self.max_degree)
        else:
            new_factors = _add_factors(self.factors, other.factors)
            new_factors = prune_factors(new_factors, self.max_rank, self.max_degree)
        result = self._replace_factors(new_factors)
        return result

    def __mul__(self, other: "RankDecomposition") -> "RankDecomposition":
        """Multiply two CP-decomposed polynomials.

        Delegates core multiplication to ``_multiply_arrays()``, then applies
        simplification and compression.
        For batched polynomials each batch element is multiplied independently.
        """
        if self.batch_shape:
            batch_size = self.batch_shape[0]
            new_factors = _batched_multiply_factors(self.factors, other.factors)
            pruned = [prune_factors(new_factors[b], self.max_rank, self.max_degree) for b in range(batch_size)]
            max_r = max(pf.shape[0] for pf in pruned)
            max_d = max(pf.shape[1] for pf in pruned)
            padded = [pad_upto(pf, max_rank=max_r, max_degree=max_d) for pf in pruned]
            new_factors = algebraic.stack(padded)
        else:
            new_factors = _multiply_factors(self.factors, other.factors)
            new_factors = prune_factors(new_factors, self.max_rank, self.max_degree)
        result = self._replace_factors(new_factors)

        return result

    def evaluate(self, points: Array | AlgebraicArray) -> AlgebraicArray:
        """Evaluate polynomial at given point.

        Parameters
        ----------
        points : Array or AlgebraicArray
            For unbatched polynomials: shape ``(num_vars,)``.
            For batched polynomials: shape ``(B, num_vars)``.

        Returns
        -------
        AlgebraicArray
            Array of shape ``(B,)`` (batched case) or a scalar array
        """
        if self.batch_shape:
            return batched_evaluate_factors(self.factors, points, self.backend)
        return evaluate_factors(self.factors, points, self.backend)

    def compose(self, replacements: Sequence["RankDecomposition"]) -> "RankDecomposition":
        """Compose polynomial with replacement polynomials.

        Parameters
        ----------
        replacements : Sequence[RankDecomposition]
            Sequence of replacement polynomials, one per variable.  For batched
            polynomials each replacement may itself be batched ``(B, R, D, N+1)``
            or unbatched ``(R, D, N+1)`` (the latter is broadcast across the batch).

        Returns
        -------
        RankDecomposition
            The composed polynomial.
        """
        if len(replacements) != self.num_vars:
            raise ValueError(
                f"Cannot compose a sequence of {len(replacements)} replacements for a polynomial with {self.num_vars} variables"
            )
        if self.batch_shape:
            replacement_factors_list = [r.factors for r in replacements]
            q_factors = prepare_replacement_factors(replacement_factors_list, self.algebra, self.batch_shape)
            result_factors = batched_compose_factors(self.factors, q_factors, self.max_rank, self.max_degree)
            return self._replace_factors(result_factors)
        replacement_factors = [r.factors for r in replacements]
        result_factors = compose_factors(self.factors, replacement_factors, self.max_rank, self.max_degree)
        result = self._replace_factors(result_factors)
        return result.normalize()

    def tree_flatten(self) -> tuple[list[AlgebraicArray], tuple[typing.Any, ...]]:
        return [self.factors], (self.algebra, self.max_rank, self.max_degree, self.max_replacement_degree, self.backend)

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[typing.Any, ...], children: Sequence[AnyPyTree]) -> "RankDecomposition":
        algebra, max_rank, max_degree, max_replacement_degree, backend = aux_data
        factors = children[0]
        assert isinstance(factors, AlgebraicArray)
        return cls(factors, max_rank, max_degree, max_replacement_degree, backend=backend)

    def _index_to_bits(self, index: int) -> tuple[int, ...]:
        """Convert flat index to n-bit tuple."""
        from bitarray.util import int2ba

        return tuple(int2ba(index, length=self.num_vars))

    def to_sparse(self) -> PolyDict:
        """Convert CP to sparse via a subset DP over variable bitmasks.

        Raises
        ------
        ValueError
            If called on a batched polynomial (``batch_shape != ()``).
            The ``PolyDict`` representation is not batched; call element-wise.

        Replaces the previous ``O((n+1)^d)`` enumeration with an
        ``O(R * D * 2^n * n)`` DP that is independent of ``degree``.
        This prevents exponential blowup when ``degree > max_degree``
        (which happens after :meth:`compose` before :meth:`normalize`
        has been applied).

        The DP assumes idempotent variable multiplication (``x_i * x_i = x_i``),
        which holds for multilinear polynomials over bounded distributive lattices.
        """
        if self.batch_shape:
            raise ValueError(
                "to_sparse() is not supported for batched RankDecomposition (batch_shape != ()). Call element-wise instead."
            )
        backend = Backend(self.backend)
        device = self.factors.device
        zero = algebraic.zeros((), semiring=self.algebra, backend=self.backend, device=device)
        one = algebraic.ones((), semiring=self.algebra, backend=self.backend, device=device)
        n = self.num_vars

        # result[bits] accumulates the coefficient for monomial prod_{i: bits[i]} x_i
        result: dict[frozenbitarray, AlgebraicArray] = {}

        for r in range(self.rank):
            # dp maps integer bitmask -> accumulated coefficient.
            # Bit i set <-> variable x_i is part of the monomial.
            # Starting state: empty monomial (bitmask 0) with coefficient 1.
            dp: dict[int, AlgebraicArray] = {0: one}

            for k in range(self.degree):
                new_dp: dict[int, AlgebraicArray] = {}

                for mask, c in dp.items():
                    # Slot k selects the constant term (index 0).
                    const_factor: AlgebraicArray = self.factors[r, k, 0]
                    if not bool(algebraic.allclose(const_factor, zero)):
                        contribution = const_factor * c
                        new_dp[mask] = (new_dp[mask] + contribution) if mask in new_dp else contribution

                    # Slot k selects variable x_i (index i+1).
                    # Idempotent rule: x_i * x_i = x_i, so mask | (1<<i) == mask when bit i is set.
                    for i in range(n):
                        var_factor: AlgebraicArray = self.factors[r, k, i + 1]
                        if bool(algebraic.allclose(var_factor, zero)):
                            continue
                        new_mask = mask | (1 << i)
                        contribution = var_factor * c
                        new_dp[new_mask] = (new_dp[new_mask] + contribution) if new_mask in new_dp else contribution

                dp = new_dp

            for mask, coeff in dp.items():
                if not bool(algebraic.allclose(coeff, zero)):
                    bits = frozenbitarray([bool(mask & (1 << i)) for i in range(n)])
                    result[bits] = (result[bits] + coeff) if bits in result else coeff

        return PolyDict(self.algebra, self.num_vars, result, backend=backend)

    @classmethod
    def from_sparse(
        cls,
        sparse: PolyDict,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
        device: object | None = None,
    ) -> Self:
        """Convert sparse to CP form (each monomial becomes rank-1 component).

        Parameters
        ----------
        sparse : algebraic.polynomials.PolyDict
            Sparse polynomial to convert.
        max_rank : int or None, optional
            Maximum rank for the result.
        max_degree : int or None, optional
            Maximum degree for the result (default: ``num_vars``).
        max_replacement_degree : int or None, optional
            Maximum degree for replacement polynomials in :meth:`compose`.
        backend : str or Backend or None, optional
            Backend to use.
        device : object or None, optional
            Target device for the underlying arrays.

        Returns
        -------
        RankDecomposition
            CP decomposition with one rank-1 component per monomial.
        """
        if max_degree is None:
            max_degree = sparse.num_vars
        algebra = sparse.algebra
        num_vars = sparse.num_vars
        backend = sparse.backend

        zero_poly = cls.constant(
            algebra.zero,
            sparse.num_vars,
            algebra,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
            device=device,
        )
        max_rank = zero_poly.max_rank
        max_degree = zero_poly.max_degree
        max_replacement_degree = zero_poly.max_replacement_degree

        if len(sparse) == 0:
            return zero_poly

        rank = len(sparse)
        factors = algebraic.zeros((rank, max_degree, num_vars + 1), semiring=algebra, backend=backend, device=device)

        for r, (monomial, coeff) in enumerate(sparse.items()):
            vars_in_monomial = [i for i, bit in enumerate(monomial) if bit]

            if len(vars_in_monomial) == 0:
                factors = factors.at[(r, 0, 0)].set(coeff)
                for k in range(1, max_degree):
                    factors = factors.at[(r, k, 0)].set(algebra.one)
            else:
                for k, var_idx in enumerate(vars_in_monomial):
                    if k < max_degree:
                        val = coeff if k == 0 else algebra.one
                        factors = factors.at[(r, k, var_idx + 1)].set(val)
                for k in range(len(vars_in_monomial), max_degree):
                    factors = factors.at[(r, k, 0)].set(algebra.one)

        return cls(
            factors,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
        )

    def __repr__(self) -> str:
        import wadler_lindig as wl

        return str(wl.pformat(self))


@pytree.register_node_class  # type: ignore[arg-type]
@dataclass
class LowRankFactors(AlgebraicPyTree):
    """CP decomposition with separated variable weights and constant bias.

    Like :class:`RankDecomposition`, but stores factors split into:
        - ``weights``: shape ``(R, D, N)`` - variable-affiliated factors
        - ``bias``: shape ``(R, D)`` - constant/bias factors

    This separation is analogous to an MLP's ``W @ x + b`` and is useful for
    training pipelines that need independent parameter groups (e.g., separate
    learning rates, freezing the bias).
    """

    weights: AlgebraicArray
    """Variable factors of shape ``(rank, degree, num_vars)``."""
    bias: AlgebraicArray
    """Constant factors of shape ``(rank, degree)``."""
    algebra: Lattice
    max_rank: int
    max_degree: int
    max_replacement_degree: int
    backend: Backend = field(default=Backend.NUMPY, kw_only=True)

    def __init__(
        self,
        weights: AlgebraicArray,
        bias: AlgebraicArray,
        max_rank: int | None = 10,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> None:
        super().__init__()

        if backend is None:
            backend = Backend.from_array(weights.data)
        elif isinstance(backend, str):
            backend = Backend(backend)

        if not isinstance(weights.semiring, Lattice):
            raise ValueError(
                f"Unsupported type for polynomial sub-algebra {type(weights.semiring)}; expected BoundedDistributiveLattice"
            )

        num_vars = weights.shape[-1]
        self.weights = weights
        self.bias = bias
        self.algebra = weights.semiring
        self.max_rank = max_rank if max_rank is not None else 10
        self.max_degree = max_degree if max_degree is not None else num_vars
        self.max_replacement_degree = max_replacement_degree if max_replacement_degree is not None else self.max_degree
        self.backend = backend

    @property
    def rank(self) -> int:
        return self.weights.shape[-3]

    @property
    def degree(self) -> int:
        return self.weights.shape[-2]

    @property
    def num_vars(self) -> int:
        return self.weights.shape[-1]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch dimensions; ``()`` for an unbatched polynomial."""
        return self.weights.shape[:-3]

    def to_merged(self) -> AlgebraicArray:
        """Merge weights and bias into a single factors array of shape ``(*batch, R, D, N+1)``."""
        return _merge_weights_bias(self.weights, self.bias)

    @classmethod
    def from_merged(
        cls,
        factors: AlgebraicArray,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> "LowRankFactors":
        """Create from merged factors of shape ``(R, D, N+1)``."""
        weights, bias = _split_merged_factors(factors)
        return cls(weights, bias, max_rank, max_degree, max_replacement_degree, backend=backend)

    def _replace_merged(self, factors: AlgebraicArray) -> "LowRankFactors":
        """Return a new instance from merged factors, preserving other attrs."""
        weights, bias = _split_merged_factors(factors)
        return LowRankFactors(weights, bias, self.max_rank, self.max_degree, self.max_replacement_degree, backend=self.backend)

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
        backend: str | Backend,
        device: object | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> "LowRankFactors":
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
        backend : str or Backend
            Backend to use.
        device : object or None, optional
            Target device for the underlying arrays.
        batch_shape : tuple[int, ...], optional
            Leading batch dimensions.

        Returns
        -------
        LowRankFactors
            A rank-1 polynomial with degree 1.
        """
        weights = algebraic.zeros((1, 1, num_vars), semiring=algebra, backend=backend, device=device)
        weights = weights.at[(0, 0, i)].set(algebra.one)
        bias = algebraic.zeros((1, 1), semiring=algebra, backend=backend, device=device)
        if batch_shape:
            weights = algebraic.broadcast_to(weights, (*batch_shape, 1, 1, num_vars))
            bias = algebraic.broadcast_to(bias, (*batch_shape, 1, 1))
        return cls(weights, bias, max_rank, max_degree, max_replacement_degree, backend=backend)

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
        backend: str | Backend,
        device: object | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> "LowRankFactors":
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
        backend : str or Backend
            Backend to use.
        device : object or None, optional
            Target device for the underlying arrays.
        batch_shape : tuple[int, ...], optional
            Leading batch dimensions.

        Returns
        -------
        LowRankFactors
            A rank-1 constant polynomial.
        """
        weights = algebraic.zeros((1, 1, num_vars), semiring=algebra, backend=backend, device=device)
        bias = algebraic.zeros((1, 1), semiring=algebra, backend=backend, device=device).at[(0, 0)].set(value)
        if batch_shape:
            weights = algebraic.broadcast_to(weights, (*batch_shape, 1, 1, num_vars))
            bias = algebraic.broadcast_to(bias, (*batch_shape, 1, 1))
        return cls(weights, bias, max_rank, max_degree, max_replacement_degree, backend=backend)

    @classmethod
    def zero(
        cls,
        num_vars: int,
        algebra: Lattice,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend,
        device: object | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> "LowRankFactors":
        return cls.constant(
            algebra.zero,
            num_vars,
            algebra,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
            device=device,
            batch_shape=batch_shape,
        )

    @classmethod
    def one(
        cls,
        num_vars: int,
        algebra: Lattice,
        max_rank: int | None = None,
        max_degree: int | None = None,
        max_replacement_degree: int | None = None,
        *,
        backend: str | Backend,
        device: object | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> "LowRankFactors":
        return cls.constant(
            algebra.one,
            num_vars,
            algebra,
            max_rank,
            max_degree,
            max_replacement_degree,
            backend=backend,
            device=device,
            batch_shape=batch_shape,
        )

    def _var_at(self, idx: int) -> "LowRankFactors":
        return LowRankFactors.variable(
            idx,
            self.num_vars,
            self.algebra,
            self.max_rank,
            self.max_degree,
            self.max_replacement_degree,
            backend=self.backend,
            device=self.weights.device,
        )

    def _make_const(self, val: Scalar) -> "LowRankFactors":
        return LowRankFactors.constant(
            val,
            self.num_vars,
            self.algebra,
            self.max_rank,
            self.max_degree,
            self.max_replacement_degree,
            backend=self.backend,
            device=self.weights.device,
        )

    def normalize(self) -> "LowRankFactors":
        """Canonicalize via truth-table round-trip, bounding degree to max_degree.

        Delegates to :meth:`RankDecomposition.normalize` via round-trip conversion.

        Raises
        ------
        ValueError
            If called on a batched polynomial (``batch_shape != ()``).
        """
        if self.batch_shape:
            raise ValueError(
                "normalize() is not supported for batched LowRankFactors "
                "(batch_shape != ()). Compose handles normalization per element internally."
            )
        rd = self.to_rank_decomposition()
        if rd.degree <= rd.max_degree:
            return self
        return LowRankFactors.from_rank_decomposition(rd.normalize())

    def __add__(self, other: "LowRankFactors | Scalar") -> "LowRankFactors":
        """Add by concatenating rank-1 components.

        For batched polynomials each batch element is added independently.
        """
        if is_scalar(other):
            other = self._make_const(other)
        assert isinstance(other, LowRankFactors)
        assert other.num_vars == self.num_vars
        validate_semiring(self.weights, other.weights)

        merged_self = self.to_merged()
        merged_other = other.to_merged()
        if self.batch_shape:
            new_factors = _batched_add_factors(merged_self, merged_other, self.max_rank, self.max_degree)
        else:
            new_factors = _add_factors(merged_self, merged_other)
            new_factors = prune_factors(new_factors, self.max_rank, self.max_degree)
        return self._replace_merged(new_factors)

    def __mul__(self, other: "LowRankFactors") -> "LowRankFactors":
        """Multiply two CP-decomposed polynomials.

        For batched polynomials each batch element is multiplied independently.
        """
        merged_self = self.to_merged()
        merged_other = other.to_merged()
        if self.batch_shape:
            batch_size = self.batch_shape[0]
            new_factors = _batched_multiply_factors(merged_self, merged_other)
            pruned = [prune_factors(new_factors[b], self.max_rank, self.max_degree) for b in range(batch_size)]
            max_r = max(pf.shape[0] for pf in pruned)
            max_d = max(pf.shape[1] for pf in pruned)
            padded = [pad_upto(pf, max_rank=max_r, max_degree=max_d) for pf in pruned]
            new_factors = algebraic.stack(padded)
        else:
            new_factors = _multiply_factors(merged_self, merged_other)
            new_factors = prune_factors(new_factors, self.max_rank, self.max_degree)
        return self._replace_merged(new_factors)

    def evaluate(self, points: Array | AlgebraicArray) -> AlgebraicArray:
        """Evaluate polynomial at given point.

        Parameters
        ----------
        points : Array or AlgebraicArray
            For unbatched polynomials: shape ``(num_vars,)``.
            For batched polynomials: shape ``(B, num_vars)``.

        Returns
        -------
        AlgebraicArray
            Array of shape ``(B,)`` (batched case) or scalar array.
        """
        merged = self.to_merged()
        if self.batch_shape:
            return batched_evaluate_factors(merged, points, self.backend)
        return evaluate_factors(merged, points, self.backend)

    def compose(self, replacements: Sequence["LowRankFactors"]) -> "LowRankFactors":
        """Compose polynomial with replacement polynomials.

        Parameters
        ----------
        replacements : Sequence[LowRankFactors]
            Sequence of replacement polynomials, one per variable.  For batched
            polynomials each replacement may itself be batched or unbatched
            (the latter is broadcast across the batch).

        Returns
        -------
        LowRankFactors
            The composed polynomial.
        """
        if len(replacements) != self.num_vars:
            raise ValueError(
                f"Cannot compose a sequence of {len(replacements)} replacements for a polynomial with {self.num_vars} variables"
            )
        if self.batch_shape:
            merged_self = self.to_merged()
            replacement_merged = [r.to_merged() for r in replacements]
            q_factors = prepare_replacement_factors(replacement_merged, self.algebra, self.batch_shape)
            result_factors = batched_compose_factors(merged_self, q_factors, self.max_rank, self.max_degree)
            return self._replace_merged(result_factors)
        merged_self = self.to_merged()
        replacement_merged = [r.to_merged() for r in replacements]
        result_factors = compose_factors(merged_self, replacement_merged, self.max_rank, self.max_degree)
        result = self._replace_merged(result_factors)
        return result.normalize()

    def to_rank_decomposition(self) -> RankDecomposition:
        """Convert to a :class:`RankDecomposition`."""
        return RankDecomposition(
            self.to_merged(), self.max_rank, self.max_degree, self.max_replacement_degree, backend=self.backend
        )

    @classmethod
    def from_rank_decomposition(cls, rd: RankDecomposition) -> "LowRankFactors":
        """Create from a :class:`RankDecomposition`."""
        return cls.from_merged(rd.factors, rd.max_rank, rd.max_degree, rd.max_replacement_degree, backend=rd.backend)

    def tree_flatten(self) -> tuple[list[AlgebraicArray], tuple[typing.Any, ...]]:
        return [self.weights, self.bias], (
            self.algebra,
            self.max_rank,
            self.max_degree,
            self.max_replacement_degree,
            self.backend,
        )

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[typing.Any, ...], children: Sequence[AnyPyTree]) -> "LowRankFactors":
        algebra, max_rank, max_degree, max_replacement_degree, backend = aux_data
        weights = children[0]
        bias = children[1]
        assert isinstance(weights, AlgebraicArray)
        assert isinstance(bias, AlgebraicArray)
        return cls(weights, bias, max_rank, max_degree, max_replacement_degree, backend=backend)

    def __repr__(self) -> str:
        import wadler_lindig as wl

        return str(wl.pformat(self))
