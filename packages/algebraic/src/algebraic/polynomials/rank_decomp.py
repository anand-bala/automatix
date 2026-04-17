"""Backend-agnostic CP (CANDECOMP/PARAFAC) decomposition of multilinear polynomials."""

import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product

from bitarray import frozenbitarray
from typing_extensions import Self

import algebraic.ops as algebraic
from algebraic.array import AlgebraicArray
from algebraic.polynomials.dok import PolyDict
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import AlgebraicPyTree, AnyPyTree, Array, Backend, Scalar, is_array, is_scalar
from algebraic.utils import validate_semiring


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
        num_vars = factors.shape[2] - 1

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
        return self.factors.shape[0]

    @property
    def degree(self) -> int:
        return self.factors.shape[1]

    @property
    def num_vars(self) -> int:
        return self.factors.shape[2] - 1

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
        factors = algebraic.zeros((1, 1, num_vars + 1), semiring=algebra, backend=backend)
        factors = factors.at[(0, 0, i + 1)].set(algebra.one)

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
        factors = algebraic.zeros((1, 1, num_vars + 1), semiring=algebra, backend=backend).at[(0, 0, 0)].set(value)

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
        backend: str | Backend,
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

    def normalize(self) -> "RankDecomposition":
        """Canonicalize via truth-table round-trip, bounding degree to max_degree.

        Converts to sparse form (enumerating all monomial evaluations) and back,
        producing a canonical CP decomposition with degree <= max_degree.
        This prevents unbounded degree growth across repeated :meth:`compose` calls.
        """
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
        """
        if is_scalar(other):
            other = self._make_const(other)
        assert isinstance(other, RankDecomposition)
        assert other.num_vars == self.num_vars
        validate_semiring(self.factors, other.factors)

        new_factors = _add_factors(self.factors, other.factors, self.algebra)
        new_factors = prune_factors(new_factors, max_rank=self.max_rank)
        result = self._replace_factors(new_factors)
        return result

    def __mul__(self, other: "RankDecomposition") -> "RankDecomposition":
        """Multiply two CP-decomposed polynomials.

        Delegates core multiplication to ``_multiply_arrays()``, then applies
        simplification and compression.
        """
        new_factors = _multiply_factors(self.factors, other.factors)
        new_factors = prune_factors(new_factors, self.max_rank)
        result = self._replace_factors(new_factors)

        return result

    def evaluate(self, points: Array | AlgebraicArray) -> "RankDecomposition":
        """Evaluate polynomial at given point.

        Parameters
        ----------
        points : Array or AlgebraicArray
            An array of shape ``(num_vars,)`` to replace each variable with.

        Returns
        -------
        RankDecomposition
            Constant polynomial after evaluation.
        """
        result_data = evaluate_factors(self.factors, points, self.algebra, self.backend)
        return self._make_const(result_data)

    def compose(self, replacements: Sequence["RankDecomposition"]) -> "RankDecomposition":
        """Compose polynomial with replacement polynomials.

        Parameters
        ----------
        replacements : Sequence[RankDecomposition]
            Sequence of replacement polynomials, one per variable.

        Returns
        -------
        RankDecomposition
            The composed polynomial.
        """
        if len(replacements) != self.num_vars:
            raise ValueError(
                f"Cannot compose a sequence of {len(replacements)} replacements for a polynomial with {self.num_vars} variables"
            )
        replacement_factors = [r.factors for r in replacements]
        result_factors = compose_factors(self.factors, replacement_factors, self.max_rank, self.algebra)
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
        """Convert CP to sparse by enumerating all monomial evaluations.

        WARNING: This is expensive ``O((n+1)^d)`` where d is degree.
        """
        backend = Backend(self.backend)

        result: dict[frozenbitarray, AlgebraicArray] = {}

        zero = algebraic.zeros((), semiring=self.algebra, backend=self.backend)
        one = algebraic.ones((), semiring=self.algebra, backend=self.backend)

        for assignment in product(range(self.num_vars + 1), repeat=self.degree):
            vars_present = frozenbitarray(
                [any(assignment[k] == i + 1 for k in range(self.degree)) for i in range(self.num_vars)]
            )

            coeff: AlgebraicArray = zero
            for r in range(self.rank):
                component: AlgebraicArray = one
                for k in range(self.degree):
                    factor_value: AlgebraicArray = self.factors[r, k, assignment[k]]
                    component = factor_value * component
                coeff = component + coeff

            if not bool(algebraic.allclose(coeff, zero)):
                if vars_present in result:
                    result[vars_present] = result[vars_present] + coeff
                else:
                    result[vars_present] = coeff

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
        )
        max_rank = zero_poly.max_rank
        max_degree = zero_poly.max_degree
        max_replacement_degree = zero_poly.max_replacement_degree

        if len(sparse) == 0:
            return zero_poly

        rank = len(sparse)
        factors = algebraic.zeros((rank, max_degree, num_vars + 1), semiring=algebra, backend=backend)

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

    one_array = algebraic.ones((1,), semiring=algebra, backend=backend)
    if is_array(points):
        points_array = algebraic.array(points, semiring=algebra, backend=backend)
    else:
        points_array = points
    selector = algebraic.concat([one_array, points_array])

    result = algebraic.zeros((), semiring=algebra, backend=backend)
    for r in range(rank):
        component_value = algebraic.ones((), semiring=algebra, backend=backend)
        for k in range(d):
            dim_value = algebraic.zeros((), semiring=algebra, backend=backend)
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

    # Collapsing the additional dimensions in the `contracted` output can cause
    # major explosion in the size of this polynomial.
    # So, before we do the outer/Khatri-Rao product to collapse the additional
    # dimensions, we will compress the contraction by essentially performing
    # a version of beam search on `contracted`.

    return contraction_compression(contracted, max_rank, algebra)


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
    num_vars = n_plus_1 - 1
    backend = Backend.from_array(replacement_factors[0].data)
    one_factors = pad_upto(
        RankDecomposition.one(num_vars, algebra, backend=backend).factors,
        max_rank=target_rank,
        max_degree=target_degree,
        algebra=algebra,
    )
    new_replacements = algebraic.stack(
        # Add the constant/bias term "replacement" to be the identity.
        [one_factors]
        + [pad_upto(q, max_rank=target_rank, max_degree=target_degree, algebra=algebra) for q in replacement_factors]
    )

    assert new_replacements.shape == (n_plus_1, target_rank, target_degree, n_plus_1)

    return new_replacements


def pad_upto(factors: AlgebraicArray, *, max_rank: int, max_degree: int, algebra: Lattice) -> AlgebraicArray:
    """Modify the factors of a :class:`RankDecomposition` such that the rank and degree are padded with identity elements up to the given maximum"""
    rank, degree, n_plus_1 = factors.shape

    if max_rank <= rank and max_degree <= degree:
        return factors

    backend = Backend.from_array(factors.data)

    new_rank = max(rank, max_rank)
    new_degree = max(degree, max_degree)

    return_shape = (new_rank, new_degree, n_plus_1)

    # First, we create a base with the padded degree axis, with every term being a polynomial.one
    # This will be the base where the factors will be replaced into
    # Then, we will pad the rank axis with polynomial.zero elements

    one_terms = algebraic.broadcast_to(
        algebraic.eye(1, n_plus_1, semiring=algebra, backend=backend),
        (rank, new_degree, n_plus_1),
    )
    degree_padded = one_terms.at[:, :degree, :].set(factors)

    zero_terms = algebraic.zeros((new_rank - rank, new_degree, n_plus_1), semiring=algebra, backend=backend)

    rank_padded = algebraic.concat((degree_padded, zero_terms), axis=0)

    assert rank_padded.shape == return_shape

    return rank_padded


def contraction_compression(contracted: AlgebraicArray, max_rank: int, algebra: Lattice) -> AlgebraicArray:
    """
    Replace exponential expansion with beam search over tensor contractions, replace norms with lattice idempotence rules.
    """

    rank1, degree1, rank2, degree2, n_plus_1 = contracted.shape
    backend = Backend.from_array(contracted.data)

    # Flatten substitution choices
    # each (rank1,degree1) has rank2 choices
    # So, we will merge the rank axes and move the degree1 axis to the front
    # (rank1, degree1, rank2, degree2, n_plus_1) -> (degree1, rank1 * rank2, degree2, n_plus_1)
    candidates = algebraic.permute_dims(contracted, (1, 0, 2, 3, 4))
    candidates = algebraic.reshape(candidates, (degree1, rank1 * rank2, degree2, n_plus_1))

    # Start with the multiplicative identity polynomial
    identity = algebraic.broadcast_to(algebraic.eye(1, n_plus_1, semiring=algebra, backend=backend), (1, 1, n_plus_1))

    # Initialize beam storage with the identity element
    # beam shape: (beam_rank, beam_degree, n+1)
    beam = identity

    for d in range(degree1):
        # Slice candidates for this degree
        candidate_d = candidates[d]  # (rank1 * rank2, degree2, n+1)

        # shape: (beam_rank * rank1 * rank2, beam_degree + degree2, n+1)
        beam = _multiply_factors(beam, candidate_d)

        beam = prune_factors(beam, max_rank)

    return beam


def deduplicate_rank_dim(factors: AlgebraicArray) -> AlgebraicArray:
    # factors shape: (rank, degree, num_vars + 1)
    # Keep only the first occurrence of each unique row.

    a = factors[:, None, :, :]  # (rank, 1, d, n+1)
    b = factors[None, :, :, :]  # (1, rank, d, n+1)

    # eq[i, j] = True iff row i equals row j (element-wise across d and n+1 axes)
    eq = algebraic.equal(a, b).all((2, 3))  # (rank, rank) raw bool array

    # earlier[i, j] = True iff i < j
    backend = Backend.from_array(factors.data)
    xp = backend.get_array_namespace()
    rank = factors.shape[0]
    arange = xp.arange(rank)
    earlier = arange[:, None] < arange[None, :]  # (rank, rank)

    # is_dup[j] = True if some earlier row i (i < j) is identical to row j
    is_dup = (eq & earlier).any(0)
    keep = ~is_dup

    return factors[keep]


def idempotence_pruning(factors: AlgebraicArray) -> AlgebraicArray:
    """Remove terms that are dominated by lattice idempotence laws"""
    # factors shape: (rank, d, n + 1)

    # Lattice structure implies:
    # p <= q if p + q == q  <- We will use this
    # or
    # p <= q if p * q == p

    # expand for pairwise comparison
    a = factors[:, None, :, :]  # (rank, 1, d, n+1)
    b = factors[None, :, :, :]  # (1, rank, d, n+1)

    added = a + b
    # Check where a + b == b (i.e., a dominates b in the lattice order)
    check = algebraic.equal(added, b).all((2, 3))
    # shape: (rank, rank)

    # check[i, j] = True means factors[i] <= factors[j] (i is dominated by j).
    # Exclude self-comparisons (diagonal is always True due to idempotence).
    # Prune i if some OTHER j (j != i) dominates i, i.e., check any j along axis 1.
    backend = Backend.from_array(factors.data)
    xp = backend.get_array_namespace()
    rank = factors.shape[0]
    arange = xp.arange(rank)
    off_diag = arange[:, None] != arange[None, :]  # True where i != j
    check = check & off_diag

    # keep[i] = True iff no other j dominates i
    keep = ~check.any(1)

    return factors[keep]


def prune_factors(factors: AlgebraicArray, max_rank: int) -> AlgebraicArray:
    factors = deduplicate_rank_dim(factors)
    factors = idempotence_pruning(factors)
    factors = factors[:max_rank]  # TODO: verify if this works...
    return factors


def _multiply_factors(p: AlgebraicArray, q: AlgebraicArray) -> AlgebraicArray:
    """Core multiplication logic on raw arrays (no simplification/compression).

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

    p_expanded = algebraic.broadcast_to(
        p[:, None, :, :],
        (rank_p, rank_q, degree_p, n_plus_1),
    )
    q_expanded = algebraic.broadcast_to(
        q[None, :, :, :],
        (rank_p, rank_q, degree_q, n_plus_1),
    )

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

    new_factors = algebraic.concat([a_padded, b_padded], axis=0)
    return new_factors


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
    # bias: (R, D) -> (R, D, 1)
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


@dataclass
class LowRankFactors(AlgebraicPyTree):
    """CP decomposition with separated variable weights and constant bias.

    Like :class:`RankDecomposition`, but stores factors split into:
        - ``weights``: shape ``(R, D, N)`` — variable-affiliated factors
        - ``bias``: shape ``(R, D)`` — constant/bias factors

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

        num_vars = weights.shape[2]
        self.weights = weights
        self.bias = bias
        self.algebra = weights.semiring
        self.max_rank = max_rank if max_rank is not None else 10
        self.max_degree = max_degree if max_degree is not None else num_vars
        self.max_replacement_degree = max_replacement_degree if max_replacement_degree is not None else self.max_degree
        self.backend = backend

    @property
    def rank(self) -> int:
        return self.weights.shape[0]

    @property
    def degree(self) -> int:
        return self.weights.shape[1]

    @property
    def num_vars(self) -> int:
        return self.weights.shape[2]

    def to_merged(self) -> AlgebraicArray:
        """Merge weights and bias into a single factors array of shape ``(R, D, N+1)``."""
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

        Returns
        -------
        LowRankFactors
            A rank-1 polynomial with degree 1.
        """
        weights = algebraic.zeros((1, 1, num_vars), semiring=algebra, backend=backend)
        weights = weights.at[(0, 0, i)].set(algebra.one)
        bias = algebraic.zeros((1, 1), semiring=algebra, backend=backend)
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

        Returns
        -------
        LowRankFactors
            A rank-1 constant polynomial.
        """
        weights = algebraic.zeros((1, 1, num_vars), semiring=algebra, backend=backend)
        bias = algebraic.zeros((1, 1), semiring=algebra, backend=backend).at[(0, 0)].set(value)
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
    ) -> "LowRankFactors":
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
        backend: str | Backend,
    ) -> "LowRankFactors":
        return cls.constant(algebra.one, num_vars, algebra, max_rank, max_degree, max_replacement_degree, backend=backend)

    def _var_at(self, idx: int) -> "LowRankFactors":
        return LowRankFactors.variable(
            idx,
            self.num_vars,
            self.algebra,
            self.max_rank,
            self.max_degree,
            self.max_replacement_degree,
            backend=self.backend,
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
        )

    def normalize(self) -> "LowRankFactors":
        """Canonicalize via truth-table round-trip, bounding degree to max_degree.

        Delegates to :meth:`RankDecomposition.normalize` via round-trip conversion.
        """
        rd = self.to_rank_decomposition()
        if rd.degree <= rd.max_degree:
            return self
        return LowRankFactors.from_rank_decomposition(rd.normalize())

    def __add__(self, other: "LowRankFactors | Scalar") -> "LowRankFactors":
        """Add by concatenating rank-1 components."""
        if is_scalar(other):
            other = self._make_const(other)
        assert isinstance(other, LowRankFactors)
        assert other.num_vars == self.num_vars
        validate_semiring(self.weights, other.weights)

        merged_self = self.to_merged()
        merged_other = other.to_merged()
        new_factors = _add_factors(merged_self, merged_other, self.algebra)
        new_factors = prune_factors(new_factors, max_rank=self.max_rank)
        return self._replace_merged(new_factors)

    def __mul__(self, other: "LowRankFactors") -> "LowRankFactors":
        """Multiply two CP-decomposed polynomials."""
        merged_self = self.to_merged()
        merged_other = other.to_merged()
        new_factors = _multiply_factors(merged_self, merged_other)
        new_factors = prune_factors(new_factors, self.max_rank)
        return self._replace_merged(new_factors)

    def evaluate(self, points: Array | AlgebraicArray) -> "LowRankFactors":
        """Evaluate polynomial at given point.

        Parameters
        ----------
        points : Array or AlgebraicArray
            An array of shape ``(num_vars,)`` to replace each variable with.

        Returns
        -------
        LowRankFactors
            Constant polynomial after evaluation.
        """
        merged = self.to_merged()
        result_data = evaluate_factors(merged, points, self.algebra, self.backend)
        return self._make_const(result_data)

    def compose(self, replacements: Sequence["LowRankFactors"]) -> "LowRankFactors":
        """Compose polynomial with replacement polynomials.

        Parameters
        ----------
        replacements : Sequence[LowRankFactors]
            Sequence of replacement polynomials, one per variable.

        Returns
        -------
        LowRankFactors
            The composed polynomial.
        """
        if len(replacements) != self.num_vars:
            raise ValueError(
                f"Cannot compose a sequence of {len(replacements)} replacements for a polynomial with {self.num_vars} variables"
            )
        merged_self = self.to_merged()
        replacement_merged = [r.to_merged() for r in replacements]
        result_factors = compose_factors(merged_self, replacement_merged, self.max_rank, self.algebra)
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
