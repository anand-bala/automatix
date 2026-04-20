"""Backend-agnostic dense tensor polynomial representations."""

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product

from array_api_compat import array_namespace
from bitarray import frozenbitarray
from jaxtyping import Shaped

import algebraic.ops as alge
from algebraic.array import AlgebraicArray
from algebraic.polynomials.dok import PolyDict
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import AlgebraicPyTree, AnyPyTree, Array, Backend, Scalar, is_array, is_scalar


@dataclass
class MonomialBasis(AlgebraicPyTree):
    """Dense, monomial basis decomposition of a multilinear polynomial.

    This class represents the coefficients of a multilinear polynomial as a tensor of
    shape ``(2,) * n``, where ``n`` is the maximum degree of the polynomial.
    """

    coeffs: AlgebraicArray
    algebra: Lattice
    backend: str | Backend = field(default=Backend.NUMPY, kw_only=True)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coeffs.shape

    @property
    def num_vars(self) -> int:
        """Number of variables/indeterminants in this multilinear polynomial."""
        return len(self.shape)

    def _replace_coeffs(self, coeffs: AlgebraicArray) -> "MonomialBasis":
        """Return a new instance with the given coefficients, preserving other attrs."""
        return MonomialBasis(coeffs.clone(), self.algebra, backend=self.backend)

    def _lift_tensor(self, tensor: AlgebraicArray, insert_axis: int) -> AlgebraicArray:
        """Lift (n-1)-dim tensor to n-dim by inserting axis and zero-padding."""
        expanded = alge.expand_dims(tensor, axis=insert_axis)
        # Build zero-padded result by stacking the expanded slice with a zeros slice
        zero_slice = alge.zeros(expanded.shape, semiring=self.algebra, backend=self.backend)
        return alge.concat([expanded, zero_slice], axis=insert_axis)

    # -- Factory methods -------------------------------------------------------

    @classmethod
    def _build_coeffs_with_one_set(
        cls,
        shape: tuple[int, ...],
        idx: tuple[int, ...],
        value: Scalar,
        algebra: Lattice,
        backend: str | Backend,
        device: object | None = None,
    ) -> AlgebraicArray:
        """Build a zero tensor with a single element set to ``value``."""
        coeffs = alge.zeros(shape, semiring=algebra, backend=backend, device=device)
        new_data = coeffs.at[idx].set(value)
        return new_data

    @classmethod
    def variable(
        cls,
        index: int,
        num_vars: int,
        algebra: Lattice,
        *,
        backend: str | Backend = Backend.NUMPY,
        device: object | None = None,
    ) -> "MonomialBasis":
        r"""Create polynomial representing a single variable :math:`x_i`.

        Parameters
        ----------
        index : int
            Variable index (0-based).
        num_vars : int
            Total number of variables.
        algebra : BoundedDistributiveLattice
            Lattice algebra governing operations.
        backend : str or Backend or None, optional
            Backend to use.

        Returns
        -------
        MonomialBasis
            A polynomial with a single variable term.

        Examples
        --------
        >>> from algebraic.polynomials import MonomialBasis
        >>> from algebraic.semirings import boolean_algebra
        >>> ba = boolean_algebra(mode="logic")
        >>> x0 = MonomialBasis.variable(0, num_vars=2, algebra=ba, backend="numpy")
        >>> x0.shape
        (2, 2)
        """
        idx = tuple(1 if i == index else 0 for i in range(num_vars))
        return cls(
            cls._build_coeffs_with_one_set((2,) * num_vars, idx, algebra.one, algebra, backend, device=device),
            algebra,
            backend=backend,
        )

    @classmethod
    def constant(
        cls,
        value: Scalar,
        num_vars: int,
        algebra: Lattice,
        *,
        backend: str | Backend = Backend.NUMPY,
        device: object | None = None,
    ) -> "MonomialBasis":
        """Create a constant polynomial.

        Parameters
        ----------
        value : Scalar
            Coefficient for the constant term.
        num_vars : int
            Number of variables.
        algebra : BoundedDistributiveLattice
            Lattice algebra governing operations.
        backend : str or Backend or None, optional
            Backend to use.

        Returns
        -------
        MonomialBasis
            A constant polynomial.

        Examples
        --------
        >>> from algebraic.polynomials import MonomialBasis
        >>> from algebraic.semirings import boolean_algebra
        >>> ba = boolean_algebra(mode="logic")
        >>> c = MonomialBasis.constant(True, num_vars=2, algebra=ba, backend="numpy")
        >>> c.coeffs[(0, 0)].data
        array(1., dtype=float32)
        """
        idx = (0,) * num_vars
        return cls(
            cls._build_coeffs_with_one_set((2,) * num_vars, idx, value, algebra, backend, device=device),
            algebra,
            backend=backend,
        )

    @classmethod
    def zero(
        cls,
        num_vars: int,
        algebra: Lattice,
        *,
        backend: str | Backend,
        device: object | None = None,
    ) -> "MonomialBasis":
        """Create the zero polynomial."""
        return cls.constant(algebra.zero, num_vars, algebra, backend=backend, device=device)

    @classmethod
    def one(
        cls,
        num_vars: int,
        algebra: Lattice,
        *,
        backend: str | Backend,
        device: object | None = None,
    ) -> "MonomialBasis":
        """Create the one polynomial."""
        return cls.constant(algebra.one, num_vars, algebra, backend=backend, device=device)

    # -- Arithmetic ------------------------------------------------------------

    def __add__(self, other: "MonomialBasis | Scalar") -> "MonomialBasis":
        """Add two polynomials by adding the monomial coefficients for identical terms."""
        if is_scalar(other):
            other_coeffs = alge.zeros(self.shape, semiring=self.algebra, backend=self.backend)
            other_coeffs = other_coeffs.at[(0,) * self.num_vars].set(other)
            coeffs = self.coeffs + other_coeffs
        else:
            assert isinstance(other, MonomialBasis)
            coeffs = self.coeffs + other.coeffs
        return self._replace_coeffs(coeffs)

    def __mul__(self, other: "MonomialBasis | Scalar") -> "MonomialBasis":
        r"""Multiply two polynomials.

        Uses the formula: :math:`c_k = \sum_{i \cup j = k} A_i \cdot B_j`
        """
        if is_scalar(other):
            other_coeffs = alge.zeros(self.shape, semiring=self.algebra, backend=self.backend)
            other_coeffs = other_coeffs.at[(0,) * self.num_vars].set(other)
            other_mb = self._replace_coeffs(other_coeffs)
        else:
            assert isinstance(other, MonomialBasis)
            other_mb = other

        # Scalar (0-var) case
        if self.num_vars == 0 or other_mb.num_vars == 0:
            coeffs = self.coeffs * other_mb.coeffs
            return self._replace_coeffs(coeffs)

        if self.num_vars != other_mb.num_vars:
            raise ValueError(
                "Multiplying two polynomials with unequal number of variables not "
                "supported unless one of them is a scalar/constant polynomial. "
                "Pad the polynomial representation to indicate the correct number "
                "of variables."
            )
        result_coeffs = _multiply_recursive(self.coeffs, other_mb.coeffs)
        return self._replace_coeffs(result_coeffs)

    # -- Evaluation / composition ----------------------------------------------

    def evaluate(
        self,
        points: Mapping[int, Scalar] | Shaped[AlgebraicArray | Array, " {self.num_vars}"],
    ) -> "MonomialBasis":
        """Evaluate polynomial at the given points using Horner-like scheme.

        Parameters
        ----------
        points : Mapping[int, Scalar] or Array
            Either a dict mapping variable indices to scalar values, or an
            array of shape ``(num_vars,)``.

        Returns
        -------
        MonomialBasis
            The evaluated (constant) polynomial.

        Examples
        --------
        >>> from algebraic.polynomials import MonomialBasis
        >>> from algebraic.semirings import boolean_algebra
        >>> ba = boolean_algebra(mode="logic")
        >>> x0 = MonomialBasis.variable(0, num_vars=2, algebra=ba, backend="numpy")
        >>> result = x0.evaluate({0: True, 1: False})
        >>> result.coeffs[(0, 0)].data
        array(1., dtype=float32)
        """
        map_points: dict[int, Scalar] = {}
        if isinstance(points, Mapping):
            map_points = dict(points)  # ty: ignore[no-matching-overload]
        else:
            assert is_array(points) or isinstance(points, AlgebraicArray)
            assert points.shape == (self.num_vars,)
            map_points = {i: typing.cast(Scalar, points[i]) for i in range(self.num_vars)}
        return self.compose(map_points)

    def compose(
        self,
        replacements: Mapping[int, "MonomialBasis | Scalar"],
    ) -> "MonomialBasis":
        """Compose polynomial with multiple substitutions.

        Returns ``p(x_1 <- q_1, ..., x_n <- q_n)`` where only specified indices
        are replaced.

        Note
        ----
        The composition should be performed simultaneously.
        """
        repl_keys: list[int] = list(sorted(replacements.keys()))

        def _compose(poly: "MonomialBasis", at: int) -> "MonomialBasis":
            if at >= len(repl_keys):
                return poly
            coeffs = poly.coeffs
            var_idx = repl_keys[at]
            # Extract cofactors of shape (2,) * (n-1)
            p_xi_0 = alge.take(coeffs, 0, axis=var_idx)
            p_xi_1 = alge.take(coeffs, 1, axis=var_idx)

            # Lift back to full shape
            p_xi_0 = self._lift_tensor(p_xi_0, var_idx)
            p_xi_1 = self._lift_tensor(p_xi_1, var_idx)

            p_xi_0_poly = poly._replace_coeffs(p_xi_0)
            p_xi_1_poly = poly._replace_coeffs(p_xi_1)

            # Recursively compose each cofactor
            p_xi_0_poly = _compose(p_xi_0_poly, at + 1)
            p_xi_1_poly = _compose(p_xi_1_poly, at + 1)

            # Merge cofactors with the replacement
            var_replacement = replacements[var_idx]
            prod = p_xi_1_poly * var_replacement
            result = p_xi_0_poly + prod
            return result

        return _compose(self, 0)

    def tree_flatten(self) -> tuple[list[AlgebraicArray], tuple[typing.Any, ...]]:
        return [self.coeffs], (self.algebra, self.backend)

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[typing.Any, ...], children: Sequence[AnyPyTree]) -> "MonomialBasis":
        algebra, backend = aux_data
        coeffs = children[0]
        assert isinstance(coeffs, AlgebraicArray)
        return cls(coeffs=coeffs, algebra=algebra, backend=backend)

    # -- Conversion ------------------------------------------------------------

    def to_sparse(self) -> PolyDict:
        """Convert to sparse (dictionary-of-keys) representation."""
        algebra = self.algebra
        backend = Backend(self.backend)
        zero = alge.zeros((), semiring=algebra, backend=backend)
        xp = array_namespace(zero.data)

        result: dict[frozenbitarray, AlgebraicArray] = {}
        for idx in product([0, 1], repeat=self.num_vars):
            coeff = self.coeffs[idx]
            assert coeff.shape == ()
            if not bool(xp.all(xp.equal(coeff.data, zero.data))):
                monomial = frozenbitarray(idx)
                result[monomial] = coeff

        return PolyDict(algebra, self.num_vars, result, backend=backend)

    @classmethod
    def from_sparse(
        cls,
        poly: PolyDict,
        *,
        backend: str | Backend | None = None,
        device: object | None = None,
    ) -> "MonomialBasis":
        """Convert from sparse (dictionary-of-keys) representation.

        Parameters
        ----------
        poly : algebraic.polynomials.PolyDict
            Sparse polynomial to convert.
        backend : str or Backend or None, optional
            Backend to use. Defaults to ``poly.backend``.

        Returns
        -------
        MonomialBasis
            Dense monomial-basis representation of *poly*.

        Examples
        --------
        >>> from algebraic.polynomials import PolyDict, MonomialBasis
        >>> from algebraic.semirings import boolean_algebra
        >>> ba = boolean_algebra(mode="logic")
        >>> sp = PolyDict.variable(0, num_vars=2, algebra=ba, backend="numpy")
        >>> mb = MonomialBasis.from_sparse(sp)
        >>> mb.shape
        (2, 2)
        """
        backend = backend or poly.backend
        backend = Backend(backend)
        coeffs = alge.zeros((2,) * poly.num_vars, semiring=poly.algebra, backend=backend, device=device)
        for monomial, coeff in poly.items():
            idx = tuple(int(bit) for bit in monomial)
            coeffs = coeffs.at[idx].set(coeff)
        return cls(coeffs, poly.algebra, backend=backend)


def _multiply_recursive(lhs: AlgebraicArray, rhs: AlgebraicArray) -> AlgebraicArray:
    """A recursive function to compute the Horner's expansion multiplication."""
    assert lhs.shape == rhs.shape
    assert lhs.semiring == rhs.semiring

    n = len(lhs.shape)
    return_shape = lhs.shape

    expected_shape_post_idx = (2,) * (n - 1)

    # Split into cofactors at the first variable
    lhs_0 = lhs[0, ...]
    lhs_1 = lhs[1, ...]
    rhs_0 = rhs[0, ...]
    rhs_1 = rhs[1, ...]

    assert lhs_0.shape == lhs_1.shape == rhs_0.shape == rhs_1.shape == expected_shape_post_idx

    # c_0 = a_0 * b_0
    # c_1 = a_0 * b_1 + a_1 * b_0 + a_1 * b_1
    if len(expected_shape_post_idx) == 0:
        ret_0 = lhs_0 * rhs_0
        ret_1 = (lhs_0 * rhs_1) + (lhs_1 * rhs_0) + (lhs_1 * rhs_1)
    else:
        ret_0 = _multiply_recursive(lhs_0, rhs_0)
        ret_1 = _multiply_recursive(lhs_0, rhs_1) + _multiply_recursive(lhs_1, rhs_0) + _multiply_recursive(lhs_1, rhs_1)
    assert ret_0.shape == ret_1.shape == expected_shape_post_idx
    ret = alge.stack([ret_0, ret_1], axis=0)
    assert ret.shape == return_shape
    return ret
