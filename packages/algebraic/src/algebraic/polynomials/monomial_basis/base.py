"""Backend-agnostic dense tensor polynomial representations."""

# NOTE: Do NOT add `from __future__ import annotations` to this module.
# The `AbstractVar` annotations from `algebraic._better_abc` are processed at class
# definition time and must not be stringified (PEP 563 lazy evaluation would break them).

import copy
import typing
from collections.abc import Mapping
from itertools import product

from bitarray import frozenbitarray
from jaxtyping import Shaped
from typing_extensions import Self

import algebraic.ops as alge
from algebraic._better_abc import AbstractClassVar, AbstractVar, BetterABCMeta
from algebraic.array import AlgebraicArray
from algebraic.polynomials.dok.base import PolyDict, _make_poly_dict
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Array, Backend, Scalar, is_array, is_scalar


class MonomialBasis(metaclass=BetterABCMeta):
    """Dense, monomial basis decomposition of a multilinear polynomial.

    This class represents the coefficients of a multilinear polynomial as a tensor of
    shape ``(2,) * n``, where ``n`` is the maximum degree of the polynomial.
    """

    coeffs: AbstractVar[AlgebraicArray]
    algebra: AbstractVar[Lattice]
    backend: AbstractClassVar[str | Backend]

    @classmethod
    def _get_backend(cls, backend: str | Backend | None) -> str | Backend:
        """Resolve backend: use explicit arg, class default, or fall back to JAX."""
        if backend is not None:
            return backend
        try:
            return cls.backend
        except AttributeError:
            return Backend.NUMPY

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coeffs.shape

    @property
    def num_vars(self) -> int:
        """Number of variables/indeterminants in this multilinear polynomial."""
        return len(self.shape)

    def _replace_coeffs(self, coeffs: AlgebraicArray) -> Self:
        """Return a new instance with the given coefficients, preserving other attrs."""
        clone = copy.copy(self)
        object.__setattr__(clone, "coeffs", coeffs)
        return clone

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
    ) -> AlgebraicArray:
        """Build a zero tensor with a single element set to ``value``."""
        coeffs = alge.zeros(shape, semiring=algebra, backend=backend)
        xp = _get_xp(coeffs)
        new_data = _set_at_index(coeffs.data, idx, value, xp)
        return coeffs._wrap(new_data)

    @classmethod
    def variable(
        cls,
        index: int,
        num_vars: int,
        algebra: Lattice,
        *,
        backend: str | Backend | None = None,
    ) -> "MonomialBasis":
        """Create polynomial representing a single variable :math:`x_i`."""
        idx = tuple(1 if i == index else 0 for i in range(num_vars))
        backend = cls._get_backend(backend)
        return _make_monomial_basis(
            cls._build_coeffs_with_one_set((2,) * num_vars, idx, algebra.one, algebra, backend),
            algebra,
            backend,
        )

    @classmethod
    def constant(
        cls,
        value: Scalar,
        num_vars: int,
        algebra: Lattice,
        *,
        backend: str | Backend | None = None,
    ) -> "MonomialBasis":
        """Create a constant polynomial."""
        idx = (0,) * num_vars
        backend = cls._get_backend(backend)
        return _make_monomial_basis(
            cls._build_coeffs_with_one_set((2,) * num_vars, idx, value, algebra, backend),
            algebra,
            backend,
        )

    @classmethod
    def zero(
        cls,
        num_vars: int,
        algebra: Lattice,
        *,
        backend: str | Backend | None = None,
    ) -> "MonomialBasis":
        """Create the zero polynomial."""
        return cls.constant(algebra.zero, num_vars, algebra, backend=backend)

    @classmethod
    def one(
        cls,
        num_vars: int,
        algebra: Lattice,
        *,
        backend: str | Backend | None = None,
    ) -> "MonomialBasis":
        """Create the one polynomial."""
        return cls.constant(algebra.one, num_vars, algebra, backend=backend)

    # -- Arithmetic ------------------------------------------------------------

    def __add__(self, other: "MonomialBasis | Scalar") -> Self:
        """Add two polynomials by adding the monomial coefficients for identical terms."""
        if is_scalar(other):
            other_coeffs = alge.zeros(self.shape, semiring=self.algebra, backend=self.backend)
            other_coeffs = other_coeffs._wrap(
                _set_at_index(
                    other_coeffs.data,
                    (0,) * self.num_vars,
                    other,
                    _get_xp(other_coeffs),
                )
            )
            coeffs = self.coeffs + other_coeffs
        else:
            assert isinstance(other, MonomialBasis)
            coeffs = self.coeffs + other.coeffs
        return self._replace_coeffs(coeffs)

    def __mul__(self, other: "MonomialBasis | Scalar") -> Self:
        r"""Multiply two polynomials.

        Uses the formula: :math:`c_k = \sum_{i \cup j = k} A_i \cdot B_j`
        """
        if is_scalar(other):
            other_coeffs = alge.zeros(self.shape, semiring=self.algebra, backend=self.backend)
            other_coeffs = other_coeffs._wrap(
                _set_at_index(
                    other_coeffs.data,
                    (0,) * self.num_vars,
                    other,
                    _get_xp(other_coeffs),
                )
            )
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
    ) -> Self:
        """Evaluate polynomial at the given points using Horner-like scheme."""
        map_points: dict[int, Scalar] = {}
        if isinstance(points, Mapping):
            map_points = dict(points)
        else:
            assert is_array(points) or isinstance(points, AlgebraicArray)
            assert points.shape == (self.num_vars,)
            map_points = {i: typing.cast(Scalar, points[i]) for i in range(self.num_vars)}
        return self.compose(map_points)

    def compose(
        self,
        replacements: Mapping[int, "MonomialBasis | Scalar"],
    ) -> Self:
        """Compose polynomial with multiple substitutions.

        Returns ``p(x_1 <- q_1, ..., x_n <- q_n)`` where only specified indices
        are replaced.

        Note
        ----
        The composition should be performed simultaneously.
        """
        repl_keys: list[int] = list(sorted(replacements.keys()))

        def _compose(poly: Self, at: int) -> Self:
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

    # -- Conversion ------------------------------------------------------------

    def to_sparse(self) -> PolyDict:
        """Convert to sparse (dictionary-of-keys) representation."""
        algebra = self.algebra
        backend = Backend(self.backend)
        zero = alge.zeros((), semiring=algebra, backend=backend)
        xp = _get_xp(zero)

        result: dict[frozenbitarray, AlgebraicArray] = {}
        for idx in product([0, 1], repeat=self.num_vars):
            coeff = self.coeffs[idx]
            assert coeff.shape == ()
            if not bool(xp.all(xp.equal(coeff.data, zero.data))):
                monomial = frozenbitarray(idx)
                result[monomial] = coeff

        return _make_poly_dict(algebra, self.num_vars, result, backend)

    @classmethod
    def from_sparse(
        cls,
        poly: PolyDict,
        *,
        backend: str | Backend | None = None,
    ) -> "MonomialBasis":
        """Convert from sparse (dictionary-of-keys) representation."""
        backend = backend or poly.backend
        backend = Backend(backend)
        coeffs = alge.zeros((2,) * poly.num_vars, semiring=poly.algebra, backend=backend)
        for monomial, coeff in poly.items():
            idx = tuple(int(bit) for bit in monomial)
            raw = coeff.data if isinstance(coeff, AlgebraicArray) else coeff
            xp = _get_xp(coeffs)
            coeffs = coeffs._wrap(_set_at_index(coeffs.data, idx, raw, xp))
        return _make_monomial_basis(coeffs, poly.algebra, backend)


# -- Module-level helpers ------------------------------------------------------


def _get_xp(arr: AlgebraicArray) -> typing.Any:  # noqa: ANN401
    """Get the array namespace for an AlgebraicArray."""
    import array_api_compat

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


def _make_monomial_basis(
    coeffs: AlgebraicArray,
    algebra: Lattice,
    backend: str | Backend,
) -> MonomialBasis:
    """Dispatch to the correct backend subclass."""
    backend = Backend(backend)
    if backend == Backend.JAX:
        from algebraic.polynomials.monomial_basis._jax import JaxMonomialBasis

        return JaxMonomialBasis(coeffs, algebra)
    if backend == Backend.TORCH:
        from algebraic.polynomials.monomial_basis._torch import TorchMonomialBasis

        return TorchMonomialBasis(coeffs, algebra)
    if backend == Backend.NUMPY:
        from algebraic.polynomials.monomial_basis._numpy import NumpyMonomialBasis

        return NumpyMonomialBasis(coeffs, algebra)
    raise ValueError(f"Unsupported backend: {backend!r}")
