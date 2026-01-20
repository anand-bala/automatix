"""JAX-based dense tensor polynomial representations."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
from bitarray import frozenbitarray
from jaxtyping import Array, ArrayLike, Scalar, Shaped

import algebraic.numpy as alge
import algebraic.polynomials.sparse as sparse_poly
from algebraic.array import AlgebraicArray
from algebraic.spec import BoundedDistributiveLattice as Lattice

K = typing.TypeVar("K", bound=Lattice)


class MonomialBasis(eqx.Module, typing.Generic[K]):
    """Dense, monomial basis decomposition of a multilinear polynomial

    This class represents the coefficients of a multilinear polynomial as a tensor of
    shape `(2,) * n`, where `n` is the maximum degree of the polynomial.
    """

    coeffs: Shaped[AlgebraicArray[K], "*2"]
    algebra: K = eqx.field(static=True)

    def __init__(self, coeffs: Shaped[AlgebraicArray[K] | ArrayLike, "*2"], algebra: None | K = None) -> None:
        if isinstance(coeffs, AlgebraicArray):
            if algebra is not None and not eqx.tree_equal(algebra, coeffs.semiring):
                raise ValueError("Provided algebra for RankDecomposition != algebra of AlgebraicArray coeffs")
            algebra = coeffs.semiring
            coeffs = coeffs.data
        elif algebra is None:
            raise ValueError("Must provide algebra if not using AlgebraicArray")
        assert algebra is not None

        self.coeffs = AlgebraicArray(coeffs, algebra)
        self.algebra = self.coeffs.semiring

    def __check_init__(self) -> None:
        if not isinstance(self.coeffs.semiring, Lattice):
            raise TypeError("Multilinear polynomial representation is only supported over BoundedDistributiveLattice algebras")

    @staticmethod
    def variable(index: int, num_vars: int, algebra: K) -> MonomialBasis[K]:
        idx = tuple(1 if i == index else 0 for i in range(num_vars))
        coeffs = alge.zeros((2,) * num_vars, algebra).at[*idx].set(algebra.one)
        return MonomialBasis(coeffs)

    @staticmethod
    def constant(value: Scalar, num_vars: int, algebra: K) -> MonomialBasis[K]:
        idx = (0,) * num_vars
        coeffs = alge.zeros((2,) * num_vars, algebra).at[idx].set(value)
        return MonomialBasis(coeffs)

    @staticmethod
    def zero(num_vars: int, algebra: K) -> MonomialBasis[K]:
        return MonomialBasis.constant(algebra.zero, num_vars, algebra)

    @staticmethod
    def one(num_vars: int, algebra: K) -> MonomialBasis[K]:
        return MonomialBasis.constant(algebra.one, num_vars, algebra)

    @property
    def shape(self) -> tuple[int, ...]:
        return typing.cast(tuple[int, ...], self.coeffs.shape)

    @property
    def num_vars(self) -> int:
        """Number of variables/indeterminants in this multilinear polynomial"""
        return len(self.shape)

    def __add__(self, other: MonomialBasis[K] | Scalar) -> MonomialBasis[K]:
        """Add two polynomials by adding the monomial coefficients for identical terms."""
        if jnp.isscalar(other):
            other = MonomialBasis(AlgebraicArray(other, self.algebra))  # type: ignore[arg-type]
        assert isinstance(other, MonomialBasis)
        coeffs = self.coeffs + other.coeffs
        return MonomialBasis[K](coeffs)

    def __mul__(self, other: MonomialBasis[K] | Scalar) -> MonomialBasis[K]:
        r"""Multiply two polynomials.

        c_k = sum_{i OR j = k} A_i * B_j

        """
        if jnp.isscalar(other):
            other = MonomialBasis(AlgebraicArray(other, self.algebra))  # type: ignore[arg-type]
        assert isinstance(other, MonomialBasis)
        # Check if either is scalar: easy case
        if self.num_vars == 0 or other.num_vars == 0:
            coeffs = self.coeffs * other.coeffs
            return MonomialBasis(coeffs)
        # Now we deal with the case where there are variables
        if self.num_vars != other.num_vars:
            raise ValueError(
                "Multiplying two polynomials with unequal number of variables not supported unless one of them is a scalar/constant polynomial. Pad the polynomial representation to indicate the correct number of variables."
            )
        result_coeffs = _multiply_recursive(self.coeffs, other.coeffs)

        # @eqx.filter_jit
        # @quax.quaxify
        # def set_at(dest: AlgebraicArray[K], at: tuple[int, ...], val: AlgebraicArray[K]) -> AlgebraicArray[K]:
        #     return dest.at[at].set(val)

        # n = self.num_vars
        # result_coeffs = alge.zeros((2,) * n, self.algebra)
        # for a_idx in range(2**n):
        #     a_bits = tuple(ba_utils.int2ba(a_idx, length=n))
        #     a_val = self.coeffs[a_bits]
        #     assert isinstance(a_val, AlgebraicArray)

        #     for b_idx in range(2**n):
        #         b_bits = tuple(ba_utils.int2ba(b_idx, length=n))
        #         b_val = other.coeffs[b_bits]
        #         assert isinstance(b_val, AlgebraicArray)

        #         # For multilinear: product monomial is S union T (bitwise OR)
        #         result_bits = tuple(a_bits[i] | b_bits[i] for i in range(n))

        #         # Accumulate coefficient
        #         product = a_val * b_val
        #         new_coeff = result_coeffs[result_bits] + product
        #         assert isinstance(new_coeff, AlgebraicArray)

        #         is_bottom = jnp.allclose(new_coeff.data, self.algebra.zero)

        #         result_with_new_coeffs = result_coeffs.at[result_bits].set(new_coeff.data)
        #         result_with_new_coeffs = set_at(result_coeffs, result_bits, new_coeff)
        #         # don't add new_coeff if it is bottom
        #         result_coeffs = alge.select([is_bottom, ~is_bottom], [result_coeffs, result_with_new_coeffs])

        # """
        # This implementation performs the OR-convolution dimension by dimension.
        # At each iteration d, the variable x_d is contracted by combining the
        # slices where i_d in {0, 1} according to:

        #     c_0 = a_0 * b_0
        #     c_1 = a_0 * b_1 + a_1 * b_0 + a_1 * b_1

        # where a_k and b_k denote the coefficients with x_d = k. The operation is
        # local to the chosen axis and preserves the overall tensor shape.

        # Axes of B are moved within the loop to align the variable being contracted,
        # while the accumulated result maintains a fixed axis-to-variable mapping.
        # All arithmetic (addition and multiplication) is delegated to the
        # underlying scalar semiring (e.g. via quax), making the function fully
        # JIT-compilable and backend-agnostic.
        # """
        # # Use einsum to compute multilinear polynomial multiplication
        # # For 2 variables: result[i,j] = sum over k,l of self[k,l] * other[i|k, j|l]
        # # where | is bitwise OR (multilinear: x*x = x)
        # #
        # # We compute this axis-by-axis using outer products

        # n = self.num_vars

        # # Start with the outer product in all dimensions
        # # Reshape to allow broadcasting: self -> [..., 1] and other -> [1, ...]
        # result_shape = tuple([2] * n)

        # # Compute full outer product by reshaping and broadcasting
        # # self.coeffs has shape (2, 2, ..., 2) with n dimensions
        # # We want outer product, so reshape self to (2,2,...,2,1,1,...,1) and other to (1,1,...,1,2,2,...,2)
        # a_shape = result_shape + tuple([1] * n)
        # b_shape = tuple([1] * n) + result_shape

        # a_expanded = alge.reshape(self.coeffs, a_shape)
        # b_expanded = alge.reshape(other.coeffs, b_shape)

        # # Outer product: shape (2,2,...,2, 2,2,...,2) with 2n dimensions
        # outer = a_expanded * b_expanded
        # assert isinstance(outer, AlgebraicArray)
        # assert outer.shape == ((2,) * (2 * n))

        # # Now reduce using OR logic for multilinear: x_i * x_i = x_i
        # # For each variable axis, combine (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1
        # # After each iteration, one dimension is removed, so axis n is always the "other" axis
        # result_coeffs = outer
        # for axis_idx in range(n):
        #     # After i iterations, we've reduced from 2n to 2n-i dimensions
        #     # Axis axis_idx from self is still at position axis_idx
        #     # Axis axis_idx from other is now at position n (after i axes have been removed)

        #     # Get slices for this variable: (0,0), (0,1), (1,0), (1,1)
        #     num_dims = len(result_coeffs.shape)

        #     slices_00 = [slice(None)] * num_dims
        #     slices_00[axis_idx] = 0
        #     slices_00[n] = 0

        #     slices_01 = [slice(None)] * num_dims
        #     slices_01[axis_idx] = 0
        #     slices_01[n] = 1

        #     slices_10 = [slice(None)] * num_dims
        #     slices_10[axis_idx] = 1
        #     slices_10[n] = 0

        #     slices_11 = [slice(None)] * num_dims
        #     slices_11[axis_idx] = 1
        #     slices_11[n] = 1

        #     # Build reduced tensor
        #     # new[..., 0, ...] = result[..., 0, 0, ...]  (only 0|0 = 0)
        #     # new[..., 1, ...] = result[..., 0, 1, ...] + result[..., 1, 0, ...] + result[..., 1, 1, ...]
        #     c0 = result_coeffs[tuple(slices_00)]
        #     c1 = result_coeffs[tuple(slices_01)] + result_coeffs[tuple(slices_10)] + result_coeffs[tuple(slices_11)]

        #     assert isinstance(c0, AlgebraicArray)
        #     assert isinstance(c1, AlgebraicArray)
        #     result_coeffs = alge.stack([c0, c1], axis=axis_idx)
        #     assert isinstance(result_coeffs, AlgebraicArray)

        return MonomialBasis(result_coeffs)

    def evaluate(self, points: Shaped[Array, " {self.num_vars}"] | Mapping[int, Scalar]) -> MonomialBasis[K]:
        """Evaluate polynomial at the given points using Horner-like scheme."""
        # Just convert the points into a set of constant Polynomials and use compose
        map_points = dict()
        if isinstance(points, Array):
            for var_idx in range(self.num_vars):
                scalar_value = points[var_idx]
                map_points[var_idx] = jnp.asarray(scalar_value)
        else:
            assert isinstance(points, Mapping)
            for var_idx, scalar_value in points.items():
                map_points[var_idx] = jnp.asarray(scalar_value)

        return self.compose(map_points)

    @eqx.filter_jit
    def compose(
        self,
        replacements: Mapping[int, MonomialBasis[K] | Scalar],
    ) -> MonomialBasis[K]:
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

        # @quax.quaxify
        def _compose(poly: MonomialBasis[K], at: int) -> MonomialBasis[K]:
            """Recursive implementation of composition.

            - `coeffs` should be an `num_vars`-dim array.
            - `at` is an index into `repl_keys`, so we can just increment it
            """
            # NOTE: Must make sure we don't go out of bounds for `at`
            if at >= len(repl_keys):
                # Return as is we there are no more variables to substitute
                return poly
            coeffs = poly.coeffs
            var_idx = repl_keys[at]
            # Extract slices of shape: (2,) * (n-1)
            p_xi_0 = alge.take(coeffs, 0, axis=var_idx)
            p_xi_1 = alge.take(coeffs, 1, axis=var_idx)
            assert isinstance(p_xi_0, AlgebraicArray), f"{type(p_xi_0)=}"
            assert isinstance(p_xi_1, AlgebraicArray), f"{type(p_xi_1)=}"

            # Lift the cofactors back to full shape by adding axis at var_idx
            p_xi_0 = self._lift_tensor(p_xi_0, var_idx)
            p_xi_1 = self._lift_tensor(p_xi_1, var_idx)

            p_xi_0_poly: MonomialBasis[K] = dataclasses.replace(poly, coeffs=p_xi_0)
            p_xi_1_poly: MonomialBasis[K] = dataclasses.replace(poly, coeffs=p_xi_1)
            # p_xi_0_poly: MonomialBasis[K] = eqx.tree_at(lambda p: p.coeffs, poly, p_xi_0)
            # p_xi_1_poly: MonomialBasis[K] = eqx.tree_at(lambda p: p.coeffs, poly, p_xi_1)
            assert isinstance(p_xi_0_poly, MonomialBasis), f"{type(p_xi_0_poly)=}"
            assert isinstance(p_xi_1_poly, MonomialBasis), f"{type(p_xi_1_poly)=}"

            # Recursively compose each cofactor
            p_xi_0_poly = _compose(p_xi_0_poly, at + 1)
            p_xi_1_poly = _compose(p_xi_1_poly, at + 1)
            assert isinstance(p_xi_0_poly, MonomialBasis), f"{type(p_xi_0_poly)=}"
            assert isinstance(p_xi_1_poly, MonomialBasis), f"{type(p_xi_1_poly)=}"

            # merge the cofactors with the replacement in place
            # Need to multiply replacement polynomial with p_xi_1_poly, then add p_xi_0_poly
            # var_repl = replacements[var_idx].coeffs
            var_replacement = replacements[var_idx]
            # var_repl_poly = MonomialBasis(var_repl)
            prod = p_xi_1_poly * var_replacement
            result = p_xi_0_poly + prod
            return result

        return _compose(self, 0)

    def _lift_tensor(self, tensor: AlgebraicArray[K], insert_axis: int) -> AlgebraicArray[K]:
        """Lift (n-1)-dim tensor to n-dim by inserting axis."""
        # Insert axis at position insert_axis
        expanded = alge.expand_dims(tensor, axis=insert_axis)
        # assert isinstance(expanded, AlgebraicArray)

        # Pad along new axis to get shape (2,) * target_ndim
        padding = [(0, 0)] * self.num_vars
        padding[insert_axis] = (0, 1)

        return alge.pad(expanded, padding, constant_values=self.algebra.zero)

    def to_sparse(self) -> sparse_poly.SparsePolynomial[K]:
        from itertools import product

        result: dict[frozenbitarray, Scalar] = dict()
        # Enumerate all 2^n possible indices
        for idx in product([0, 1], repeat=self.num_vars):
            coeff = self.coeffs[idx]
            assert coeff.shape == ()
            # Only include non-zero coefficients
            if not alge.allclose(coeff, self.algebra.zero, rtol=1e-5):
                monomial = frozenbitarray(idx)
                result[monomial] = coeff.data  # Store the raw JAX array

        return sparse_poly.SparsePolynomial(self.algebra, self.num_vars, result)

    @classmethod
    def from_sparse(cls, poly: sparse_poly.SparsePolynomial[K]) -> MonomialBasis[K]:
        coeffs = alge.zeros((2,) * poly.num_vars, poly.algebra)
        for monomial, coeff in poly.items():
            # Convert bitarray to index tuple
            idx = tuple(int(bit) for bit in monomial)
            coeffs = coeffs.at[idx].set(coeff)
        return MonomialBasis(coeffs)


def _multiply_recursive(
    lhs: Shaped[AlgebraicArray[K], "*2"], rhs: Shaped[AlgebraicArray[K], "*2"]
) -> Shaped[AlgebraicArray[K], "*2"]:
    """A recursive function to compute the Horner's expansion multiplication."""
    assert lhs.shape == rhs.shape
    assert eqx.tree_equal(lhs.semiring, rhs.semiring)

    n = len(lhs.shape)
    return_shape = lhs.shape

    expected_shape_post_idx = (2,) * (n - 1)

    # We want to split the lhs and rhs into the cofactors of the variable at the given level
    lhs_0 = lhs[0, ...]  # 0 cofactor of LHS at x_i
    lhs_1 = lhs[1, ...]  # 1 cofactor of LHS at x_i

    rhs_0 = rhs[0, ...]  # 0 cofactor of RHS at x_i
    rhs_1 = rhs[1, ...]  # 1 cofactor of RHS at x_i

    assert lhs_0.shape == lhs_1.shape == rhs_0.shape == rhs_1.shape == expected_shape_post_idx

    # Now we will use this formula for finding the cofactors of the resultant
    # c_0 = a_0 * b_0
    # c_1 = a_0 * b_1 + a_1 * b_0 + a_1 * b_1
    if len(expected_shape_post_idx) == 0:
        # Scalars! just deal with them as single variable things
        ret_0 = lhs_0 * rhs_0
        ret_1 = (lhs_0 * rhs_1) + (lhs_1 * rhs_0) + (lhs_1 * rhs_1)
    else:
        ret_0 = _multiply_recursive(lhs_0, rhs_0)
        ret_1 = _multiply_recursive(lhs_0, rhs_1) + _multiply_recursive(lhs_1, rhs_0) + _multiply_recursive(lhs_1, rhs_1)
    assert ret_0.shape == ret_1.shape == expected_shape_post_idx
    # Put it back together
    ret: AlgebraicArray[K] = alge.stack([ret_0, ret_1], axis=0)
    assert ret.shape == return_shape
    return ret
