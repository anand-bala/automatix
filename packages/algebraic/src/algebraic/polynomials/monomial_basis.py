"""JAX-based dense tensor polynomial representations."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

import operator
import typing
from collections.abc import Mapping

import bitarray.util as ba_utils
import equinox as eqx
import jax.lax
import jax.numpy as jnp
import quax
from jaxtyping import Array, Scalar, Shaped

import algebraic.array.core as alge
from algebraic.array.core import AlgebraicArray
from algebraic.spec import BoundedDistributiveLattice as Lattice


class MonomialBasis[K: Lattice](eqx.Module):
    """Dense, monomial basis decomposition of a multilinear polynomial

    This class represents the coefficients of a multilinear polynomial as a tensor of
    shape `(2,) * n`, where `n` is the maximum degree of the polynomial.
    """

    coeffs: Shaped[AlgebraicArray[K], "*2"]
    algebra: K

    def __init__(self, coeffs: Shaped[AlgebraicArray[K], "*2"]) -> None:
        self.coeffs = coeffs
        self.algebra = self.coeffs.semiring

    def __check_init__(self) -> None:
        if not isinstance(self.coeffs.semiring, Lattice):
            raise TypeError("Multilinear polynomial representation is only supported over BoundedDistributiveLattice algebras")

    @staticmethod
    def variable(index: int, num_vars: int, algebra: K) -> MonomialBasis:
        idx = jnp.zeros(num_vars, dtype=jnp.int32).at[index].set(1)
        coeffs = alge.zeros((2,) * num_vars, algebra).at[*idx].set(algebra.one)
        return MonomialBasis(coeffs)

    @staticmethod
    def constant(value: Scalar, num_vars: int, algebra: K) -> MonomialBasis:
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
            other = MonomialBasis(AlgebraicArray(other, self.algebra))  # type: ignore[arg-type, assignment]
        assert isinstance(other, MonomialBasis)
        coeffs = quax.quaxify(operator.add)(self.coeffs, other.coeffs)
        return MonomialBasis[K](coeffs)

    def __mul__(self, other: MonomialBasis[K] | Scalar) -> MonomialBasis[K]:
        r"""Multiply two polynomials.

        c_k = sum_{i OR j = k} A_i * B_j

        """
        if jnp.isscalar(other):
            other = MonomialBasis(AlgebraicArray(other, self.algebra))  # type: ignore[arg-type, assignment]
        assert isinstance(other, MonomialBasis)
        # Check if either is scalar: easy case
        if self.num_vars == 0 or other.num_vars == 0:
            coeffs = quax.quaxify(operator.mul)(self.coeffs, other.coeffs)
            return MonomialBasis(coeffs)
        # Now we deal with the case where there are variables
        if self.num_vars != other.num_vars:
            raise ValueError(
                "Multiplying two polynomials with unequal number of variables not supported unless one of them is a scalar/constant polynomial. Pad the polynomial representation to indicate the correct number of variables."
            )

        allclose = quax.quaxify(jnp.allclose)

        n = self.num_vars
        result_coeffs = alge.zeros((2,) * n, self.algebra)
        for a_idx in range(2**n):
            a_bits = tuple(ba_utils.int2ba(a_idx, length=n))
            a_val = self.coeffs[a_bits]
            assert isinstance(a_val, AlgebraicArray)

            for b_idx in range(2**n):
                b_bits = tuple(ba_utils.int2ba(b_idx, length=n))
                b_val = other.coeffs[b_bits]
                assert isinstance(b_val, AlgebraicArray)

                # For multilinear: product monomial is S union T (bitwise OR)
                result_bits = tuple(a_bits[i] | b_bits[i] for i in range(n))

                # Accumulate coefficient
                product = a_val * b_val
                new_coeff = result_coeffs[result_bits] + product

                is_bottom = jnp.allclose(new_coeff.data, self.algebra.zero)
                result_with_new_coeffs = result_coeffs.at[result_bits].set(new_coeff.data)
                # don't add new_coeff if it is bottom
                result_coeffs = quax.quaxify(jnp.select)([is_bottom, ~is_bottom], [result_coeffs, result_with_new_coeffs])

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
        # moveaxis = quax.quaxify(jnp.moveaxis)
        # stack = quax.quaxify(jnp.stack)

        # n = self.num_vars
        # result_coeffs = self.coeffs
        # for axis in range(n):
        #     # Move active axis to the front
        #     a = moveaxis(result_coeffs, axis, 0)  # type: ignore[arg-type]
        #     b = moveaxis(other.coeffs, axis, 0)  # type: ignore[arg-type]

        #     a0, a1 = a[0], a[1]
        #     b0, b1 = b[0], b[1]
        #     c = stack(
        #         (
        #             a0 * b0,
        #             a0 * b1 + a1 * b0 + a1 * b1,
        #         ),
        #         axis=0,
        #     )
        #     result_coeffs = moveaxis(c, 0, axis)  # type: ignore[assignment]

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
        replacements: Mapping[int, MonomialBasis | Scalar],
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

        @quax.quaxify
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
            p_xi_0 = jnp.take(coeffs, 0, axis=var_idx)  # type: ignore[arg-type]
            p_xi_1 = jnp.take(coeffs, 1, axis=var_idx)  # type: ignore[arg-type]

            # Lift the cofactors back to full shape by adding axis at var_idx
            p_xi_0 = self._lift_tensor(p_xi_0, var_idx)  # type: ignore[arg-type,assignment]
            p_xi_1 = self._lift_tensor(p_xi_1, var_idx)  # type: ignore[arg-type,assignment]

            p_xi_0_poly = eqx.tree_at(lambda p: p.coeffs, poly, p_xi_0)
            p_xi_1_poly = eqx.tree_at(lambda p: p.coeffs, poly, p_xi_1)

            # Recursively compose each cofactor
            p_xi_0_poly = _compose(p_xi_0_poly, at + 1)  # type: ignore[arg-type]
            p_xi_1_poly = _compose(p_xi_1_poly, at + 1)  # type: ignore[arg-type]

            # merge the cofactors with the replacement in place
            # Need to multiply replacement polynomial with p_xi_1_poly, then add p_xi_0_poly
            # var_repl = replacements[var_idx].coeffs
            var_replacement = replacements[var_idx]
            # var_repl_poly = MonomialBasis(var_repl)
            prod = p_xi_1_poly.__mul__(var_replacement)
            result = p_xi_0_poly + prod
            return result

        return _compose(self, 0)

    def _lift_tensor(self, tensor: AlgebraicArray[K], insert_axis: int) -> AlgebraicArray[K]:
        """Lift (n-1)-dim tensor to n-dim by inserting axis."""
        jnp_expand_dims = quax.quaxify(jnp.expand_dims)
        jnp_pad = quax.quaxify(jnp.pad)
        # Insert axis at position insert_axis
        expanded = jnp_expand_dims(tensor, axis=insert_axis)  # type: ignore[arg-type]
        # assert isinstance(expanded, AlgebraicArray)

        # Pad along new axis to get shape (2,) * target_ndim
        padding = [(0, 0)] * self.num_vars
        padding[insert_axis] = (0, 1)

        return jnp_pad(expanded, padding, constant_values=self.algebra.zero)  # type: ignore[return-value]
