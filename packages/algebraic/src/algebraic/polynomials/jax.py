"""JAX-based dense tensor polynomial representations."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
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

    num_vars: int = eqx.field(metadata=dict(static=True))
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
                map_points[var_idx] = points[var_idx].item()
        else:
            assert isinstance(points, Mapping)
            map_points.update(points)

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
            p_xi_0 = jnp.take(poly.coeffs, 0, axis=var_idx)  #
            p_xi_1 = jnp.take(poly.coeffs, 1, axis=var_idx)

            # Lift the cofactors back to full shape by adding axis at var_idx
            p_xi_0 = self._lift_tensor(p_xi_0, var_idx)
            p_xi_1 = self._lift_tensor(p_xi_1, var_idx)

            # Recursively compose each cofactor
            p_xi_0_poly = _compose(p_xi_0, at + 1)
            p_xi_1_poly = _compose(p_xi_1, at + 1)

            # merge the cofactors with the replacement in place
            return MonomialBasis(self.module.add(p_xi_0_poly, self.module.mul(var_repl, p_xi_1_poly)))

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

    def to_sparse_algebra(self) -> SparsePolynomialAlgebra[Array, K]:
        """Convert to a corresponding SparsePolynomialAlgebra"""
        return SparsePolynomialAlgebra(algebra=self.module.algebra, degree=self.num_vars)

    def to_sparse(self, poly: MonomialBasis) -> SparsePolynomial[Array]:
        """Convert to sparse representation."""
        from bitarray import frozenbitarray

        zero = self.module.zeros(())
        result = {
            frozenbitarray(idx): poly.coeffs[idx]
            # Iterate over all indices in the polynomial
            for idx in jnp.argwhere(
                # where the coefficient value is **not close** to the additive identity
                jnp.logical_not(jnp.isclose(poly.coeffs, zero, atol=1e-10))
            )
        }

        return SparsePolynomial(result)

    def from_sparse[S](
        self,
        poly: SparsePolynomial[S],
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
    """Tensor rank decomposition of multilinear polynomial."""

    factors: Array
    """The factors are an array of shape `(n+1,) * degree`, where `n` is the number of variables in the polynomial.
    """

    @property
    def degree(self) -> int:
        return self.factors.ndim

    @property
    def num_vars(self) -> int:
        return self.factors.shape[0] - 1


class RankDecompositionAlgebra[K: Lattice](eqx.Module):
    num_vars: int = eqx.field(metadata=dict(static=True))
    """Number of variables/indeterminants in this multilinear polynomial"""
    module: JaxBiModule[K]
    """The underlying algebraic module to define the tensor algebra over"""

    def variable(self, i: int, coefficient: None | Scalar = None) -> RankDecomposition:
        """Create a polynomial representing a single variable x_i"""
        n = self.num_vars
        coefficient = coefficient if coefficient is not None else self.module.algebra.one
        factor = (
            # Initialize the order-1 tensor (degree=1) with 0
            self.module.zeros((n + 1,))
            # Set the position for the variable to the coefficient if available.
            # We do i+1 as 0 is constant, and the i-th variable should be indexed by
            # 1 in this representation.
            .at[i + 1]
            .set(coefficient)
        )
        return RankDecomposition(factor)

    def constant(self, value: Scalar) -> RankDecomposition:
        """Create a polynomial representing a single variable x_i"""
        n = self.num_vars
        factor = (
            # Initialize the order-1 tensor (degree=1) with 0
            self.module.zeros((n + 1,))
            # Set the position for the constant term (all 0's) to the scalar value
            .at[0]
            .set(value)
        )
        return RankDecomposition(factor)

    @property
    def zero(self) -> RankDecomposition:
        return self.constant(self.module.algebra.zero)

    @property
    def one(self) -> RankDecomposition:
        return self.constant(self.module.algebra.one)

    def _pad_to_degree(self, poly: RankDecomposition, target_degree: int) -> RankDecomposition:
        """
        Pad a polynomial tensor to a higher degree.
        """
        if poly.degree == target_degree:
            return poly

        num_new_dims = target_degree - poly.degree

        # Add singleton dimensions at the end
        new_shape = poly.factors.shape + (1,) * num_new_dims
        expanded = jnp.reshape(poly.factors, new_shape)

        # Build pad_width: no padding for existing dims, pad new dims to (n_vars+1)
        pad_width = [(0, 0)] * poly.degree + [(0, self.num_vars)] * num_new_dims

        return RankDecomposition(jnp.pad(expanded, pad_width, mode="constant", constant_values=self.module.algebra.zero))

    def add(self, a: RankDecomposition, b: RankDecomposition) -> RankDecomposition:
        """Add two polynomials by adding the monomial coefficients for identical terms."""
        assert a.num_vars == self.num_vars
        assert b.num_vars == self.num_vars
        # Make sure both the polynomials have the same order/degree
        degree = max(a.degree, b.degree)
        a = self._pad_to_degree(a, degree)
        b = self._pad_to_degree(b, degree)

        factors = self.module.add(a.factors, b.factors)
        return RankDecomposition(factors)

    def mul(self, a: RankDecomposition, b: RankDecomposition) -> RankDecomposition:
        """
        Multiply two polynomial tensors.

        Parameters
        ----------
        a : Array of shape (n+1,) * d1
        b : Array of shape (n+1,) * d2

        Returns
        -------
        Product polynomial of shape (n+1,) * (d1 + d2)
        """
        d1 = a.factors.ndim
        d2 = b.factors.ndim

        # The outer product gives us shape (n+1,) * (d1 + d2)
        result = self.module.tensordot(a.factors, b.factors, axes=0)
        assert result.shape == (self.num_vars + 1,) * (d1 + d2)

        # TODO(anand): We need to handle idempotent simplification
        # For example, x_i * x_i = x_i means merging duplicate variable indices

        return RankDecomposition(result)

    def evaluate(self, poly: RankDecomposition, point: Array | Mapping[int, Array]) -> Array:
        """Evaluate polynomial at a point using Horner-like scheme."""
        # Convert to dense array if needed
        if isinstance(point, Mapping):
            point_dense = jnp.zeros(self.num_vars)
            for idx, val in point.items():
                point_dense = point_dense.at[idx].set(val)
            point = point_dense

        # Horner's method from last variable to first
        result = poly.coeffs
        for var_idx in reversed(range(self.num_vars)):
            # result = result[..., 0] + x[var_idx] * result[..., 1]
            part_0 = jnp.take(result, 0, axis=var_idx)
            part_1 = jnp.take(result, 1, axis=var_idx)
            result = self.module.add(
                part_0,
                self.module.mul(point[var_idx], part_1),
            )

        return result

    def compose(
        self,
        poly: RankDecomposition,
        replacements: Shaped[Array, "*{self.num_vars + 1}"] | Mapping[int, RankDecomposition],
    ) -> RankDecomposition:
        """Compose polynomial with multiple substitutions.

        Compose polynomial p with substitution polynomials g_0, g_1, ..., g_n.

        Computes h(y) = p(g_1(y), g_2(y), ..., g_n(y)) where we also include
        g_0 as the constant polynomial (identity for the constant basis element).

        Computes h(y) = p(g_1(y), g_2(y), ..., g_n(y)) where:
        - P[i_1,...,i_d] is the coefficient of x_{i_1} * ... * x_{i_d} in p
        - G[j, a_1,...,a_d] is the coefficient of y_{a_1} * ... * y_{a_d} in g_j

        The composition formula is:
        H[a_1^(1),...,a_d^(1), ..., a_1^(d),...,a_d^(d)] =
            sum_{i_1,...,i_d} P[i_1,...,i_d] * G[i_1,a_1^(1),...,a_d^(1)] * ... * G[i_d,a_1^(d),...,a_d^(d)]

        Note
        ----
        The composition should be performed simultaneously. If not, this is a bug.
        """
        if isinstance(replacements, Mapping):
            degree = max(poly.degree, max(repl_poly.degree for repl_poly in replacements.values()))

            # Turn the replacements into a list, with index 0 being that for a constant and
            # index n being that of variable n-1.
            # Each polynomial should be padded to the same degree
            repl_list = jnp.empty((self.num_vars + 1,) * (degree + 1))
            repl_list = repl_list.at[0].set(self._pad_to_degree(self.one, degree))
            for i in range(self.num_vars):
                if i in replacements:
                    repl_list = repl_list.at[i + 1].set(self._pad_to_degree(replacements[i], degree))
                else:
                    repl_list = repl_list.at[i + 1].set(self._pad_to_degree(self.variable(i), degree))
        else:
            degree = len(replacements.shape) - 1
            assert replacements.shape == (self.num_vars + 1,) * (degree + 1)
            repl_list = replacements

        # pad the input poly
        poly = self._pad_to_degree(poly, degree)
        # Build the einsum specification programmatically
        # We'll use integer indices to track dimensions

        # poly has indices i_1, ..., i_d (these will be contracted)
        # We'll use integers 0, 1, ..., d-1 for these contraction indices
        p = poly.factors
        p_idx = list(range(degree))

        # We need d copies of G, one for each mode of P
        # Each G copy has indices [i_k, α_1^(k), ..., α_d^(k)]
        # where i_k contracts with P, and the α indices become output dimensions

        # Start building the einsum operands list
        # Format: [array1, indices1, array2, indices2, ..., output_indices]
        einsum_args = [p, p_idx]

        # Counter for assigning fresh output indices
        # We've used 0 to d-1 for contraction indices
        next_index = degree

        # Track all output indices in order
        output_indices = []

        # Add d copies of G to the einsum
        for k in range(degree):
            # This copy of G contracts on dimension i_k (which is index k in P_indices)
            contraction_index = k

            # This copy of G contributes d new output dimensions
            g_output_indices = list(range(next_index, next_index + degree))
            next_index += degree

            # G's full index list: [contraction_index, *g_output_indices]
            g_indices = [contraction_index] + g_output_indices

            # Add this copy of G to einsum specification
            einsum_args.extend([repl_list, g_indices])

            # Add this copy's output indices to the overall output
            output_indices.extend(g_output_indices)

        # Add the output specification at the end
        einsum_args.append(output_indices)

        # Perform the contraction
        result = self.module.einsum(*einsum_args)

        return result

    def _index_to_bits(self, index: int) -> tuple[int, ...]:
        """Convert flat index to n-bit tuple."""
        from bitarray.util import int2ba

        return tuple(int2ba(index, length=self.num_vars))

    def to_sparse_algebra(self) -> SparsePolynomialAlgebra[Array, K]:
        """Convert to a corresponding SparsePolynomialAlgebra"""
        return SparsePolynomialAlgebra(algebra=self.module.algebra, degree=self.num_vars)

    def to_sparse(self, poly: RankDecomposition) -> SparsePolynomial[Array]:
        """Convert to sparse representation."""
        from bitarray import bitarray, frozenbitarray

        zero = self.module.zeros(())
        one = self.module.ones(())

        # where the coefficient value is **not close** to the additive identity
        non_zero_indices = jnp.argwhere(jnp.logical_not(jnp.isclose(poly.factors, zero, atol=1e-10)))

        result = dict[frozenbitarray, Scalar]()
        # We will have to convert each index into a monomial and perform addition separately.
        for idx in non_zero_indices:
            assert len(idx) == poly.degree
            coefficient = poly.factors[*idx]
            monomial = bitarray(self.num_vars)
            monomial[[i - 1 for i in idx if i != 0]] = 1
            monomial = frozenbitarray(monomial)
            result[monomial] = self.module.mul(result.get(monomial, one), coefficient)

        return SparsePolynomial(result)

    def from_sparse[S](
        self,
        poly: SparsePolynomial[S],
    ) -> RankDecomposition:
        """Convert from sparse representation.

        Note
        ----
        Assumes that the user is passing a `SparsePolynomial` in the same domain as the current `RankDecompositionAlgebra`.
        """
        ret = self.zero
        for monomial, coeff in poly.items():
            if len(monomial) != self.num_vars:
                raise ValueError(
                    f"Cannot convert sparse polynomial with {len(monomial)} variables to monomial basis with {self.num_vars} variables"
                )
            if monomial.count() == 0:
                # Constant term
                ret = self.add(ret, self.constant(coeff))
            else:
                # Get the indices with 1 and add 1 to them
                idx = sorted((i + 1 for i, v in enumerate(monomial) if v))
                coeffs = [coeff] + ([None] * (len(idx) - 1))
                # And then do a product of them
                ret = self.add(
                    ret,
                    functools.reduce(
                        self.mul,
                        (self.variable(var, c) for var, c in zip(idx, coeffs)),
                    ),
                )
        return ret
