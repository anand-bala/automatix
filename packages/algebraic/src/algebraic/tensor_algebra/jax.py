"""JAX-based semiring implementations.

This module contains all semiring implementations using JAX arrays and operations.
All semirings here are designed to work with jax.jit, jax.vmap, and automatic
differentiation.
"""
# mypy: disable-error-code="no-any-return"

import functools
from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Num
from typing_extensions import final, overload, override

import algebraic.kernels.jax as kernels
import algebraic.utils as utils
from algebraic.spec import (
    AlgebraicStructure,
    BiModule,
    BinaryOp,
    BooleanAlgebra,
    BoundedDistributiveLattice,
    DeMorganAlgebra,
    HeytingAlgebra,
    MatmulFn,
    MaybeAxis,
    ReductionOp,
    Ring,
    Semiring,
    Shape,
    StoneAlgebra,
    VdotFn,
)
from algebraic.spec import BoundedDistributiveLattice as Lattice

# Register all spec dataclasses as PyTrees
for dataclass_type in [
    AlgebraicStructure,
    Semiring,
    BoundedDistributiveLattice,
    Ring,
    DeMorganAlgebra,
    HeytingAlgebra,
    StoneAlgebra,
    BooleanAlgebra,
    BiModule,
]:
    jtu.register_dataclass(dataclass_type)

type JaxSemiring = Semiring[Array]


@final
class JaxBiModule[S: JaxSemiring](eqx.Module, BiModule[S]):
    """JAX wrappers for the vector interface for algebraic structures"""

    _vdot: None | VdotFn = eqx.field(default=None, kw_only=True)
    _matmul: None | MatmulFn = eqx.field(default=None, kw_only=True)

    @overload
    def zeros(self, shape: int) -> Num[Array, " {shape}"]: ...

    @overload
    def zeros(self, shape: Sequence[int]) -> Num[Array, " {*shape}"]: ...

    @override
    def zeros(self, shape: Shape) -> Num[Array, "*shape"]:
        """Return an array of given shape filled with the additive identity (zero)"""
        return jnp.full(shape, self.algebra.zero)

    @overload
    def ones(self, shape: int) -> Num[Array, " {shape}"]: ...

    @overload
    def ones(self, shape: Sequence[int]) -> Num[Array, " {*shape}"]: ...

    @override
    def ones(self, shape: Shape) -> Num[Array, "*shape"]:
        """Return an array of given shape filled with the multiplicative identity (one)"""
        return jnp.full(shape, self.algebra.one)

    @override
    def vdot(self, a: Num[Array, " n"], b: Num[Array, " n"]) -> Num[Array, ""]:
        """Compute semiring dot product of two 1D arrays."""
        if self._vdot is not None:
            return self._vdot(a, b)

        result: Array
        products: Array = self.mul(a, b)
        if self.sum is not None:
            # Call sum with axis parameter if available
            result = self.sum(products, axis=None)
        else:
            result = jax.lax.reduce(
                products,
                self.algebra.zero,
                self.add,
                (0,),
            )
        return result

    @override
    def matmul(self, a: Num[Array, "n k"], b: Num[Array, "k m"]) -> Num[Array, "n m"]:
        """Compute semiring matrix multiplication."""
        if self._matmul is not None:
            return self._matmul(a, b)
        a = jnp.asarray(a)
        b = jnp.asarray(b)

        # Create pairwise sums: shape (n, k, m)
        # a[:, :, None] has shape (n, k, 1)
        # b[None, :, :] has shape (1, k, m)
        # Broadcasting gives (n, k, m)
        elementwise_mul = self.mul(a[:, :, None], b[None, :, :])

        # Take sum over k dimension to get (n, m)
        if self.sum is not None:
            # Call sum with axis parameter if available
            result = self.sum(elementwise_mul, axis=1)
        else:
            result = jax.lax.reduce(
                elementwise_mul,
                self.algebra.zero,
                self.add,
                (1,),
            )
        return result

    @override
    def transpose(self, a: Array, axes: Sequence[int] | None = None) -> Array:
        """Transpose array axes."""
        return jnp.transpose(a, axes)

    @override
    def tensordot(
        self,
        a: Array,
        b: Array,
        axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    ) -> Array:
        """Compute semiring tensor contraction."""
        # Normalize axes specification
        if isinstance(axes, int):
            # axes=n means: contract last n axes of a with first n axes of b
            n = axes
            axes_a = tuple(range(a.ndim - n, a.ndim))  # Last n axes
            axes_b = tuple(range(0, n))  # First n axes
        else:
            axes_a_, axes_b_ = axes
            # Normalize negative indices
            axes_a = tuple(ax if ax >= 0 else a.ndim + ax for ax in axes_a_)
            axes_b = tuple(ax if ax >= 0 else b.ndim + ax for ax in axes_b_)

        if len(axes_a) != len(axes_b):
            raise ValueError("axes_a and axes_b must have same length")

        # Get shapes
        shape_a = a.shape
        shape_b = b.shape

        # Check contracted dimensions match
        for ax_a, ax_b in zip(axes_a, axes_b):
            if shape_a[ax_a] != shape_b[ax_b]:
                raise ValueError(
                    f"Contracted dimensions don't match: a.shape[{ax_a}]={shape_a[ax_a]} vs b.shape[{ax_b}]={shape_b[ax_b]}"
                )

        # Determine output dimensions
        free_a = [i for i in range(a.ndim) if i not in axes_a]
        free_b = [i for i in range(b.ndim) if i not in axes_b]

        # Rearrange a: [free_a, contracted]
        a_perm = free_a + list(axes_a)
        a_trans = jnp.transpose(a, a_perm)

        # Rearrange b: [contracted, free_b]
        b_perm = list(axes_b) + free_b
        b_trans = jnp.transpose(b, b_perm)

        # Flatten to matrix multiplication form
        # After transpose:
        # a_trans has shape [free_a dims..., contracted dims...]
        # b_trans has shape [contracted dims..., free_b dims...]

        # Reshape to 2D for matmul
        n_free_a = len(free_a)
        n_contracted = len(axes_a)
        n_free_b = len(free_b)

        # Compute flattened sizes
        size_free_a: int = 1
        if n_free_a > 0:
            for i in range(n_free_a):
                size_free_a *= a_trans.shape[i]

        size_contracted: int = 1
        if n_contracted > 0:
            for i in range(n_free_a, n_free_a + n_contracted):
                size_contracted *= a_trans.shape[i]

        size_free_b: int = 1
        if n_free_b > 0:
            for i in range(n_contracted, n_contracted + n_free_b):
                size_free_b *= b_trans.shape[i]

        # Flatten: a_trans -> (size_free_a, size_contracted)
        #          b_trans -> (size_contracted, size_free_b)
        a_flat = a_trans.reshape((size_free_a, size_contracted))
        b_flat = b_trans.reshape((size_contracted, size_free_b))

        # Perform semiring matrix multiplication
        result_flat = self.matmul(a_flat, b_flat)

        # Reshape to output
        shape_out_a = [shape_a[i] for i in free_a]
        shape_out_b = [shape_b[i] for i in free_b]
        output_shape = shape_out_a + shape_out_b

        if output_shape:
            result = result_flat.reshape(output_shape)
        else:
            # Scalar output
            result = result_flat.squeeze()

        return result

    @overload
    def einsum(self, subscript: str, *operands: Array) -> Array: ...

    @overload
    def einsum(self, *args: tuple[Array, Sequence[int]] | Sequence[int]) -> Array: ...

    def einsum[**P](self, *args: P.args, **kwargs: P.kwargs) -> Array:
        """Compute einsum using semiring operations.

        Uses opt_einsum for parsing and optimal contraction ordering, but performs
        all arithmetic operations using semiring add/mul instead of standard ops.

        Supports standard einsum notation:
            'ij,jk->ik': matrix multiplication
            'ii->i': diagonal extraction
            'ij->': sum all elements
            'i,i->': dot product
            'ijk,ikl->ijl': batched matmul

        Parameters
        ----------
        subscripts : str
            Einsum subscript string
        *operands : Array
            Input arrays

        Returns
        -------
        Array
            Result of semiring einsum operation
        """
        import opt_einsum
        from opt_einsum.parser import parse_einsum_input

        operands: list[Array]
        input_str, output_str, operands = parse_einsum_input(args)
        contraction_str = f"{input_str}->{output_str}"
        operand_shapes = [op.shape for op in operands]

        # Parse the einsum expression and get optimized contraction path
        # This returns a ContractExpression object
        expr = opt_einsum.contract_expression(contraction_str, *operand_shapes)

        # Execute the contraction path using semiring operations
        # expr.contraction_list is a list of (input_indices, rm_indices, equation, remaining_indices, do_blas)
        contraction_indices: Sequence[int]
        for contraction_indices, _rm_idx, einsum_str, *_ in expr.contraction_list:
            # Extract operands for this contraction
            contract_ops: list[Array] = [operands.pop(i) for i in contraction_indices]

            # Perform the pairwise contraction using semiring operations
            if len(contract_ops) == 1:
                # Single operand: transpose/reduction
                result = self._execute_unary(einsum_str, contract_ops[0])
            elif len(contract_ops) == 2:
                # Binary contraction: use tensordot
                result = self._execute_binary(einsum_str, contract_ops[0], contract_ops[1])
            else:
                # Shouldn't happen with pairwise contractions, but handle gracefully
                raise NotImplementedError(f"opt_einsum returned {len(contract_ops)}-way contraction; expected unary or binary")

            operands.append(result)

        # Should have exactly one operand left
        assert len(operands) == 1
        return operands[0]

    def _execute_unary(self, einsum_str: str, operand: Array) -> Array:
        """Execute a unary einsum operation (transpose/reduction).

        Parameters
        ----------
        einsum_str : str
            Einsum string like 'ijk->ik' or 'ii->i'
        operand : Array
            Input array
        """
        if "->" not in einsum_str:
            # No reduction, just return
            return operand

        input_spec, output_spec = einsum_str.split("->")

        # Find dimensions to reduce
        reduce_dims = tuple(i for i, idx in enumerate(input_spec) if idx not in output_spec)

        result = operand

        # Reduce dimensions (in reverse order to preserve indices)
        if reduce_dims:
            for dim in sorted(reduce_dims, reverse=True):
                if self.sum is not None:
                    result = self.sum(result, axis=dim)
                else:
                    result = jax.lax.reduce(result, self.algebra.zero, self.add, (dim,))

        # Transpose to match output order
        if output_spec:
            remaining_spec = "".join(idx for idx in input_spec if idx in output_spec)
            if remaining_spec != output_spec:
                perm = tuple(remaining_spec.index(idx) for idx in output_spec)
                result = self.transpose(result, perm)

        return result

    def _execute_binary(self, einsum_str: str, a: Array, b: Array) -> Array:
        """Execute a binary einsum operation using tensordot.

        Parameters
        ----------
        einsum_str : str
            Einsum string like 'ij,jk->ik' or 'ab,bc->ac'
        a, b : Array
            Input arrays
        """
        # Parse the einsum string
        parts = einsum_str.split("->")
        inputs_str = parts[0]
        output_str = parts[1] if len(parts) > 1 else ""

        spec_a, spec_b = inputs_str.split(",")

        # Find contracted indices
        contracted = set(spec_a) & set(spec_b)

        # Build axes for tensordot
        axes_a = tuple(i for i, idx in enumerate(spec_a) if idx in contracted)
        axes_b = tuple(i for i, idx in enumerate(spec_b) if idx in contracted)

        if not axes_a:
            # Pure outer product
            a_expanded = a.reshape(a.shape + (1,) * b.ndim)
            b_expanded = b.reshape((1,) * a.ndim + b.shape)
            result = self.mul(a_expanded, b_expanded)
            result_spec = spec_a + spec_b
        else:
            # Contract using tensordot
            result = self.tensordot(a, b, axes=(axes_a, axes_b))

            # Determine result spec: [free_a, free_b]
            free_a = "".join(idx for i, idx in enumerate(spec_a) if i not in axes_a)
            free_b = "".join(idx for i, idx in enumerate(spec_b) if i not in axes_b)
            result_spec = free_a + free_b

        # Handle output specification
        if output_str:
            # Sum over dimensions not in output
            reduce_dims = tuple(i for i, idx in enumerate(result_spec) if idx not in output_str)

            if reduce_dims:
                for dim in sorted(reduce_dims, reverse=True):
                    if self.sum is not None:
                        result = self.sum(result, axis=dim)
                    else:
                        result = jax.lax.reduce(result, self.algebra.zero, self.add, (dim,))

                # Update result_spec
                result_spec = "".join(idx for idx in result_spec if idx in output_str)

            # Transpose to match output order
            if result_spec != output_str:
                perm = tuple(result_spec.index(idx) for idx in output_str)
                result = self.transpose(result, perm)

        return result


def counting_semiring() -> JaxBiModule[JaxSemiring]:
    r"""Implementation of the counting semiring (R, +, *, 0, 1)."""

    def add(x1: Num[Array, "*#n"], x2: Num[Array, "*#n"]) -> Num[Array, "*#n"]:
        return x1 + x2

    def multiply(x1: Num[Array, "*#n"], x2: Num[Array, "*#n"]) -> Num[Array, "*#n"]:
        return x1 * x2

    def sum(a: Num[Array, " ..."], axis: MaybeAxis = None) -> Num[Array, " ..."]:
        return jnp.sum(a, axis=axis)

    def prod(a: Num[Array, " ..."], axis: MaybeAxis = None) -> Num[Array, " ..."]:
        return jnp.prod(a, axis=axis)

    return JaxBiModule(
        algebra=Semiring(
            add=add,
            mul=multiply,
            zero=jnp.asarray(0.0),
            one=jnp.asarray(1.0),
        ),
        sum=sum,
        prod=prod,
        _vdot=lambda a, b: jnp.vdot(a, b),
        _matmul=lambda a, b: jnp.matmul(a, b),
    )


@overload
def max_min_algebra(
    *,
    smooth: bool = False,
    only: None = None,
    temperature: float = 1.0,
) -> JaxBiModule[DeMorganAlgebra]: ...


@overload
def max_min_algebra(
    *,
    smooth: bool,
    only: Literal["negative", "positive"],
    temperature: float,
) -> JaxBiModule[Lattice]: ...


def max_min_algebra(
    *,
    smooth: bool = False,
    only: None | Literal["negative", "positive"] = None,
    temperature: float = 1.0,
) -> JaxBiModule[Lattice] | JaxBiModule[DeMorganAlgebra]:
    """Implementation of the min-max semiring on reals (R cup {-inf, inf}, max, min, -inf, inf).

    Parameters
    ----------
    smooth : bool
        If `True`, use the logsumexp approximation of max and min.
    only : "negative", "positive", None (default)
        Restrict the semiring to either the negative or positive extended reals. If
        `None`, returns a full complemented max-min algebra (with negation).
    temperature : float, default 1.0
        Temperature closer to infinity is closer to true max/min

    """
    add_kernel: BinaryOp
    sum_kernel: ReductionOp
    mul_kernel: BinaryOp
    prod_kernel: ReductionOp

    if smooth:
        add_kernel = utils.wrap_binary_op(kernels.smooth_maximum, temperature=temperature)
        sum_kernel = utils.wrap_reduction_op(kernels.smooth_max, temperature=temperature)
        mul_kernel = utils.wrap_binary_op(kernels.smooth_minimum, temperature=temperature)
        prod_kernel = utils.wrap_reduction_op(kernels.smooth_min, temperature=temperature)
    else:
        add_kernel = utils.wrap_binary_op(jnp.maximum)
        sum_kernel = jnp.max
        mul_kernel = utils.wrap_binary_op(jnp.minimum)
        prod_kernel = jnp.min

    zero = jnp.asarray(0.0 if only == "positive" else -jnp.inf)
    one = jnp.asarray(-0.0 if only == "negative" else jnp.inf)

    def add(x1: Num[Array, "*#n"], x2: Num[Array, "*#n"]) -> Num[Array, "*#n"]:
        return add_kernel(x1, x2)

    def sum(a: Num[Array, " ..."], axis: MaybeAxis = None) -> Num[Array, " ..."]:
        return sum_kernel(a, axis)

    def multiply(x1: Num[Array, "*#n"], x2: Num[Array, "*#n"]) -> Num[Array, "*#n"]:
        return mul_kernel(x1, x2)

    def prod(a: Num[Array, " ..."], axis: MaybeAxis = None) -> Num[Array, " ..."]:
        return prod_kernel(a, axis)

    def complement(x: Num[Array, " ..."]) -> Num[Array, " ..."]:
        return -x

    algebra: DeMorganAlgebra | Lattice
    if only is None:
        # We can return complemented algebra
        algebra = DeMorganAlgebra(
            add=jnp.maximum,
            mul=jnp.minimum,
            zero=zero,
            one=one,
            complement=complement,
        )
    else:
        algebra = Lattice(
            add=jnp.maximum,
            mul=jnp.minimum,
            zero=zero,
            one=one,
        )
    return JaxBiModule(
        algebra=algebra,
        sum=sum,
        prod=prod,
        _vdot=None,
        _matmul=None,
    )


def tropical_semiring(*, minplus: bool = True, smooth: bool = False, temperature: float = 1.0) -> JaxBiModule[JaxSemiring]:
    """The min-plus tropical semiring

    The choice of `minplus` determines if the output is the min-plus semiring (R_>=0 cup
    {-inf, inf}, min, +, inf, 0) or the max-plus tropical semiring (R_<=0 cup {-inf,
    inf}, max, +, -inf, 0).

    Parameters
    ----------
    minplus: bool
        If `True`, returns the min-plus tropical semiring. Else, the maxplus semiring.
    smooth : bool
        If `True`, use the logsumexp approximation of max and min.
    only : "negative", "positive", None (default)
        Restrict the semiring to either the negative or positive extended reals. If
        `None`, returns a full complemented max-min algebra (with negation).
    temperature : float, default 1.0
        Temperature for the smooth approximation; closer to infinity is closer to true max/min
    """
    add_kernel: BinaryOp
    sum_kernel: ReductionOp
    if smooth:
        if minplus:
            add_kernel = utils.wrap_binary_op(kernels.smooth_minimum, temperature=temperature)
            sum_kernel = functools.partial(kernels.smooth_min, temperature=temperature)
        else:
            add_kernel = utils.wrap_binary_op(kernels.smooth_maximum, temperature=temperature)
            sum_kernel = functools.partial(kernels.smooth_max, temperature=temperature)
    else:
        if minplus:
            add_kernel = utils.wrap_binary_op(jnp.minimum)
            sum_kernel = jnp.min
        else:
            add_kernel = utils.wrap_binary_op(jnp.maximum)
            sum_kernel = jnp.max

    if minplus:
        zero = jnp.asarray(jnp.inf)
        one = jnp.asarray(0.0)
    else:
        zero = jnp.asarray(-jnp.inf)
        one = jnp.asarray(-0.0)

    def add(x1: Num[Array, "*#n"], x2: Num[Array, "*#n"]) -> Num[Array, "*#n"]:
        return add_kernel(x1, x2)

    def sum(a: Num[Array, " ..."], axis: MaybeAxis = None) -> Num[Array, " ..."]:
        return sum_kernel(a, axis)

    def multiply(x1: Num[Array, "*#n"], x2: Num[Array, "*#n"]) -> Num[Array, "*#n"]:
        return x1 + x2

    def prod(a: Num[Array, " ..."], axis: MaybeAxis = None) -> Num[Array, " ..."]:
        return jnp.sum(a, axis=axis)

    return JaxBiModule(
        algebra=Semiring(
            add=add,
            mul=multiply,
            zero=zero,
            one=one,
            properties={"idempotent_add", "commutative", "simple"},
        ),
        sum=sum,
        prod=prod,
        _vdot=None,
        _matmul=None,
    )


def boolean_algebra(
    mode: Literal["logic", "soft", "smooth", "ste"] = "soft",
    temperature: float = 1.0,
) -> JaxBiModule[BooleanAlgebra]:
    """Create a differentiable Boolean kernel.

    Parameters
    ----------
    mode : {"logic", "soft", "smooth", "ste"}
        Differentiation mode:
        - "logic": non-differentiable
        - "soft": Soft Boolean using multiplication and addition (fastest, smoothest)
        - "smooth": Smooth Boolean using sigmoid with temperature
        - "ste": Straight-Through Estimator (biased gradients, but works generally)
    temperature : float, optional
        Temperature parameter for "smooth" mode (default: 1.0)


    Notes
    -----
    The differentiable modes work best with inputs in [0,1] closer to the boundaries.
    """

    zero = jnp.asarray(0.0)
    one = jnp.asarray(1.0)

    # TODO: Need to add the reduction version of these
    sum = None
    prod = None
    vdot = None
    matmul = None
    match mode:
        case "logic":
            add = utils.wrap_binary_op(jnp.logical_or)
            mul = utils.wrap_binary_op(jnp.logical_and)
            neg = utils.wrap_unary_op(jnp.logical_not)
        case "soft":
            add = utils.wrap_binary_op(kernels.soft_boolean_or)
            mul = utils.wrap_binary_op(kernels.soft_boolean_and)
            neg = kernels.soft_boolean_not
        case "smooth":
            add = utils.wrap_binary_op(kernels.smooth_boolean_or, temperature=temperature)
            mul = utils.wrap_binary_op(kernels.smooth_boolean_and, temperature=temperature)
            neg = functools.partial(kernels.smooth_boolean_not, temperature=temperature)
        case "ste":
            add = utils.wrap_binary_op(jnp.max)
            mul = utils.wrap_binary_op(jnp.min)
            neg = utils.wrap_unary_op(lambda x: 1 - x)
        case _:
            raise ValueError(f"Unknown mode: {mode}. Use 'logic', 'soft', 'smooth', or 'ste'.")
    return JaxBiModule(
        algebra=BooleanAlgebra(
            zero=zero,
            one=one,
            add=add,
            mul=mul,
            complement=neg,
        ),
        prod=prod,
        sum=sum,
        _vdot=vdot,
        _matmul=matmul,
    )
