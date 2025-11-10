"""GPU-optimized kernel representations of algebraic structures.

This module defines AlgebraicStructure and related classes that represent
semirings, De Morgan algebras, and other structures in a form optimized
for JAX/GPU computation.

Key design: Kernels are frozen dataclasses (JAX pytrees), enabling:
- vmap over different algebras
- Dynamic algebra selection in jitted functions
- Composable algebra combinations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Protocol, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num

Axis: TypeAlias = None | int | tuple[int, ...]
Shape: TypeAlias = int | tuple[int, ...]


class _ReductionOp(Protocol):
    def __call__(self, _: Num[Array, "..."], axis: Axis = None) -> Num[Array, "..."]: ...


@dataclass(frozen=True)
class AlgebraicStructure:
    """Kernel representation of a semiring or algebraic structure.

    This dataclass encodes the essential operations of an algebraic
    structure in a form suitable for GPU computation. It supports:
    - All semirings (tropical, Boolean, counting, etc.)
    - De Morgan algebras (adds negation)
    - Lattice algebras
    - Foundational for polynomial rings

    Attributes
    ----------
    add : Callable[[Array, Array], Array]
        Semiring addition operation (oplus).
        Examples: jnp.add, jnp.maximum, jnp.minimum
    mul : Callable[[Array, Array], Array]
        Semiring multiplication operation (otimes).
        Examples: jnp.multiply, jnp.add, jnp.minimum
    zero : float | Array
        Additive identity of the semiring.
        Examples: 0.0, -inf, 1.0
    one : float | Array
        Multiplicative identity of the semiring.
        Examples: 1.0, 0.0, inf
    sum : None | Callable[[Array, int | None], Array]
        Reduction using addition (derived from add via fold).
        If None, will use jax.lax.reduce with add operation.
        Examples: jnp.sum, jnp.max, jnp.min
    prod : None | Callable[[Array, int | None], Array]
        Reduction using multiplication (derived from mul via fold).
        If None, will use jax.lax.reduce with mul operation.
        Examples: jnp.prod, jnp.min, jnp.max
    negate : Callable[[Array], Array]
        Negation operation (NOT x) for De Morgan algebras.
        Must satisfy: negate(negate(x)) = x
        Examples: lambda x: 1 - x, lambda x: -x
    properties : FrozenSet[str]
        Set of algebraic properties.
        Valid values: "idempotent_add", "idempotent_mul", "commutative",
        "simple", "distributive", "has_negation"
    """

    add: Callable[[Array, Array], Array]
    mul: Callable[[Array, Array], Array]
    zero: float | Array
    one: float | Array
    sum: _ReductionOp | None = None
    prod: _ReductionOp | None = None
    negate: Callable[[Array], Array] | None = None
    properties: FrozenSet[str] = field(default_factory=frozenset)

    def is_idempotent_add(self) -> bool:
        """Check if a oplus a = a (additive idempotence)."""
        return "idempotent_add" in self.properties

    def is_idempotent_mul(self) -> bool:
        """Check if a otimes a = a (multiplicative idempotence)."""
        return "idempotent_mul" in self.properties

    def is_commutative(self) -> bool:
        """Check if a oplus b = b oplus a and a otimes b = b otimes a."""
        return "commutative" in self.properties

    def is_simple(self) -> bool:
        """Check if structure is simple (all properties hold)."""
        return "simple" in self.properties

    def has_negation(self) -> bool:
        """Check if structure has a negation operation."""
        return self.negate is not None

    def zeros(self, shape: int | tuple[int, ...]) -> Num[Array, "..."]:
        """Create an array filled with the additive identity."""
        if isinstance(self.zero, (int, float)):
            return jnp.full(shape, self.zero)
        else:
            return jnp.full(shape, self.zero)

    def ones(self, shape: int | tuple[int, ...]) -> Num[Array, "..."]:
        """Create an array filled with the multiplicative identity."""
        if isinstance(self.one, (int, float)):
            return jnp.full(shape, self.one)
        else:
            return jnp.full(shape, self.one)

    def vdot(
        self,
        a: Num[Array, " n"],
        b: Num[Array, " n"],
    ) -> Num[Array, ""]:
        """Compute the dot product of two 1D arrays using the semiring.

        Computes: oplus_i (a_i otimes b_i)
        In LaTeX: bigoplus_{i} a_i otimes b_i
        """
        result: Array
        products: Array
        products = self.mul(a, b)
        if self.sum is not None:
            # Call sum with axis parameter if available
            result = self.sum(products, axis=None)
            return result
        else:
            result = jax.lax.reduce(products, self.zero, self.add, (0,))
            return result

    def matmul(
        self,
        a: Num[Array, "n k"],
        b: Num[Array, "k m"],
    ) -> Num[Array, "n m"]:
        """Compute matrix-semiring product of two arrays.

        Uses element-wise vdot for each row-column pair.
        Default implementation; subclasses can override for optimization.

        Formula: C[i,j] = oplus_k (A[i,k] otimes B[k,j])
        In LaTeX: C_{ij} = bigoplus_k A_{ik} otimes B_{kj}
        """
        mv = jax.vmap(self.vdot, (0, None), 0)
        mm = jax.vmap(mv, (None, 1), 1)
        return mm(a, b)

    def __repr__(self) -> str:
        """String representation of the algebra."""
        props = ", ".join(sorted(self.properties)) if self.properties else "generic"
        return f"AlgebraicStructure(zero={self.zero}, one={self.one}, properties={{{props}}})"
