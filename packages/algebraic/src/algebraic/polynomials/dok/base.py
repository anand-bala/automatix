"""Sparse polynomial representation using dictionary-based storage."""

# NOTE: Do NOT add `from __future__ import annotations` to this module.
# The `AbstractVar` annotations from `algebraic._better_abc` are processed at class
# definition time and must not be stringified (PEP 563 lazy evaluation would break them).

import copy
import functools
import typing
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping

import bitarray.util as ba_util
from bitarray import bitarray, frozenbitarray
from jaxtyping import Shaped
from typing_extensions import Self

import algebraic.ops as alge
from algebraic._backend_registry import BackendClassRegistry
from algebraic._better_abc import AbstractClassVar, AbstractVar, BetterABCMeta
from algebraic.array import AlgebraicArray
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Array, Backend, Scalar, is_array


class PolyDict(metaclass=BetterABCMeta):
    """Sparse polynomial represented as monomial -> coefficient mapping."""

    algebra: AbstractVar[Lattice]
    num_vars: AbstractVar[int]
    data: AbstractVar[Mapping[frozenbitarray, AlgebraicArray]]
    backend: AbstractClassVar[str | Backend]
    """The specific backend from the derived class"""

    def __getitem__(self, monomial: bitarray | str | Iterable[int]) -> AlgebraicArray:
        """Return the coefficient of the monomial with the given binary powers."""
        return self.data[frozenbitarray(monomial)]

    def __iter__(self) -> Iterator[frozenbitarray]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: frozenbitarray | str | int) -> bool:
        match key:
            case str():
                key = frozenbitarray(key)
            case int():
                key = frozenbitarray(ba_util.int2ba(key, length=self.num_vars))
        return key in self.data

    def get(self, key: frozenbitarray | str | int, default: AlgebraicArray | None = None, /) -> AlgebraicArray | None:
        match key:
            case str():
                key = frozenbitarray(key)
            case int():
                key = frozenbitarray(ba_util.int2ba(key, length=self.num_vars))
        return self.data.get(key, default)

    def keys(self) -> typing.KeysView[frozenbitarray]:
        return self.data.keys()

    def values(self) -> typing.ValuesView[AlgebraicArray]:
        return self.data.values()

    def items(self) -> typing.ItemsView[frozenbitarray, AlgebraicArray]:
        return self.data.items()

    @classmethod
    def _get_backend(cls, backend: str | Backend | None) -> str | Backend:
        """Resolve backend: use explicit arg, class default, or fall back to NumPy."""
        if backend is not None:
            return backend
        try:
            return cls.backend
        except AttributeError:
            return Backend.NUMPY

    @classmethod
    def constant(cls, value: Scalar, num_vars: int, *, algebra: Lattice, backend: str | Backend | None = None) -> "PolyDict":
        """Create a constant polynomial with the given *value*.

        Parameters
        ----------
        value : Scalar
            Coefficient for the constant term.
        num_vars : int
            Number of variables in the polynomial.
        algebra : BoundedDistributiveLattice
            Lattice algebra governing operations.
        backend : str or Backend or None, optional
            Backend to use. Falls back to the class default or NumPy.

        Returns
        -------
        PolyDict
            A constant polynomial.

        Examples
        --------
        >>> import numpy as np
        >>> from algebraic.semirings import boolean_algebra
        >>> from algebraic.polynomials import PolyDict
        >>> alg = boolean_algebra('ste')
        >>> p = PolyDict.constant(np.array(True), num_vars=3, algebra=alg, backend="numpy")
        >>> p['000'].data
        array(True)
        """
        zeros_idx = frozenbitarray(ba_util.zeros(num_vars))
        backend = cls._get_backend(backend)
        coeff = alge.array(value, semiring=algebra, backend=backend)
        return _make_poly_dict(algebra, num_vars, {zeros_idx: coeff}, backend=backend)

    @classmethod
    def zero(cls, num_vars: int, *, algebra: Lattice, backend: str | Backend) -> "PolyDict":
        return cls.constant(algebra.zero, num_vars, algebra=algebra, backend=backend)

    @classmethod
    def one(cls, num_vars: int, *, algebra: Lattice, backend: str | Backend) -> "PolyDict":
        return cls.constant(algebra.one, num_vars, algebra=algebra, backend=backend)

    @classmethod
    def variable(cls, index: int, num_vars: int, *, algebra: Lattice, backend: str | Backend | None = None) -> "PolyDict":
        r"""Create a polynomial representing a single variable :math:`x_i`.

        Parameters
        ----------
        index : int
            Variable index (0-based).
        num_vars : int
            Total number of variables.
        algebra : BoundedDistributiveLattice
            Lattice algebra governing operations.
        backend : str or Backend or None, optional
            Backend to use. Falls back to the class default or NumPy.

        Returns
        -------
        PolyDict
            A polynomial with a single monomial ``x_index``.
        """
        monomial = ba_util.zeros(num_vars)
        monomial[index] = 1
        backend = cls._get_backend(backend)
        coefficient = alge.ones((), semiring=algebra, backend=backend)
        return _make_poly_dict(algebra, num_vars, {frozenbitarray(monomial): coefficient}, backend)

    def _replace_attr(self, name: str, value: object) -> Self:
        """Create a new instance with one attribute changed (backend-specific)."""
        clone = copy.copy(self)
        object.__setattr__(clone, name, value)
        return clone

    def _wrap(self, data: Mapping[frozenbitarray, AlgebraicArray]) -> Self:
        return self._replace_attr("data", data)

    def __add__(self, other: Self | Scalar) -> Self:
        # This will essentially merge the two polynomials by adding the monomial coefficients where they are common, or using the additive identity where one isn't available.
        other_data: Mapping[frozenbitarray, AlgebraicArray | Scalar]
        if isinstance(other, PolyDict):
            assert self.algebra == other.algebra
            assert self.num_vars == other.num_vars
            # TODO: maybe this is too strong...
            assert self.backend == other.backend
            other_data = other.data
        else:
            # Make a dict out of it
            other_data = {frozenbitarray(self.num_vars): other}

        identity = alge.zeros((), semiring=self.algebra, backend=self.backend)

        ret = {
            key: (self.data.get(key, identity) + other_data.get(key, identity)) for key in self.data.keys() | other_data.keys()
        }
        return self._wrap(ret)

    def __mul__(self, other: Self | Scalar) -> Self:
        r"""Multiply two polynomials.

        $(\sum_{S \in a} c_S x^S) \cdot (\sum_{T \in b} d_T x^T) = sum_{S,T} (c_S * d_T) x^{S \cup T}$
        """
        # if jnp.isscalar(other):
        #     other = PolyDict.constant(typing.cast(Scalar, other), self.num_vars, self.algebra)
        other_data: Mapping[frozenbitarray, AlgebraicArray | Scalar]
        if isinstance(other, PolyDict):
            assert self.algebra == other.algebra
            assert self.num_vars == other.num_vars
            # TODO: maybe this is too strong...
            assert self.backend == other.backend
            other_data = other.data
        else:
            # Make a dict out of it
            other_data = {frozenbitarray(self.num_vars): other}

        zero = alge.zeros((), semiring=self.algebra, backend=self.backend)

        ret: defaultdict[frozenbitarray, AlgebraicArray] = defaultdict(lambda: zero)  # initialize with additive identity
        for m_a, c_a in self.data.items():
            for m_b, c_b in other_data.items():
                new_monom = frozenbitarray(m_a | m_b)
                new_coeff = c_a * c_b
                ret[new_monom] = ret[new_monom] + new_coeff
        return self._wrap(ret)

    def evaluate(self, point: Shaped[Array, " {self.num_vars}"] | Mapping[int, Scalar]) -> Self:
        """Evaluate polynomial at a point.

        Parameters
        ----------
        point : Array or Mapping[int, Scalar]
            Either an array of shape ``(num_vars,)`` or a dict mapping
            variable indices to scalar values.

        Returns
        -------
        PolyDict
            The evaluated (constant) polynomial.

        Examples
        --------
        >>> import numpy as np
        >>> from algebraic.semirings import boolean_algebra
        >>> from algebraic.polynomials import PolyDict
        >>> num_vars = 2
        >>> algebra = boolean_algebra(mode='logic')
        >>> x_0 = PolyDict.variable(0, num_vars, algebra=algebra, backend="numpy")
        >>> x_1 = PolyDict.variable(1, num_vars, algebra=algebra, backend="numpy")
        >>> p = x_0 * x_1  # x_0 AND x_1
        >>> e1 = p.evaluate(dict(enumerate(np.array([True, True]))))
        >>> e1.isscalar()
        True
        >>> e1['00'].data
        array(True)
        """
        cls = type(self)
        if isinstance(point, Mapping):
            replacements = {
                i: cls.constant(val, self.num_vars, algebra=self.algebra, backend=self.backend)  # ty: ignore[invalid-argument-type]
                for i, val in point.items()
            }
        else:
            assert is_array(point)
            replacements = {
                i: cls.constant(point[i], self.num_vars, algebra=self.algebra, backend=self.backend)
                for i in range(self.num_vars)
            }
        return self.compose(replacements)  # ty: ignore[invalid-argument-type]

    def compose(self, replacements: Mapping[int, Self]) -> Self:
        """Compose polynomial with multiple substitutions.

        Returns ``p(x_1 <- q_1, ..., x_n <- q_n)`` where only specified
        indices are replaced.

        Parameters
        ----------
        replacements : Mapping[int, PolyDict]
            Dict mapping variable indices to replacement polynomials.

        Returns
        -------
        PolyDict
            The composed polynomial.

        Note
        ----
        The composition is performed simultaneously. If not, this is a bug.
        """
        cls = type(self)
        result = cls.zero(self.num_vars, algebra=self.algebra, backend=self.backend)  # initialize with additive identity

        def var_at(idx: int) -> Self:
            return cls.variable(idx, self.num_vars, algebra=self.algebra, backend=self.backend)

        def as_const(val: AlgebraicArray | Scalar) -> Self:
            raw = val.data if isinstance(val, AlgebraicArray) else val
            return cls.constant(raw, self.num_vars, algebra=self.algebra, backend=self.backend)

        for monomial, coeff in self.items():
            # make a new term by replacing the monomial terms with either the replacement if it exists, or a plain variable.
            term = functools.reduce(
                lambda a, b: a * b,
                (replacements.get(idx, var_at(idx)) for idx, deg in enumerate(monomial) if deg == 1),
                as_const(coeff),
            )
            result = result + term

        return result

    def isscalar(self) -> bool:
        return len(self) == 0 or (len(self) == 1 and self.get(frozenbitarray(self.num_vars)) is not None)


_poly_dict_registry = BackendClassRegistry("PolyDict")
_poly_dict_registry.register(
    Backend.JAX,
    lambda: __import__("algebraic.polynomials.dok._jax", fromlist=["JaxPolyDict"]).JaxPolyDict,
)
_poly_dict_registry.register(
    Backend.TORCH,
    lambda: __import__("algebraic.polynomials.dok._torch", fromlist=["TorchPolyDict"]).TorchPolyDict,
)
_poly_dict_registry.register(
    Backend.NUMPY,
    lambda: __import__("algebraic.polynomials.dok._numpy", fromlist=["NumpyPolyDict"]).NumpyPolyDict,
)


def _make_poly_dict(
    algebra: Lattice, num_vars: int, data: Mapping[frozenbitarray, AlgebraicArray], backend: str | Backend
) -> PolyDict:
    cls = _poly_dict_registry[backend]
    return cls(algebra, num_vars, data)
