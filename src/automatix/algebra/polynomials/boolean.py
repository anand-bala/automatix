from typing import Mapping, Self, TypeAlias, final

import dd.autoref as bddlib
from typing_extensions import override

from automatix.algebra.abc import AbstractPolynomial, PolynomialManager

# if TYPE_CHECKING:
#     import dd.autoref as bddlib
# else:
#     try:
#         import dd.cudd as bddlib  # pyright: ignore[reportMissingImports]
#     except ImportError:
#         import dd.autoref as bddlib

_Poly: TypeAlias = "BooleanPolynomial"


@final
class BooleanPolyCtx(PolynomialManager[_Poly, bool]):
    """Polynomial manager for Boolean polynomials."""

    def __init__(self, manager: None | bddlib.BDD = None) -> None:
        super().__init__()

        self._bdd = manager or bddlib.BDD()

    def _wrap(self, expr: bddlib.Function) -> _Poly:
        assert self._bdd == expr.bdd
        poly = BooleanPolynomial(self)
        poly._expr = expr
        return poly

    @property
    @override
    def top(self) -> _Poly:
        return self._wrap(self._bdd.true)

    @property
    @override
    def bottom(self) -> _Poly:
        return self._wrap(self._bdd.false)

    @override
    def is_top(self, poly: _Poly) -> bool:
        return poly._expr == self._bdd.true

    @override
    def is_bottom(self, poly: _Poly) -> bool:
        return poly._expr == self._bdd.false

    @override
    def const(self, value: bool) -> _Poly:
        return self._wrap(self._bdd.true if value else self._bdd.false)

    @override
    def var(self, name: str) -> _Poly:
        return self._wrap(self._bdd.var(name))

    @override
    def declare(self, var: str) -> _Poly:
        self._bdd.declare(var)
        return self.var(var)

    @override
    def let(self, poly: _Poly, mapping: Mapping[str, "bool | BooleanPolynomial"]) -> _Poly:
        new_mapping = {name: val if isinstance(val, bool) else val._expr for name, val in mapping.items()}
        new_func: bddlib.Function = self._bdd.let(new_mapping, poly._expr)  # type: ignore
        return self._wrap(new_func)

    @override
    def negate(self, poly: _Poly) -> _Poly:
        return self._wrap(~poly._expr)

    @override
    def add(self, lhs: _Poly, rhs: _Poly) -> _Poly:
        return self._wrap(self._bdd.apply("or", lhs._expr, rhs._expr))

    @override
    def multiply(self, lhs: _Poly, rhs: _Poly) -> _Poly:
        return self._wrap(self._bdd.apply("and", lhs._expr, rhs._expr))


@final
class BooleanPolynomial(AbstractPolynomial[bool]):
    """A Polynomial over the Boolean algebra.

    A Boolean polynomial is defined over the Boolean algebra, where addition is defined by logical OR and multiplication by
    logical AND.
    """

    def __init__(self, manager: BooleanPolyCtx) -> None:
        super().__init__()

        self._manager = manager
        self._bdd = self._manager._bdd
        self._expr: bddlib.Function = self._manager._bdd.false

    @property
    @override
    def context(self) -> BooleanPolyCtx:
        return self._manager

    @property
    @override
    def support(self) -> set[str]:
        return self._bdd.support(self._expr)

    @override
    def declare(self, var: str) -> "BooleanPolynomial":
        return self.context.declare(var)

    @override
    def top(self) -> "BooleanPolynomial":
        return self.context.top

    @override
    def bottom(self) -> "BooleanPolynomial":
        return self.context.bottom

    @override
    def is_top(self) -> bool:
        return self._expr == self._bdd.true

    @override
    def is_bottom(self) -> bool:
        return self._expr == self._bdd.false

    @override
    def const(self, value: bool) -> "BooleanPolynomial":
        return self.context.const(value)

    @override
    def let(self, mapping: Mapping[str, bool | Self]) -> "BooleanPolynomial":
        return self.context.let(self, mapping)

    @override
    def eval(self, mapping: Mapping[str, bool]) -> bool:
        assert self.support.issubset(mapping.keys())
        evald = self.let(mapping)
        if evald.is_top():
            return True
        elif evald.is_bottom():
            return False
        else:
            raise RuntimeError("Evaluated polynomial is not constant, even with full support")

    @override
    def negate(self) -> "BooleanPolynomial":
        return self.context.negate(self)

    @override
    def add(self, other: bool | Self) -> "BooleanPolynomial":
        if isinstance(other, bool):
            wrapped = self.const(other)
        else:
            wrapped = other
        return self.context.add(self, wrapped)

    @override
    def multiply(self, other: bool | Self) -> "BooleanPolynomial":
        if isinstance(other, bool):
            wrapped = self.const(other)
        else:
            wrapped = other
        return self.context.multiply(self, wrapped)

    def __str__(self) -> str:
        return str(self._expr.to_expr())
