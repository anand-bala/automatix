from typing import Union

from jaxtyping import Array, ArrayLike, Num
from typing_extensions import TypeAlias

_Axis: TypeAlias = Union[None, int, tuple[int, ...]]
_ArrayLike: TypeAlias = Num[ArrayLike, " ..."]
_Array: TypeAlias = Num[Array, " ..."]

def logsumexp(a: _ArrayLike, axis: _Axis = None) -> _Array: ...
