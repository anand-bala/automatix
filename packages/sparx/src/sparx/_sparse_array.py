from abc import abstractmethod
from typing import Self

import equinox as eqx
import quax
from equinox import AbstractVar
from jaxtyping import Scalar


class SparseArray(quax.ArrayValue):
    """An abstract base class for all the sparse array classes."""

    _shape: AbstractVar[tuple[int, ...]]
    allow_materialize: AbstractVar[bool]
    """Flag to control if the quax'd array should be materialized into a dense array """

    def __check_init__(self) -> None:
        if not all(isinstance(sh, int) and int(sh) >= 0 for sh in self._shape):
            raise ValueError("shape must be an non-negative integer or a tuple of non-negative integers.")

    @property
    @abstractmethod
    def nnz(self) -> int:
        """
        The number of nonzero elements in this array.
        """

    @abstractmethod
    def prune(self, *, value: Scalar | None = None) -> Self:
        """Prune the given value from the sparse array and compress it further."""

    def enable_materialise(self) -> Self:
        return eqx.tree_at(lambda arr: arr.allow_materialize, self, True)

    @property
    def density(self) -> float:
        """The ratio of nonzero to all elements in this array."""
        return self.nnz / self.size

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"<{cls_name}: shape={self.shape!s}, dtype={self.dtype!s}, nnz={self.nnz:d}>"
