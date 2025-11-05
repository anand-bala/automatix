from typing import Generic, Protocol, TypeVar, runtime_checkable

from automatix.algebra.spec import AbstractPolynomial

Alph = TypeVar("Alph", contravariant=True)
Q = TypeVar("Q", contravariant=True)
K = TypeVar("K")


@runtime_checkable
class AbstractTransition(Protocol[Alph, Q, K]):
    def __call__(self, input: Alph, state: Q) -> AbstractPolynomial[K]: ...


class AFA(Generic[Alph, Q, K]):
    def __init__(self, transitions: AbstractTransition[Alph, Q, K]) -> None:
        self._transitions = transitions

    @property
    def transitions(self) -> AbstractTransition[Alph, Q, K]:
        """Get the transition relation"""
        return self._transitions
