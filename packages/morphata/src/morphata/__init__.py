"""Morphata: Flexible automata representations for regular and omega-regular languages.

This package provides:
- Pure structural automaton interfaces (Automaton, Domain, TransitionRelation)
- Acceptance condition expressions (morphata.acceptance)
- HOA format parser (morphata.hoa.parser)
- Example implementations (morphata.examples)
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from typing_extensions import override

from morphata.spec import AcceptanceCondition as AcceptanceCondition
from morphata.spec import (
    AlternatingTransitions,
    BoolExpr,
    DeterministicTransitions,
    NonDeterministicTransitions,
    UniversalTransitions,
)
from morphata.spec import Automaton as Automaton
from morphata.spec import Domain as Domain
from morphata.spec import InitialState as InitialState
from morphata.spec import TransitionRelation as TransitionRelation


@dataclass
class DeterministicTransitionRelation[Q: Hashable, S: Hashable](DeterministicTransitions[Q, S]):
    data: Mapping[Q, Mapping[S, Q]]

    @override
    def __call__(self, state: Q, symbol: S) -> Q:
        return self.data[state][symbol]


@dataclass
class NonDeterministicTransitionRelation[Q: Hashable, S: Hashable](NonDeterministicTransitions[Q, S]):
    data: Mapping[Q, Mapping[S, AbstractSet[Q]]]

    @override
    def __call__(self, state: Q, symbol: S) -> AbstractSet[Q]:
        return self.data[state][symbol]


@dataclass
class UniversalTransitionRelation[Q: Hashable, S: Hashable](UniversalTransitions[Q, S]):
    data: Mapping[Q, Mapping[S, AbstractSet[Q]]]

    @override
    def __call__(self, state: Q, symbol: S) -> AbstractSet[Q]:
        return self.data[state][symbol]


@dataclass
class AlternatingTransitionRelation[Q: Hashable, S: Hashable](AlternatingTransitions[Q, S]):
    data: Mapping[Q, Mapping[S, BoolExpr[Q]]]

    @override
    def __call__(self, state: Q, symbol: S) -> BoolExpr[Q]:
        return self.data[state][symbol]


__all__ = [
    "Domain",
    "InitialState",
    "AcceptanceCondition",
    "TransitionRelation",
    "DeterministicTransitions",
    "NonDeterministicTransitions",
    "UniversalTransitions",
    "AlternatingTransitions",
    "Automaton",
]
