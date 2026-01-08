"""Acceptance conditions for omega-automata.

Defines acceptance condition types including Büchi, co-Büchi, Rabin, Streett, and Muller.
"""

from __future__ import annotations

from collections.abc import Hashable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class Finite[Q: Hashable]:
    """Finite accepting condition: a finite run, r, is accepting iff it ends in an accepting state"""

    accepting: AbstractSet[Q]


@dataclass(frozen=True)
class Buchi[Q: Hashable]:
    """Büchi condition: a run, r, is accepting iff inf(r) intersects with `accepting`"""

    accepting: AbstractSet[Q]


@dataclass(frozen=True)
class GeneralizedBuchi[Q: Hashable]:
    """Generalized Büchi condition: a run, r, is accepting iff inf(r) intersects with `accepting[i]` for some i"""

    accepting: tuple[AbstractSet[Q], ...]


@dataclass(frozen=True)
class CoBuchi[Q: Hashable]:
    """co-Büchi condition: a run, r, is accepting iff inf(r) does not intersect with `rejecting`"""

    rejecting: AbstractSet[Q]


@dataclass(frozen=True)
class GeneralizedCoBuchi[Q: Hashable]:
    """Generalized co-Büchi condition: a run, r, is accepting iff inf(r) does not intersect with `rejecting[i]` for some i"""

    rejecting: tuple[AbstractSet[Q]]


class AccPair[Q: Hashable](NamedTuple):
    """Pair of accepting and rejecting state sets for Rabin/Streett conditions."""

    rejecting: AbstractSet[Q]
    """States that must not appear infinitely often"""
    accepting: AbstractSet[Q]
    """States that must appear infinitely often"""


@dataclass(frozen=True)
class Streett[Q: Hashable]:
    """Streett condition: a run, r, is accpting iff _for all_ `i`, we have that inf(r) does not intersect with `pairs[i].rejecting` and does intersect with `pairs[i].accepting`"""

    pairs: tuple[AccPair[Q], ...]

    @property
    def index(self) -> int:
        """Number of pairs in this condition."""
        return len(self.pairs)


@dataclass(frozen=True)
class Rabin[Q: Hashable]:
    """Rabin condition: a run, r, is accpting iff _for some_ `i`, we have that inf(r) does not intersect with `pairs[i].rejecting` and does intersect with `pairs[i].accepting`"""

    pairs: tuple[AccPair[Q], ...]

    @property
    def index(self) -> int:
        """Number of pairs in this condition."""
        return len(self.pairs)


@dataclass(frozen=True)
class Muller[Q: Hashable]:
    """Muller condition: a run, r, is accepting iff for some `i`, we have that inf(r) is exactly `sets[i]`"""

    sets: tuple[AbstractSet[Q]]
