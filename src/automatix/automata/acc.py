from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class Finite:
    """Finite accepting condition: a finite run, r, is accepting iff it ends in an accepting state"""

    accepting: frozenset[int]


@dataclass(frozen=True)
class Buchi:
    """Büchi condition: a run, r, is accepting iff inf(r) intersects with `accepting`"""

    accepting: frozenset[int]


@dataclass(frozen=True)
class GeneralizedBuchi:
    """Generalized Büchi condition: a run, r, is accepting iff inf(r) intersects with `accepting[i]` for some i"""

    accepting: tuple[frozenset[int], ...]


@dataclass(frozen=True)
class CoBuchi:
    """co-Büchi condition: a run, r, is accepting iff inf(r) does not intersect with `rejecting`"""

    rejecting: frozenset[int]


@dataclass(frozen=True)
class GeneralizedCoBuchi:
    """Generalized co-Büchi condition: a run, r, is accepting iff inf(r) does not intersect with `rejecting[i]` for some i"""

    rejecting: tuple[frozenset[int]]


class AccPair(NamedTuple):
    rejecting: frozenset[int]
    accepting: frozenset[int]


@dataclass(frozen=True)
class Streett:
    """Streett condition: a run, r, is accpting iff _for all_ `i`, we have that inf(r) does not intersect with `pairs[i].rejecting` and does intersect with `pairs[i].accepting`"""

    pairs: tuple[AccPair, ...]

    @property
    def index(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class Rabin:
    """Rabin condition: a run, r, is accpting iff _for some_ `i`, we have that inf(r) does not intersect with `pairs[i].rejecting` and does intersect with `pairs[i].accepting`"""

    pairs: tuple[AccPair, ...]

    @property
    def index(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class Muller:
    """Muller condition: a run, r, is accepting iff for some `i`, we have that inf(r) is exactly `sets[i]`"""

    sets: tuple[frozenset[int]]
