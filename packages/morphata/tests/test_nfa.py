"""Tests for morphata NFA implementation.

These tests focus on basic structural properties without weighted semantics.
"""

import pytest
from logic_asts import base, ltl

from morphata.automata.nfa import NFA
from morphata.spec import FiniteAcceptance


def test_nfa_creation():
    """Test basic NFA creation and properties."""
    nfa = NFA[str]()

    assert len(nfa) == 0
    assert list(nfa) == []
    assert nfa.initial_state == frozenset()


def test_nfa_add_locations():
    """Test adding locations to NFA."""
    nfa = NFA[str]()

    nfa.add_location(0, initial=True)
    nfa.add_location(1, final=True)
    nfa.add_location(2)

    assert len(nfa) == 3
    assert set(nfa) == {0, 1, 2}
    assert nfa.initial_state == frozenset({0})
    assert nfa.final_locations == frozenset({1})


def test_nfa_duplicate_location_error():
    """Test that adding duplicate location raises error."""
    nfa = NFA[str]()
    nfa.add_location(0)

    with pytest.raises(ValueError, match="already exists"):
        nfa.add_location(0)


def test_nfa_add_transitions():
    """Test adding transitions with guards."""
    nfa = NFA[str]()
    nfa.add_location(0, initial=True)
    nfa.add_location(1, final=True)

    # Create guards using logic_asts
    guard_a = base.Variable("a")
    guard_not_a = base.Not(base.Variable("a"))

    nfa.add_transition(0, 0, guard_not_a)
    nfa.add_transition(0, 1, guard_a)

    # Check guards
    guards_from_0 = nfa.guards(0)
    assert len(guards_from_0) == 2
    assert 0 in guards_from_0
    assert 1 in guards_from_0

    # Check specific guard
    assert nfa.guards(0, 1) == guard_a


def test_nfa_duplicate_transition_error():
    """Test that adding duplicate transition raises error."""
    nfa = NFA[str]()
    nfa.add_location(0)
    nfa.add_location(1)

    guard = base.Variable("a")
    nfa.add_transition(0, 1, guard)

    with pytest.raises(ValueError, match="already exists"):
        nfa.add_transition(0, 1, guard)


def test_nfa_acceptance_condition():
    """Test acceptance condition is FiniteAcceptance."""
    nfa = NFA[str]()
    nfa.add_location(0, initial=True)
    nfa.add_location(1, final=True)

    acc = nfa.acceptance_condition
    assert isinstance(acc, FiniteAcceptance)
    assert acc.accepting == frozenset({1})


def test_nfa_transitions_iterable():
    """Test transitions property returns all transitions."""
    nfa = NFA[str]()
    nfa.add_location(0)
    nfa.add_location(1)
    nfa.add_location(2)

    guard1 = base.Variable("a")
    guard2 = base.Variable("b")

    nfa.add_transition(0, 1, guard1)
    nfa.add_transition(1, 2, guard2)

    transitions = list(nfa.transitions)
    assert len(transitions) == 2
    assert (0, 1, guard1) in transitions
    assert (1, 2, guard2) in transitions


def test_nfa_call_basic():
    """Test basic NFA transition function call."""
    nfa = NFA[str]()
    nfa.add_location(0, initial=True)
    nfa.add_location(1, final=True)

    # Transition on 'a'
    guard_a = base.Variable("a")
    nfa.add_transition(0, 1, guard_a)

    # Test transition with input containing 'a'
    accepting, next_state = nfa({"a"}, frozenset({0}))
    assert next_state == frozenset({1})
    assert accepting is True  # reached final state

    # Test transition with input not containing 'a'
    accepting, next_state = nfa({"b"}, frozenset({0}))
    assert next_state == frozenset()  # no successors
    assert accepting is False


def test_nfa_temporal_guard_error():
    """Test that guards with temporal operators are rejected."""
    nfa = NFA[str]()
    nfa.add_location(0)
    nfa.add_location(1)

    # Create a temporal guard (Next operator)
    temporal_guard = ltl.Next(base.Variable("a"))

    with pytest.raises(ValueError, match="temporal operators"):
        nfa.add_transition(0, 1, temporal_guard)
