Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install morphata

Building an NFA
---------------

The :class:`~morphata.examples.nfa.NFA` class provides a graph-based
nondeterministic finite automaton with guard-labelled transitions:

.. code-block:: python

   from morphata.examples.nfa import NFA
   import logic_asts as logic

   # Create an NFA over string-valued atomic propositions
   nfa: NFA[str] = NFA()

   # Add locations (states)
   nfa.add_location(0, initial=True)
   nfa.add_location(1, final=True)

   # Add a guarded transition: 0 → 1 when "a" holds
   nfa.add_transition(0, 1, logic.Variable("a"))

   # Convert to an Automaton
   aut = nfa.to_automaton()

LTL to Alternating Automaton
-----------------------------

Convert an LTL formula to an alternating finite automaton:

.. code-block:: python

   from morphata.examples.ltl import ltl_to_automaton
   import logic_asts as logic

   # F(a & b) — "eventually a and b"
   a = logic.Variable("a")
   b = logic.Variable("b")
   formula = logic.Eventually(logic.And(a, b))

   # Convert to AFA with finite-word acceptance
   aut = ltl_to_automaton(formula, finite=True)

   # Evaluate transitions
   successor = aut.delta(0, {"a": True, "b": False})

Parsing HOA Format
------------------

Read automata from the standard HOA text format:

.. code-block:: python

   from morphata.hoa.parser import parse

   hoa_string = """
   HOA: v1
   States: 2
   Start: 0
   acc-name: Buchi
   Acceptance: 1 Inf(0)
   AP: 1 "a"
   --BODY--
   State: 0
     [0] 1
   State: 1 {0}
     [t] 1
   --END--
   """

   automaton = parse(hoa_string)

Acceptance Conditions
---------------------

morphata provides concrete acceptance conditions for both finite-word and
omega-regular languages:

.. code-block:: python

   from morphata.acceptance import Finite, Buchi, Rabin, AccPair

   # Finite-word acceptance
   fin = Finite(frozenset({1, 3}))

   # Büchi acceptance (visit set infinitely often)
   buchi = Buchi(frozenset({1}))

   # Rabin acceptance (pair-based)
   pair = AccPair(rejecting=frozenset({0}), accepting=frozenset({1}))
   rabin = Rabin(pairs=(pair,))
