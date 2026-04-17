Automaton Specification
=======================

morphata provides a layered set of interfaces for defining automata.
An :class:`~morphata.spec.Automaton` is assembled from four components:
a :class:`~morphata.spec.Domain`, an initial configuration, a transition
relation, and an acceptance condition.

Domain
------

A :class:`~morphata.spec.Domain` describes the state space and input alphabet
of an automaton. It is *capability-based*: properties return ``None`` when
the domain is symbolic or otherwise not enumerable.

.. code-block:: python

   from morphata.spec import Domain

   class MyDomain(Domain[int, str]):
       @property
       def states(self):
           return range(3)

       @property
       def symbols(self):
           return ["a", "b"]

Transition Relations
--------------------

morphata supports several flavours of transition relation, each modelling a
different branching mode:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Class
     - Successor type
     - Use case
   * - :class:`~morphata.spec.DeterministicTransitions`
     - single ``State``
     - DFA
   * - :class:`~morphata.spec.NonDeterministicTransitions`
     - ``Iterable[State]``
     - NFA
   * - :class:`~morphata.spec.AlternatingTransitions`
     - ``BoolExpr[State]``
     - AFA

All three satisfy the :data:`~morphata.spec.TransitionRelation` type alias and
can be passed to :class:`~morphata.spec.Automaton`.

Boolean Expressions
^^^^^^^^^^^^^^^^^^^

Alternating transitions return a :data:`~morphata.spec.BoolExpr` - a positive
boolean formula over state variables built from the ``logic_asts`` library:

.. code-block:: python

   import logic_asts as logic

   q0 = logic.Variable(0)
   q1 = logic.Variable(1)
   expr = q0 & q1          # Both q0 AND q1 must accept
   expr = q0 | q1          # Either q0 OR q1 must accept

Automaton
---------

The :class:`~morphata.spec.Automaton` class ties everything together:

.. code-block:: python

   from morphata.spec import Automaton

   aut = Automaton(
       domain=my_domain,
       initial=0,                # single initial state
       delta=my_transitions,     # any TransitionRelation
       acceptance=my_acceptance, # any AcceptanceCondition
   )

The initial configuration can be:

- A single state (deterministic)
- An iterable of states (nondeterministic - stored as ``frozenset``)
- A :data:`~morphata.spec.BoolExpr` (alternating only)

Initial State
^^^^^^^^^^^^^

The :data:`~morphata.spec.InitialState` type alias captures these options:

.. code-block:: python

   type InitialState[State] = State | AbstractSet[State] | BoolExpr[State]

Concrete Transition Relations
-----------------------------

The top-level ``morphata`` package provides simple dataclass-based transition
relations backed by nested mappings:

.. code-block:: python

   import morphata

   # Deterministic: state x symbol -> state
   delta = morphata.DeterministicTransitionRelation(
       data={0: {"a": 1, "b": 0}, 1: {"a": 1, "b": 0}}
   )

   # Nondeterministic: state x symbol -> set of states
   delta = morphata.NonDeterministicTransitionRelation(
       data={0: {"a": {0, 1}, "b": {0}}, 1: {"a": set(), "b": set()}}
   )

For richer automaton implementations (guard-labelled NFAs, LTL-based AFAs),
see the :doc:`/api/morphata.examples.nfa` and :doc:`/api/morphata.examples.ltl`
modules.
