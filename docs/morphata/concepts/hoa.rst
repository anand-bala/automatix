HOA Format
==========

The `Hanoi Omega-Automata (HOA) <https://adl.github.io/hoaf/>`_ format is a
standard text representation for omega-automata. morphata provides a parser
and exporter for HOA v1, extended with a ``Final(n)`` operator for finite-word
acceptance.

Parsing
-------

Use :func:`~morphata.hoa.parser.parse` to read an HOA string:

.. code-block:: python

   from morphata.hoa.parser import parse

   hoa_str = """
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
   automaton = parse(hoa_str)

The returned :class:`~morphata.spec.Automaton` has integer states and
:data:`~morphata.spec.NonDeterministicTransitions`.

Exporting
---------

Use :func:`~morphata.hoa.exporter.to_hoa` to serialise an automaton back to
HOA format:

.. code-block:: python

   from morphata.hoa.exporter import to_hoa

   hoa_output = to_hoa(automaton)

The ``Final(n)`` Extension
--------------------------

Standard HOA defines ``Inf(n)`` (visit set *n* infinitely often) and ``Fin(n)``
(visit set *n* finitely often) for omega-regular acceptance. morphata adds
``Final(n)``  for finite-word automata:

- ``Final(n)``: Accept if the run **ends in** a state marked with acceptance
  set *n*.

This is **not part of the standard HOA v1 specification** but provides a
natural way to express regular-language acceptance in the HOA syntax.

.. code-block:: python

   finite_hoa = """
   HOA: v1
   States: 2
   Start: 0
   acc-name: Finite
   Acceptance: 1 Final(0)
   AP: 1 "a"
   --BODY--
   State: 0
     [0] 1 {0}
     [!0] 0
   State: 1
     [t] 1
   --END--
   """
   automaton = parse(finite_hoa)
   # Accepts finite words ending with 'a'

Acceptance Expressions
----------------------

Internally, the parser represents acceptance formulas using classes from
:mod:`morphata.hoa.acc_expr`. These expression trees are converted to concrete
:class:`~morphata.spec.AcceptanceCondition` instances during parsing.
