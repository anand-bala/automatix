Acceptance Conditions
=====================

An :class:`~morphata.spec.AcceptanceCondition` determines when a run of an
automaton is considered accepting. morphata supports both **finite-word** and
**omega-regular** acceptance.

Finite-Word Acceptance
----------------------

:class:`~morphata.acceptance.Finite` accepts a run that **ends in** a state
belonging to the accepting set:

.. code-block:: python

   from morphata.acceptance import Finite

   acc = Finite(frozenset({1, 3}))
   assert not acc.is_omega_regular()

This is the standard acceptance for DFAs and NFAs over finite words.

Omega-Regular Acceptance
------------------------

Omega-regular conditions reason about the set of states visited **infinitely
often** during an infinite run, denoted :math:`\mathit{inf}(r)`.

Büchi
^^^^^

:class:`~morphata.acceptance.Buchi` accepts iff
:math:`\mathit{inf}(r) \cap F \neq \emptyset`:

.. code-block:: python

   from morphata.acceptance import Buchi
   acc = Buchi(frozenset({1}))

co-Büchi
^^^^^^^^

:class:`~morphata.acceptance.CoBuchi` accepts iff
:math:`\mathit{inf}(r) \cap R = \emptyset`:

.. code-block:: python

   from morphata.acceptance import CoBuchi
   acc = CoBuchi(frozenset({2}))  # state 2 must be visited finitely often

Generalized Variants
^^^^^^^^^^^^^^^^^^^^

:class:`~morphata.acceptance.GeneralizedBuchi` requires intersection with
*every* accepting set. :class:`~morphata.acceptance.GeneralizedCoBuchi`
requires avoidance of *every* rejecting set.

Rabin and Streett
^^^^^^^^^^^^^^^^^

These use pairs of state sets ``(rejecting, accepting)``:

- :class:`~morphata.acceptance.Rabin`: Accepts iff *for some* pair, the
  rejecting set is visited finitely often **and** the accepting set is
  visited infinitely often.
- :class:`~morphata.acceptance.Streett`: Accepts iff *for all* pairs, the
  same condition holds. (Dual of Rabin.)

.. code-block:: python

   from morphata.acceptance import Rabin, AccPair

   pair = AccPair(rejecting=frozenset({0}), accepting=frozenset({1}))
   acc = Rabin(pairs=(pair,))

Muller
^^^^^^

:class:`~morphata.acceptance.Muller` accepts iff :math:`\mathit{inf}(r)`
equals exactly one of the given state sets.

Factory Function
----------------

Use :func:`~morphata.acceptance.acc_from_name` to construct conditions by
name string:

.. code-block:: python

   from morphata.acceptance import acc_from_name

   acc = acc_from_name("Buchi", {1, 2})
   acc = acc_from_name("Rabin", ({0}, {1}), ({2}, {3}))
