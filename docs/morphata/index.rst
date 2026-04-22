morphata
========

**morphata** is a Python library for constructing, manipulating, and translating
automata over regular and omega-regular languages.
It provides flexible graph-based representations for automata without committing
to any specific model checking or monitoring algorithm.

Features
--------

- **Graph-based automata implementations**

  - Nondeterministic Finite Automata (NFA) for finite words
  - STREL automata for spatio-temporal specifications

- **HOA format parser**

  - Extended HOA v1 format with finite-word acceptance support
  - Standard acceptance conditions:
    Buchi, co-Buchi, Rabin, Streett, Parity, Muller
  - Extension:
    ``Final(n)`` operator for finite-word automata (not in standard HOA v1)
  - Validation and error reporting

- **Acceptance conditions**

  - Expression algebra for omega-regular conditions
  - Classical acceptance types (Buchi, generalized-Buchi, co-Buchi, Rabin,
    Streett, Muller, Parity)
  - Finite-word acceptance for regular languages

- **Pure structural interfaces**

  - Base automaton interfaces without weighted semantics
  - NetworkX-based graph representations
  - Clean separation from quantitative monitoring (provided by
    `automatix <../automatix/index.html>`_)

- **LTL to Alternating Finite Automaton conversion**

  - LTL/LTLf formulas to AFAs via ``ltl_to_automaton``
  - Finite-word (``Finite``) and infinite-word (``Buchi``) acceptance

.. toctree::
   :maxdepth: 2

   quick-start
   concepts/index
   api/modules
   changelog
