"""An Extended HOA (Hanoi Omega-Automata) format.

Provides parsing of automaton definitions in HOA format and data structures
for representing parsed automata.

This parser supports the standard HOA v1 format with the following extension:
    - Final(n) operator: For finite-word automata acceptance conditions
      (not part of the standard HOA v1 specification)

Standard HOA v1 acceptance operators (Inf, Fin, Buchi, co-Buchi, Rabin, Streett,
Parity, Muller) are fully supported. The Final(n) operator provides a natural
way to express finite-word acceptance in HOA syntax.

Example extended HOA with finite acceptance:
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

import morphata.hoa.parser as parser
from morphata.hoa.parser import ParsedAutomaton as ParsedAutomaton
from morphata.hoa.parser import parse as parse

__all__ = ["parse", "parser", "ParsedAutomaton"]
