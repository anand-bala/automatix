# Morphata

Morphata is a Python library for constructing, manipulating, and translating
automata over regular and omega-regular languages.
It provides flexible graph-based representations for automata without committing
to any specific model checking or monitoring algorithm.

## Features

- **Graph-based automata implementations**
  - Nondeterministic Finite Automata (NFA) for finite words
  - STREL automata for spatio-temporal specifications

- **HOA format parser**
  - Extended HOA v1 format with finite-word acceptance support
  - Standard acceptance conditions:
    Buchi, co-Buchi, Rabin, Streett, Parity, Muller
  - Extension:
    `Final(n)` operator for finite-word automata (not in standard HOA v1)
  - Validation and error reporting

- **Acceptance conditions**
  - Expression algebra for omega-regular conditions
  - Classical acceptance types (Buchi, generalized-Buchi, co-Buchi, Rabin,
    Streett, Muller, Parity)
  - Finite-word acceptance for regular languages

- **Pure structural interfaces**
  - Base automaton interfaces without weighted semantics
  - NetworkX-based graph representations
  - Clean separation from quantitative monitoring (provided by automatix)

## Architecture

Morphata serves as the foundation for the automatix library:

- **morphata**:
  Graph-based automata, HOA parsing, acceptance conditions
- **automatix**:
  Weighted automata over semirings (depends on morphata)
- **algebraic**:
  Pure semiring algebra (independent)

This layered architecture allows morphata to be used independently for
structural automata operations, or as a foundation for weighted semantics.

## Installation

Morphata is part of the automatix workspace.
Install using uv:

```bash
uv pip install -e packages/morphata
```

## Quick Example

```python
from morphata.automata import NFA
from logic_asts import base

# Create an NFA
nfa = NFA[str]()
nfa.add_location(0, initial=True)
nfa.add_location(1, final=True)

# Add transition with guard
guard_a = base.Variable("a")
nfa.add_transition(0, 1, guard_a)

# Use the automaton
accepting, next_state = nfa({"a"}, nfa.initial_state)
print(f"Accepting: {accepting}, Next state: {next_state}")
```

## HOA Parser Example

### Standard HOA Format (Omega-Automata)

```python
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
print(f"Acceptance: {automaton.header.acc}")
```

### Extended Format: Finite-Word Acceptance

Morphata extends the HOA v1 format with a `Final(n)` operator for finite-word
automata.
This is **not part of the standard HOA specification** but provides a natural
way to express finite-word acceptance in the HOA syntax.

```python
from morphata.hoa.parser import parse

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
```

**Note**:
The `Final(n)` operator semantics differ from `Fin(n)` and `Inf(n)`:
- `Inf(n)`:
  Accept if acceptance set n is visited **infinitely often** (omega-regular)
- `Fin(n)`:
  Accept if acceptance set n is visited **finitely often** (omega-regular)
- `Final(n)`:
  Accept if the run **ends in** a state marked with acceptance set n (regular)

## Development

Run tests:
```bash
python -m pytest packages/morphata/tests/ -v
```

Type checking:
```bash
python -m mypy packages/morphata/src/morphata/ --strict
```

## License

Part of the automatix project.
