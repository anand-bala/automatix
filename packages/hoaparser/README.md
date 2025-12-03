# HOA Parser: Hanoi Omega-Automata Format Parser

A Python parser for the **HOA (Hanoi Omega-Automata) format** - a standardized
specification for describing omega-automata.

## Overview

This package provides a complete parser for the HOA format, a human-readable
format for specifying omega-automata with various acceptance conditions (Büchi,
co-Büchi, Rabin, Streett, parity, etc.).

## Features

- **Complete HOA v1 Support**:
  Full parsing of HOA format version 1.0
- **Multiple Acceptance Conditions**:
  Büchi, co-Büchi, Rabin, Streett, parity, Muller, generalized variants
- **Flexible Label Expressions**:
  Support for complex boolean predicates on transitions
- **Type Safe**:
  Full mypy strict type checking
- **Lark-Based**:
  Uses Lark parser generator for robust grammar-based parsing

## Installation

The package is part of the automatix workspace:

```bash
pip install git+https://github.com/anand-bala/automatix
```

## Quick Start

### Basic Parsing

```python
from hoaparser import parse

# HOA format string
hoa_spec = """
HOA: v1
name: "simple-buchi"
States: 2
Start: 0
AP: 1 "a"
Acceptance: 1 Inf(0)
--BODY--
State: 0
[0] 1 {0}
[!0] 0
State: 1
[0] 1 {0}
[!0] 0
--END--
"""

# Parse
automaton = parse(hoa_spec)

# Access parsed structure
print(f"Name: {automaton.header.name}")
print(f"States: {automaton.header.num_states}")
print(f"Acceptance: {automaton.header.acc}")

# Iterate over transitions
for state, transitions in automaton.body.items():
    print(f"From state {state.idx}:")
    for trans in transitions:
        print(f"  To {trans.dst} on {trans.label}")
```

### Working with Parsed Data

```python
from hoaparser import parse, State

hoa_spec = """HOA: v1
States: 2
Start: 0
AP: 1 "request"
Acceptance: 1 Inf(0)
--BODY--
State: 0
[0] 1 {0}
State: 1
[!0] 0
--END--
"""

automaton = parse(hoa_spec)

# Access header information
print(f"Initial states: {automaton.header.initial}")
print(f"Predicates: {automaton.header.predicates}")
print(f"Acceptance condition: {automaton.header.acc}")

# Access body (state-transition graph)
for state, transitions in automaton.body.items():
    if state in automaton.header.initial:
        print(f"Initial state: {state.idx}")
    for trans in transitions:
        print(f"  {state.idx} -> {trans.dst[0]}: {trans.label}")
```

## Data Structures

### Header

Contains automaton metadata:

```python
@dataclass
class Header:
    acc: AcceptanceCondition          # Acceptance condition spec
    name: str | None                  # Automaton name
    num_states: int | None            # Number of states
    initial: list[list[int]]          # Initial state(s)
    predicates: list[str]             # Atomic propositions
    aliases: dict[str, LabelExpr]     # Label macros
    properties: list[str]             # Automaton properties
```

### State

Represents a single automaton state:

```python
@dataclass(frozen=True)
class State:
    idx: int                          # State index
    label: LabelExpr | None           # State labeling (if any)
    acc_set: list[int] | None         # Acceptance set membership
    description: str | None           # Human-readable description
```

### Transition

Represents a transition between states:

```python
@dataclass(frozen=True)
class Transition:
    dst: list[int]                    # Destination state(s)
    label: LabelExpr | None           # Transition label/guard
    acc_set: list[int] | None         # Acceptance set membership
```

### ParsedAutomaton

The complete parsed automaton structure:

```python
@dataclass(frozen=True)
class ParsedAutomaton:
    header: Header                    # Metadata
    body: dict[State, list[Transition]]  # State graph
```

## Acceptance Conditions

The parser supports all HOA acceptance conditions:

```python
from hoaparser.omega import (
    Buchi, CoBuchi,                   # Single-set conditions
    GeneralizedBuchi, GeneralizedCoBuchi,  # Multiple-set conditions
    Rabin, Streett,                   # Pairwise conditions
    Parity, Muller,                   # Other conditions
    GenericCondition                  # Generic formula-based
)

# Access acceptance condition from parsed automaton
acc = automaton.header.acc
if isinstance(acc, Buchi):
    print(f"Büchi condition on set {acc.sets[0]}")
elif isinstance(acc, Rabin):
    print(f"Rabin with {len(acc.pairs)} pairs")
```

## Label Expressions

Transitions are labeled with boolean expressions over atomic propositions:

```python
from logic_asts.base import And, Or, Not, Variable

# Example from parsing: [a & b | !c]
# Results in: Or(And(Variable('a'), Variable('b')), Not(Variable('c')))

# Access in parsed automaton
for state, transitions in automaton.body.items():
    for trans in transitions:
        if trans.label is not None:
            # trans.label is a logic_asts expression
            print(f"Transition labeled: {trans.label}")
```

## Exception Handling

The parser provides specific exceptions for different error conditions:

```python
from hoaparser import (
    HoaSyntaxError,
    IncorrectVersionError,           # HOA version not v1
    DuplicateHeaderError,             # Duplicate header field
    DuplicateAliasError,              # Duplicate alias definition
    MissingHeaderError,               # Missing required field
    UndefinedAliasError               # Undefined alias used
)

try:
    automaton = parse(hoa_string)
except IncorrectVersionError:
    print("Only HOA v1 is supported")
except DuplicateHeaderError as e:
    print(f"Error: {e}")
except MissingHeaderError as e:
    print(f"Missing required field: {e}")
```

## CLI Interface

The package provides a command-line interface for testing:

```bash
# Parse a HOA file and pretty-print the result
python -m hoaparser automaton.hoa
```

## HOA Format Overview

The HOA format has this structure:

```
HOA: v1
name: "automaton-name"
States: 3
Start: 0
AP: 2 "a" "b"
Acceptance: 2 Inf(0) | Fin(1)
--BODY--
State: 0
  [a & b] 1 {0}
  [!a] 2
State: 1
  [b] 1 {0}
State: 2
  [a] 0 {1}
--END--
```

## Module Structure

```
hoaparser/
├── __init__.py                      # Main parser and data classes
├── __main__.py                      # CLI interface
├── omega.py                         # Acceptance condition expressions
└── grammar.lark                     # Lark parser grammar
```

## Integration with Automatix

The hoaparser is designed to integrate with automatix automata:

```python
from hoaparser import parse
from automatix.operators import MatrixOperator
from automatix.weights.guard_weights import ExprWeightFn
from algebraic.tensor_algebra.jax import tropical_semiring

# Parse HOA specification
automaton_spec = parse(hoa_string)

# Extract information
header = automaton_spec.header
states = automaton_spec.body

# Use with automatix for weighted evaluation
semiring = tropical_semiring(semiring_type="MaxPlus")
# ... convert to automatix.nfa.NFA, create weight functions, etc.
```

## Limitations

- **Boolean only**:
  Acceptance conditions are expressed as boolean formulas
- **No optimization**:
  Parser focuses on correctness, not performance
- **HOA v1 only**:
  Earlier versions are not supported

## Testing

```bash
# Run tests
python -m pytest tests/ -v

# Type check
python -m mypy src/hoaparser/ --strict
```

## Design Notes

- **Lark Parser**:
  Uses the Lark parsing toolkit for robust, maintainable grammar
- **Immutable Data**:
  `State` and other key classes are frozen dataclasses
- **Expression Trees**:
  Labels are represented as logic_asts expression trees
- **Type Safe**:
  Full typing with jaxtyping for numpy operations

## Future Directions

- **Optimization**:
  Lazy parsing for large automata
- **Serialization**:
  Write back to HOA format
- **Validation**:
  Check acceptance conditions and structural properties
- **Simplification**:
  Minimize/optimize parsed automata

## References

- HOA Format Specification:
  http://adl.github.io/hoaf/
- Accepted Tool Papers:
  https://adl.github.io/hoaf/papers.html
- Related Tools:
  Spot, ltl2ba, ltl3ba

## License

See LICENSE file for details.
