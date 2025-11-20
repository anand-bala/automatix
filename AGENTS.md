# Automatix: Agent Context

**Version**: v0.4.0
**Last Updated**: 2025-11-20
**Status**: NFA operators with guard-based weight functions; STREL automaton (Boolean-only); hoaparser for HOA format parsing

## Quick Start for Agents

### Project Purpose
Automatix is a library for **weighted automata over semirings** with a focus on:
- Flexible weight functions mapping `(input, guard) -> semiring_value`
- Multi-backend support (JAX-first; PyTorch/NumPy planned for future versions)
- Spatio-temporal specifications via STREL automata (Boolean-only currently)
- Quantitative monitoring and synthesis

### Essential Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Type check (strict mode)
python -m mypy src/automatix/ --strict

# Check workspace status
jj status

# Commit changes (use jj, not git)
jj commit -m "your message"
```

### Workspace Structure
This is a Jujutsu monorepo with three packages:

```
automatix/
├── src/automatix/              # Main package
│   ├── spec.py                 # Abstract interfaces (Automaton, WeightFunction, Guard, AcceptanceCondition)
│   ├── operators.py            # MatrixOperator for finite-word automata
│   ├── acc.py                  # Acceptance conditions (Finite, Buchi, CoBuchi, Rabin, Streett, etc.)
│   ├── automata/
│   │   ├── nfa.py              # NFA implementation
│   │   └── strel.py            # STREL automaton (Boolean-only)
│   └── weights/
│       └── guard_weights.py    # Guard-based weight functions (Predicate, And, Or, ExprWeightFn)
│
├── packages/algebraic/         # Separate package: semiring algebra
│   ├── spec.py                 # Semiring interfaces (Semiring, BoundedDistributiveLattice, etc.)
│   └── backends/
│       ├── jax.py              # JAX implementations
│       └── torch.py            # PyTorch stub
│
├── packages/hoaparser/         # Separate package: HOA format parser
│   ├── grammar.lark            # Lark grammar for HOA syntax
│   └── parser.py               # Parser implementation
│
└── examples/                   # Example applications
    ├── weight_functions_demo.py
    ├── motion_planning/
    └── swarm-monitoring/
```

## Core Architecture

### Main Modules

**spec.py**:
- Abstract interfaces: `AbstractAutomaton`, `SizedAutomaton`, `WeightFunction`, `Guard`, `AcceptanceCondition`
- Defines contracts for automata and weight functions

**operators.py**:
- `MatrixOperator`: Weighted finite-word automaton operator
- `MatrixOperator.make()`: Factory method for creating operators from NFA + weight functions

**acc.py**:
- `Finite`: Accept after reaching final state
- `Buchi`, `CoBuchi`: Accepting transitions in infinite runs
- `GeneralizedBuchi`, `GeneralizedCoBuchi`
- `Rabin`, `Streett`: Pairwise conditions
- `Muller`: Set-based acceptance

**automata/nfa.py**:
- NFA implementation with location and transition management
- Implements `SizedAutomaton` interface
- Methods: `add_location()`, `add_transition()`, `guards()`

**automata/strel.py**:
- STREL (Spatio-Temporal Requirement Elicitation Language) automaton
- Boolean-only (no semiring support currently)
- Expression-based polynomial structure (~750 lines)
- Uses `_NodeMap` and helper classes for internal representation

**weights/guard_weights.py**:
- `AbstractPredicate`: Base class for predicates
- `Predicate`: Atomic predicates with boolean evaluation
- `And`, `Or`: Logical operators over predicates
- `ExprWeightFn`: Guard-based weight function class

### Separate Packages

**packages/algebraic/**:
- Pure semiring algebra (no automata)
- Interfaces: `Semiring`, `Ring`, `BiModule`, `BoundedDistributiveLattice`, `DeMorganAlgebra`, etc.
- JAX implementations:
  - `counting_semiring()`: Count semiring
  - `tropical_semiring()`: MinPlus/MaxPlus with smooth options
  - `max_min_algebra()`: MaxMin algebra (robustness semantics)
  - `boolean_algebra()`: Standard Boolean algebra

**packages/hoaparser/**:
- Hanoi Omega-Automata (HOA) format parser
- Uses Lark parser generator
- Parses automaton structure, states, transitions, acceptance conditions

## Design Decisions (Locked In)

1. **Weight Function Scope**: Guard-based weight functions (input + boolean expression -> semiring value)
2. **Runtime Validation**: Validation at automaton construction time
3. **Semiring-Agnostic**: Weight functions don't know target semiring
4. **Breaking Changes OK**: No v0.3.0 backward compatibility required
5. **Version Control**: Use `jj` (Jujutsu), not git
6. **Code Style**: No emojis/unicode; use LaTeX for math symbols (inline code blocks)

## Module Integration

```
weights/guard_weights.py (predicates + ExprWeightFn)
    |
automata/nfa.py (NFA structure)
    |
operators.py (MatrixOperator.make())
    |
algebraic/ (semiring implementations)
```

**Dependency hierarchy**: Clean, no circular imports.

## Current Implementation Status

### What Works
- NFA operators with guard-based weight functions
- JAX semiring implementations (counting, tropical, max-min, boolean)
- STREL automaton (Boolean-only)
- HOA format parser
- Example applications (weight functions, motion planning, swarm monitoring)

### What's Out of Scope (v0.4.0)
- General semiring support for STREL (Boolean-only currently)
- PyTorch backend (JAX only)
- Omega-regular automata (Buchi, co-Buchi)

### Known Limitations
- STREL is Boolean-only due to polynomial representation complexity
- No polynomial evaluation over arbitrary semirings yet
- Examples may have outdated import paths (check before running)

## Important Notes for Agents

1. **Single User**: Anand is the primary user; breaking changes are acceptable
2. **JAX-First**: Optimize for JAX initially
3. **No Deprecation**: Delete old code instead of wrapping it
4. **Type Safety**: All new code must pass `mypy --strict`
5. **Prefer Refactoring**: Update existing patterns rather than special-casing
6. **Documentation**: Keep design decisions documented in markdown files
7. **Commit Format**: Use present tense (`add`, `fix`, `refactor`); reference issues/papers when relevant

## Testing

- **Framework**: pytest with jaxtyping
- **Type Safety**: 100% mypy --strict
- **JAX Integration**: JIT, vmap, gradient operations tested

## Examples

Located in `examples/`:
- `weight_functions_demo.py`: Guard-based weight function patterns
- `motion_planning/`: Dynamics and planning examples
- `swarm-monitoring/`: Multi-agent coordination specs

Run examples:
```bash
python examples/FILENAME.py
```

## Future Directions (Deferred)

### Polynomial Semirings (v0.6.0+)
- Custom polynomial representations for general semirings
- STREL support beyond Boolean
- De Morgan algebra implementations
- Requires significant algebra infrastructure

### Multi-Backend Support (v0.7.0+)
- PyTorch implementations for semiring operations
- Cross-backend testing and benchmarking
- NumPy deferred (JAX on CPU sufficient for now)

### Omega-Regular Automata (v0.8.0+)
- Buchi and co-Buchi support
- Limit-based weight semantics
- OmegaAutomatonOperator

## References

- **Implementation planning**: `.cache/AFA_POLYNOMIAL_ARCHITECTURE.md`
- **Workspace structure**: `pyproject.toml`
- **Type specifications**: Code with jaxtyping decorators
