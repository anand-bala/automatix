# Automatix: Agent Context and Refactoring Roadmap

**Version**:
v0.5.0 (Phase 2 Complete) **Last Updated**:
2025-11-07 **Status**:
Core NFA weight functions complete; AFA deferred to v0.8.0

## Quick Start for Agents

### Project Purpose
Automatix is a library for **weighted automata over semirings** with a focus on:
- Flexible weight functions mapping `(input_symbol, guard) -> semiring_value`
- Multi-backend support (JAX-first, PyTorch/NumPy planned)
- Spatio-temporal specifications (STREL-based AFA for future versions)
- Quantitative monitoring and synthesis

### Essential Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Type check (strict mode)
python -m mypy src/automatix/ --strict

# View available semirings
python -c "from automatix.algebra import list_semirings; print(list_semirings('jax'))"

# Commit changes (use jj, not git)
jj commit -m "your message"

# Check status
jj status
```

### Directory Structure
```
src/automatix/
├── predicates.py           # Predicate classes (And, Or, Predicate, AbstractPredicate)
├── weights.py              # Weight functions + make_atomic_predicate_weight_function factory
├── nfa/
│   ├── automaton.py        # NFA, AutomatonOperator, make_automaton_operator
│   ├── predicate.py        # Backward compatibility re-exports
│   └── __init__.py
├── afa/
│   ├── automaton.py        # Generic AFA[Alph, Q, K] (abstract, deferred)
│   └── strel.py            # STREL->AFA translation (Boolean only, v0.8.0+)
└── algebra/
    ├── spec.py             # AbstractSemiring interface
    ├── backends/
    │   ├── jax_.py         # 11 JAX semirings + kernels
    │   ├── torch_.py       # v0.6.0 stub
    │   └── numpy_.py       # v0.6.0 stub
    └── polynomials/        # Polynomial implementations (Boolean only currently)
```

## Architecture Overview

### Core Concepts

**Weight Functions**:
Map `(input, guard) -> weight` where:
- `input`:
  Concrete data (vector in state space)
- `guard`:
  Boolean expression (from logic_asts)
- `weight`:
  Value in target semiring (float, array, etc.)

**Semiring Operations**:
- AND (`land`):
  Uses semiring multiplication (otimes)
- OR (`lor`):
  Uses semiring addition (oplus)

**Key Semirings**:
- **Tropical**:
  MinPlus (min as addition), MaxPlus (max as addition)
- **Robustness**:
  MaxMin semiring for STL robustness degrees
- **Boolean**:
  Standard logical operations
- **Others**:
  Counting, Log, LSE variants

### Module Integration

```
predicates.py (base classes)
    |
weights.py (weight functions + factory)
    |
nfa/automaton.py (NFA + make_automaton_operator)
    |
algebra/ (semiring implementations)
```

**No circular imports** - clean dependency hierarchy.

## Phase Completion Status

### Phase 1: Foundation (COMPLETE - Nov 5, 2025)
- Pure interface definitions (spec.py)
- Modular backend structure
- Centralized registry system
- JAX kernels infrastructure
- Type safety:
  100% mypy strict
- Tests:
  18/18 passing

### Phase 2: Weight Functions (COMPLETE - Nov 6, 2025)
- Weight function architecture finalized
- Factory function:
  `make_atomic_predicate_weight_function`
- NFA integration via `make_automaton_operator`
- Tests:
  13/13 passing
- Examples:
  6 working patterns

### Phase 2.5: Backward Compatibility (COMPLETE - Nov 7, 2025)
- Predicates moved to separate `predicates.py` module
- NFA tests fully rewritten with new API
- Examples for tropical semirings and robustness semantics
- Type safety maintained:
  100% mypy strict
- Tests:
  36/36 passing

## Known Blockers and Deferments

### AFA Module (Deferred to v0.8.0)

**Status**:
Currently Boolean-only via `dd` package (Binary Decision Diagrams)

**Architecture Mismatch**:
- NFA:
  Guards on transitions, scalar weight functions
- AFA:
  STREL specifications, polynomial-based transitions over semirings

**Blockers for General Semiring Support**:

1. **Polynomial Representations**:
   - Current:
     Uses `dd` package (reduced ordered BDDs)
   - Problem:
     BDDs require properties not satisfied by arbitrary semirings
   - Need:
     Custom polynomial representations for each semiring family

2. **Algebra Classes**:
   - Required:
     Implementations for De Morgan algebras (min-max, etc.)
   - Required:
     Boolean algebra specializations
   - Scope:
     Significant implementation effort

3. **Labeling Function**:
   - Current pattern:
     `label_fn(spatial_model, location, atom) -> K`
   - AFA uses this for atomic predicates, not weight functions
   - Integration path unclear until polynomial representations exist

4. **STREL Specificity**:
   - Current implementation is tightly coupled to STREL
   - Needs generalization for other temporal logics
   - Deferred until core polynomial infrastructure in place

**Path Forward for v0.8.0**:
1. Design and implement custom polynomial representations
2. Create De Morgan algebra implementations
3. Add Boolean algebra specializations
4. Generalize AFA beyond STREL
5. Integrate with weight function abstraction (if applicable)

## Design Decisions (Locked In)

1. **Weight Function Scope**:
   Core patterns only (guard-based + global)
2. **Runtime Validation**:
   Validation at automaton construction time
3. **Semiring-Agnostic**:
   Weight functions don't know which semiring they're used with
4. **Backward Compatibility**:
   No v0.4.0 support required (breaking changes OK)
5. **Version Control**:
   Use `jj` (Jujutsu) not git
6. **Code Style**:
   No emojis/unicode; use LaTeX for math symbols

## Testing

- **Framework**:
  pytest with jaxtyping
- **Coverage**:
  All 36 tests passing (26 existing + 7 new + 3 HOA parser)
- **Type Safety**:
  100% mypy --strict across all modules
- **JAX Integration**:
  JIT, vmap, gradient flow all tested

## Examples

Located in `examples/`:
- `weight_functions_demo.py`:
  6 weight function patterns
- `tropical_semirings_demo.py`:
  MinPlus/MaxPlus applications
- `robustness_semantics_demo.py`:
  STL robustness with MaxMin

Run examples:
`python examples/FILENAME.py`

## Important Notes for Agents

1. **Single User**:
   Anand is the primary user; breaking changes are acceptable
2. **JAX-First**:
   Optimize for JAX initially; other backends later
3. **No Deprecation**:
   Delete old code instead of wrapping it
4. **Type Safety**:
   All new code must pass `mypy --strict`
5. **Prefer Refactoring**:
   Update existing code patterns rather than special-casing
6. **Documentation**:
   Keep design decisions documented in markdown files
7. **Commit Message Format**:
   Use present tense, reference papers/issues when relevant

## Next Steps (v0.6.0 and Beyond)

1. **Phase 3 (v0.6.0)**:
   Multi-backend support
   - PyTorch backend implementation
   - NumPy backend implementation
   - Cross-backend testing

2. **Phase 4 (v0.7.0)**:
   Polish and Release
   - Final testing and documentation
   - Migration guide for v0.5.0 -> v0.6.0
   - Release preparation

3. **Phase 5 (v0.8.0)**:
   AFA Generalization
   - Custom polynomial representations
   - De Morgan algebra implementations
   - AFA extension beyond STREL
   - Weight function integration (if applicable)

## References

- **Architecture Reference**:
  `ARCHITECTURE.md`
