# Automatix: Agent Context


## Core Architecture

### Main Modules (automatix)

**spec.py** :
- Re-exports `BoolExpr as Guard` from morphata.spec
- `WeightFunction[In, AP]` protocol: Maps (input, guard) -> semiring value
- `AcceptanceCondition` type alias: Union of runtime acceptance conditions

**operators/** :
- `MatrixOperator[S: Semiring, In]` (operators/matrix.py): Operator for weighted
  finite-word NFAs
  - Fields: `initial_weights` , `final_weights` , `cost_transitions`
  - Factory: `MatrixOperator.make(aut, semiring, weight_function=...)`
  - Returns `algebraic.AlgebraicArray` instances for semiring-aware operations
- `PolynomialOperator[Symbol]` (operators/polynomial.py): Operator for alternating
  finite automata
  - Fields: `initial_poly` , `accepting_states` , `num_states` , `algebra` ,
    `_transition_cache`
  - Methods: `accepts()` , `run_polynomial()` , `step()` , `evaluate_at_accepting()`
  - Factory: `from_afa(aut, algebra, *, cache_transitions=True)`
  - Convenience: `from_ltl(formula, algebra, *, finite=True, cache_transitions=True)`
  - Represents AFA transitions/runs as multilinear polynomials using
    `algebraic.polynomials.RankDecomposition`

**acc.py** : Runtime acceptance condition implementations:
- `Finite[Q]` : Accept after reaching final state
- `Buchi[Q]` , `CoBuchi[Q]` : Omega-regular conditions
- `GeneralizedBuchi[Q]` , `GeneralizedCoBuchi[Q]` : Generalized variants
- `Rabin[Q]` , `Streett[Q]` , `Muller[Q]` : Pairwise/set-based conditions

**weights/guard_weights.py** :
- `AbstractPredicate[S]` : Base class for predicates
- `Predicate` : Atomic predicates wrapping user functions
- `And` , `Or` : Logical operators over predicates (semiring mul/add)
- `ExprWeightFn` : Recursive guard evaluation with memoization

### Morphata Package (Foundation)

**Purpose** : Pure structural automata representations (no semirings)

**spec.py** :
- `Domain[State, Symbol]` : Capability-based domain (may be symbolic)
- `TransitionRelation[State, Symbol]` : Protocol for transitions
- `AlternatingTransitions` : Returns `BoolExpr[State]` (alternating semantics)
- `NonDeterministicTransitions` : Returns `Iterable[State]`
- `DeterministicTransitions` : Returns single `State`
- `AcceptanceCondition` : Abstract base for acceptance

**automaton.py** :
- `Automaton[State, Symbol]` : Frozen dataclass with domain, initial, delta, acceptance
- Core data structure representing automata

**utils.py** :
- Utility functions for automata operations
- Helper methods for common automata manipulations

**acceptance.py** : Expression-based acceptance conditions for HOA specifications:
- `Finite` , `Buchi` , `CoBuchi` , `Parity` , etc.
- Used for parsing HOA format files

**examples/nfa.py** :
- `NFA[AP]` : Graph-based NFA with guards as boolean expressions
- States are integer locations
- Transitions: (src, dst, guard) triples
- Methods: `add_location()` , `add_transition()` , `to_automaton()`

**examples/strel.py** :
- `STRELAutomaton` : Spatio-temporal reach-escape logic
- Operators: Reach, Escape, Somewhere, Everywhere
- Inputs: Graphs with labeled nodes
- Boolean-only currently (no general semiring support)

**examples/ltl.py** :
- `ltl_to_automaton(formula, finite=True)` : LTL/LTLf to AFA conversion
- Returns `morphata.Automaton[int, Input[AP]]`
- States are integer indices representing LTL subformulas
- Finite=True: AFA with `Finite` acceptance
- Finite=False: ABWA with `Buchi` acceptance

**hoa/parser.py** :
- HOA (Hanoi Omega-Automata) v1 format parser
- Supports finite and omega-regular automata
- Returns `morphata.Automaton` instances

**hoa/exporter.py** :
- HOA format exporter
- Converts automata to HOA format for external tools

**hoa/hoa.lark** :
- LALR grammar for HOA format parsing
- Complete v1 specification support

### Algebraic Package (Semiring Algebra)

**Purpose** : Multi-backend semiring algebra (independent of automata)

**Public API** :
- Prefer `import algebraic` as the canonical entry point
- The top-level package re-exports array operations, semiring constructors,
  polynomial types, and spec types
- Use `backend=` to select the array backend (`"numpy"`, `"jax"`, or `"torch"`)
- Do not use removed public paths such as `algebraic.numpy`; prefer top-level calls
  like `algebraic.array(...)`, `algebraic.zeros(...)`, `algebraic.ones(...)`,
  `algebraic.sum(...)`, and `algebraic.matmul(...)`

**Top-level `algebraic` package** :
- Re-exports `algebraic.array()` , `algebraic.zeros()` , `algebraic.ones()` ,
  `algebraic.sum()` , `algebraic.matmul()` and related array operations
- Re-exports `algebraic.semirings` and `algebraic.polynomials`
- Re-exports core types including `AlgebraicArray` , `Semiring` ,
  `BooleanAlgebra` , `BoundedDistributiveLattice` , `DeMorganAlgebra` ,
  `HeytingAlgebra` , `Ring` , and `StoneAlgebra`

**ops/** :
- Canonical array-API-like semiring operations used by the top-level package
- Backend-aware implementations for NumPy, JAX, and PyTorch

**semirings.py** : Concrete algebra implementations:
- `counting_semiring()`
- `tropical_semiring()`
- `max_min_algebra()`
- `boolean_algebra()`

**array/** :
- `AlgebraicArray` : Backend-aware arrays with semiring semantics
- Overrides `+` , `*` , `@` to use semiring operations

**polynomials/** : Three complementary multilinear polynomial representations:
1. `PolyDict`
   - Sparse dictionary-backed representation
   - Public path: `algebraic.polynomials.PolyDict`
2. `MonomialBasis`
   - Dense tensor basis for small multilinear problems
   - Public path: `algebraic.polynomials.MonomialBasis`
3. `RankDecomposition`
   - Structured low-rank representation for larger problems
   - Public path: `algebraic.polynomials.RankDecomposition`

**spec.py** :
- `AlgebraicStructure` : Base class with properties (idempotent, commutative, etc.)
- `Semiring` : add, mul, zero, one operations
- `BoundedDistributiveLattice`
- `DeMorganAlgebra`
- `BooleanAlgebra`
- `HeytingAlgebra`
- `Ring`
- `StoneAlgebra`

## Package Architecture

**Layered Design:**

```
morphata (foundation: pure structural automata)
    |
automatix (weighted semantics with semirings)
    |
applications
```

**Key Separations:**
1. **morphata.spec** : Pure structural interfaces (no semirings)
2. **automatix.spec** : Extends with WeightFunction protocol and automatix-specific types
3. **morphata.acceptance** : Expression-based (for HOA specifications)
4. **automatix.acc** : State-set-based (for runtime checking)
5. **morphata.examples** : Concrete implementations (NFA, STREL, LTL)
6. **automatix.operators** : Weighted operators (MatrixOperator, PolynomialOperator)

**Dependencies:**
- morphata: logic-asts, networkx, attrs, lark
- automatix: morphata, algebraic, JAX, equinox
- algebraic: bitarray plus backend integrations for NumPy, JAX, and PyTorch

**Note** : `automatix` operators are JAX/equinox-based; `algebraic` itself is
multi-backend.

**No circular dependencies** : morphata does not import from automatix.

## Design Principles

1. **Weight Function Scope** : Guard-based weight functions (input + boolean expression
   -> semiring value)
2. **Runtime Validation** : Validation at automaton construction time
3. **Semiring-Agnostic** : Weight functions don't know target semiring
4. **API Stability** : v0.x; breaking changes may occur between releases
5. **Public Algebraic API** : Prefer top-level `import algebraic` in examples and
   user-facing code
6. **Import Paths** : Use `morphata.examples` (not `morphata.automata` )

## Usage Notes and Limitations

- STREL automata use Boolean semantics; semiring-valued STREL is not available.
- MatrixOperator targets finite-word acceptance; omega-regular acceptance is not
  implemented.
- PolynomialOperator usage is tested with Boolean algebra; other semirings may require
  additional validation.

## Important Notes for Agents

1. **Canonical Public APIs** : Prefer current public entry points (`import algebraic`,
   `algebraic.semirings`, `algebraic.polynomials`) over internal submodules in examples,
   docs, and new code
2. **Multi-Backend Awareness** : `algebraic` is backend-agnostic via `backend=`; do not
   add backend-only assumptions to algebraic code or docs
3. **No Deprecation** : Delete old code instead of wrapping it
4. **Type Safety** : All new code must pass `mypy --strict`
5. **Prefer Refactoring** : Update existing patterns rather than special-casing
6. **Documentation** : Keep design decisions documented in markdown files

## Testing

- **Framework** : pytest with jaxtyping
- **Type Safety** : 100% mypy --strict for all packages
- **Coverage** : Unit tests for all core modules

**Test Locations** :
- `tests/` : Automatix tests
- `packages/morphata/tests/` : Morphata tests (examples, HOA parser)
- `packages/algebraic/tests/` : Algebraic tests (polynomials, semirings, arrays)

## Examples

Located in `examples/` :
- `weight_functions_demo.py` : Guard-based weight function patterns
- `motion_planning/` : Dynamics and planning examples
- `swarm-monitoring/` : Multi-agent coordination specs with STREL

Run examples:
```bash
python examples/weight_functions_demo.py
python examples/swarm-monitoring/monitoring_example.py
```

## Common Patterns

### Creating a Weighted NFA Operator

```python
import algebraic
import jax.numpy as jnp
from morphata.examples.nfa import NFA
from automatix.operators import MatrixOperator
import logic_asts as logic

# Create NFA
aut: NFA[str] = NFA()
aut.add_location(0, initial=True)
aut.add_location(1, final=True)
aut.add_transition(0, 1, guard=logic.Variable("a"))

# Define weight function
def weight_fn(x, guard):
    return float(x[0])  # Example: use first element

# Create operator
maxplus = algebraic.semirings.tropical_semiring(minplus=False)
operator = MatrixOperator.make(aut, maxplus, weight_function=weight_fn)

# Evaluate
x = jnp.array([2.0])
transitions = operator.cost_transitions(x)
# `transitions` is an `algebraic.AlgebraicArray`
```

### Working with AlgebraicArray

```python
import algebraic

algebra = algebraic.semirings.max_min_algebra()
arr1 = algebraic.zeros((3, 3), semiring=algebra, backend="numpy")
arr2 = algebraic.ones((3, 3), semiring=algebra, backend="numpy")

# Operations use semiring semantics
result = arr1 + arr2
product = arr1 * arr2
total = algebraic.sum(arr2)
matrix_product = algebraic.matmul(arr1, arr2)

# `backend` can be "numpy", "jax", or "torch"
```

### Converting LTL to AFA

```python
from morphata.examples.ltl import ltl_to_automaton
import logic_asts as logic

# Define LTL formula: F(a & b)
a = logic.Variable("a")
b = logic.Variable("b")
formula = logic.Eventually(logic.And(a, b))

# Convert to AFA
aut = ltl_to_automaton(formula, finite=True)

# Access components
print(aut.initial)  # Initial states
print(aut.acceptance)  # Finite acceptance condition

# Evaluate transitions
input_dict = {"a": True, "b": False}
successor = aut.delta(0, input_dict)  # Returns BoolExpr[int]
```

### Working with Polynomials

```python
import algebraic

algebra = algebraic.semirings.boolean_algebra(mode="logic")
num_vars = 3

# Create variables with the sparse polynomial representation
x0 = algebraic.polynomials.PolyDict.variable(
    0, num_vars=num_vars, algebra=algebra, backend="numpy"
)
x1 = algebraic.polynomials.PolyDict.variable(
    1, num_vars=num_vars, algebra=algebra, backend="numpy"
)
x2 = algebraic.polynomials.PolyDict.variable(
    2, num_vars=num_vars, algebra=algebra, backend="numpy"
)

# Build polynomial: (x0 * x1) + x2
poly = (x0 * x1) + x2

# Evaluate at a point
result = poly.evaluate({0: True, 1: True, 2: False})
```

### Using PolynomialOperator with LTL

```python
import algebraic
from automatix.operators import from_ltl, from_afa
from morphata.examples.ltl import ltl_to_automaton
import logic_asts as logic

# Method 1: Direct from LTL formula
algebra = algebraic.semirings.boolean_algebra()
formula = logic.Eventually(logic.Variable("a"))  # F(a)

poly_op = from_ltl(formula, algebra, finite=True)

# Check if word is accepted
word = [{"a": False}, {"a": True}]  # ~a, a
result = poly_op.accepts(word)
print(result)  # Should be True (eventually a is satisfied)

# Method 2: From existing AFA
aut = ltl_to_automaton(formula, finite=True)
poly_op = from_afa(aut, algebra, cache_transitions=True)

# Access internal representation
print(f"Number of states: {poly_op.num_states}")
print(f"Accepting states: {poly_op.accepting_states}")

# Get run polynomial for a word (intermediate representation)
run_poly = poly_op.run_polynomial(word)
print(f"Run polynomial rank: {run_poly.rank}")
```

## Key Files Reference

**Core automatix** :
- `src/automatix/spec.py` : WeightFunction protocol, type definitions
- `src/automatix/operators/matrix.py` : MatrixOperator implementation (weighted NFAs)
- `src/automatix/operators/polynomial.py` : PolynomialOperator implementation (AFAs)
- `src/automatix/weights/guard_weights.py` : Guard evaluation logic

**Morphata foundation** :
- `packages/morphata/src/morphata/spec.py` : Base automaton interfaces
- `packages/morphata/src/morphata/examples/nfa.py` : NFA implementation
- `packages/morphata/src/morphata/examples/ltl.py` : LTL to AFA conversion
- `packages/morphata/src/morphata/examples/strel.py` : STREL automaton

**Algebraic operations** :
- `packages/algebraic/src/algebraic/__init__.py` : Canonical public API / re-exports
- `packages/algebraic/src/algebraic/ops/` : Backend-aware array operations
- `packages/algebraic/src/algebraic/spec.py` : Semiring and algebra interfaces
- `packages/algebraic/src/algebraic/semirings.py` : Concrete semiring implementations
- `packages/algebraic/src/algebraic/array/` : `AlgebraicArray` implementation
- `packages/algebraic/src/algebraic/polynomials/` : `PolyDict` , `MonomialBasis` ,
  `RankDecomposition`

**Configuration** :
- `pyproject.toml` : Workspace configuration, dependencies, tool settings
