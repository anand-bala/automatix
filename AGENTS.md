# Automatix: Agent Context

**Version**: v0.5.0  
**Last Updated**: 2026-01-25

## Quick Start for Agents

### Project Purpose
Automatix is a library for **weighted automata over semirings** with a focus on:
- Flexible weight functions mapping `(input, guard) -> semiring_value`
- JAX implementations with automatic differentiation support
- Polynomial representations for alternating automata
- LTL/LTLf temporal logic specifications
- Quantitative monitoring and synthesis

### Build, Lint, and Test Commands

**Testing**:
```bash
# Run all tests (uses --lf for last-failed)
just test
# OR
uv run --dev --frozen pytest --lf

# Run all tests verbosely without --lf
uv run --dev --frozen pytest -v

# Run a single test file
uv run --dev --frozen pytest tests/test_weight_functions.py -v

# Run a single test function
uv run --dev --frozen pytest tests/test_weight_functions.py::TestWeightFunctionBasics::test_constant_weight_function -v

# Run tests for specific package
uv run --dev --frozen pytest packages/morphata/tests/ -v
uv run --dev --frozen pytest packages/algebraic/tests/ -v

# Run with coverage
uv run --dev --frozen pytest --cov=automatix --cov-report=term
```

**Linting and Formatting**:
```bash
# Format and lint (fixes issues automatically)
just fmt
# OR
uv run --frozen ruff format
uv run --frozen ruff check --output-format concise --fix --exit-non-zero-on-fix

# Format + type check
just lint
```

**Type Checking**:
```bash
# Run all type checkers
just type-check

# Run specific type checkers
uv run --frozen mypy --strict
uv run --frozen ty check --output-format concise
uv run --frozen pyrefly check --output-format min-text
```

**Development Setup**:
```bash
# Set up dev environment (sync dependencies)
just dev
# OR
uv sync --all-packages --frozen --inexact --dev

# For CUDA support (set CUDA_VERSION=12 or 13)
CUDA_VERSION=12 just dev
```

## Code Style Guidelines

### General Principles
- **Type Safety**: All code must pass `mypy --strict`. Use explicit type annotations.
- **No Unicode/Emojis**: Use plain ASCII text. For math symbols, use LaTeX in inline code blocks.
- **Line Length**: Maximum 127 characters (configured in ruff).
- **Indentation**: 4 spaces (never tabs).
- **Breaking Changes OK**: v0.x has no backward compatibility guarantees - prefer refactoring over deprecation.

### Imports
- Always use `from __future__ import annotations` at the top.
- Order: stdlib, third-party, local (ruff handles this automatically).
- Use absolute imports from package roots:
  - `from automatix.operators import MatrixOperator`
  - `from morphata.examples.nfa import NFA` (NOT `morphata.automata`)
  - `from algebraic.semirings import tropical_semiring`
- Prefer importing specific items over module imports.
- Type-only imports should use `typing.TYPE_CHECKING` when needed to avoid circular dependencies.

### Naming Conventions
- Classes: `PascalCase` (e.g., `MatrixOperator`, `WeightFunction`)
- Functions/methods: `snake_case` (e.g., `cost_transitions`, `add_location`)
- Constants: `UPPER_SNAKE_CASE` for true constants
- Type variables: Single uppercase letter or descriptive name (e.g., `S`, `In`, `State`)
- Private: Prefix with `_` (e.g., `_transition_cache`)

### Type Annotations
- Use modern type syntax: `type` aliases, `|` for unions, generic classes with `[T]`
- JAX array types: Use `jaxtyping` annotations (e.g., `Num[Array, "..."]`, `Shaped[Array, "q q"]`)
- Protocol classes: Mark with `@runtime_checkable` when appropriate
- Generic classes: Use PEP 695 syntax `class Foo[T]:`
- Return types: Always specify, including `-> None`

### Documentation
- Use triple-quoted docstrings for all public classes, functions, and methods.
- Format: NumPy-style docstrings with sections: Parameters, Returns, Notes, Examples.
- Keep module-level docstrings concise (1-3 sentences).
- Inline comments: Use sparingly, prefer self-documenting code.

### Error Handling
- Prefer explicit errors over silent failures.
- Use built-in exception types when appropriate.
- For JAX code: Be aware of tracer semantics - avoid Python control flow on traced values.
- Validate inputs at construction time, not runtime (design decision).

### JAX-Specific Patterns
- Use `equinox.Module` for JAX-compatible classes (not dataclasses).
- Wrap transformations with `quax.quaxify` when using `AlgebraicArray`.
- Use `@eqx.filter_jit` for method JIT compilation.
- Functional style: Avoid mutation, return new objects.
- Array operations: Use `jax.numpy`, not `numpy`.

### Testing
- Test files: `test_*.py` in `tests/` or `packages/*/tests/`
- Test classes: `class Test<Feature>:` with methods `def test_<case>(self) -> None:`
- Use descriptive test names that explain what is being tested.
- Disable specific mypy errors in test files with `# mypy: disable-error-code="..."`
- Use `assert` for test assertions (pytest style).

### Workspace Structure
This is a monorepo with the main `automatix` package and two workspace packages:

```
automatix/
├── src/automatix/              # Main package (weighted automata)
│   ├── spec.py                 # WeightFunction protocol, AcceptanceCondition
│   ├── acc.py                  # State-set acceptance conditions (runtime checking)
│   ├── operators/
│   │   ├── __init__.py         # Re-exports MatrixOperator and PolynomialOperator
│   │   ├── matrix.py           # MatrixOperator for weighted NFAs
│   │   └── polynomial.py       # PolynomialOperator for AFAs
│   ├── automata/
│   │   ├── nfa.py              # Re-exports morphata.examples.nfa
│   │   └── strel.py            # Re-exports morphata.examples.strel
│   └── weights/
│       └── guard_weights.py    # Guard-based weight functions
│
├── packages/morphata/          # Foundation: graph-based automata (pure structural)
│   ├── src/morphata/
│   │   ├── spec.py             # Base automaton interfaces (Domain, TransitionRelation, etc.)
│   │   ├── acceptance.py       # Expression-based acceptance conditions (HOA specs)
│   │   ├── automaton.py        # Core automaton dataclass
│   │   ├── utils.py            # Utility functions
│   │   ├── examples/
│   │   │   ├── nfa.py          # NFA implementation (graph-based)
│   │   │   ├── strel.py        # STREL automaton (spatio-temporal specs)
│   │   │   └── ltl.py          # LTL/LTLf to AFA conversion
│   │   └── hoa/
│   │       ├── parser.py       # HOA format parser
│   │       ├── acc_expr.py     # HOA acceptance expressions
│   │       ├── exporter.py     # HOA format exporter
│   │       └── hoa.lark        # HOA grammar file
│   └── tests/
│       ├── examples/           # Tests for NFA, STREL, LTL
│       └── hoa/                # HOA format test files
│
├── packages/algebraic/         # Semiring algebra and polynomials (JAX-focused)
│   ├── src/algebraic/
│   │   ├── spec.py             # Semiring, BoundedDistributiveLattice interfaces
│   │   ├── semirings.py        # Semiring implementations
│   │   ├── array/
│   │   │   ├── core.py         # AlgebraicArray (semiring-aware JAX arrays)
│   │   │   └── _index_update.py # Indexing operations
│   │   ├── numpy.py            # Semiring-aware numpy operations
│   │   ├── kernels/            # JAX kernels for semiring operations
│   │   └── polynomials/
│   │       ├── sparse.py       # SparsePolynomial (dict-based)
│   │       ├── monomial_basis.py # MonomialBasis (dense tensor)
│   │       └── rank_decomp.py  # RankDecomposition (CP decomposition)
│   └── tests/
│       ├── array/              # AlgebraicArray tests
│       └── polynomials/        # Polynomial tests
│
├── tests/                      # Automatix tests
│   ├── test_weight_functions.py
│   ├── operators/
│   │   └── test_polynomial_operator.py
│   └── nfa/
│       └── test_jax_automaton_operator.py
│
└── examples/                   # Example applications
    ├── weight_functions_demo.py
    ├── motion_planning/
    └── swarm-monitoring/
```

## Core Architecture

### Main Modules (automatix)

**spec.py**:
- Re-exports `BoolExpr as Guard` from morphata.spec
- `WeightFunction[In, AP]` protocol:
  Maps (input, guard) -> semiring value
- `AcceptanceCondition` type alias:
  Union of runtime acceptance conditions

**operators/**:
- `MatrixOperator[S:
  Semiring, In]` (operators/matrix.py):
  JAX module for weighted finite-word NFAs
  - Fields:
    `initial_weights`, `final_weights`, `cost_transitions`
  - Factory:
    `MatrixOperator.make(aut, semiring, weight_function=...)`
  - Returns AlgebraicArray instances for semiring-aware operations
- `PolynomialOperator[Symbol]` (operators/polynomial.py):
  JAX module for alternating finite automata
  - Fields:
    `initial_poly`, `accepting_states`, `num_states`, `algebra`,
    `_transition_cache`
  - Methods:
    `accepts()`, `run_polynomial()`, `step()`, `evaluate_at_accepting()`
  - Factory:
    `from_afa(aut, algebra, *, cache_transitions=True)`
  - Convenience:
    `from_ltl(formula, algebra, *, finite=True, cache_transitions=True)`
  - Represents AFA transitions/runs as multilinear polynomials using
    RankDecomposition

**acc.py**:
Runtime acceptance condition implementations:
- `Finite[Q]`:
  Accept after reaching final state
- `Buchi[Q]`, `CoBuchi[Q]`:
  Omega-regular conditions
- `GeneralizedBuchi[Q]`, `GeneralizedCoBuchi[Q]`:
  Generalized variants
- `Rabin[Q]`, `Streett[Q]`, `Muller[Q]`:
  Pairwise/set-based conditions

**weights/guard_weights.py**:
- `AbstractPredicate[S]`:
  Base class for predicates
- `Predicate`:
  Atomic predicates wrapping user functions
- `And`, `Or`:
  Logical operators over predicates (semiring mul/add)
- `ExprWeightFn`:
  Recursive guard evaluation with memoization

### Morphata Package (Foundation)

**Purpose**:
Pure structural automata representations (no JAX, no semirings)

**spec.py**:
- `Domain[State, Symbol]`:
  Capability-based domain (may be symbolic)
- `TransitionRelation[State, Symbol]`:
  Protocol for transitions
- `AlternatingTransitions`:
  Returns `BoolExpr[State]` (alternating semantics)
- `NonDeterministicTransitions`:
  Returns `Iterable[State]`
- `DeterministicTransitions`:
  Returns single `State`
- `AcceptanceCondition`:
  Abstract base for acceptance

**automaton.py**:
- `Automaton[State, Symbol]`:
  Frozen dataclass with domain, initial, delta, acceptance
- Core data structure representing automata

**utils.py**:
- Utility functions for automata operations
- Helper methods for common automata manipulations

**acceptance.py**:
Expression-based acceptance conditions for HOA specifications:
- `Finite`, `Buchi`, `CoBuchi`, `Parity`, etc.
- Used for parsing HOA format files

**examples/nfa.py**:
- `NFA[AP]`:
  Graph-based NFA with guards as boolean expressions
- States are integer locations
- Transitions:
  (src, dst, guard) triples
- Methods:
  `add_location()`, `add_transition()`, `to_automaton()`

**examples/strel.py**:
- `STRELAutomaton`:
  Spatio-temporal reach-escape logic
- Operators:
  Reach, Escape, Somewhere, Everywhere
- Inputs:
  Graphs with labeled nodes
- Boolean-only currently (no general semiring support)

**examples/ltl.py**:
- `ltl_to_automaton(formula, finite=True)`:
  LTL/LTLf to AFA conversion
- Returns `morphata.Automaton[int, Input[AP]]`
- States are integer indices representing LTL subformulas
- Finite=True:
  AFA with `Finite` acceptance
- Finite=False:
  ABWA with `Buchi` acceptance

**hoa/parser.py**:
- HOA (Hanoi Omega-Automata) v1 format parser
- Supports finite and omega-regular automata
- Returns `morphata.Automaton` instances

**hoa/exporter.py**:
- HOA format exporter
- Converts automata to HOA format for external tools

**hoa/hoa.lark**:
- LALR grammar for HOA format parsing
- Complete v1 specification support

### Algebraic Package (Semirings)

**Purpose**:
Pure semiring algebra (no automata)

**spec.py**:
- `AlgebraicStructure`:
  Base class with properties (idempotent, commutative, etc.)
- `Semiring`:
  add, mul, zero, one operations
- `BoundedDistributiveLattice`:
  Semiring with join/meet semantics
- `DeMorganAlgebra`:
  With complement operation
- `BooleanAlgebra`:
  Full complement + De Morgan laws

**semirings.py**:
Concrete semiring implementations:
- `counting_semiring()`:
  Standard (N, +, *, 0, 1)
- `tropical_semiring(minplus=True)`:
  Min-plus or Max-plus
  - Options:
    smooth approximation, temperature parameter
- `max_min_algebra()`:
  MaxMin algebra (robustness semantics)
- `boolean_algebra(mode="soft")`:
  Boolean logic with soft/hard mode

**array/core.py**:
- `AlgebraicArray`:
  JAX arrays with semiring semantics
- Overrides `+`, `*`, `@` operators to use semiring operations
- Factory functions:
  `zeros(shape, semiring)`, `ones(shape, semiring)`
- Full JAX transformation support (jit, vmap, grad)
- Use `quax.quaxify` to wrap JAX transformations when crossing JIT boundaries

**array/_index_update.py**:
- Indexing operations for AlgebraicArray
- Efficient updates using JAX's indexing primitives

**numpy.py**:
- Semiring-aware numpy operations
- Drop-in replacement for numpy with semiring arithmetic
- Type stubs provided in `numpy.pyi`

**kernels/**:
- JAX kernels for semiring operations
- Optimized implementations of semiring primitives

**polynomials/**:
Three complementary polynomial representations:

1. **SparsePolynomial** (`sparse.py`):
   - Storage:
     `dict[frozenbitarray, coefficient]`
   - Best for:
     Sparse polynomials, any size
   - Space:
     O(number of monomials)

2. **MonomialBasis** (`monomial_basis.py`):
   - Storage:
     Dense tensor of shape (2,) * n
   - Best for:
     Small n (≤ 15), multilinear over lattices
   - Space:
     O(2^n)

3. **RankDecomposition** (`rank_decomp.py`):
   - Storage:
     CP decomposition (rank, degree, num_vars+1)
   - Best for:
     Large n, memory-efficient structured polynomials
   - Space:
     O(rank * degree * num_vars)
   - Key methods:
     - `variable(i, num_vars, algebra)`:
       Create x_i
     - `constant(value, num_vars, algebra)`:
       Create constant
     - `__add__`, `__mul__`:
       Polynomial operations
     - `evaluate(points)`:
       Evaluate at points
     - `compose(replacements)`:
       Variable substitution
   - Works with any `BoundedDistributiveLattice`

## Package Architecture

**Layered Design:**

```
morphata (foundation: pure structural automata)
    ↓
automatix (weighted semantics with semirings)
    ↓
applications
```

**Key Separations:**
1. **morphata.spec**:
   Pure structural interfaces (no JAX, no semirings)
2. **automatix.spec**:
   Extends with WeightFunction protocol and jaxtyping
3. **morphata.acceptance**:
   Expression-based (for HOA specifications)
4. **automatix.acc**:
   State-set-based (for runtime checking)
5. **morphata.examples**:
   Concrete implementations (NFA, STREL, LTL)
6. **automatix.operators**:
   Weighted operators (MatrixOperator with semirings)

**Dependencies:**
- morphata:
  logic-asts, networkx, attrs, lark
- automatix:
  morphata, algebraic, JAX, equinox, quax
- algebraic:
  JAX, quax, bitarray (independent of automata)

**No circular dependencies**:
morphata does not import from automatix.

## Design Principles

1. **Weight Function Scope**:
   Guard-based weight functions (input + boolean expression -> semiring value)
2. **Runtime Validation**:
   Validation at automaton construction time
3. **Semiring-Agnostic**:
   Weight functions don't know target semiring
4. **API Stability**:
   v0.x; breaking changes may occur between releases
5. **Code Style**:
   No emojis/unicode; use LaTeX for math symbols (inline code blocks)
6. **Import Paths**:
   Use `morphata.examples` (not `morphata.automata`)

## Usage Notes and Limitations

- STREL automata use Boolean semantics; semiring-valued STREL is not available.
- MatrixOperator targets finite-word acceptance; omega-regular acceptance is not implemented.
- PolynomialOperator usage is tested with Boolean algebra; other semirings may require additional validation.

## Important Notes for Agents

1. **JAX-First**:
   Optimize for JAX initially
2. **No Deprecation**:
   Delete old code instead of wrapping it
3. **Type Safety**:
   All new code must pass `mypy --strict`
4. **Prefer Refactoring**:
   Update existing patterns rather than special-casing
5. **Documentation**:
   Keep design decisions documented in markdown files
6. **quax.quaxify**:
   Always wrap JAX transformations (jit, vmap, scan) when working with
   AlgebraicArray

## Testing

- **Framework**:
  pytest with jaxtyping
- **Type Safety**:
  100% mypy --strict for all packages
- **JAX Integration**:
  JIT, vmap, gradient operations tested
- **Coverage**:
  Unit tests for all core modules

**Test Locations**:
- `tests/`:
  Automatix tests
- `packages/morphata/tests/`:
  Morphata tests (examples, HOA parser)
- `packages/algebraic/tests/`:
  Algebraic tests (polynomials, semirings, arrays)

## Examples

Located in `examples/`:
- `weight_functions_demo.py`:
  Guard-based weight function patterns
- `motion_planning/`:
  Dynamics and planning examples
- `swarm-monitoring/`:
  Multi-agent coordination specs with STREL

Run examples:
```bash
python examples/weight_functions_demo.py
python examples/swarm-monitoring/monitoring_example.py
```

## Common Patterns

### Creating a Weighted NFA Operator

```python
from algebraic.semirings import tropical_semiring
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
maxplus = tropical_semiring(minplus=False)
operator = MatrixOperator.make(aut, maxplus, weight_function=weight_fn)

# Evaluate
import jax.numpy as jnp
x = jnp.array([2.0])
transitions = operator.cost_transitions(x)
# Access data: transitions.data[i, j]
```

### Working with AlgebraicArray

```python
import jax
import quax
from algebraic.array.core import zeros, ones
from algebraic.semirings import max_min_algebra

algebra = max_min_algebra()
arr1 = zeros((3, 3), algebra)
arr2 = ones((3, 3), algebra)

# Operations use semiring semantics
result = arr1 + arr2  # Uses algebra.add
product = arr1 * arr2  # Uses algebra.mul

# JAX transformations need quax.quaxify
@jax.jit
def compute(x):
    return x + x

# Use quax.quaxify when AlgebraicArray crosses JIT boundary
compute_wrapped = quax.quaxify(compute)
output = compute_wrapped(arr1)
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
from algebraic.polynomials.rank_decomp import RankDecomposition
from algebraic.semirings import boolean_algebra
import jax.numpy as jnp

algebra = boolean_algebra()
num_vars = 3

# Create variables
x0 = RankDecomposition.variable(0, num_vars, algebra)
x1 = RankDecomposition.variable(1, num_vars, algebra)
x2 = RankDecomposition.variable(2, num_vars, algebra)

# Build polynomial: (x0 * x1) + x2
poly = (x0 * x1) + x2

# Evaluate at point
point = jnp.array([1.0, 1.0, 0.0])
result = poly.evaluate(point)
# Extract scalar: result.factors[0, 0, 0].data

# Compose (substitute variables)
composed = poly.compose({0: x2})  # Replace x0 with x2
```

### Using PolynomialOperator with LTL

```python
from automatix.operators import from_ltl, from_afa
from algebraic.semirings import boolean_algebra
from morphata.examples.ltl import ltl_to_automaton
import logic_asts as logic

# Method 1: Direct from LTL formula
algebra = boolean_algebra()
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

**Core automatix**:
- `src/automatix/spec.py`:
  WeightFunction protocol, type definitions
- `src/automatix/operators/matrix.py`:
  MatrixOperator implementation (weighted NFAs)
- `src/automatix/operators/polynomial.py`:
  PolynomialOperator implementation (AFAs)
- `src/automatix/weights/guard_weights.py`:
  Guard evaluation logic

**Morphata foundation**:
- `packages/morphata/src/morphata/spec.py`:
  Base automaton interfaces
- `packages/morphata/src/morphata/examples/nfa.py`:
  NFA implementation
- `packages/morphata/src/morphata/examples/ltl.py`:
  LTL to AFA conversion
- `packages/morphata/src/morphata/examples/strel.py`:
  STREL automaton

**Algebraic operations**:
- `packages/algebraic/src/algebraic/spec.py`:
  Semiring interfaces
- `packages/algebraic/src/algebraic/semirings.py`:
  Concrete semirings
- `packages/algebraic/src/algebraic/array/core.py`:
  AlgebraicArray
- `packages/algebraic/src/algebraic/polynomials/rank_decomp.py`:
  CP decomposition

**Configuration**:
- `pyproject.toml`:
  Workspace configuration, dependencies, tool settings
