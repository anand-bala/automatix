# Automatix Architecture Documentation

## Overview

The `automatix` library (v0.5.0) provides:

1. **Top-level abstractions** (weights.py) - WeightFunction definitions
2. **Algebra module** (algebra/) - Pure interface definitions and semiring registry
3. **NFA module** (nfa/) - Finite automaton implementations
4. **Clear separation of concerns** - Guards, weights, and semirings are decoupled

This architecture enables:
- Pluggable backend support (JAX, PyTorch, NumPy)
- Flexible weight function patterns
- Clean public API

---

## Weight Functions Architecture (Phase 2)

**Location**: `src/automatix/weights.py` (top-level module)

**Key Types**:
- `InputSymbol` - Concrete input data (vector in state space)
- `Guard` - Guard expression (str or logic_asts.Expr)
- `SemiringValue` - Weight value in target semiring
- `WeightFunction = Callable[[InputSymbol, Guard], SemiringValue]`

**Rationale for Top-Level Placement**:
- Weight functions bridge both NFA and AFA modules
- They explicitly reason about guards (fundamental to automata)
- Provides natural import: `from automatix import WeightFunction`

---

## Automatix Algebra Module Architecture

## Directory Structure

```
src/automatix/algebra/
├── __init__.py                 Public API exports
├── spec.py                     Pure interface definitions
├── registry.py                 Semiring registry and factory
├── abc.py                      Deprecated (kept for reference)
│
├── backends/                   Backend implementations
│   ├── __init__.py
│   ├── _base.py               Shared utilities
│   ├── jax_.py                JAX semirings (production)
│   ├── torch_.py              PyTorch semirings (v0.6.0)
│   ├── numpy_.py              NumPy semirings (v0.6.0)
│   └── jax_kernels/           JAX-specific optimizations
│       ├── __init__.py
│       ├── logsumexp.py       Custom logsumexp with proper gradients
│       ├── logsumexp.pyi
│       ├── utils.py           Shared kernel utilities
│       ├── maxplus.py         MaxPlus kernels (planned v0.6.0+)
│       └── log_semiring.py    LogSemiring kernels (planned v0.6.0+)
│
├── abstract/                   Non-backend-specific abstractions
│   ├── __init__.py
│   └── polynomial.py          Polynomial abstractions
│
├── semiring/                   Legacy module (deprecated - will be removed)
│   ├── __init__.py            Re-exports from backends/jax_.py
│   ├── jax_backend.py         Re-exports from backends/jax_.py
│   ├── numpy_backend.py       Deprecated (empty)
│   ├── torch_backend.py       Deprecated (empty)
│   └── utils/                 Moved to backends/jax_kernels/
│       ├── __init__.py
│       ├── logsumexp.py       (MOVED - kept for backward compat reference only)
│       └── logsumexp.pyi
│
└── polynomials/               Polynomial implementations
    ├── __init__.py
    ├── boolean.py             BooleanPolynomial with BDD backend
    └── demorgan.py            Deprecated (empty)
```

## Core Interfaces (spec.py)

### AbstractSemiring

The base interface for all semiring implementations.

```python
class AbstractSemiring(ABC):
    # Creation
    @staticmethod
    def zeros(shape) -> Array: ...
    @staticmethod
    def ones(shape) -> Array: ...

    # Operations
    @classmethod
    def add(cls, x1, x2) -> Array: ...      # Semiring addition (+)
    @classmethod
    def multiply(cls, x1, x2) -> Array: ... # Semiring multiplication (*)
    @classmethod
    def sum(cls, a, axis=None) -> Array: ...
    @classmethod
    def prod(cls, a, axis=None) -> Array: ...

    # Derived operations
    @classmethod
    def vdot(cls, a, b) -> Array: ...       # Semiring dot product
    @classmethod
    def matmul(cls, a, b) -> Array: ...     # Semiring matrix product

    # Semiring properties
    is_additively_idempotent: bool
    is_multiplicatively_idempotent: bool
    is_commutative: bool
    is_simple: bool
```

### AbstractNegation

Interface for negation operation (~) on semiring elements.

```python
class AbstractNegation(ABC):
    @classmethod
    def negate(cls, x: Array) -> Array: ... # Involution operation
```

### AbstractDeMorganAlgebra

Combines semiring + negation with specific properties.

```python
class AbstractDeMorganAlgebra(AbstractSemiring, AbstractNegation):
    # Automatically sets all idempotence/commutativity flags to True
    is_additively_idempotent = True
    is_multiplicatively_idempotent = True
    is_commutative = True
    is_simple = True
```

## Backend Implementations (backends/)

### JAX Backend (jax_.py) - PRODUCTION

11 semiring implementations designed for automatic differentiation:

1. **CountingSemiring** - Standard real arithmetic (R, +, *, 0, 1)
2. **MaxMinSemiring** - Min-max lattice (R cup {-inf, inf}, max, min, -inf, inf)
3. **LeftMaxMinSemiring** - Min-max on negative reals (R_<=0 cup ...)
4. **RightMaxMinSemiring** - Min-max on positive reals (R_>=0 cup ...)
5. **MaxMinAlgebra** - De Morgan variant with negation
6. **LSEMaxMinSemiring** - Smooth min-max using logsumexp
7. **LeftLSEMaxMinSemiring** - LSE variant for negative reals
8. **RightLSEMaxMinSemiring** - LSE variant for positive reals
9. **MaxPlusSemiring** - Tropical algebra (R_<=0, max, +, -inf, 0)
10. **LogSemiring** - Log-domain arithmetic (R_<=0, logsumexp, +, -inf, 0)
11. **LatticeAlgebra** - Simple lattice with De Morgan properties

All JAX semirings:
- Support `jax.jit` compilation
- Support `jax.vmap` for batch operations
- Have proper gradient definitions for autodiff
- Use pure functional JAX operations (no Python loops)

### PyTorch Backend (torch_.py) - PLANNED (v0.6.0)

Stub for PyTorch tensor-based semirings. Will include:
- Device placement support (CPU/GPU)
- torch.nn.Module integration for learnable weights
- torch.optim compatibility
- Gradient computation via standard autograd

### NumPy Backend (numpy_.py) - PLANNED (v0.6.0)

Stub for NumPy-based semirings. Will include:
- CPU-only operations
- Efficient einsum-based matrix operations
- Lower memory footprint for large automata

### JAX Kernels (jax_kernels/) - EXTENSIBLE

Dedicated package for custom forward/backward pass implementations:

Current implementations:
- **logsumexp.py** - Custom logsumexp with proper gradient handling using jax.custom_vjp
- **utils.py** - Shared kernel utilities (decorator, helpers for gradient computation)

Planned optimizations (v0.6.0+):
- **maxplus.py** - MaxPlus-specific forward/backward kernels
- **log_semiring.py** - LogSemiring numerical stability optimizations

This structure scales for adding specialized kernels:

```python
from automatix.algebra.backends.jax_kernels import kernel, logsumexp

@kernel
def forward_maxplus_custom(x, y):
    # Custom optimized implementation
    return jnp.maximum(x, y)

@kernel
def backward_maxplus_custom(grad_out, x, y):
    # Efficient gradient computation
    ...
```

## Semiring Registry (registry.py)

Centralized factory for discovering and instantiating semirings.

### Getting Semirings

```python
from automatix.algebra import get_semiring

# Get a semiring by name
MaxPlus = get_semiring("MaxPlus", backend="jax")
Counting = get_semiring("Counting", backend="jax")

# All available semirings
semirings = list_semirings("jax")
# ['Counting', 'LeftLSEMaxMin', 'LeftMaxMin', 'Lattice', 'Log', 'LSEMaxMin',
#  'MaxMin', 'MaxMinAlgebra', 'MaxPlus', 'RightLSEMaxMin', 'RightMaxMin']

# Available backends
backends = get_available_backends()
# ['jax']  (v0.6.0: ['jax', 'torch', 'numpy'])
```

### Registering New Semirings

```python
from automatix.algebra import register
from automatix.algebra.spec import AbstractSemiring

@register("MyCustomSemiring", backend="jax")
class MyCustomSemiring(AbstractSemiring):
    @staticmethod
    def zeros(shape):
        ...
    # ... implement other methods
```

## Public API (\_\_init\_\_.py)

The main entry point for all algebra functionality.

```python
# Abstractions
from automatix.algebra import (
    AbstractSemiring,
    AbstractNegation,
    AbstractDeMorganAlgebra,
    AbstractPolynomial,
    PolynomialManager,
)

# Registry
from automatix.algebra import (
    get_semiring,
    register,
    unregister,
    list_semirings,
    get_available_backends,
)

# JAX semirings (convenience imports)
from automatix.algebra import (
    CountingSemiring,
    MaxPlusSemiring,
    LogSemiring,
    LatticeAlgebra,
    # ... and others
)

# Polynomials
from automatix.algebra import (
    BooleanPolynomial,
    BooleanPolyCtx,
)
```

## Backward Compatibility

Old import paths still work but are deprecated:

```python
# Old way (still works but deprecated)
from automatix.algebra.semiring.jax_backend import MaxPlusSemiring
from automatix.algebra.abc import AbstractPolynomial

# New way (recommended)
from automatix.algebra import MaxPlusSemiring
from automatix.algebra.spec import AbstractPolynomial
```

## Design Principles

### 1. Pure Interfaces (spec.py)

- No implementation code
- Only abstract base classes and protocols
- Backend-agnostic
- Serves as single source of truth for API contract

### 2. Modular Backends

- Each backend is independent
- Can be developed/tested separately
- No dependencies between backends
- Future: PyTorch and NumPy backends coexist with JAX

### 3. Centralized Registry

- Single place to discover available semirings
- String-based lookup enables configuration-driven code
- Extensible: users can register custom semirings
- Metadata: backend info, properties, etc.

### 4. JAX-First Approach (v0.5.0)

- JAX backend is mature and production-ready
- Multi-backend support (Phase 3) deferred to v0.6.0
- Research can start immediately without backend maintenance burden

### 5. Clean Public API

- One import path: `from automatix.algebra import ...`
- No need to know about backend internals
- Consistent with Python conventions

## Key Dependencies

- **JAX** - Primary backend for v0.5.0
- **Equinox** - For learnable weight functions (Phase 2)
- **logic-asts** - Expression handling for predicates
- **networkx** - Graph representation of automata
- **typing_extensions** - Modern type hints (Python 3.11+)
- **jaxtyping** - Type hints for array shapes

## Migration from v0.4.0

See MIGRATION.md (created in Phase 4: Polish & Release) for detailed upgrade guide.

Quick changes:
1. Replace `from automatix.algebra.semiring.jax_backend import ...` with `from automatix.algebra import ...`
2. Use `get_semiring()` factory instead of direct class imports when appropriate
3. Updated class paths but all functionality is the same

## Weight Functions Architecture (Phase 2 - In Design)

### Design Pattern: Runtime Validation

Weight functions are implemented using a **runtime validation pattern** rather than
semiring-parametric types:

**Key Principles**:
1. **Semiring-Agnostic Design** - Weight functions are pure data structures that don't
   know about semirings
2. **Output Domain Flexibility** - Weight functions output values in any domain; the
   semiring defines how those values are interpreted
3. **Fail-Fast Construction** - Validation happens when a weight function is attached to
   an automaton with a specific semiring
4. **Clear Separation of Concerns** - Automata handle structure, weight functions
   handle value generation, semirings handle algebraic operations

**Architecture Pattern**:
```python
# Weight function: pure, semiring-agnostic
class GuardWeightFunction:
    def __call__(self, guard: Guard) -> Any:
        # Returns a value in some domain (user's responsibility)
        return some_value

# Automaton: expects weights and a semiring instance
class NFA(Generic[S]):
    def __init__(
        self,
        semiring: type[S],
        weights: GuardWeightFunction,
        ...
    ):
        # Validate at construction: weights' output domain matches semiring
        self._validate_weights(weights, semiring)
        self.weights = weights
        self.semiring = semiring

# Usage
maxplus = get_semiring("MaxPlus", backend="jax")
weights = GuardWeightFunction(...)
nfa = NFA(semiring=maxplus, weights=weights)  # Validation happens here
```

**Advantages**:
- Users don't need to understand semiring type parameters
- Weight functions are simple callables with clear I/O contracts
- Easy to document: "This weight function should output MaxPlus values (real numbers)"
- Semirings and weights are truly decoupled
- Simpler testing of weight functions in isolation

**Validation Approach**:
- Check weight function outputs on a sample of inputs
- Validate against semiring's expected domain (documented per semiring)
- Clear error messages for mismatches
- Optional: strict type hints for additional safety

### Future Patterns (v0.6.0+)

1. **Learnable Weights** - Equinox modules with JAX pytree structure
2. **Weight Composition** - Combine guard + global + learnable weights
3. **Numerical Stability** - Domain-specific validators for log-domain, tropical, etc.

## Future Enhancements

### v0.6.0 (Phase 3-4)
- PyTorch backend implementation
- NumPy backend implementation
- Cross-backend testing and validation
- Performance benchmarking
- Advanced weight function patterns

### v0.7.0+
- Sparse matrix support (COO/CSR formats)
- Backend-specific optimizations
- Custom gradient implementations for special cases
- Integration with external libraries (TensorFlow, etc.)

## Type Checking

All code should pass `mypy` strict mode:

```bash
mypy src/automatix/algebra/ --strict
```

Array types use `jaxtyping` for precise shapes:

```python
from jaxtyping import Num, Array

def vdot(a: Num[Array, " n"], b: Num[Array, " n"]) -> Num[Array, ""]:
    return sum_i (a_i * b_i)
```

## Testing Strategy

- **Unit tests** - Each semiring individually
- **Integration tests** - Multiple semirings in pipelines
- **Equivalence tests** - JAX vs PyTorch vs NumPy (v0.6.0+)
- **Property-based tests** - Semiring axioms (hypothesis library)
- **Gradient tests** - Autodiff correctness

See `tests/algebra/` for detailed test organization.

---

## Polynomial Representations (Future Work - v0.8.0)

### Current State (v0.5.0)

The AFA module uses polynomial representations via the `dd` package:
- **Current**: Boolean Decision Diagrams (BDDs) for Boolean polynomials
- **Limitation**: BDD properties (canonicity, efficiency) don't hold for arbitrary semirings
- **Scope**: AFA module is STREL-specific and Boolean-only

### Design Blockers for Generalization

To support AFA with arbitrary semirings, we need:

1. **Custom Polynomial Representations**
   - BDDs require properties: idempotence, commutativity, specific absorption laws
   - Not all semirings satisfy these properties
   - Example: Min-Max (tropical) semiring needs different representation

2. **Algebra Class Implementations**
   - De Morgan algebras (min-max, soft min-max)
   - Boolean algebra specializations
   - Ring algebra support
   - Each requires custom polynomial encoding

3. **Operations**
   - Polynomial operations over semiring operations
   - Negation handling (De Morgan's laws)
   - Simplification/canonicalization strategies

### Path to v0.8.0

1. **Design Phase**
   - Research polynomial representations for different algebra classes
   - Document requirements per semiring family
   - Define abstract polynomial interface

2. **Implementation Phase**
   - Implement custom polynomial classes for key semiring families
   - Create De Morgan algebra support
   - Add Boolean algebra specializations

3. **Integration Phase**
   - Generalize AFA beyond STREL
   - Connect with weight function abstraction (if applicable)
   - Create comprehensive test suite

### References

- STREL-AFA paper: `.cache/strel-afa/` (archived)
- Current AFA implementation: `src/automatix/afa/strel.py` (Boolean-only)
- Polynomial theory: Golan, Kuich references in paper bibliography
