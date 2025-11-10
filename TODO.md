# Automatix TODO - v0.6.0 AFA Implementation Roadmap

**Status**: Updated Nov 10, 2025 - Kernel architecture complete (Phase 2.75). Ready to begin Phase 3: AFA Polynomial Operations.

**Current Phase**: v0.6.0 - AFA Implementation with Multilinear Polynomials

**Test Results**: 129/129 tests passing. All modules 100% mypy --strict compliant.

---

## Phase Overview

### Phase 1: Foundation (COMPLETE - Nov 5, 2025)
- Pure interface definitions (spec.py)
- Modular backend structure
- Centralized registry system
- JAX kernels infrastructure
- Type safety: 100% mypy strict
- Tests: 18/18 passing

### Phase 2: Weight Functions (COMPLETE - Nov 6, 2025)
- Weight function architecture finalized
- Factory function: make_atomic_predicate_weight_function
- NFA integration via make_automaton_operator
- Tests: 13/13 passing
- Examples: 6 working patterns

### Phase 2.5: Backward Compatibility + Kernels (COMPLETE - Nov 10, 2025)
- AlgebraicStructure kernel abstraction (kernels.py)
- Backward-compatible adapter layer (_compat.py)
- normalize_semiring() for class/kernel conversion
- Extended registry with kernel support
- AbstractSemiring.to_kernel() method for all 11 JAX semirings
- Differentiable Boolean kernels (soft/smooth/STE)
- Updated predicates.py: And, Or, ExprWeightFn accept both class and kernel
- Updated finite_word.py: make_automaton_operator supports both class and kernel
- Full backward compatibility maintained
- Tests: 39 new kernel tests + 90 existing = 129/129 passing

### Phase 3: AFA Implementation (NEXT - Nov 10+)
**Goals for Week 2-4**:
- Polynomial substitution and like-term collection
- AFA class finalization and execution
- Basic STREL to AFA translation (Boolean only)
- Comprehensive test coverage

---

## Critical Path: Week 2 Tasks (Polynomial Operations)

These are the blockers for Week 3-4 AFA integration. Must be completed and tested before proceeding.

### Task 1: Polynomial Multiplication (CRITICAL)

**File**: `src/automatix/algebra/polynomials/substitution.py:17-92`

**Current State**: Skeleton with nested loop over all monomial pairs. No coefficient accumulation logic.

**Decision**: Distribute-all-terms approach (Option A from TODO.md review)

**Semantics**:
For multilinear polynomials, multiply using:
```
(a*x_i + b) * (c*x_j + d) = ac*x_i*x_j + (ad)*x_i + (bc)*x_j + bd

Then apply multilinear constraint x^2 = x (idempotent):
(a*x_i + b) * (c*x_i + d) = ac*x_i + (ad+bc)*x_i + bd = (ac+ad+bc)*x_i + bd
```

**Implementation Checklist**:
- [ ] Implement monomial multiplication using bitwise OR of state indices
- [ ] Accumulate coefficients using semiring.multiply for individual terms
- [ ] Apply multilinear constraint (if result_idx has repeated bits, reduce)
- [ ] Collect like terms using semiring.add
- [ ] Handle zero coefficients correctly
- [ ] Test on Boolean and MaxMin semirings

**Test Requirements**:
- (1 + x₀) * (1 + x₁) = 1 + x₀ + x₁ + x₀*x₁ (Boolean)
- (1 + x₀) * (1 + x₀) = 1 + x₀ (idempotent: x₀² = x₀)
- Zero polynomial cases
- Different semirings (Boolean, MaxMin, etc.)

**Complexity**: O(2^{2q}) where q = num_states. Acceptable for q <= 6.

---

### Task 2: Polynomial Substitution (CRITICAL)

**File**: `src/automatix/algebra/polynomials/substitution.py:95-169`

**Current State**: Skeleton with validation but no substitution logic.

**Decision**: Brute-force like-term collection with idempotent simplification (Option A)

**Algorithm**:
```python
result = zero_polynomial(num_states)

for monomial_idx in range(2^num_states):
    coeff_m = poly.coefficients[monomial_idx]
    alpha_m = int_to_alpha(monomial_idx)

    # For this monomial, compute product of successor polynomials
    # For each state i where alpha_m[i] = 1, include successor[i]
    term_poly = constant_1_polynomial(num_states)
    for state_i where alpha_m[i] = 1:
        term_poly = polynomial_multiply(term_poly, successor[state_i])

    # Scale by coefficient and accumulate
    scaled_poly = polynomial_scalar_mult(coeff_m, term_poly)
    result = polynomial_add(result, scaled_poly)

return result
```

**Key Insight**: Idempotent semirings (a ⊕ a = a) mean multiple accumulations automatically simplify.

**Implementation Checklist**:
- [ ] Implement int_to_alpha and alpha_to_int conversions
- [ ] Compute successor polynomial product for each monomial's state set
- [ ] Implement polynomial_scalar_mult (coefficient scaling)
- [ ] Implement polynomial_add (like-term collection via accumulation)
- [ ] Handle sparse successors (missing → zero polynomial with warning)
- [ ] Test idempotence: substitution of same poly multiple times

**Test Requirements**:
- Single variable substitution: P[x₀ ← Q] = correct evaluation
- Multiple variables: P[x₀ ← Q₀, x₁ ← Q₁]
- Identity: P[x₀ ← x₀] = P (within numerical precision)
- Sparse successors produce warnings and zero contributions
- Idempotence verification: a ⊕ a = a in results

**Complexity**: O(2^q * 2^q) = O(4^q) in worst case (for each monomial, multiply with successor). Deferred optimization: batch operations kernel.

---

### Task 3: Sparse Successor Handling

**File**: `src/automatix/algebra/polynomials/substitution.py:145-148`

**Current State**: Raises ValueError if successor missing.

**Decision**: Default to zero polynomial (Option B - rejecting sink semantics)

**Change Required**:
```python
# OLD:
for state_idx in range(poly.num_states):
    if state_idx not in successors:
        raise ValueError(f"Missing successor for state q_{state_idx}...")

# NEW:
for state_idx in range(poly.num_states):
    if state_idx not in successors:
        logger.warning(f"No successor for state q_{state_idx}, using zero polynomial")
        # Continue (zero will be multiplied in, contributing nothing)
```

**Rationale**: In AFA semantics, missing successor = no outgoing transition = rejecting sink. Zero polynomial is correct identity element.

**Implementation Checklist**:
- [ ] Remove ValueError check
- [ ] Add logger.warning for missing successors
- [ ] Verify zero polynomial semantics (0 * x = 0, 0 + x = x)
- [ ] Test partial successor dicts

---

### Task 4: Helper Functions for Polynomial Operations

**File**: `src/automatix/algebra/polynomials/ring_polynomials.py` or new utilities module

**Required Functions** (may already exist in MultilinearPolynomial class):
- [ ] `polynomial_add(p1, p2, algebra) -> Polynomial` - element-wise addition
- [ ] `polynomial_scalar_mult(scalar, poly, algebra) -> Polynomial` - multiply all coefficients by scalar
- [ ] `zeros(num_states, algebra) -> Polynomial` - zero polynomial
- [ ] `ones(num_states, algebra) -> Polynomial` - constant 1 polynomial
- [ ] `int_to_alpha(idx, q) -> Tuple[int, ...]` - convert index to binary tuple (or verify existing impl)
- [ ] `alpha_to_int(alpha) -> int` - convert binary tuple to index (already in MultilinearPolynomial)

**Status Check**: These may already exist in MultilinearPolynomial or as factory methods. Verify before implementing.

---

### Task 5: Comprehensive Polynomial Tests

**File**: `tests/afa/test_polynomial_substitution.py` (create new)

**Test Coverage**:

1. **Multiplication Tests**:
   - `test_multiply_identity`: P * 1 = P
   - `test_multiply_zero`: P * 0 = 0
   - `test_multiply_commutativity`: P * Q = Q * P (if semiring commutative)
   - `test_multiply_distributivity`: P * (Q + R) = P*Q + P*R
   - `test_multiply_multilinear_idempotence`: (1 + x₀) * (1 + x₀) = 1 + x₀
   - `test_multiply_cross_variable`: (1 + x₀) * (1 + x₁) = 1 + x₀ + x₁ + x₀*x₁
   - `test_multiply_boolean_semiring`: Boolean-specific operations
   - `test_multiply_maxmin_semiring`: MaxMin-specific operations

2. **Substitution Tests**:
   - `test_substitute_single_variable`: P[x₀ ← Q] correct
   - `test_substitute_multiple_variables`: P[x₀ ← Q₀, x₁ ← Q₁]
   - `test_substitute_identity`: P[x₀ ← x₀] = P
   - `test_substitute_constant`: Substituting with constant polynomials
   - `test_substitute_repeated`: Multiple substitution steps maintain semantics
   - `test_substitute_sparse_successors`: Missing successors handled correctly
   - `test_substitute_sparse_warnings`: Warnings logged appropriately

3. **Idempotence Verification** (key for distributive lattices):
   - `test_idempotence_multiplication`: a ⊗ a = a in polynomial multiplication
   - `test_idempotence_addition`: a ⊕ a = a in like-term collection
   - `test_idempotence_substitution`: Idempotent semiring semantics preserved

4. **Integration Tests**:
   - `test_polynomial_evaluation_cycle`: Substitute → evaluate → verify output
   - `test_complex_polynomial_operation`: Multi-step operations on complex polynomials

5. **Edge Cases**:
   - Zero polynomials
   - Constant polynomials (all states have coefficient 0 except constant term)
   - Single-variable polynomials
   - Full multi-variable polynomials (all 2^q terms nonzero)

**Test Organization**:
```python
# tests/afa/test_polynomial_substitution.py
class TestPolynomialMultiplication:
    # Multiplication tests

class TestPolynomialSubstitution:
    # Substitution tests

class TestIdempotenceProperties:
    # Idempotence verification tests

class TestIntegration:
    # End-to-end tests
```

---

## Week 3 Tasks: AFA Class Integration

These depend on Week 2 polynomial operations being complete.

### Task 6: AFA Class Finalization

**File**: `src/automatix/automata/afa/automaton.py`

**Current State**: Placeholder class with minimal structure.

**Implementation Requirements**:

1. **Constructor Parameters**:
   ```python
   class AFA(Generic[Alph, Q]):
       def __init__(
           self,
           initial_states: List[Q],
           weight_function: Callable[[Alph, str], float],  # (input, guard) → weight
           final_states: List[Q],
           semiring: AbstractSemiring | AlgebraicStructure,
           state_list: List[Q] = None,  # Optional explicit state ordering
       ):
   ```

2. **State Representation**:
   - Map Q (user states) to indices 0..num_states-1
   - Store bidirectional mapping for conversion

3. **Initial Polynomial**:
   - Construct from initial_states list
   - Initial polynomial: sum of x_i for each initial state i
   - Example: if initial_states = [0, 2], initial_poly = x₀ + x₂

4. **Acceptance Semantics**:
   - Accept if any final state has nonzero coefficient in result polynomial
   - Example: final_states = [1, 3], accept iff result.coefficients[1] != 0 or result.coefficients[3] != 0

5. **Weight Function**:
   - Must support normalize_semiring (accept both class and kernel)
   - Called during execution: weight_fn(input, guard) → semiring_value

---

### Task 7: PolynomialAutomatonOperator Implementation

**File**: `src/automatix/automata/afa/operators.py`

**Current State**: Placeholder with skeleton methods.

**Implementation Requirements**:

1. **execute() Method**:
   ```python
   def execute(self, inputs: List[Alph]) -> float:
       # Start with initial polynomial
       P = afa.initial_polynomial

       # For each input symbol
       for input_symbol in inputs:
           # Compute successor polynomials Q_i for each state
           successors = {}
           for state_i in afa.states:
               # weight_fn determines coefficient
               Q_i = successor_polynomial(state_i, input_symbol)
               successors[state_i_index] = Q_i

           # Substitute P(Q_0, Q_1, ..., Q_q)
           P = polynomial_substitute(P, successors)

       # Evaluate at final states
       result = 0
       for final_state_idx in afa.final_states:
           result += P.coefficients[final_state_idx]

       return result
   ```

2. **Successor Polynomial Computation**:
   - For each (state, input_symbol) pair
   - Evaluate weight_fn(input_symbol, guard) for relevant guards
   - Build successor polynomial from guard evaluations

3. **Error Handling**:
   - Lazy validation during execute (don't fail in constructor)
   - Clear error messages for missing guards or invalid successors

---

## Week 4 Tasks: STREL Translation

Deferred pending completion of Tasks 1-7.

### Task 8: Basic STREL to AFA Translation (Boolean only)

**File**: `src/automatix/afa/strel.py` (already exists but empty)

**Current State**: Imports only, no implementation.

**Scope**: Boolean semiring only (v0.8.0+ will handle other semirings)

**Key Operations**:
- STL always → AFA (all states must be true)
- STL eventually → AFA (some state path must be true)
- STL until → AFA temporal constraint
- Guard predicates → weight functions

---

## Code Review Items for User Approval

### Item A: Polynomial Multiplication Edge Cases

**Issue**: Should polynomial multiplication with zero polynomials be special-cased?

**Options**:
- A) Return zero (algebraically correct, let idempotence handle)
- B) Add explicit zero check (clearer intent, slight performance cost)
- C) Document assumption and trust idempotence

**Recommendation**: Option C - document the assumption, trust idempotence.

---

### Item B: Successor Evaluation in AFA

**Issue**: How to compute successor polynomial Q_i from weight_function output?

**Options**:
- A) weight_function returns scalar, we build polynomial from predicate matching
- B) weight_function returns full successor polynomial directly
- C) weight_function returns state-to-state transition weights

**Decision from TODO.md**: Option A (matches NFA pattern).

---

## Testing Checklist

- [ ] Task 1: Polynomial multiplication tests (all combinations)
- [ ] Task 2: Polynomial substitution tests (all cases)
- [ ] Task 3: Sparse successor handling tests
- [ ] Task 4: Helper function tests (if new ones created)
- [ ] Task 5: Comprehensive polynomial test suite passing
- [ ] Run full test suite: `python -m pytest tests/ -v`
- [ ] Type check: `python -m mypy src/automatix/ --strict`
- [ ] All 129+ tests passing before Week 3

---

## Known Constraints

1. **De Morgan Algebras Only**: Multilinear polynomials only for distributive lattice semirings
2. **Idempotent Operations**: a ⊕ a = a, a ⊗ a = a in target semirings
3. **Dense Representation**: (2^q,) coefficient vectors (sparse deferred)
4. **JAX-First**: Optimize for JAX compilation
5. **No Tropical**: Tropical semirings deferred to v0.8.0+

---

## Deferred Items (v0.7.0+)

- Batch operations kernel integration (optimization)
- Sparse polynomial representation
- PyTorch backend implementation
- Omega-regular automata (Büchi, co-Büchi)
- Tree automata exploration
- Tropical semiring support

---

## Success Criteria for v0.6.0 Completion

1. All polynomial operations (multiply, substitute, evaluate) implemented and tested
2. AFA class fully specified with proper semantics
3. PolynomialAutomatonOperator executing correctly on input sequences
4. Basic STREL to AFA translation (Boolean only)
5. 150+ tests passing (polynomial + AFA + existing)
6. 100% mypy --strict compliance
7. Examples demonstrating polynomial evaluation workflow
8. Documentation of polynomial semantics in place

---

**Last Updated**: 2025-11-10
**Next Session**: Begin Week 2 tasks (polynomial operations)
**Maintainer**: Anand (primary user)
