# Code Review Checklist - Quick Reference

**Purpose**:
Quick lookup for what needs review and why.

**Last Updated**:
Nov 10, 2025

**Status**:
Kernel architecture refactoring COMPLETE.
Items 3, 7, 8 now resolved.
Focus remaining on polynomial substitution and AFA semantics.

**How to use**:
1. Open this file when reviewing code
2. Find the section/item you're reviewing
3. Check the suggested decision
4. Review the code with decision in mind
5. Update decision in code comments if needed

---

## Critical Path (Items 1, 2, 4, 5, 6)

These are the remaining blocking items for Week 2 AFA implementation.

- [X] **Item 1** (HIGH):
  Basis value pre-computation in Algorithm 4
  - **Decision**:
    On-the-fly computation (Week 1 decided)
  - **Status**:
    IMPLEMENTED

- [X] **Item 2** (HIGH):
  Computing (1-x) in non-field semirings
  - **Decision**:
    Add error check for non-field semirings
  - **Verify**:
    tensor_encoding.py line 275 has proper validation

- [ ] **Item 4** (MEDIUM):
  Like-term collection during substitution
  - **Decision**:
    Brute-force loop for now, optimize later
  - **Verify**:
    Test correctness with simple examples

- [ ] **Item 5** (MEDIUM):
  Sparse successor handling
  - **Decision**:
    Require all states have successors
  - **Verify**:
    substitution.py validates at construction

- [ ] **Item 6** (MEDIUM):
  Batch operations kernel integration
  - **Decision**:
    Skip kernel for now, use naive loop
  - **Verify**:
    Add TODO comment for future optimization

---

## RESOLVED ITEMS (Nov 10, 2025)

The kernel architecture refactoring has resolved the original critical path
items:

- [x] **Item 3** (WAS CRITICAL):
  Polynomial multiplication semantics - RESOLVED
  - **Resolution**:
    AlgebraicStructure enables flexible polynomial operations
  - **Implementation**:
    DecidedProgramming pattern with kernel abstraction

- [x] **Item 7** (WAS CRITICAL):
  Transition representation - RESOLVED
  - **Resolution**:
    Kernel-based weight functions enable both scalar and polynomial weights
  - **Implementation**:
    normalize_semiring() supports both patterns

- [x] **Item 8** (WAS CRITICAL):
  Weight function interface - RESOLVED
  - **Resolution**:
    normalize_semiring() adapter for transparent class/kernel support
  - **Implementation**:
    Both Type[AbstractSemiring] and AlgebraicStructure now accepted

---

## Algorithm 4 (Items 1-2)

**Status**:
Item 1 IMPLEMENTED (Week 1), Item 2 TODO (Add validation)

### Item 1: Basis Pre-computation (DONE)
- **File**:
  `tensor_encoding.py:239-241`
- **Decision**:
  On-the-fly computation (no pre-computation)
- **Status**:
  IMPLEMENTED in Week 1
- **Verification**:
  Run test suite to confirm Algorithm 4 works correctly

### Item 2: Computing (1-x) in Semirings (TODO)
- **File**:
  `tensor_encoding.py:275-278`
- **Type**:
  Correctness validation
- **Decision**:
  Add error check for non-field semirings
- **TODO**:
  Add validation that semiring is field before using negation
- **Check**:
  Line 275 should validate semiring has negate operation or is field

---

## Polynomial Substitution (Items 4, 5, 6)

**Status**:
Item 3 RESOLVED via kernel architecture, Items 4-6 TODO for Week 2

### Item 4: Like-Term Collection (WEEK 2 TODO)
- **File**:
  `substitution.py:72-78`
- **Type**:
  Implementation + Optimization
- **Decision**:
  Brute-force loop first, optimize later
- **TODO**:
  Implement simple loop-based like-term collection
- **Check**:
  Test with multiple monomials combining to same index

### Item 5: Sparse Successors (WEEK 2 TODO)
- **File**:
  `substitution.py:52-53`
- **Type**:
  API validation
- **Decision**:
  Require all states have successors (strict)
- **TODO**:
  Validate all states in range(num_states) have entry in successors dict
- **Check**:
  Clear error message when successor is missing

### Item 6: Batch Operations Integration (DEFERRED)
- **File**:
  `substitution.py:68-70`
- **Type**:
  Optimization (not critical for correctness)
- **Decision**:
  Skip kernel integration for now
- **TODO**:
  Use naive loop for coefficient accumulation
- **Note**:
  Add TODO comment for future optimization with batch_operations kernel

---

## AFA Architecture (Items 9-14)

**Status**:
Items 7-8 RESOLVED via kernel architecture, Items 9-14 TODO for Week 2

### Item 7: Transition Representation (RESOLVED)
- **Status**:
  RESOLVED via kernel architecture
- **Resolution**:
  normalize_semiring() enables flexible weight functions
- **Implementation**:
  Can use both scalar and polynomial weights transparently

### Item 8: Weight Function Interface (RESOLVED)
- **Status**:
  RESOLVED via kernel architecture
- **Resolution**:
  normalize_semiring() adapter accepts both class and kernel
- **Implementation**:
  Backward compatible, no breaking changes

### Item 9: Initial Polynomial (WEEK 2 TODO)
- **File**:
  `automaton.py:82`
- **Decision**:
  Derive from `initial_states` list
- **TODO**:
  Constructor accepts `initial_states:
  List[Q]` and builds polynomial

### Item 10: Acceptance Condition (WEEK 2 TODO)
- **File**:
  `automaton.py:87`
- **Decision**:
  Check nonzero at final states
- **TODO**:
  Accept if any final state has nonzero coefficient

### Item 11: Execute Output (WEEK 2 TODO)
- **File**:
  `operators.py:103`
- **Decision**:
  Return scalar weight (sum of final coefficients)
- **TODO**:
  `execute()` returns K (semiring value), not polynomial

### Item 12: Validation (DEFERRED)
- **File**:
  `operators.py:73`
- **Decision**:
  Lazy validation during execute()
- **Note**:
  Validate AFA state/transitions when first executed, not at construction

### Item 13: Pre-computation (DEFERRED)
- **File**:
  `operators.py:76`
- **Decision**:
  No pre-computation in Week 2
- **Note**:
  Add TODO comment for future JIT/caching optimization

### Item 14: Final States Meaning (DEFERRED)
- **File**:
  `automaton.py:87`
- **Decision**:
  Acceptance only (clear semantics)
- **Note**:
  Final states only used to determine acceptance condition

---

## Week 1 Items (Already Complete)

### Item 15: Polynomial Degree Zero-Check (DONE)
- **File**:
  `ring_polynomials.py:159-177`
- **Status**:
  IMPLEMENTED in Week 1
- **Verification**:
  Works for numeric types (int, float, JAX arrays)

### Item 16: Batch Operations Assumptions (DONE)
- **File**:
  `batch_operations.py:83-91`
- **Status**:
  IMPLEMENTED in Week 1
- **Note**:
  Code correctly assumes commutative semirings for accumulation

---

## Test Coverage Checklist

**Algorithm 4 Tests** (`tests/afa/test_algorithm_4.py` - to create):
- [ ] Correctness:
  Compare Algorithm 4 vs Algorithm 1 results
- [ ] Performance:
  Verify 10-20x speedup
- [ ] Semiring handling:
  Test Boolean and MaxMin
- [ ] Edge cases:
  Constants, single-state polynomials

**Polynomial Substitution Tests** (`tests/afa/test_polynomial_substitution.py` -
to create):
- [ ] Multiplication:
  Simple examples like (a*x + b) * (c*x + d)
- [ ] Substitution:
  x_0 -> Q_0 produces correct result
- [ ] Like-term collection:
  Multiple monomials combine correctly
- [ ] Sparse successors:
  Error handling for missing successor

**AFA Tests** (`tests/afa/test_afa_class.py` - to create):
- [ ] Construction:
  AFA creation with states and weight function
- [ ] Execution:
  execute() on simple inputs
- [ ] Acceptance:
  Verify final_states determine acceptance
- [ ] Output:
  execute() returns correct scalar value

---

## Common Questions While Reviewing

**Q:
Is polynomial multiplication correct?** A:
Check against manual calculation for simple example like (1 + x_0) * (1 + x_1)
Expected:
1 + x_0 + x_1 + x_0*x_1

**Q:
Are coefficients accumulating correctly?** A:
Verify like-term collection groups same monomials and uses semiring.add()

**Q:
Is Algorithm 4 faster?** A:
Run benchmark:
`python examples/polynomial_evaluation_benchmark.py` Should show 10-20x speedup
on q=10

**Q:
Does AFA execute correctly?** A:
Test with single-state automaton accepting simple string patterns

**Q:
Are acceptance states working?** A:
Verify non-final states don't contribute to acceptance

---

## Decision Template

When making a decision on a REVIEW NEEDED item, document it like this:

```python
# REVIEW NEEDED DECISION: [Item number]
# Decision: [Option chosen, e.g., "Option A: ..."]
# Reason: [Why this decision was made]
# Alternative: [What alternative was considered]
# Test: [How to verify this works]
```

Example:
```python
# REVIEW NEEDED DECISION: Item 1
# Decision: Compute basis values on-the-fly
# Reason: Simpler, no pre-computation needed
# Alternative: Pre-compute dict of basis values
# Test: Benchmark shows no performance regression
```

---

## Priority Order for Review

**Session 1** (Critical):
1. Review Item 3 (polynomial multiplication)
2. Review Item 7 (transition representation)
3. Review Item 8 (weight function)

**Session 2** (High):
4.
Review Item 1 (basis pre-computation) 5.
Review Item 2 (semiring negation) 6.
Review Items 4-6 (substitution details)

**Session 3** (Medium):
7.
Review Items 9-11 (AFA API) 8.
Review Items 12-14 (validation, optimization, semantics)

**Session 4** (Low):
9.
Review Items 15-16 (week 1 code details)

---

## Links to Detailed Information

- Full documentation:
  `.cache/WEEK2_STARTUP_SUMMARY.md`
- Implementation plan:
  `.cache/WEEK2_IMPLEMENTATION_PLAN.md`
- Full TODO details:
  `./TODO.md` (this repo)
- Polynomial theory:
  `.cache/AFA_POLYNOMIAL_ARCHITECTURE.md`

---

## Status Indicators

- ✅ Ready to review
- ⏳ Waiting for decision on dependency
- ❌ Blocking issue
- 📝 Documented decision

Update these as you review!
