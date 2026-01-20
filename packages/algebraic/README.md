# Algebraic: Semiring Algebra for JAX

A Python package providing **semiring algebra implementations** optimized for
JAX with differentiable operations.

## Overview

This package provides abstract semiring interfaces and concrete implementations
for:

- **Tropical semirings** (MinPlus, MaxPlus) with smooth variants for
  differentiability
- **Max-Min algebras** for robustness semantics
- **Boolean algebras** with De Morgan and Heyting algebra variants
- **Counting semirings**
- **Custom semirings** via the extensible interface

## Features

- **AlgebraicArray**:
  JAX arrays with semiring semantics - override `+`, `*`, `@` to use custom
  algebras
- **JAX-First**:
  Optimized for JAX with JIT compilation, vmap, and automatic differentiation
- **Differentiable Kernels**:
  Smooth approximations of boolean and tropical operations for neural networks
- **Rich Semiring Library**:
  Tropical, Boolean, Max-Min, Counting, and custom semirings
- **Polynomial Algebras**:
  Sparse and dense multilinear polynomials over semirings


## Quick Start

### Recommended Imports

For the best experience with `algebraic`, use these imports:

```python
import algebraic.numpy as alge  # For array operations and creation
from algebraic import jit, vmap  # For JAX transformations
from algebraic.semirings import tropical_semiring, boolean_algebra  # For semirings
```

These provide a seamless interface that automatically handles `AlgebraicArray`
integration with JAX without manual `quax.quaxify` calls.

### Basic Semiring Operations

```python
from algebraic.semirings import tropical_semiring, max_min_algebra, boolean_algebra

# Tropical semiring (MaxPlus: max is addition, + is multiplication)
maxplus = tropical_semiring(minplus=False)
a = maxplus.add(2.0, 3.0)  # max(2, 3) = 3
b = maxplus.mul(2.0, 3.0)  # 2 + 3 = 5

# Tropical semiring (MinPlus: min is addition, + is multiplication)
minplus = tropical_semiring(minplus=True)  # or just tropical_semiring()
c = minplus.add(2.0, 3.0)  # min(2, 3) = 2
d = minplus.mul(2.0, 3.0)  # 2 + 3 = 5

# Max-Min algebra (for robustness/STL semantics)
maxmin = max_min_algebra()
e = maxmin.add(-0.5, 0.2)  # max(-0.5, 0.2) = 0.2
f = maxmin.mul(-0.5, 0.2)  # min(-0.5, 0.2) = -0.5

# Boolean algebra
bool_alg = boolean_algebra(mode="logic")
true = bool_alg.one
false = bool_alg.zero
result = bool_alg.add(true, false)  # True OR False = True
```

### AlgebraicArray: JAX Arrays with Semiring Semantics

The `AlgebraicArray` class wraps JAX arrays and overrides arithmetic operations
to use semiring semantics.
It integrates seamlessly with JAX transformations like `jit`, `vmap`, and
`grad`.

**Recommended**:
Use `algebraic.numpy` (imported as `alge`) for array creation and operations:

```python
import algebraic.numpy as alge
from algebraic.semirings import tropical_semiring

# Create algebraic arrays with tropical semiring
tropical = tropical_semiring(minplus=True)
a = alge.array([1.0, 2.0, 3.0], tropical)
b = alge.array([4.0, 5.0, 6.0], tropical)

# Element-wise operations use semiring semantics
c = a + b  # Tropical addition: [min(1,4), min(2,5), min(3,6)] = [1, 2, 3]
d = a * b  # Tropical multiplication: [1+4, 2+5, 3+6] = [5, 7, 9]

# Reductions use semiring operations (via algebraic.numpy)
total = alge.sum(a)  # min(1, 2, 3) = 1
product = alge.prod(a)  # 1 + 2 + 3 = 6

# Matrix multiplication with @ operator
A = alge.array([[1.0, 2.0], [3.0, 4.0]], tropical)
B = alge.array([[5.0, 6.0], [7.0, 8.0]], tropical)
C = A @ B  # Tropical matmul: C[i,j] = min_k(A[i,k] + B[k,j])
# Result: [[6, 7], [8, 9]]
```

### Boolean Algebra for Graph and Logic Operations

```python
import algebraic.numpy as alge
from algebraic.semirings import boolean_algebra

# Boolean algebra for reachability
bool_alg = boolean_algebra(mode="logic")

# Adjacency matrix: edge from i to j
adj = alge.array([
    [False, True,  False],
    [False, False, True],
    [True,  False, False]
], bool_alg)

# Matrix multiplication computes 2-step reachability
reach_2 = adj @ adj
# reach_2[i,j] = True if there's a path of length 2 from i to j

# Transitive closure: adj + adj^2 + adj^3 + ...
reach = adj
for _ in range(3):
    reach = reach + (reach @ adj)
# reach[i,j] = True if there's any path from i to j
```

### Smooth Boolean Operations for Learning

```python
import algebraic.numpy as alge
from algebraic.semirings import boolean_algebra

# Differentiable boolean operations for neural networks
smooth_bool = boolean_algebra(mode="smooth", temperature=10.0)
soft_bool = boolean_algebra(mode="soft")

# Example: Soft logical operations on continuous values
x = alge.array([0.9, 0.8, 0.1], soft_bool)
y = alge.array([0.7, 0.3, 0.2], soft_bool)

# Soft AND: element-wise multiplication
z_and = x * y  # [0.63, 0.24, 0.02]

# Soft OR: probabilistic OR formula
z_or = x + y  # [0.97, 0.86, 0.28]
```

### JAX Transformations

`AlgebraicArray` works seamlessly with JAX's transformation system.

**Recommended**:
Use the wrapped transformations from `algebraic` instead of manually using
`quax.quaxify`:

```python
import jax
import jax.numpy as jnp  # For jnp.inf
import algebraic.numpy as alge
from algebraic import jit, vmap  # Use these instead of jax.jit/jax.vmap with quax.quaxify
from algebraic.semirings import tropical_semiring, boolean_algebra

tropical = tropical_semiring(minplus=True)

# JIT compilation - use algebraic.jit
@jit
def shortest_paths(dist_matrix):
    """Compute all-pairs shortest paths using tropical matrix multiplication."""
    n = dist_matrix.shape[0]
    result = dist_matrix
    for _ in range(n - 1):
        result = result @ dist_matrix
    return result

D = alge.array([[0., 1., jnp.inf],
                [jnp.inf, 0., 1.],
                [1., jnp.inf, 0.]], tropical)
shortest = shortest_paths(D)

# Vectorization with vmap - use algebraic.vmap
bool_alg = boolean_algebra(mode="soft")

@vmap
def process_graph(adj):
    """Process a single graph with AlgebraicArray."""
    result = adj
    for _ in range(1):  # Compute 2-step reachability
        result = result @ adj
    return result

# Create batched AlgebraicArray (shape: [batch, rows, cols])
batch_adj = alge.array([
    [[0.0, 1.0], [1.0, 0.0]],
    [[1.0, 0.0], [0.0, 1.0]]
], bool_alg)

# vmap over batch dimension
batch_reach = process_graph(batch_adj)

# Automatic differentiation (with differentiable modes)
smooth_bool = boolean_algebra(mode="smooth", temperature=10.0)

@jax.grad
def loss_fn(x):
    """Example: compute gradient of a soft boolean expression."""
    result = alge.sum(x * x)  # Soft AND reduction
    # Extract underlying data for gradient computation
    from algebraic import AlgebraicArray
    return result.data if isinstance(result, AlgebraicArray) else result

x = alge.array([0.9, 0.8, 0.7], smooth_bool)
gradient = loss_fn(x)
```

**Note**:
For operations on `AlgebraicArray` that need to be JIT-compiled or vectorized,
use `from algebraic import jit, vmap` instead of `jax.jit` and `jax.vmap`.
These wrappers automatically handle the `quax.quaxify` integration for you.

### Advanced Features

#### Functional Index Updates

`AlgebraicArray` supports JAX-style functional index updates with semiring
operations:

```python
import algebraic.numpy as alge
from algebraic.semirings import tropical_semiring

tropical = tropical_semiring(minplus=True)
arr = alge.array([1.0, 2.0, 3.0, 4.0], tropical)

# Functional updates (returns new array)
new_arr = arr.at[1].set(0.5)  # Set index 1 to 0.5

# Add using semiring addition (min for tropical)
updated = arr.at[1].add(1.5)  # arr[1] = min(2.0, 1.5) = 1.5

# Multiply using semiring multiplication (+ for tropical)
scaled = arr.at[2].multiply(2.0)  # arr[2] = 3.0 + 2.0 = 5.0
```

#### Multilinear Polynomials

Work with sparse and dense polynomial representations over semirings:

```python
from algebraic.polynomials import SparsePolynomial, MonomialBasis
from algebraic.semirings import boolean_algebra

bool_alg = boolean_algebra(mode="logic")

# Sparse representation (efficient for few terms)
x0 = SparsePolynomial.variable(0, num_vars=3, algebra=bool_alg)
x1 = SparsePolynomial.variable(1, num_vars=3, algebra=bool_alg)
p = x0 * x1 + x1  # Polynomial: (x0 AND x1) OR x1

# Evaluate at a point
result = p.evaluate({0: True, 1: False, 2: True})

# Dense monomial basis (efficient for many terms)
mb0 = MonomialBasis.variable(0, num_vars=2, algebra=bool_alg)
mb1 = MonomialBasis.variable(1, num_vars=2, algebra=bool_alg)
q = mb0 * mb1  # Represented as dense tensor
```

## Core Concepts

### Semirings

A semiring $(S, \oplus, \otimes, \mathbf{0}, \mathbf{1})$ consists of:

- **Addition** ($\oplus$):
  Combines alternative paths/outcomes
- **Multiplication** ($\otimes$):
  Combines sequential compositions
- **Additive identity** ($\mathbf{0}$):
  Identity for $\oplus$
- **Multiplicative identity** ($\mathbf{1}$):
  Identity for $\otimes$

### Lattices

Bounded distributive lattices specialize semirings where:

- **Join** ($\lor$) = Addition ($\oplus$)
- **Meet** ($\land$) = Multiplication ($\otimes$)
- **Top** = Multiplicative identity ($\mathbf{1}$)
- **Bottom** = Additive identity ($\mathbf{0}$)

## Available Semirings

| Name | Addition | Multiplication | Use Case |
|------|----------|----------------|----------|
| **Boolean** | Logical OR | Logical AND | Logic, SAT |
| **Tropical (MaxPlus)** | max | + | Optimization, path problems |
| **Tropical (MinPlus)** | min | + | Shortest paths, distances |
| **Max-Min** | max | min | Robustness degrees, STL |
| **Counting** | + | $\times$ | Counting paths |

## Use Cases

### Graph Algorithms

- **Shortest paths**:
  Use tropical semirings for Floyd-Warshall algorithm
- **Reachability**:
  Boolean algebra for transitive closure
- **Path counting**:
  Counting semiring for enumeration

### Formal Verification

- **Temporal logic**:
  Signal Temporal Logic (STL) with max-min algebra
- **Automata theory**:
  Weighted automata with tropical semirings
- **Model checking**:
  Boolean polynomials for state space exploration

### Machine Learning

- **Differentiable logic**:
  Soft/smooth boolean operations for neural networks
- **Attention mechanisms**:
  Tropical attention for robust aggregation
- **Graph neural networks**:
  Semiring-based message passing

### Optimization

- **Dynamic programming**:
  Tropical semirings for Bellman equations
- **Constraint satisfaction**:
  Boolean algebra for SAT solving
- **Resource allocation**:
  Max-min algebra for bottleneck optimization
