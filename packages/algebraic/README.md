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

- **JAX-First**:
  Optimized for JAX with JIT compilation and automatic differentiation
- **Differentiable Kernels**:
  Smooth approximations of boolean operations for learning
- **Flexible**:
  Use semirings as objects or as kernels in JAX pytrees


## Quick Start

### Basic Semiring Operations

```python
from algebraic.backends.jax import tropical_semiring, max_min_algebra

# Tropical semiring (MaxPlus: max is addition, + is multiplication)
maxplus = tropical_semiring(semiring_type="MaxPlus")
a = maxplus.add(2.0, 3.0)  # max(2, 3) = 3
b = maxplus.mul(2.0, 3.0)  # 2 + 3 = 5

# Max-Min algebra (for robustness/STL semantics)
maxmin = max_min_algebra()
c = maxmin.add(-0.5, 0.2)  # max(-0.5, 0.2) = 0.2
d = maxmin.mul(-0.5, 0.2)  # min(-0.5, 0.2) = -0.5

# Boolean algebra
from algebraic.backends.jax import boolean_algebra
bool_alg = boolean_algebra()
true = bool_alg.one
false = bool_alg.zero
result = bool_alg.add(true, false)  # True OR False = True
```

### Array Operations with Semirings

```python
import jax.numpy as jnp
from algebraic.backends.jax import tropical_semiring

semiring = tropical_semiring(semiring_type="MinPlus")

# Dot product in semiring
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
result = semiring.vdot(a, b)  # min(1+4, 2+5, 3+6) = min(5, 7, 9) = 5

# Matrix multiplication in semiring
A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
C = semiring.matmul(A, B)  # Shortest path computation
```

### Smooth Boolean Operations for Learning

```python
from algebraic.backends.jax import boolean_algebra

# Smooth AND/OR for backpropagation
smooth_kernel = boolean_algebra(mode="smooth", temperature=0.1)
soft_and = smooth_kernel.mul(0.9, 0.8)  # Smooth AND: 0.9 * 0.8 H 0.72
soft_or = smooth_kernel.add(0.1, 0.2)   # Smooth OR: 0.1 + 0.2 H 0.27
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
