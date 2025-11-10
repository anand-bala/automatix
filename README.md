# Symbolic (Weighted) Automata Monitoring

This project implements different automata I use in my research, including
nondeterministic weighted automata and alternating weighted automata.

### Differentiable Automata in [JAX](https://github.com/google/jax)

The `automatix.nfa` module implements differentiable automata in JAX. The `automatix.algebra`
module provides GPU-optimized semiring kernels using frozen dataclasses as JAX pytrees,
enabling dynamic algebra selection in jitted functions.

Semirings can be used either as classes or as kernels:

```python
from automatix.algebra import get_semiring, normalize_semiring

# Class-based API (backward compatible)
MaxPlus = get_semiring("MaxPlus", backend="jax")
weights = MaxPlus.zeros((3, 3))

# Kernel-based API (GPU optimized)
kernel = get_kernel("MaxPlus", "jax")
weights = kernel.zeros((3, 3))

# Both work with predicates and automata
from automatix.predicates import And, Predicate
pred = Predicate(lambda x: x > 0)
and_pred = And(args=[pred, pred], semiring=MaxPlus)  # Normalizes to kernel
```

Differentiable Boolean kernels are available for learning:

```python
from automatix.algebra import create_boolean_kernel

soft = create_boolean_kernel(mode="soft")        # Multiplicative relaxation
smooth = create_boolean_kernel(mode="smooth")    # Sigmoid-based
ste = create_boolean_kernel(mode="ste")          # Straight-through estimator
```

### Alternating Automata as Ring Polynomials

The `automatix.afa` module implements weighted alternating finite automata over
algebra defined in `automatix.algebra`.

## Using the project

If you are just using it as a library, the Git repository should be installable pretty
easily using

```bash
pip install git+https://github.com/anand-bala/automatix
```

## Developing the project

The project is a standard Python package. I use [`uv`](https://docs.astral.sh/uv/) to
develop it, as it is the most straightforward Python packaging tool I have used.

## Examples

You can look into the `examples` folder for some examples, and generally hack away at
the code.
