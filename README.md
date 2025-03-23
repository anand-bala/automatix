# Symbolic (Weighted) Automata Monitoring

This project implements different automata I use in my research, including
nondeterministic weighted automata and alternating weighted automata.

### Differentiable Automata in [JAX](https://github.com/google/jax)

The `automatix.nfa` module implements differentiable automata in JAX, along with
`automatix.algebra.semiring.jax_backend`.
Specifically, it does so by defining _matrix operators_ on the automata transitions,
which can then be interpreted over a semiring to yield various acceptance and weighted
semantics.

### Alternating Automata as Ring Polynomials

The `automatix.afa` module implements weighted alternating finite automata over
algebra defined in `automatix.algebra.semiring`.

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
