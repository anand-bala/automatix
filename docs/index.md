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

If you want develop the project, you will need to install [Pixi](https://pixi.sh/latest/).
Once you install it for you platform, you can install the project development
dependencies and the current project as an editable install using:

```bash
pixi install -e dev
```

Then, you can use the following command to activate the pixi environment:
```bash
pixi shell -e dev
```

From here, you can look into the `examples` folder for some examples, and generally hack
away at the code.

