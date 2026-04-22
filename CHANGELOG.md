## [unreleased]

### Documentation

- remove non-ASCII characters that I did not type

### Testing

- add pytest-timeout to dev dependencies
- added regression tests for the step operator

### Miscellaneous

- cleanup some typing issues
## [0.7.1] - 2026-04-17

### Features

- add an "operator" that is just a BDD transition system
- add helpers to convert between Polynomials and BDDs

### Refactor

- move all operator tests under the same dir

### Documentation

- remove French spacing from all documentation
## [0.7.0] - 2026-04-17

### Features

- dd as a dependency with customized wrapper + helpers
- add symbolic polynomial operator + conversions to tensor

### Documentation

- update the urls for all packages

### Testing

- add tests for PolynomialOperator operations

### Build

- add linters for restructured text
## [0.6.0] - 2026-04-09

### Bug Fixes

- some type checker issues

### Other

- add a default cooldown period for all dependncies

### Refactor

- rename packages and use hatchling build backend instead

### Documentation

- add sphinx-based documentation

### Build

- separate out the linting/formatting configs for better root-detection

### Miscellaneous

- update the minimum version of the `uv_build` backend
- upgrate automatix to new algebraic API
- [**breaking**] port all operators to the latest `algebraic` API
## [0.5.1] - 2026-01-26

### Miscellaneous

- PolynomialOperator transition cache should not be static
## [0.5.0] - 2026-01-23

### Features

- extend predicate with logical expressions
- add some bounded distributive lattices
- implement boolean polynomials as BDDs
- add alternating automaton representation
- use lark to parse STREL expressions into an AST
- translation of STREL to AFA (except spatial formula).
- allow unbounded distance intervals
- add transitions for spatial operators
- don't add unnecessary states to the automaton
- add accepting states in the STREL AFA
- add initial poly and final mapping functions
- add a function to compute the output of a trace
- make transitions lazily create BDD variables
- add a polynomial context abstract class to pass around
- add Linear Temporal Logic grammar
- parse the Hanoi Omega-Automata format
- standardize acceptance conditions
- generic interface for automata that can be used downstream
- add helper methods for acceptance and label expressions
- add support for python 3.10
- version bump
- add AbstractNegation and 0-1 max-min LatticeAlgebra for JAX
- add De Morgan algebras
- [**breaking**] NFA uses logic_ast.Expr guards
- [**breaking**] move algebraic structures to `algebraic`
- [**breaking**] move HOA parsing related things into `hoaparser` package
- PolynomialOperator for AFA derived from LTLf specs

### Bug Fixes

- correctly handle "true" and "false" as constants in STREL
- correctly handle identifier and constant nodes
- compare BooleanPolynomial.eval against top and bot
- don't treat aliases as new states
- error in selecting top and bottom for boolean polynomials
- correctly parse state conjunctions and make State hashable
- lints reported by basedpyright
- update uv.lock
- support more versions of python
- make mypy stop complaining
- fix parsing issues in HOA `num_states`

### Other

- add more semirings
- add pytest dependency
- rename project to automatix
- add exponent to semiring
- add boolean monitoring example script
- cleanup STREL monitoring examples
- remove pixi and use uv instead
- clean up and format all files
- bump dependency versions
- add weight function abstractions and implementations
- bump minimium python version and dependencies
- remove mkdocs
- clean up older automaton API and show some (inefficient) examples
- update documentation
- remove cuda as an extra and allow end-user to install using pip
- Add tests and fixes for tensor polynomials
- refresh uv.lock
- mypy "packages" refer to module names by the looks of it
- use just instead of make as a task runner
- remove `sparx` from this repository
- clean up the dev dependencies and type checker configs

### Refactor

- move to an automaton-focused library
- move files and change Predicate name
- move differentiable stuff into nfa submodule
- move semiring things into a different module
- rename automata to AFA and NFA
- get rid of `sys` import.
- change polynomial methods to be consistent with `top` and `bottom`
- move base automata definitions to top-level
- use logic-asts instead of hand-rolling
- refactor algebra module architecture

- Create pure interface definitions (spec.py)
- Registry for semirings where you can specify backends (registry.py)
- Separated backends
   - backends/jax_.py: moved existing code here
   - backends/jax_kernels: moved logsumexp here; will place optimized
     kernels within such directories (e.g., maxplus).
   - backends/torch_.py, backends/numpy_.py: stubs for torch and numpy
   - backends/_base.py: shared utilities
- move NFA to finite_word.py, delete word_aut.py
- migrate semiring architecture to GPU-optimized kernels
- create `algebraic` package
- automatix now has the algebraic stuff separated from the automata stuff
- split kernels and tensor algebra implementation
- dependencies

### Documentation

- integrate with jupytext
- bring back the docs

### Styling

- fix some ruff lints
- run linting and formatter
- format all files

### Testing

- add initial tests

### Build

- add typing-extensions as dependency
- add networkx dependency
- merge pyproject and pixi
- temporarily remove brax and cuda stuff
- add dd and optree dependencies
- clean up dependencies
- don't handle Semiring matrices of size 1000...

### Miscellaneous

- make pyright configrations public
- [**breaking**] update python minimum version to 3.11
- run linting and cleanup
- remove duplicate settings for pyright/basedpyright
- simplify makefile and python build backend
- update description and bump version
- run formatter
- rename project due to PyPI name clash
- version bump
- bump version to 0.3.5
- update dependencies
- bump version to 0.3.6
- bump version to v0.4.0
- add ty and zuban to dev-dependencies
- use quax to make algebraic tensors look like regular tensors
- move NFA and STRELAutomaton implementations from automatix
- make automatix depend on morphata
- migrate over to using morphata
- run formatter
- remove AcceptanceCondition from automatix
- rename algebraic -> algebraic-jax for PyPI
- run aggressive ruff fixes
- appease the type checkers
