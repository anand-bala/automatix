## [1.3.6] - 2026-05-07

### Features

- expose JIT-safe paths for polynomial composition and evaluation

### Bug Fixes

- einsum for composition; replace with reshape-broadcast

### Documentation

- roadmap/plan for speeding up rank decomp with sparse arrays
- add docs for training with rank decomposition
## [1.3.5] - 2026-04-25

### Features

- pruning a rank decomposition should allow for short circuiting and approx-equal checks
- add a "packed" identity pruning strategy that is more aggressive
- unify the polynomial factors' arithmetic for batch and unbatched

### Other

- add a scratch testing doc to check polynomial composition perf

### Documentation

- move all the tradeoffs and other things into a developer notes doc
## [1.3.4] - 2026-04-25

### Bug Fixes

- rank decomposition polynomial composition wasn't batched

### Documentation

- add pip install instructions
## [1.3.3] - 2026-04-23

### Features

- add batched evaluation for rank decomposition factors
- handled potentially batched rank decomposition tensors

### Bug Fixes

- AlgebraicArray doctest needs to accomodate the new pretty repr
- make AlgebraicStructure properties hashable for PyTree-ness

### Refactor

- don't pass around `algebra` for the polynomial helpers
- [**breaking**] rank decomposed polynomial `evaluate` returns AlgebraicArray

### Testing

- add testing for batched rank decomposition representations
- adapt tests to handle the new `evaluate` API

### Miscellaneous

- move opt_einsum mypy config into global conf instead of inline
## [1.3.2] - 2026-04-22

### Features

- reexport namespaced optree API as `algebraic.utils.pytree`
- register existing AlgebraicPyTree via `algebraic.utils.pytree`
- `algebraic.utils.torch` exports modules with easy `algebraic` interface
- add wadler-lindig-based repr methods

### Build

- handle markdown headings parsing as comments in release
- ignore typing for `wadler_lindig`
## [1.3.1] - 2026-04-22

### Documentation

- generate CHANGELOGs
- add changelogs to the sphinx documentation

### Build

- update cliff settings to process some older conventions
- add shared artifacts in the root directory for git-cliff
- handle releasing multiple packages in one commit
## [1.3.0] - 2026-04-22

### Features

- add batched composition for rank decomposition factors
- group all polynomial utilities together as public API
- add functions to reason about device types and coersion

### Build

- add git-cliff support for CHANGELOG generation
- add a `make-release` script for the lazy
## [1.2.8] - 2026-04-20

### Bug Fixes

- coerce things to an appropriate device where possible
## [1.2.7] - 2026-04-20

### Bug Fixes

- device and dtype of reduce and prefix_scan identity
## [1.2.6] - 2026-04-20

### Bug Fixes

- pass `device` parameter through all construction functions
## [1.2.5] - 2026-04-17

### Bug Fixes

- conversion of larger degree CP polynomials to sparse

### Documentation

- remove non-ASCII characters that I did not type

### Testing

- add pytest-timeout to dev dependencies
## [1.2.4] - 2026-04-17

### Features

- dd as a dependency with customized wrapper + helpers

### Bug Fixes

- degree blowup and padding mismatch when doing serial compositions
- some typing errors by using a type guard

### Documentation

- remove French spacing from all documentation

### Build

- add mypy config for `dd` BDD library
## [1.2.3] - 2026-04-15

### Features

- add constant term separated rank decomposition poly

### Documentation

- update the docs on composition
- clean up old modules and add proper toctrees
- initial documentation
- update the urls for all packages

### Build

- add linters for restructured text
## [1.2.2] - 2026-04-09

### Bug Fixes

- torch gradient loss due to `copy.copy`

### Miscellaneous

- ensure mypy doesn't error out due to missing imports
## [1.2.1] - 2026-04-09

### Bug Fixes

- issues with torch gradients not being propagated correctly

### Documentation

- update README and add the missing LICENSE file

### CI

- make sure uv is installing all extras
- only use `mypy` for type checking

### Miscellaneous

- [**breaking**] port all operators to the latest `algebraic` API
## [1.2.0] - 2026-04-09

### Features

- clean out the justfile and add feature to upload to HF
- torch and jax as soft/optional dependencies
- `numpy` and `array-api-compat` as explicit deps
- abstract algebraic array, with (partial) Array API
- a common Dispatcher for all operations
- the jax, torch, and numpy array backends
- required custom kernel for torch
- annotations for `_better_abc`
- a centralized utils module
- class annotations for TorchAlgebraicArray
- an initial version of `jit` and `vmap` (WIP)
- an implementation of `allclose` and `isclose`
- `einsum` implementation
- testing utilities that use numpy.testing
- new tests and port old tests for the new API

### Bug Fixes

- some type checker issues
- allclose and isclose with proper unwrap
- previously quaxified code in monomial basis
- some typing issues in rank decomp + bug fix
- and test jaxify and torchify for the polynomial classes
- a bunch of typing issues to satisfy mypy

### Other

- add a CONTRIBUTING.md file
- add a default cooldown period for all dependncies

### Refactor

- rename packages and use hatchling build backend instead
- Semiring hierarchy to use better dataclasses from `equinox` 
- documentation for new API
- docstrings to be NumPy + rst
- some repetition by moving things to common modules
- README
- to version 1.2.0

### Documentation

- add sphinx-based documentation
- clean up the API Reference layout

### Build

- separate out the linting/formatting configs for better root-detection

### CI

- remove --package for publishing

### Miscellaneous

- update the minimum version of the `uv_build` backend
- freeze semiring dataclasses for better equality check
- [**breaking**] let the semiring specifications be backend-agnostic
- [**breaking**] remove the `scan` and `reduce` abstract methods from `AlgebraicArray`
- remove all older API files
- [**breaking**] make `AlgebraicArray` not generic
- handle builtin scalars with numpy fallback for `algebraic.array`
- [**breaking**] port `algebraic.polynomials` to support multiple backends
- `algebraic` top-level module is all a user needs
- use `arr.at[...].<fn>` pattern for all backends
- array operations should promote the result to the correct type
- polynomial factory methods should handle `cls.backend` properly
- upgrate automatix to new algebraic API
- [**breaking**] deduplicate a lot of backend-specific things
- [**breaking**] remove the `jax.jit` and `torch.compile`
## [1.0.0] - 2026-01-23

### Features

- implement boolean polynomials as BDDs
- use lark to parse STREL expressions into an AST
- translation of STREL to AFA (except spatial formula).
- add transitions for spatial operators
- add support for python 3.10
- version bump
- [**breaking**] move algebraic structures to `algebraic`
- [**breaking**] move HOA parsing related things into `hoaparser` package
- matmul helper for AlgebraicArray
- TypeGuard functions to help with BooleanAlgebra

### Bug Fixes

- restore READMEs and examples from history
- lints reported by basedpyright
- update uv.lock
- support more versions of python
- make mypy stop complaining
- change README.md encoding
- issue with multiplying by zero polynomial.
- Semiring not being static, and Quax invariants not being maintained.
- some typing lints in tests

### Other

- add more semirings
- add pytest dependency
- rename project to automatix
- add boolean monitoring example script
- cleanup STREL monitoring examples
- remove pixi and use uv instead
- bump dependency versions
- add weight function abstractions and implementations
- bump minimium python version and dependencies
- remove mkdocs
- clean up older automaton API and show some (inefficient) examples
- register algebraic structures as JAX PyTrees within backend-specific module
- update documentation
- remove cuda as an extra and allow end-user to install using pip
- add implementations for various multilinear polynomial representations
- Add tests and fixes for tensor polynomials
- refresh uv.lock
- make sure pytest picks all package tests and source files
- mypy "packages" refer to module names by the looks of it
- use just instead of make as a task runner
- remove `sparx` from this repository
- update justfile to be less strict
- clean up the dev dependencies and type checker configs

### Refactor

- move to an automaton-focused library
- move semiring things into a different module
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
- migrate semiring architecture to GPU-optimized kernels
- create `algebraic` package
- split kernels and tensor algebra implementation
- type signatures to satisfy linter
- documentation
- dependencies
- README
- test for changed type error format in 3.14

### Documentation

- integrate with jupytext
- bring back the docs

### Styling

- run linting and formatter

### Build

- add typing-extensions as dependency
- add networkx dependency
- merge pyproject and pixi
- temporarily remove brax and cuda stuff
- add dd and optree dependencies
- clean up dependencies

### CI

- add split Github Actions pipelines for each package
- change the working directory for each package
- don't just call just, make CI verbose
- make sure uv installs dependencies for all packages
- remove pyrefly for subpackages

### Miscellaneous

- make pyright configrations public
- [**breaking**] update python minimum version to 3.11
- remove duplicate settings for pyright/basedpyright
- simplify makefile and python build backend
- update description and bump version
- rename project due to PyPI name clash
- version bump
- bump version to 0.3.5
- update dependencies
- bump version to 0.3.6
- bump version to v0.4.0
- add ty and zuban to dev-dependencies
- use quax to make algebraic tensors look like regular tensors
- make indexing helper plain classes
- move NFA and STRELAutomaton implementations from automatix
- remove references to older API
- (chore) run formatter
- migrate over to using morphata
- rename algebraic -> algebraic-jax for PyPI
- run aggressive ruff fixes
- use the correct order of args for Poly.constant(...)
- correctly overload `max_min_algebra` for type checker
- minor change in the layout of _IndexUpdateRef
- let "std-fuzzy" be an alias for the standard fuzzy algebra
- satisfy the type checker and linters
- define new modules to hide the use of quax
- allow AlgebraicArray to be materialized as data
- pytest should check for the correct error type when argument is missing
- appease the type checkers
