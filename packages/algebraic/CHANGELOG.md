## [unreleased]

### Build

- update cliff settings to process some older conventions
## [1.3.0] - 2026-04-22

### Features

- add batched composition for rank decomposition factors
- group all polynomial utilities together as public API
- add functions to reason about device types and coersion

### Build

- add git-cliff support for CHANGELOG generation
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
## [1.2.4] - 2026-04-17

### Bug Fixes

- degree blowup and padding mismatch when doing serial compositions
- some typing errors by using a type guard

### Documentation

- remove French spacing from all documentation
## [1.2.3] - 2026-04-15

### Features

- add constant term separated rank decomposition poly

### Documentation

- update the urls for all packages
## [1.2.2] - 2026-04-09

### Bug Fixes

- torch gradient loss due to `copy.copy`
## [1.2.1] - 2026-04-09

### Bug Fixes

- issues with torch gradients not being propagated correctly
## [1.2.0] - 2026-04-09

### Features

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

- add a default cooldown period for all dependncies

### Refactor

- rename packages and use hatchling build backend instead
- Semiring hierarchy to use better dataclasses from `equinox` 
- documentation for new API
- docstrings to be NumPy + rst
- some repetition by moving things to common modules
- to version 1.2.0

### Documentation

- clean up the API Reference layout

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
- [**breaking**] deduplicate a lot of backend-specific things
- [**breaking**] remove the `jax.jit` and `torch.compile`
## [1.0.0] - 2026-01-23

### Features

- [**breaking**] move algebraic structures to `algebraic`
- matmul helper for AlgebraicArray
- TypeGuard functions to help with BooleanAlgebra

### Bug Fixes

- change README.md encoding
- issue with multiplying by zero polynomial.
- Semiring not being static, and Quax invariants not being maintained.
- some typing lints in tests

### Other

- register algebraic structures as JAX PyTrees within backend-specific module
- update documentation
- remove cuda as an extra and allow end-user to install using pip
- add implementations for various multilinear polynomial representations
- Add tests and fixes for tensor polynomials

### Refactor

- create `algebraic` package
- split kernels and tensor algebra implementation
- type signatures to satisfy linter
- documentation
- README
- test for changed type error format in 3.14

### Miscellaneous

- use quax to make algebraic tensors look like regular tensors
- make indexing helper plain classes
- move NFA and STRELAutomaton implementations from automatix
- remove references to older API
- (chore) run formatter
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
