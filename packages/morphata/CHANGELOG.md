## [1.0.3] - 2026-04-22

### Documentation

- remove non-ASCII characters that I did not type
- generate CHANGELOGs
- add changelogs to the sphinx documentation

### Testing

- add pytest-timeout to dev dependencies

### Build

- add mypy config for `dd` BDD library
- add git-cliff support for CHANGELOG generation
- add a `make-release` script for the lazy
- update cliff settings to process some older conventions
- add shared artifacts in the root directory for git-cliff
- handle releasing multiple packages in one commit
## [1.0.2] - 2026-04-17

### Features

- dd as a dependency with customized wrapper + helpers

### Bug Fixes

- incorrect pattern matching from queue for `ltl_to_automaton`

### Documentation

- update README and add the missing LICENSE file
- initial documentation
- initial documentation
- update the urls for all packages

### Build

- add linters for restructured text

### CI

- make sure uv is installing all extras
- only use `mypy` for type checking

### Miscellaneous

- [**breaking**] port all operators to the latest `algebraic` API
- ensure mypy doesn't error out due to missing imports
- remove empty submodules
## [1.0.1] - 2026-04-09

### Features

- implement boolean polynomials as BDDs
- use lark to parse STREL expressions into an AST
- translation of STREL to AFA (except spatial formula).
- add transitions for spatial operators
- add support for python 3.10
- version bump
- [**breaking**] move algebraic structures to `algebraic`
- [**breaking**] move HOA parsing related things into `hoaparser` package
- LTL/LTLf -> ABWA/AFA
- clean out the justfile and add feature to upload to HF
- torch and jax as soft/optional dependencies
- new tests and port old tests for the new API

### Bug Fixes

- restore READMEs and examples from history
- lints reported by basedpyright
- update uv.lock
- support more versions of python
- make mypy stop complaining
- some type checker issues
- some formatting/typing issues

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
- update documentation
- remove cuda as an extra and allow end-user to install using pip
- Add tests and fixes for tensor polynomials
- refresh uv.lock
- make sure pytest picks all package tests and source files
- mypy "packages" refer to module names by the looks of it
- use just instead of make as a task runner
- remove `sparx` from this repository
- update justfile to be less strict
- clean up the dev dependencies and type checker configs
- add a CONTRIBUTING.md file
- add a default cooldown period for all dependncies

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
- dependencies
- rename packages and use hatchling build backend instead
- README

### Documentation

- integrate with jupytext
- bring back the docs
- add sphinx-based documentation

### Styling

- run linting and formatter

### Build

- add typing-extensions as dependency
- add networkx dependency
- merge pyproject and pixi
- temporarily remove brax and cuda stuff
- add dd and optree dependencies
- clean up dependencies
- separate out the linting/formatting configs for better root-detection

### CI

- add split Github Actions pipelines for each package
- change the working directory for each package
- don't just call just, make CI verbose
- make sure uv installs dependencies for all packages
- remove pyrefly for subpackages
- remove --package for publishing

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
- a consolidated library for automata structure
- copy hoaparser as submodule
- move NFA and STRELAutomaton implementations from automatix
- make the Automaton interface clean and tiny
- move hoa parser and refactor tests
- format README
- migrate over to using morphata
- rename algebraic -> algebraic-jax for PyPI
- make the type checkers happy
- appease the type checkers
- update the minimum version of the `uv_build` backend
- upgrate automatix to new algebraic API
