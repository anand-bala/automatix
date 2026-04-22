## [unreleased]

### Documentation

- remove non-ASCII characters that I did not type

### Build

- add git-cliff support for CHANGELOG generation
- update cliff settings to process some older conventions
## [1.0.2] - 2026-04-17

### Bug Fixes

- incorrect pattern matching from queue for `ltl_to_automaton`

### Documentation

- initial documentation
- update the urls for all packages

### Miscellaneous

- remove empty submodules
## [1.0.1] - 2026-04-09

### Features

- LTL/LTLf -> ABWA/AFA

### Bug Fixes

- some formatting/typing issues

### Refactor

- rename packages and use hatchling build backend instead

### Miscellaneous

- a consolidated library for automata structure
- copy hoaparser as submodule
- move NFA and STRELAutomaton implementations from automatix
- make the Automaton interface clean and tiny
- move hoa parser and refactor tests
- format README
- make the type checkers happy
- update the minimum version of the `uv_build` backend
