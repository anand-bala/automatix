# Automatix: Weighted Automata over Semirings

A Python library for weighted automata over semirings.
The library currently includes:

- NFA with Guard-Based Weights:
  Nondeterministic finite automata with weight functions mapping `(input, guard)
  -> semiring_value`
- Semiring-Agnostic Design:
  Support for counting, tropical (min/max-plus), max-min, and Boolean semirings
- JAX Integration:
  Differentiable operations via JAX with JIT compilation support
- Alternating Finite Automata for Spatio-Temporal Reach Escape Logic (STREL)


This repository also contains the following packages in the `packages/`
subdirectory:

- `algebraic`:
  Implementations of algebraic structures as tensors.

- `hoaparser`:
  A parser for the
  [Hanoi Omega-Automata format](https://adl.github.io/hoaf/index.html).
  It supports the entire `v1` format, but does not implement any of the
  semantics of the automata.
  The goal is to use it as a base package to handle parsing HOA files outputted
  by the tools linked in <https://adl.github.io/hoaf/support.html> without
  having to generate the automata yourself.

## Quick Start

### Installation

```bash
pip install git+https://github.com/anand-bala/automatix
```

### Basic Usage

For an example of using the matrix operator, see the file in
<tests/nfa/test_jax_automaton_operator.py>

## Citation

If you are using the matrix operator or this package in general, you should cite
one of the following papers:

- For differentiable weighted automata in general:

```bibtex
@inproceedings{balakrishnan2024differentiable,
  title = {Differentiable {{Weighted Automata}}},
  booktitle = {{{ICML}} 2024 {{Workshop}} on {{Differentiable Almost Everything}}: {{Differentiable Relaxations}}, {{Algorithms}}, {{Operators}}, and {{Simulators}}},
  author = {Balakrishnan, Anand and Deshmukh, Jyotirmoy V.},
  year = 2024,
  month = jun,
  url = {https://openreview.net/forum?id=k2hIQYqHTh},
  copyright = {All rights reserved},
  langid = {english}
}
```

- If you are using weighted automata for motion planning

```bibtex
@inproceedings{balakrishnan2024motion,
  title = {Motion {{Planning}} for {{Automata-based Objectives}} Using {{Efficient Gradient-based Methods}}},
  booktitle = {2024 {{IEEE}}/{{RSJ International Conference}} on {{Intelligent Robots}} and {{Systems}} ({{IROS}})},
  author = {Balakrishnan, Anand and Atasever, Merve and Deshmukh, Jyotirmoy V.},
  year = 2024,
  month = oct,
  pages = {13734--13740},
  issn = {2153-0866},
  doi = {10.1109/IROS58592.2024.10802177}
}
```

- If you are using alternating weighted automata for multi-agent systems.

```bibtex
@inproceedings{balakrishnan2025monitoring,
  title = {Monitoring {{Spatially Distributed Cyber-Physical Systems}} with {{Alternating Finite Automata}}},
  booktitle = {Proceedings of the 28th {{ACM International Conference}} on {{Hybrid Systems}}: {{Computation}} and {{Control}}},
  author = {Balakrishnan, Anand and Paul, Sheryl and Silvetti, Simone and Nenzi, Laura and Deshmukh, Jyotirmoy V.},
  year = 2025,
  month = may,
  pages = {1--11},
  publisher = {ACM},
  address = {Irvine CA USA},
  doi = {10.1145/3716863.3718033},
  isbn = {979-8-4007-1504-4},
  langid = {english}
}
```

## License

See LICENSE file for details.
