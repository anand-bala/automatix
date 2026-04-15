# Automatix: Weighted Automata over Semirings

A Python library for weighted automata over semirings, with support for differentiable
operations via JAX/PyTorch.

This repository also contains the following packages in the `packages/` subdirectory:

- `algebraic` : Multi-backend semiring algebra (NumPy, JAX, PyTorch) with
  `AlgebraicArray` , concrete semiring implementations (tropical, max-min, boolean,
  counting), and multilinear polynomial representations.

- `morphata` : Pure structural automata representations -- graph-based NFAs, alternating
  finite automata, HOA v1 format parser/exporter, and acceptance condition algebra --
  without any weighted semantics.

## Quick Start

### Installation

```bash
pip install argus-automatix
```

### Basic Usage

For an example of using the matrix operator, see the file in
<tests/nfa/test_jax_automaton_operator.py>

## Citation

If you are using the matrix operator or this package in general, you should cite one of
the following papers:

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
