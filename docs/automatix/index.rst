automatix
=========

**automatix** is a library for symbolic weighted automata monitors, built on the
:external:ref:`morphata` (structural automata) and :external:ref:`algebraic` (semiring
algebra) foundation packages.

Features
--------

- **NFA with Guard-Based Weights:**
  Nondeterministic finite automata with weight functions mapping ``(input, guard) -> semiring_value``
- **Semiring-Agnostic Design:**
  Support for counting, tropical (min/max-plus), max-min, and Boolean semirings
- **JAX Integration:**
  Differentiable operations via JAX with JIT compilation support
- **Alternating Finite Automata for Spatio-Temporal Reach Escape Logic (STREL)**

Citation
--------

If you are using the matrix operator or this package in general, you should cite
one of the following papers:

For differentiable weighted automata in general:


.. code-block:: bibtex

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

If you are using weighted automata for motion planning:

.. code-block:: bibtex

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

If you are using alternating weighted automata for multi-agent systems:

.. code-block:: bibtex

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

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
