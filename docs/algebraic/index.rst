algebraic
=========

**algebraic** is a Python package providing **semiring algebra implementations** with
support for NumPy, JAX, and PyTorch backends.

Overview
--------

This package provides abstract semiring interfaces and concrete implementations for:

- **Tropical semirings** (MinPlus, MaxPlus) with smooth variants for differentiability
- **Max-Min algebras** for robustness semantics
- **Boolean algebras** with De Morgan and Heyting algebra variants
- **Counting semirings**
- **Custom semirings** via the extensible interface

Features
--------

- **AlgebraicArray**: Arrays with semiring semantics - override ``+``, ``*``, ``@`` to
  use custom algebras
- **Multi-Backend**: Supports NumPy, JAX, and PyTorch backends with a unified API
- **Differentiable Kernels**: Smooth approximations of boolean and tropical operations
  for neural networks
- **Rich Semiring Library**: Tropical, Boolean, Max-Min, Counting, and custom semirings
- **Polynomial Algebras**: Sparse and dense multilinear polynomials over semirings


Algebraic Structures
--------------------

A semiring :math:`(S, \oplus, \otimes, \mathbf{0}, \mathbf{1})` consists of:

- **Addition** (:math:`\oplus`): Combines alternative paths/outcomes
- **Multiplication** (:math:`\otimes`): Combines sequential compositions
- **Additive identity** (:math:`\mathbf{0}`): Identity for :math:`\oplus`
- **Multiplicative identity** (:math:`\mathbf{1}`): Identity for :math:`\otimes`

Bounded distributive lattices specialize semirings where:

- **Join** (:math:`\lor`) = Addition (:math:`\oplus`)
- **Meet** (:math:`\land`) = Multiplication (:math:`\otimes`)
- **Top** = Multiplicative identity (:math:`\mathbf{1}`)
- **Bottom** = Additive identity (:math:`\mathbf{0}`)

.. list-table:: Available Semirings
   :header-rows: 1
   :widths: 15 20 20 25

   * - Name
     - Addition
     - Multiplication
     - Use Case
   * - **Boolean**
     - Logical OR
     - Logical AND
     - Logic, SAT
   * - **Tropical (MaxPlus)**
     - max
     - \+
     - Optimization, path problems
   * - **Tropical (MinPlus)**
     - min
     - \+
     - Shortest paths, distances
   * - **Max-Min**
     - max
     - min
     - Robustness degrees, STL
   * - **Counting**
     - \+
     - :math:`\times`
     - Counting paths

Use Cases
---------

Graph Algorithms
^^^^^^^^^^^^^^^^

- **Shortest paths**: Use tropical semirings for Floyd-Warshall algorithm
- **Reachability**: Boolean algebra for transitive closure
- **Path counting**: Counting semiring for enumeration

Formal Verification
^^^^^^^^^^^^^^^^^^^

- **Temporal logic**: Signal Temporal Logic (STL) with max-min algebra
- **Automata theory**: Weighted automata with tropical semirings
- **Model checking**: Boolean polynomials for state space exploration

Machine Learning
^^^^^^^^^^^^^^^^

- **Differentiable logic**: Soft/smooth boolean operations for neural networks
- **Attention mechanisms**: Tropical attention for robust aggregation
- **Graph neural networks**: Semiring-based message passing

Optimization
^^^^^^^^^^^^

- **Dynamic programming**: Tropical semirings for Bellman equations
- **Constraint satisfaction**: Boolean algebra for SAT solving
- **Resource allocation**: Max-min algebra for bottleneck optimization

.. toctree::
   :maxdepth: 2

   quick-start
   concepts/index
   api/modules
   developer-notes
   changelog

