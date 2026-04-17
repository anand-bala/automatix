Automaton Operators
===================

Operators combine structural automata with semiring algebra to perform
weighted evaluation of input words. automatix provides two operators
targeting different automaton models.

MatrixOperator (Weighted NFAs)
------------------------------

:class:`~automatix.operators.matrix.MatrixOperator` implements weighted
finite-word NFA semantics using matrix multiplication over a semiring.

Given an NFA with :math:`q` states, each input symbol :math:`x` produces a
:math:`q \times q` transition matrix :math:`M(x)` where entry
:math:`M(x)_{ij}` is the weight of the transition from state :math:`i` to
state :math:`j`. The weight of a word is computed as:

.. math::

   w = \mathbf{i}^T \cdot M(x_1) \otimes M(x_2) \otimes \cdots \otimes M(x_n) \cdot \mathbf{f}

where :math:`\mathbf{i}` and :math:`\mathbf{f}` are the initial and final
weight vectors, and :math:`\otimes` denotes semiring matrix multiplication.

Construction
^^^^^^^^^^^^

Use :meth:`~automatix.operators.matrix.MatrixOperator.make`:

.. code-block:: python

   import algebraic
   from morphata.examples.nfa import NFA
   from automatix.operators import MatrixOperator
   import logic_asts as logic

   # Build an NFA
   nfa: NFA[str] = NFA()
   nfa.add_location(0, initial=True)
   nfa.add_location(1, final=True)
   nfa.add_transition(0, 1, logic.Variable("a"))

   # Define a weight function
   def weight_fn(x, guard):
       return float(x[0])

   # Create the operator
   maxplus = algebraic.semirings.tropical_semiring(minplus=False)
   op = MatrixOperator.make(nfa, maxplus, weight_function=weight_fn)

Evaluation
^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp

   x = jnp.array([2.0])
   M = op.cost_transitions(x)  # q × q AlgebraicArray

The returned ``M`` is an :class:`~algebraic.AlgebraicArray` — arithmetic on it
uses the semiring operations automatically.

PolynomialOperator (Alternating Automata)
-----------------------------------------

:class:`~automatix.operators.polynomial.PolynomialOperator` represents AFA
transitions and run states as **multilinear polynomials** over a bounded
distributive lattice (typically Boolean algebra).

Each state variable :math:`x_i` represents whether state :math:`i` is in the
current run tree. A transition :math:`\delta(q, \sigma)` returning a boolean
expression is converted to a
:class:`~algebraic.polynomials.RankDecomposition` polynomial.

Construction
^^^^^^^^^^^^

From an LTL formula (convenience):

.. code-block:: python

   import algebraic
   from automatix.operators import PolynomialOperator
   from automatix.operators.polynomial import from_ltl
   import logic_asts as logic

   algebra = algebraic.semirings.boolean_algebra()
   formula = logic.Eventually(logic.Variable("a"))

   op = from_ltl(formula, algebra, finite=True)

From an existing AFA:

.. code-block:: python

   from morphata.examples.ltl import ltl_to_automaton
   from automatix.operators.polynomial import from_afa

   aut = ltl_to_automaton(formula, finite=True)
   op = from_afa(aut, algebra, cache_transitions=True)

Evaluation
^^^^^^^^^^

Check acceptance of a word:

.. code-block:: python

   word = [{"a": False}, {"a": True}]
   result = op.accepts(word)  # True — eventually "a" is satisfied

Access intermediate representations:

.. code-block:: python

   run_poly = op.run_polynomial(word)  # RankDecomposition
   print(f"Rank: {run_poly.rank}")

SymbolicPolynomialOperator (BDD-based Alternating Automata)
-----------------------------------------------------------

:class:`~automatix.operators.symbolic_polynomial.SymbolicPolynomialOperator`
is an alternative to
:class:`~automatix.operators.polynomial.PolynomialOperator` that routes each
boolean transition formula through a **reduced ordered BDD** (via
:mod:`automatix.operators._bdd`) before tensorisation. This canonicalises
shared sub-functions so they are tensorised exactly once, which can reduce
redundant computation for automata with structurally shared transitions.

The pipeline is:

.. code-block:: text

   morphata.Automaton (AlternatingTransitions)
       → BoolExpr[int]
       → reduced ordered BDD (dd)
       → tensorised polynomial
       → RankDecomposition or LowRankFactors

The runtime API mirrors ``PolynomialOperator`` — :meth:`accepts`,
:meth:`run_polynomial`, :meth:`step`, and :meth:`evaluate_at_accepting` all
work identically.

Construction
^^^^^^^^^^^^

From an LTL formula (convenience):

.. code-block:: python

   import algebraic
   from automatix.operators.symbolic_polynomial import SymbolicPolynomialOperator
   import logic_asts as logic

   algebra = algebraic.semirings.boolean_algebra()
   formula = logic.Eventually(logic.Variable("a"))

   op = SymbolicPolynomialOperator.from_ltl(
       formula, algebra, backend="numpy", finite=True
   )

From an existing AFA:

.. code-block:: python

   from morphata.examples.ltl import ltl_to_automaton
   from automatix.operators.symbolic_polynomial import SymbolicPolynomialOperator

   aut = ltl_to_automaton(formula, finite=True)
   op = SymbolicPolynomialOperator.from_afa(
       aut, algebra, backend="numpy", cache_transitions=True
   )

The ``output`` parameter selects the polynomial representation —
``"rank_decomposition"`` (default) or ``"low_rank_factors"``. An optional
``var_order`` controls the BDD variable ordering.

Evaluation
^^^^^^^^^^

.. code-block:: python

   word = [{"a": False}, {"a": True}]
   result = op.accepts(word)  # True — eventually "a" is satisfied

   run_poly = op.run_polynomial(word)
   print(f"Rank: {run_poly.rank}")

JAX and PyTorch Integration
---------------------------

Both operators implement ``AlgebraicPyTree``, making them compatible with JAX
and PyTorch functional transforms. Use ``algebraic.utils.jax.jaxify()`` or
``algebraic.utils.torch.torchify()`` for backend integration:

.. code-block:: python

   from algebraic.utils.jax import jaxify
   import jax

   jax_op = jaxify(op)
   jit_cost = jax.jit(jax_op.cost_transitions)
