Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install argus-automatix

Weighted NFA
------------

Create a weighted NFA monitor using the tropical (max-plus) semiring:

.. code-block:: python

   import algebraic
   import jax.numpy as jnp
   from morphata.examples.nfa import NFA
   from automatix.operators import MatrixOperator
   import logic_asts as logic

   # 1. Build an NFA
   nfa: NFA[str] = NFA()
   nfa.add_location(0, initial=True)
   nfa.add_location(1, final=True)
   nfa.add_transition(0, 1, guard=logic.Variable("a"))

   # 2. Define a weight function
   def weight_fn(x, guard):
       return float(x[0])

   # 3. Create the operator
   maxplus = algebraic.semirings.tropical_semiring(minplus=False)
   op = MatrixOperator.make(nfa, maxplus, weight_function=weight_fn)

   # 4. Evaluate
   x = jnp.array([2.0])
   transitions = op.cost_transitions(x)

LTL Monitoring with Polynomial Operator
----------------------------------------

Monitor an LTL property using alternating automata and Boolean polynomials:

.. code-block:: python

   import algebraic
   from automatix.operators.polynomial import from_ltl
   import logic_asts as logic

   # Define LTL formula: F(a) — "eventually a"
   algebra = algebraic.semirings.boolean_algebra()
   formula = logic.Eventually(logic.Variable("a"))

   # Create polynomial operator
   op = from_ltl(formula, algebra, finite=True)

   # Check acceptance of a word
   word = [{"a": False}, {"a": True}]
   result = op.accepts(word)
   print(f"Accepted: {result}")  # True

   # Inspect the operator
   print(f"States: {op.num_states}")
   print(f"Accepting: {op.accepting_states}")
