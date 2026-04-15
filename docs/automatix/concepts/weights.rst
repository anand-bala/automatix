Weight Functions
================

A weight function bridges structural automata and semiring algebra.
It maps each ``(input, guard)`` pair on a transition to a value in the
target semiring.

The WeightFunction Protocol
----------------------------

Any callable with the signature ``(x, guard) -> scalar_or_array`` satisfies
the :class:`~automatix.spec.WeightFunction` protocol:

.. code-block:: python

   from automatix.spec import WeightFunction

   # A plain function works
   def my_weight(x, guard) -> float:
       return float(x[0])

   # So does a callable class
   class LearnableWeight:
       def __call__(self, x, guard):
           return x @ self.params

Guards
------

A **guard** is a :data:`~morphata.spec.BoolExpr` labelling a transition in an
NFA. Guards are built from atomic propositions using ``logic_asts``:

.. code-block:: python

   import logic_asts as logic

   a = logic.Variable("a")
   b = logic.Variable("b")
   guard = a & ~b  # "a and not b"

The weight function receives this guard and decides how to interpret it over
the input.

Composable Predicates
---------------------

The :mod:`automatix.weights.guard_weights` module provides composable
predicate classes that evaluate guards using semiring operations:

- :class:`~automatix.weights.guard_weights.Predicate`: Wraps a user function.
- :class:`~automatix.weights.guard_weights.And`: Conjunction via semiring
  multiplication.
- :class:`~automatix.weights.guard_weights.Or`: Disjunction via semiring
  addition.

.. code-block:: python

   from automatix.weights.guard_weights import Predicate, And, Or, ExprWeightFn
   import algebraic

   algebra = algebraic.semirings.tropical_semiring(minplus=False)

   # Wrap user predicates
   p_a = Predicate(algebra=algebra, fn=lambda x: x[0])
   p_b = Predicate(algebra=algebra, fn=lambda x: x[1])

   # Compose using semiring operations
   p_and = And(algebra=algebra, children=(p_a, p_b))  # mul
   p_or  = Or(algebra=algebra, children=(p_a, p_b))   # add

ExprWeightFn
^^^^^^^^^^^^

:class:`~automatix.weights.guard_weights.ExprWeightFn` recursively evaluates a
guard expression tree, mapping each atomic proposition to a predicate and
composing them with semiring AND/OR. It implements memoisation for repeated
subexpressions.

.. code-block:: python

   predicates = {"a": p_a, "b": p_b}
   weight_fn = ExprWeightFn(algebra=algebra, predicates=predicates)

   # Evaluate: the guard is parsed, predicates looked up, results composed
   value = weight_fn(x, guard)
