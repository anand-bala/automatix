"""Tests for weight function abstractions and integration with automata.

This test module covers:
- WeightFunction type definition and usage
- Integration of weight functions with MatrixOperator.make
- Weight functions with different automaton structures
"""
# mypy: disable-error-code="no-untyped-call, no-any-return"

import jax.numpy as jnp
import logic_asts as logic
from algebraic.semirings import tropical_semiring
from jaxtyping import Array, Scalar, ScalarLike

from automatix import Guard
from automatix.automata.nfa import NFA
from automatix.operators import MatrixOperator

type InputSymbol = Array
type SemiringValue = Array | Scalar | ScalarLike

maxplus = tropical_semiring(minplus=False)


def parse_guard(expr: str) -> Guard[str]:
    return logic.parse_expr(expr)


class TestWeightFunctionBasics:
    """Test basic weight function definition and calling."""

    def test_constant_weight_function(self) -> None:
        """A constant weight function should always return the same value."""

        def constant_weight(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return 1.0

        # Should be callable with any input and guard
        assert constant_weight(jnp.array([1.0, 2.0]), logic.Literal(True)) == 1.0
        assert constant_weight(jnp.array([3.0, 4.0]), parse_guard("x")) == 1.0

    def test_guard_dependent_weight(self) -> None:
        """A weight function can depend on the guard expression."""

        def guard_weight(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            # Return weight based on guard type
            if isinstance(guard, logic.Variable):
                return 1.0
            elif isinstance(guard, logic.Literal):
                return 2.0
            return 0.5

        assert guard_weight(jnp.array([1.0]), parse_guard("x")) == 1.0
        assert guard_weight(jnp.array([1.0]), logic.Literal(True)) == 2.0

    def test_input_dependent_weight(self) -> None:
        """A weight function can depend on the input symbol."""

        def input_weight(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            # Return weight based on input magnitude
            return float(jnp.linalg.norm(x))

        x1 = jnp.array([3.0, 4.0])
        x2 = jnp.array([0.0, 0.0])

        assert input_weight(x1, logic.Variable("dummy")) == 5.0
        assert input_weight(x2, logic.Variable("dummy")) == 0.0

    def test_lambda_weight_function(self) -> None:
        """Weight functions can be defined as lambdas."""

        def wf(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return 1.5

        assert wf(jnp.array([1.0]), logic.Variable("any_guard")) == 1.5


class TestAutomatonOperatorIntegration:
    """Test weight functions integrated with automaton operators."""

    def test_simple_automaton_with_weight_function(self) -> None:
        """Create a simple automaton and compute transitions with weight function."""
        # Create a 2-state automaton
        aut = NFA[str]()
        aut.add_location(0, initial=True)
        aut.add_location(1, final=True)
        aut.add_transition(0, 1, guard=logic.Variable("a"))
        aut.add_transition(1, 1, guard=logic.Literal(True))

        # Define a simple weight function
        def weight_fn(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return 1.0

        # Create automaton operator
        operator = MatrixOperator.make(
            aut,
            maxplus,
            weight_function=weight_fn,
        )

        # Check initial and final weights
        assert operator.initial_weights.shape == (2,)
        assert operator.final_weights.shape == (2,)

        # Check cost_transitions
        x = jnp.array([1.0])
        transitions = operator.cost_transitions(x)
        assert transitions.shape == (2, 2)
        # Should have transitions at (0,1) and (1,1)
        assert transitions.data[0, 1] == 1.0
        assert transitions.data[1, 1] == 1.0

    def test_automaton_with_input_dependent_weights(self) -> None:
        """Test automaton where weights depend on input values."""
        aut: NFA[str] = NFA()
        aut.add_location(0, initial=True)
        aut.add_location(1, final=True)
        aut.add_transition(0, 1, guard=logic.Variable("a"))

        # Weight function that depends on input
        def input_weight(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return float(x[0])  # Return first element of input

        operator = MatrixOperator.make(
            aut,
            maxplus,
            weight_function=input_weight,
        )

        # Test with different inputs
        x1 = jnp.array([2.0])
        transitions1 = operator.cost_transitions(x1)
        assert transitions1.data[0, 1] == 2.0

        x2 = jnp.array([5.0])
        transitions2 = operator.cost_transitions(x2)
        assert transitions2.data[0, 1] == 5.0

    def test_automaton_with_multiple_transitions(self) -> None:
        """Test automaton with multiple guards and transitions."""
        aut: NFA[str] = NFA()
        aut.add_location(0, initial=True)
        aut.add_location(1)
        aut.add_location(2, final=True)

        aut.add_transition(0, 1, guard=logic.Variable("a"))
        aut.add_transition(0, 2, guard=logic.Variable("b"))
        aut.add_transition(1, 2, guard=logic.Literal(True))

        # Weight function returning constant
        def constant_weight(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return 1.0

        operator = MatrixOperator.make(
            aut,
            maxplus,
            weight_function=constant_weight,
        )

        x = jnp.array([3.0])
        transitions = operator.cost_transitions(x)
        assert transitions.shape == (3, 3)
        assert transitions.data[0, 1] == 1.0
        assert transitions.data[0, 2] == 1.0
        assert transitions.data[1, 2] == 1.0


class TestWeightFunctionSignature:
    """Test that weight functions follow the correct signature."""

    def test_weight_function_takes_two_arguments(self) -> None:
        """Weight function must accept (InputSymbol, Guard[str])."""

        # This should work
        def valid_wf(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return 1.0

        aut: NFA[str] = NFA()
        aut.add_location(0, initial=True)
        aut.add_location(1, final=True)
        aut.add_transition(0, 1, guard=logic.Variable("a"))

        # Should not raise
        operator = MatrixOperator.make(
            aut,
            maxplus,
            weight_function=valid_wf,
        )
        assert operator is not None

    def test_weight_function_returns_semiring_value(self) -> None:
        """Weight function should return a valid semiring value."""

        def float_weight(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return 3.14

        def int_weight(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return 42

        aut: NFA[str] = NFA()
        aut.add_location(0, initial=True)
        aut.add_location(1, final=True)
        aut.add_transition(0, 1, guard=logic.Variable("a"))

        # Both should work
        op1 = MatrixOperator.make(aut, maxplus, weight_function=float_weight)
        op2 = MatrixOperator.make(aut, maxplus, weight_function=int_weight)

        x = jnp.array([1.0])
        assert op1.cost_transitions(x).data[0, 1] == 3.14
        assert op2.cost_transitions(x).data[0, 1] == 42


class TestWeightFunctionWithDifferentGuardAPTypes:
    """Test weight functions with different guard representations."""

    def test_weight_function_with_string_guards(self) -> None:
        """Weight functions should handle string guard representations."""
        aut: NFA[str] = NFA()
        aut.add_location(0, initial=True)
        aut.add_location(1, final=True)
        # Add transition with string guard
        aut.add_transition(0, 1, guard=logic.Variable("a"))

        def weight_fn(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            # Guard[str] should be parsed to Expr
            assert isinstance(guard, logic.Expr)
            return 1.0

        operator = MatrixOperator.make(
            aut,
            maxplus,
            weight_function=weight_fn,
        )

        x = jnp.array([1.0])
        operator.cost_transitions(x)

    def test_weight_function_with_expr_guards(self) -> None:
        """Weight functions should handle Expr guard representations."""
        aut: NFA[str] = NFA()
        aut.add_location(0, initial=True)
        aut.add_location(1, final=True)
        # Add transition with Expr guard
        aut.add_transition(0, 1, guard=logic.Literal(True))

        def weight_fn(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            assert isinstance(guard, logic.Expr)
            return 1.0

        operator = MatrixOperator.make(
            aut,
            maxplus,
            weight_function=weight_fn,
        )

        x = jnp.array([1.0])
        operator.cost_transitions(x)


class TestWeightFunctionClosures:
    """Test weight functions that capture external state."""

    def test_weight_function_with_captured_variable(self) -> None:
        """Weight functions can capture variables from their closure."""
        scale = 2.5

        def scaled_weight(x: InputSymbol, guard: Guard[str]) -> SemiringValue:
            return scale

        aut: NFA[str] = NFA()
        aut.add_location(0, initial=True)
        aut.add_location(1, final=True)
        aut.add_transition(0, 1, guard=logic.Variable("a"))

        operator = MatrixOperator.make(
            aut,
            maxplus,
            weight_function=scaled_weight,
        )

        x = jnp.array([1.0])
        assert operator.cost_transitions(x).data[0, 1] == 2.5

    def test_weight_function_with_object_state(self) -> None:
        """Weight functions can be methods capturing object state."""

        class WeightGenerator:
            def __init__(self, multiplier: float) -> None:
                self.multiplier = multiplier

            def __call__(self, x: InputSymbol, guard: Guard[str]) -> SemiringValue:
                return self.multiplier * float(x[0])

        gen = WeightGenerator(10.0)
        aut: NFA[str] = NFA()
        aut.add_location(0, initial=True)
        aut.add_location(1, final=True)
        aut.add_transition(0, 1, guard=logic.Variable("a"))

        operator = MatrixOperator.make(
            aut,
            maxplus,
            weight_function=gen,
        )

        x = jnp.array([3.0])
        assert operator.cost_transitions(x).data[0, 1] == 30.0
