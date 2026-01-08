"""

This example demonstrates the correct weight function design where:
- Weight functions map (input_symbol, guard) to semiring values
- They implement lambda(x, Delta) from weighted automata theory
- They are integrated into automaton operators via make_automaton_operator

Examples shown:
1. Constant weight functions
2. Input-dependent weight functions
3. Guard-dependent weight functions
4. Combined input and guard-dependent weights
5. Using weight functions with automaton operators
"""

import jax.numpy as jnp
import logic_asts.base as logic
from algebraic.semirings import tropical_semiring
from jaxtyping import Array, Num
from morphata.examples.nfa import NFA

from automatix.operators import MatrixOperator
from automatix.spec import Guard, WeightFunction

# Type aliases for clarity
InputSymbol = Num[Array, "..."]
SemiringValue = float

# Create semiring instance (MaxPlus tropical semiring)
maxplus = tropical_semiring(minplus=False)


def example_1_constant_weight_function() -> None:
    """Example 1: Simple constant weight function.

    A constant weight function returns the same weight for all transitions,
    regardless of input or guard. This is useful for uniform weighting schemes.
    """
    print("\n" + "=" * 70)
    print("Example 1: Constant Weight Function")
    print("=" * 70)

    def constant_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        """Weight function that always returns 1.0."""
        return 1.0

    # Create a simple 2-state automaton
    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1, final=True)
    aut.add_transition(0, 1, guard="a")

    print("\nAutomaton structure:")
    print(f"  States: {aut.num_locations}")
    print(f"  Initial: {aut.initial_locations}")
    print(f"  Final: {aut.final_locations}")

    # Create automaton operator with weight function
    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=constant_weight,
    )

    # Compute transitions for a sample input
    x = jnp.array([1.0, 2.0])
    transitions = operator.cost_transitions(x)

    print("\nConstant weight function: lambda(x, guard) = 1.0")
    print(f"Transition matrix shape: {transitions.shape}")
    print(f"Transition matrix:\n{transitions}")


def example_2_input_dependent_weights() -> None:
    """Example 2: Weight function depending on input data.

    The weight can depend on the concrete input symbol x.
    This is useful for data-dependent transition costs.
    """
    print("\n" + "=" * 70)
    print("Example 2: Input-Dependent Weight Function")
    print("=" * 70)

    def input_norm_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        """Weight function based on norm of input vector."""
        return float(jnp.linalg.norm(x))

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1)
    aut.add_location(2, final=True)
    aut.add_transition(0, 1, guard="a")
    aut.add_transition(1, 2, guard="b")

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=input_norm_weight,
    )

    print("\nWeight function: lambda(x, guard) = ||x||")
    print("\nTesting with different inputs:")

    # Test with different inputs
    test_inputs = [
        jnp.array([3.0, 4.0]),  # norm = 5.0
        jnp.array([1.0, 0.0, 0.0]),  # norm = 1.0
        jnp.array([0.0, 0.0]),  # norm = 0.0
    ]

    for x in test_inputs:
        transitions = operator.cost_transitions(x)
        print(f"\n  Input x = {x}")
        print(f"  ||x|| = {jnp.linalg.norm(x):.1f}")
        print("  Non-zero transitions:")
        for i in range(transitions.shape[0]):
            for j in range(transitions.shape[1]):
                if transitions.data[i, j] != maxplus.zero:
                    print(f"    ({i}, {j}) -> {transitions[i, j]:.1f}")


def example_3_guard_dependent_weights() -> None:
    """Example 3: Weight function depending on guard expressions.

    The weight can depend on the guard expression type or content.
    This is useful for assigning different costs to different conditions.
    """
    print("\n" + "=" * 70)
    print("Example 3: Guard-Dependent Weight Function")
    print("=" * 70)

    def guard_type_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        """Weight function based on guard expression type."""
        if isinstance(guard, logic.Literal):
            # True/False literals get weight 0.5
            return 0.5
        else:
            # Other expressions get weight 1.0
            return 1.0

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1)
    aut.add_location(2, final=True)
    aut.add_transition(0, 1, guard="a")  # Parsed as atom/variable
    aut.add_transition(1, 2, guard=logic.Literal(True))  # Literal

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=guard_type_weight,
    )

    x = jnp.array([1.0])
    transitions = operator.cost_transitions(x)

    print("\nWeight function: lambda(x, guard) = 0.5 if guard is Literal else 1.0")
    print("\nTransition weights:")
    print(f"  (0, 1) -> {transitions.data[0, 1]:.1f} (atom guard)")
    print(f"  (1, 2) -> {transitions.data[1, 2]:.1f} (literal guard)")


def example_4_combined_input_guard_weights() -> None:
    """Example 4: Weight function depending on both input and guard.

    The weight can depend on both the input symbol x and the guard expression.
    This provides the full power of lambda(x, Delta) from automata theory.
    """
    print("\n" + "=" * 70)
    print("Example 4: Combined Input and Guard-Dependent Weights")
    print("=" * 70)

    def combined_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        """Weight based on both input norm and guard type."""
        input_contribution = float(jnp.linalg.norm(x))

        guard_contribution = 0.5 if isinstance(guard, logic.Literal) else 1.0

        return input_contribution * guard_contribution

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1, final=True)
    aut.add_transition(0, 1, guard="a")

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=combined_weight,
    )

    print("\nWeight function: lambda(x, guard) = ||x|| * guard_factor")
    print("  where guard_factor = 0.5 for Literal, 1.0 otherwise")

    print("\nTesting with different inputs:")
    test_inputs = [
        jnp.array([3.0, 4.0]),  # norm = 5.0
        jnp.array([2.0, 0.0]),  # norm = 2.0
    ]

    for x in test_inputs:
        transitions = operator.cost_transitions(x)
        norm = float(jnp.linalg.norm(x))
        expected = norm * 1.0  # guard "a" is not a Literal
        print(f"\n  Input x = {x}")
        print(f"  ||x|| = {norm:.1f}")
        print(f"  Weight (0, 1) = {transitions.data[0, 1]:.1f}")
        print(f"  Expected = {expected:.1f}")


def example_5_closure_weight_function() -> None:
    """Example 5: Weight function capturing external state.

    Weight functions can be closures that capture parameters.
    This is useful for parameterized weight functions and learning scenarios.
    """
    print("\n" + "=" * 70)
    print("Example 5: Parameterized Weight Function (Closure)")
    print("=" * 70)

    # Weight function parameterized by a scaling factor
    def make_scaled_weight(scale: float) -> WeightFunction:
        """Create a weight function with a given scale factor."""

        def scaled_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
            return scale * float(jnp.linalg.norm(x))

        return scaled_weight

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1, final=True)
    aut.add_transition(0, 1, guard="a")

    # Create operators with different scale factors
    scales = [0.5, 1.0, 2.0]
    operators = [MatrixOperator.make(aut, maxplus, weight_function=make_scaled_weight(s)) for s in scales]

    x = jnp.array([3.0, 4.0])
    norm = float(jnp.linalg.norm(x))

    print("\nCreated weight functions with different scale factors:")
    print(f"Input x = {x}, ||x|| = {norm:.1f}")

    for scale, operator in zip(scales, operators):
        transitions = operator.cost_transitions(x)
        weight = transitions.data[0, 1]
        print(f"\n  Scale factor: {scale}")
        print(f"  Weight (0, 1) = {weight:.1f}")
        print(f"  Expected = {scale * norm:.1f}")


def example_6_learnable_weights() -> None:
    """Example 6: Learning-based weight function.

    Weight functions can use learnable parameters indexed by guard.
    This is a sketch of how to implement learned transition costs.
    """
    print("\n" + "=" * 70)
    print("Example 6: Learnable Weight Function")
    print("=" * 70)

    # Simple learned model: map guards to fixed weights
    learned_weights = {
        "a": 1.5,
        "b": 2.5,
        "true": 0.0,  # True literal
    }

    def learned_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        """Look up weight based on guard string representation."""
        guard_str = str(guard)

        # In a real scenario, this would be indexed by guard structure
        # and might use neural networks or other learning models
        return learned_weights.get(guard_str, 1.0)

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1)
    aut.add_location(2, final=True)
    aut.add_transition(0, 1, guard="a")
    aut.add_transition(1, 2, guard="b")

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=learned_weight,
    )

    x = jnp.array([1.0])
    transitions = operator.cost_transitions(x)

    print("\nLearned weights mapping:")
    for guard_str, weight in learned_weights.items():
        print(f"  '{guard_str}' -> {weight}")

    print("\nTransitions in automaton:")
    print(f"  (0, 1) with guard 'a' -> weight {transitions.data[0, 1]}")
    print(f"  (1, 2) with guard 'b' -> weight {transitions.data[1, 2]}")


def main() -> None:
    """Run all examples."""
    print("\n" + "#" * 70)
    print("# Weight Function Demonstrations (Phase 2)")
    print("# lambda(x, Delta) as (InputSymbol, Guard) -> SemiringValue")
    print("#" * 70)

    example_1_constant_weight_function()
    example_2_input_dependent_weights()
    example_3_guard_dependent_weights()
    example_4_combined_input_guard_weights()
    example_5_closure_weight_function()
    example_6_learnable_weights()

    print("\n" + "#" * 70)
    print("# All examples completed successfully!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
