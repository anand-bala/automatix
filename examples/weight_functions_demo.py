"""
Weight function demonstrations for automatix.

Shows how weight functions map (input_symbol, guard) to semiring values,
implementing lambda(x, Delta) from weighted automata theory.

Examples shown:
1. Constant weight functions
2. Input-dependent weight functions
3. Guard-dependent weight functions
4. Combined input and guard-dependent weights
5. Parameterized weight functions (closures)
6. Learnable weight function sketch
"""

import numpy as np
import logic_asts.base as logic
from algebraic.semirings import tropical_semiring
from morphata.examples.nfa import NFA

from automatix.operators import MatrixOperator
from automatix.spec import Guard, WeightFunction

# Type aliases for clarity
InputSymbol = object
SemiringValue = float

# Create semiring instance (MaxPlus tropical semiring)
maxplus = tropical_semiring(minplus=False)


def example_1_constant_weight_function() -> None:
    """Example 1: Simple constant weight function."""
    print("\n" + "=" * 70)
    print("Example 1: Constant Weight Function")
    print("=" * 70)

    def constant_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        return 1.0

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1, final=True)
    aut.add_transition(0, 1, guard=logic.Variable("a"))

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=constant_weight,
        backend="numpy",
    )

    x = np.array([1.0, 2.0])
    transitions = operator.cost_transitions(x)

    print("\nConstant weight function: lambda(x, guard) = 1.0")
    print(f"Transition matrix shape: {transitions.shape}")
    print(f"Transition matrix:\n{transitions.data}")


def example_2_input_dependent_weights() -> None:
    """Example 2: Weight function depending on input data."""
    print("\n" + "=" * 70)
    print("Example 2: Input-Dependent Weight Function")
    print("=" * 70)

    def input_norm_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        return float(np.linalg.norm(x))

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1)
    aut.add_location(2, final=True)
    aut.add_transition(0, 1, guard=logic.Variable("a"))
    aut.add_transition(1, 2, guard=logic.Variable("b"))

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=input_norm_weight,
        backend="numpy",
    )

    print("\nWeight function: lambda(x, guard) = ||x||")
    print("\nTesting with different inputs:")

    test_inputs = [
        np.array([3.0, 4.0]),    # norm = 5.0
        np.array([1.0, 0.0, 0.0]),  # norm = 1.0
        np.array([0.0, 0.0]),    # norm = 0.0
    ]

    for x in test_inputs:
        transitions = operator.cost_transitions(x)
        print(f"\n  Input x = {x}, ||x|| = {np.linalg.norm(x):.1f}")
        print("  Non-zero transitions:")
        for i in range(transitions.shape[0]):
            for j in range(transitions.shape[1]):
                if transitions.data[i, j] != maxplus.zero:
                    print(f"    ({i}, {j}) -> {transitions.data[i, j]:.1f}")


def example_3_guard_dependent_weights() -> None:
    """Example 3: Weight function depending on guard expressions."""
    print("\n" + "=" * 70)
    print("Example 3: Guard-Dependent Weight Function")
    print("=" * 70)

    def guard_type_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        if isinstance(guard, logic.Literal):
            return 0.5
        return 1.0

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1)
    aut.add_location(2, final=True)
    aut.add_transition(0, 1, guard=logic.Variable("a"))
    aut.add_transition(1, 2, guard=logic.Literal(True))

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=guard_type_weight,
        backend="numpy",
    )

    x = np.array([1.0])
    transitions = operator.cost_transitions(x)

    print("\nWeight function: lambda(x, guard) = 0.5 if guard is Literal else 1.0")
    print(f"  (0, 1) -> {transitions.data[0, 1]:.1f} (atom guard)")
    print(f"  (1, 2) -> {transitions.data[1, 2]:.1f} (literal guard)")


def example_4_combined_input_guard_weights() -> None:
    """Example 4: Weight function depending on both input and guard."""
    print("\n" + "=" * 70)
    print("Example 4: Combined Input and Guard-Dependent Weights")
    print("=" * 70)

    def combined_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        input_contribution = float(np.linalg.norm(x))
        guard_contribution = 0.5 if isinstance(guard, logic.Literal) else 1.0
        return input_contribution * guard_contribution

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1, final=True)
    aut.add_transition(0, 1, guard=logic.Variable("a"))

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=combined_weight,
        backend="numpy",
    )

    print("\nWeight function: lambda(x, guard) = ||x|| * guard_factor")
    for x in [np.array([3.0, 4.0]), np.array([2.0, 0.0])]:
        transitions = operator.cost_transitions(x)
        norm = float(np.linalg.norm(x))
        print(f"\n  Input x = {x}, ||x|| = {norm:.1f}")
        print(f"  Weight (0, 1) = {transitions.data[0, 1]:.1f}  (expected {norm * 1.0:.1f})")


def example_5_closure_weight_function() -> None:
    """Example 5: Parameterized weight function (closure)."""
    print("\n" + "=" * 70)
    print("Example 5: Parameterized Weight Function (Closure)")
    print("=" * 70)

    def make_scaled_weight(scale: float) -> WeightFunction:
        def scaled_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
            return scale * float(np.linalg.norm(x))
        return scaled_weight  # type: ignore[return-value]

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1, final=True)
    aut.add_transition(0, 1, guard=logic.Variable("a"))

    x = np.array([3.0, 4.0])
    norm = float(np.linalg.norm(x))
    print(f"\nInput x = {x}, ||x|| = {norm:.1f}")

    for scale in [0.5, 1.0, 2.0]:
        op = MatrixOperator.make(aut, maxplus, weight_function=make_scaled_weight(scale), backend="numpy")
        weight = op.cost_transitions(x).data[0, 1]
        print(f"  scale={scale}: weight(0,1) = {weight:.1f}  (expected {scale * norm:.1f})")


def example_6_learnable_weights() -> None:
    """Example 6: Learnable weight function sketch."""
    print("\n" + "=" * 70)
    print("Example 6: Learnable Weight Function")
    print("=" * 70)

    learned_weights = {"a": 1.5, "b": 2.5, "true": 0.0}

    def learned_weight(x: InputSymbol, guard: Guard) -> SemiringValue:
        return learned_weights.get(str(guard), 1.0)

    aut = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1)
    aut.add_location(2, final=True)
    aut.add_transition(0, 1, guard=logic.Variable("a"))
    aut.add_transition(1, 2, guard=logic.Variable("b"))

    operator = MatrixOperator.make(
        aut,
        maxplus,
        weight_function=learned_weight,
        backend="numpy",
    )

    x = np.array([1.0])
    transitions = operator.cost_transitions(x)

    print("\nLearned weights:", learned_weights)
    print(f"  (0, 1) with guard 'a' -> weight {transitions.data[0, 1]}")
    print(f"  (1, 2) with guard 'b' -> weight {transitions.data[1, 2]}")


def main() -> None:
    print("\n" + "#" * 70)
    print("# Weight Function Demonstrations")
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
