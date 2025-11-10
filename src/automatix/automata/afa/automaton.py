"""Alternating Finite Automaton (AFA) class definition.

An AFA is defined over:
- Alphabet (input symbol type)
- States (automaton states)
- Semiring K (values for weights and polynomials)

Core idea: States become polynomial indeterminants, transitions define
successor polynomials based on input and guards.

Status: PLACEHOLDER - Full implementation deferred to Week 2.
"""

from typing import Generic, TypeVar, List

# Type variables for genericity
Alph = TypeVar("Alph")  # Alphabet - input symbol type
Q = TypeVar("Q")        # State type


class AFA(Generic[Alph, Q]):
    r"""Alternating Finite Automaton with multilinear polynomial semantics.

    Generic Parameters
    ------------------
    Alph : Type
        Input alphabet. Examples:
        - str: for string processing
        - np.ndarray: for vector inputs
        - Custom types for domain-specific alphabets

    Q : Type
        State type. Usually:
        - int: numeric states (0, 1, 2, ...)
        - str: named states ("initial", "accept", ...)
        - Custom dataclass for complex state structures

    Semantics
    ---------

    An AFA evaluates multilinear polynomials where:
    - Indeterminants x_q correspond to automaton states q
    - Coefficients c_alpha come from weight function evaluations
    - At each step, we substitute current states with successors:
      P_new(x) = P_current(Q_0(x), Q_1(x), ..., Q_q(x))

    Where Q_i(x) is the successor polynomial for state i given input x.

    Key Properties
    ---------------
    - Multilinear: Each variable appears at most once per monomial
    - Semiring-based: All operations in K (addition, multiplication)
    - Polynomial semantics: Natural composition of multiple transitions
    - Alternation: Universal and existential branching via polynomial union

    REVIEW NEEDED: Complete the class definition
    - [ ] Define transition representation
    - [ ] Clarify weight function interface
    - [ ] Specify acceptance condition
    - [ ] Document state initialization
    - [ ] Error handling and validation

    Attributes (PLACEHOLDER)
    -------------------------
    states : List[Q]
        List of automaton states.
    initial_polynomial : ???
        REVIEW NEEDED: Initial polynomial representation
        Option A: Constant polynomial (all probability in state 0)
        Option B: Specified explicitly
        Option C: Derived from initial state set

    final_states : List[Q]
        States considered "accepting" (semantics depends on application).
        REVIEW NEEDED: What is the acceptance condition?
        - Polynomial evaluated at final states only?
        - All states reach certain threshold?
        - Specific polynomial degree?

    transitions : ???
        REVIEW NEEDED: How are transitions represented?
        Option A: Dict[Tuple[Q, Alph, guard], MultilinearPolynomial]
           - Each transition is explicitly a polynomial
           - Guard determines if transition applies
        Option B: weight_function + successor_mapping
           - weight_function: (input, guard) -> semiring value
           - successor_mapping: Q -> Q (deterministic successor)
           - Polynomials constructed on-the-fly
        Option C: Lazy evaluation
           - Compute successors from predicates at execution time
           - No pre-computed transition table

    Examples
    --------
    PLACEHOLDER: Add concrete example once structure is finalized.

    References
    ----------
    Gillespie (2023): Multilinear Polynomial Evaluation
    AFA_POLYNOMIAL_ARCHITECTURE.md: Polynomial semantics
    WEEK2_IMPLEMENTATION_PLAN.md: v0.6.0 roadmap
    """

    def __init__(self, **kwargs):
        """Initialize AFA.

        REVIEW NEEDED: Define proper constructor signature

        Current placeholder accepts arbitrary kwargs for flexibility.
        """
        # PLACEHOLDER IMPLEMENTATION
        self.states: List[Q] = []
        # self.transitions = None
        # self.initial_polynomial = None
        # self.final_states = []
        # self.weight_function = None

    def __repr__(self) -> str:
        """String representation."""
        return f"AFA(states={len(self.states)} states)"


__all__ = ["AFA"]
