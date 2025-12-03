# mypy: disable-error-code="no-untyped-call, no-any-return"

import jax
import jax.nn
import jax.numpy as jnp
import logic_asts as logic
import pytest
from algebraic import Semiring
from algebraic.tensor_algebra import jax as absalg
from algebraic.tensor_algebra.jax import JaxBiModule as BiModule
from jaxtyping import Array, Num, Scalar
from typing_extensions import TypeAlias

from automatix import Guard, WeightFunction
from automatix.automata.nfa import NFA
from automatix.operators import MatrixOperator
from automatix.weights.guard_weights import ExprWeightFn, Predicate

Box: TypeAlias = Num[Array, "4"]
Circle: TypeAlias = Num[Array, "3"]
Point: TypeAlias = Num[Array, "2"]


# Obstacle locations
RED_BOX: Box = jnp.array([0.0, 0.9, -1.0, -0.5])
"""red box in bottom right corner in the format `[x1, x2, y1, y2]`"""
GREEN_BOX: Box = jnp.array([0.2, 0.7, 0.8, 1.2])
"""green box in top right corner in the format `[x1, x2, y1, y2]`"""
ORANGE_BOX: Box = jnp.array([-1.0, -0.7, -0.2, 0.5])
"""orange box on the left in the format `[x1, x2, y1, y2]`"""

BLUE_CIRCLE: Circle = jnp.array([0.0, 0.0, 0.4])
"""blue circle in the center in the format `[x, y, radius]`"""


def signed_dist_box(point: Point, box: Box) -> Scalar:
    """Get the signed distance from a point to a box.

    If positive, the point is outside the box; otherwise, it is within the box.
    """
    # See: https://stackoverflow.com/a/30545544

    # Get the signed distance to the borders
    bottom_left = box[jnp.array([0, 2])] - point
    top_right = point - box[jnp.array([1, 3])]
    # Get the signed distance to the _closest_ border
    closest_border = jnp.maximum(bottom_left, top_right)
    dist = jnp.sqrt(jnp.sum(jax.nn.relu(closest_border) ** 2, axis=-1)) + jax.nn.relu(-jnp.amax(closest_border, axis=-1))

    # dist = jnp.linalg.vector_norm(jax.nn.relu(closest_border), axis=-1) + jax.nn.relu(-jnp.amax(closest_border, axis=-1))
    return dist


def test_signed_dist_to_box() -> None:
    point = jnp.array([-1.0, -0.75])
    dist = signed_dist_box(point, RED_BOX)

    assert dist.shape == ()
    assert jnp.allclose(dist, 1.0)
    print(dist)

    points = jnp.repeat(jnp.expand_dims(point, 0), 20, axis=0)
    assert points.shape == (20, 2)
    dists = jax.vmap(signed_dist_box, (0, None), 0)(points, RED_BOX)
    assert dists.shape == (20,)
    assert jnp.allclose(dists, 1.0)
    print(dists)


def make_box_predicate[S: Semiring](algebra: BiModule[S], box: Box) -> tuple[Predicate, Predicate]:
    """Make the predicate for being inside and outside a box."""

    @jax.jit
    def inside(x: Num[Array, " n"]) -> Scalar:
        dist = signed_dist_box(x, box)
        return -jax.nn.relu(dist)

    @jax.jit
    def outside(x: Num[Array, " n"]) -> Scalar:
        dist = signed_dist_box(x, box)
        return -jax.nn.relu(-dist)

    return Predicate(algebra, inside), Predicate(algebra, outside)


def make_circle_predicate[S: Semiring](algebra: BiModule[S], circle: Circle) -> tuple[Predicate, Predicate]:
    """Make the predicates for being inside and outside a circle"""

    @jax.jit
    def inside(x: Num[Array, " n"]) -> Scalar:
        center = circle[:-1]
        radius = circle[-1]
        signed_dist = jnp.linalg.norm(x - center) - radius
        return -jax.nn.relu(signed_dist)

    @jax.jit
    def outside(x: Num[Array, " n"]) -> Scalar:
        center = circle[:-1]
        radius = circle[-1]
        signed_dist = jnp.linalg.norm(x - center) - radius
        return -jax.nn.relu(-signed_dist)

    return Predicate(algebra, inside), Predicate(algebra, outside)


def parse_guard(expr: str) -> Guard[str]:
    return logic.parse_expr(expr)  # type: ignore[call-overload]


@pytest.fixture(
    params=[
        absalg.counting_semiring(),
        absalg.tropical_semiring(minplus=False),
        absalg.max_min_algebra(),
        absalg.tropical_semiring(minplus=True),
    ],
    ids=["CountingSemiring", "MaxPlusSemiring", "MaxMinSemiring", "MinPlusSemiring"],
)
def sequential_aut[S: Semiring](
    request: pytest.FixtureRequest,
) -> tuple[NFA[str], dict[str, Predicate], dict[str, Predicate], BiModule[S]]:
    aut: NFA[str] = NFA()
    aut.add_location(0, initial=True)
    aut.add_location(1)
    aut.add_location(2)
    aut.add_location(3, final=True)

    aut.add_transition(0, 0, guard=parse_guard("~red"))
    aut.add_transition(0, 1, guard=parse_guard("red"))
    aut.add_transition(1, 1, guard=parse_guard("~green"))
    aut.add_transition(1, 2, guard=parse_guard("green"))
    aut.add_transition(2, 2, guard=parse_guard("~orange"))
    aut.add_transition(2, 3, guard=parse_guard("orange"))
    aut.add_transition(3, 3, guard=logic.Literal(True))

    algebra: BiModule[S] = request.param
    assert isinstance(algebra, BiModule)

    in_red, out_red = make_box_predicate(algebra, RED_BOX)
    in_green, out_green = make_box_predicate(algebra, GREEN_BOX)
    in_orange, out_orange = make_box_predicate(algebra, ORANGE_BOX)

    atoms = dict(red=in_red, green=in_green, orange=in_orange)
    neg_atoms = dict(red=out_red, green=out_green, orange=out_orange)
    return aut, atoms, neg_atoms, algebra


def test_expr_weight_fn[S: Semiring](
    sequential_aut: tuple[NFA[str], dict[str, Predicate], dict[str, Predicate], BiModule[S]],
) -> None:
    aut, atoms, neg_atoms, algebra = sequential_aut
    weight_fn = ExprWeightFn[S, str](
        algebra=algebra,
        atoms=atoms,
        neg_atoms=neg_atoms,
    )
    assert isinstance(weight_fn, WeightFunction)
    operator = MatrixOperator.make(
        aut,
        algebra,
        weight_function=weight_fn,
    )

    assert operator.initial_weights.shape == (4,)
    assert operator.initial_weights[0] == algebra.ones(()).item()
    assert operator.final_weights.shape == (4,)
    assert operator.final_weights[3] == algebra.ones(()).item()

    transitions = jax.jit(operator.cost_transitions)

    n_timesteps = 1500

    # Generate a trajectory of a circle of radius 0.75 starting at theta = -pi/2 to pi
    angles = jnp.linspace(-jnp.pi / 2, jnp.pi * 3 / 2, n_timesteps)
    xs = jnp.cos(angles)
    ys = jnp.sin(angles)
    trajectory = jnp.stack((xs, ys), axis=0).T
    assert trajectory.shape == (n_timesteps, 2)

    assert transitions(trajectory[0]).shape == (4, 4)

    deltas = jax.vmap(transitions)(trajectory)
    assert deltas.shape == (n_timesteps, 4, 4)

    weights, _ = jax.lax.scan(lambda x, y: (algebra.matmul(x, y), None), operator.initial_weights.reshape(1, -1), deltas)
    weight = algebra.vdot(weights.squeeze(), operator.final_weights)
    assert weight.size == 1
