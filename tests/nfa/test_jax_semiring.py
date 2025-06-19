from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import pytest

from automatix.algebra.semiring.jax_backend import CountingSemiring


@pytest.mark.parametrize(
    "batch,n,m,l",
    [
        (None, None, 5, 20),
        (None, 1, 100, 200),
        (None, 20, 500, 10),
        (1, 1, 5, 20),
        (40, 1, 250, 250),
        (500, 1, 500, 500),
        # (1000, 1, 1000, 1000),
    ],
)
def test_real_semiring(batch: Optional[int], n: Optional[int], m: int, l: int) -> None:  # noqa: E741
    expected_shape: Tuple[int, ...]
    rng = jax.random.key(42)
    rng, key_a, key_b = jax.random.split(rng, 3)
    if batch:
        n = n if n is not None else 1
        mat1 = jax.random.normal(key_a, (batch, n, m))
        mat2 = jax.random.normal(key_b, (batch, m, l))
        expected_shape = (batch, n, l)
    else:
        if n is None:
            mat1 = jax.random.normal(key_a, (1, m))
            expected_shape = (1, l)
        else:
            mat1 = jax.random.normal(key_a, (n, m))
            expected_shape = (n, l)
        mat2 = jax.random.normal(key_b, (m, l))

    expected = mat1 @ mat2
    assert expected.shape == expected_shape, f"{expected.shape} != {expected_shape}"

    if batch:
        ours = jax.vmap(CountingSemiring.matmul, (0, 0), 0)(mat1, mat2)
    else:
        ours = CountingSemiring.matmul(mat1, mat2)
    assert ours.shape == expected.shape, f"{ours.shape} != {expected.shape}"
    assert jnp.allclose(ours, expected, rtol=1e-4, atol=1e-4).item()
