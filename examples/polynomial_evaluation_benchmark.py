"""Simple benchmark for polynomial evaluation performance.

This script establishes performance baselines for Algorithm 1 evaluation
across different polynomial sizes (q = num_states).

Usage:
    python examples/polynomial_evaluation_benchmark.py

Results are printed to stdout and should be captured for trend analysis.
"""

import time
from typing import Dict, Tuple

import jax

from automatix.algebra.backends.jax_ import LatticeAlgebra
from automatix.algebra.polynomials import MultilinearPolynomial
from automatix.algebra.polynomials.tensor_encoding import eval_algorithm_1


def benchmark_polynomial_evaluation(
    num_states: int,
    num_evals: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    r"""Benchmark polynomial evaluation for a given polynomial size.

    Creates a random polynomial with q states and evaluates it num_evals
    times at random points, measuring wall-clock time.

    Parameters
    ----------
    num_states : int
        Number of states (polynomial indeterminants).
    num_evals : int
        Number of evaluations to run (for timing).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        (mean_time_microseconds, std_dev_microseconds) per evaluation.
    """
    # Create a random polynomial with some nonzero coefficients
    key = jax.random.PRNGKey(seed)
    coeffs = jax.random.uniform(key, shape=(2**num_states,))
    poly = MultilinearPolynomial(
        LatticeAlgebra,
        coefficients=coeffs,
        num_states=num_states,
        max_degree=None,
    )

    # Create random evaluation points
    key = jax.random.fold_in(key, 1)
    evaluation_points = jax.random.uniform(key, shape=(num_evals, num_states))

    # Warm up JIT compilation
    test_values = {i: evaluation_points[0, i] for i in range(num_states)}
    _ = eval_algorithm_1(poly, LatticeAlgebra, test_values)

    # Run benchmark
    start_time = time.perf_counter()
    for i in range(num_evals):
        values = {j: evaluation_points[i, j] for j in range(num_states)}
        _ = eval_algorithm_1(poly, LatticeAlgebra, values)
    end_time = time.perf_counter()

    total_time_seconds = end_time - start_time
    mean_time_us = (total_time_seconds / num_evals) * 1e6
    std_dev_us = 0.0  # Placeholder (would need multiple runs)

    return mean_time_us, std_dev_us


def main() -> None:
    """Run benchmarks for various polynomial sizes."""
    print("Polynomial Evaluation Performance Baseline (Algorithm 1)")
    print("=" * 60)
    print()
    print(f"{'num_states':<15} {'2^q':<15} {'Time (µs)':<15} {'Status':<20}")
    print("-" * 60)

    # Test sizes up to q=12 (2^12 = 4096 coefficients)
    test_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    results: Dict[int, Tuple[float, float]] = {}

    for q in test_sizes:
        num_coeffs = 2**q
        try:
            mean_us, std_us = benchmark_polynomial_evaluation(q, num_evals=50)
            results[q] = (mean_us, std_us)

            # Determine status
            if mean_us < 10:
                status = "FAST"
            elif mean_us < 100:
                status = "ACCEPTABLE"
            elif mean_us < 1000:
                status = "SLOW"
            else:
                status = "VERY SLOW"

            print(f"{q:<15} {num_coeffs:<15} {mean_us:<15.2f} {status:<20}")

        except Exception as e:
            print(f"{q:<15} {num_coeffs:<15} {'ERROR':<15} {str(e)[:20]:<20}")

    print()
    print("Performance Analysis:")
    print("-" * 60)

    # Estimate scaling factor
    if len(results) >= 2:
        q1, (t1, _) = list(results.items())[0]
        q2, (t2, _) = list(results.items())[1]
        scaling = (t2 / t1) ** (1 / (q2 - q1))
        print(f"Estimated scaling factor per additional state: {scaling:.2f}x")
        print()

    # Print recommendations
    print("Recommendations for Week 1:")
    print("-" * 60)
    max_acceptable_time = 100  # microseconds
    for q, (mean_us, _) in results.items():
        if mean_us > max_acceptable_time:
            print(f"q={q}: {mean_us:.1f}µs - Consider optimization for larger problems")
            break
    else:
        print(f"All tested sizes < {max_acceptable_time}µs - No immediate optimization needed")

    print()
    print("Next steps:")
    print("  - Rerun this benchmark weekly to track improvements")
    print("  - If q=10 exceeds 100µs, implement Algorithm 4 (v0.6.0)")
    print("  - Profile hotspots with JAX profiler: jax.profiler")


if __name__ == "__main__":
    main()
