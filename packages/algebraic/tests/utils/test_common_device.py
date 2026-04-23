"""Tests for common_device and to_common_device utilities."""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import algebraic
import numpy as np
import pytest
from algebraic.semirings import counting_semiring
from algebraic.types import is_jax_array, is_numpy_array, is_torch_array
from algebraic.utils import common_device, to_common_device

# ---------------------------------------------------------------------------
# CPU-only tests (no GPU required)
# ---------------------------------------------------------------------------


class TestCommonDeviceCPU:
    """Tests for common_device that don't require a GPU."""

    def test_all_scalars_returns_cpu(self) -> None:
        assert common_device(1.0, 2, 3.5) == "cpu"

    def test_numpy_array(self) -> None:
        arr = np.array([1.0, 2.0])
        dev = common_device(arr)
        assert str(dev).lower() == "cpu"

    def test_numpy_array_and_scalar(self) -> None:
        arr = np.array([1.0])
        dev = common_device(arr, 5.0)
        assert str(dev).lower() == "cpu"

    def test_two_numpy_arrays(self) -> None:
        a = np.array([1.0])
        b = np.array([2.0])
        assert str(common_device(a, b)).lower() == "cpu"


class TestToCommonDeviceCPU:
    """Tests for to_common_device that don't require a GPU."""

    def test_all_scalars_become_numpy(self) -> None:
        a, b, c = to_common_device(1.0, 2, 3.5)
        assert is_numpy_array(a)
        assert is_numpy_array(b)
        assert is_numpy_array(c)

    def test_scalar_converted_to_sibling_backend(self) -> None:
        arr = np.array([1.0, 2.0])
        out_arr, out_scalar = to_common_device(arr, 5.0)
        assert is_numpy_array(out_arr)
        assert is_numpy_array(out_scalar)
        np.testing.assert_array_equal(np.asarray(out_scalar), np.asarray(5.0))

    def test_numpy_arrays_unchanged(self) -> None:
        a = np.array([1.0])
        b = np.array([2.0])
        out_a, out_b = to_common_device(a, b)
        assert out_a is a
        assert out_b is b


# ---------------------------------------------------------------------------
# GPU tests - skipped when no CUDA device is available
# ---------------------------------------------------------------------------

_has_jax_gpu = False
try:
    import jax

    _has_jax_gpu = any(d.platform == "gpu" for d in jax.devices())
except ImportError:
    pass

_has_torch_gpu = False
try:
    import torch as _torch

    _has_torch_gpu = _torch.cuda.is_available()
except ImportError:
    pass

requires_jax_gpu = pytest.mark.skipif(not _has_jax_gpu, reason="JAX GPU not available")
requires_torch_gpu = pytest.mark.skipif(not _has_torch_gpu, reason="Torch CUDA not available")
requires_both_gpu = pytest.mark.skipif(not (_has_jax_gpu and _has_torch_gpu), reason="Both JAX GPU and Torch CUDA required")


@requires_jax_gpu
class TestCommonDeviceJAXGPU:
    """common_device tests that require a JAX GPU."""

    def test_jax_gpu_array_returns_gpu_device(self) -> None:
        import jax.numpy as jnp

        arr = jnp.array([1.0])
        dev = common_device(arr)
        assert "cuda" in str(dev).lower() or "gpu" in str(dev).lower()

    def test_jax_gpu_beats_cpu_scalar(self) -> None:
        import jax.numpy as jnp

        arr = jnp.array([1.0])
        dev = common_device(arr, 3.0)
        assert "cuda" in str(dev).lower() or "gpu" in str(dev).lower()

    def test_jax_gpu_beats_numpy_cpu(self) -> None:
        import jax.numpy as jnp

        jax_arr = jnp.array([1.0])
        np_arr = np.array([2.0])
        dev = common_device(jax_arr, np_arr)
        assert "cuda" in str(dev).lower() or "gpu" in str(dev).lower()

    def test_jax_gpu_beats_numpy_cpu_reverse_order(self) -> None:
        import jax.numpy as jnp

        jax_arr = jnp.array([1.0])
        np_arr = np.array([2.0])
        dev = common_device(np_arr, jax_arr)
        assert "cuda" in str(dev).lower() or "gpu" in str(dev).lower()


@requires_torch_gpu
class TestCommonDeviceTorchGPU:
    """common_device tests that require a Torch CUDA device."""

    def test_torch_gpu_array_returns_gpu_device(self) -> None:
        import torch

        arr = torch.tensor([1.0], device="cuda")
        dev = common_device(arr)
        assert "cuda" in str(dev).lower()

    def test_torch_gpu_beats_cpu_scalar(self) -> None:
        import torch

        arr = torch.tensor([1.0], device="cuda")
        dev = common_device(arr, 3.0)
        assert "cuda" in str(dev).lower()

    def test_torch_gpu_beats_torch_cpu(self) -> None:
        import torch

        gpu = torch.tensor([1.0], device="cuda")
        cpu = torch.tensor([2.0], device="cpu")
        dev = common_device(cpu, gpu)
        assert "cuda" in str(dev).lower()


@requires_both_gpu
class TestCommonDeviceCrossFramework:
    """Mixing JAX and Torch arrays must raise TypeError."""

    def test_jax_gpu_and_torch_gpu_raises(self) -> None:
        import jax.numpy as jnp
        import torch

        j = jnp.array([1.0])
        t = torch.tensor([1.0], device="cuda")
        with pytest.raises(TypeError, match="different frameworks"):
            common_device(j, t)

    def test_jax_gpu_and_torch_cpu_raises(self) -> None:
        import jax.numpy as jnp
        import torch

        j = jnp.array([1.0])
        t = torch.tensor([1.0])
        with pytest.raises(TypeError, match="different frameworks"):
            common_device(j, t)


@requires_jax_gpu
class TestToCommonDeviceJAXGPU:
    """to_common_device tests that require a JAX GPU."""

    def test_scalar_becomes_jax_array(self) -> None:
        import jax.numpy as jnp

        arr = jnp.array([1.0])
        out_arr, out_scalar = to_common_device(arr, 5.0)
        assert is_jax_array(out_arr)
        assert is_jax_array(out_scalar)

    def test_numpy_moved_to_jax_gpu(self) -> None:
        import jax.numpy as jnp

        jax_arr = jnp.array([1.0])
        np_arr = np.array([2.0])
        out_jax, out_np = to_common_device(jax_arr, np_arr)
        assert is_jax_array(out_jax)
        assert is_jax_array(out_np)
        np.testing.assert_allclose(np.asarray(out_np), [2.0])

    def test_algebraic_array_moved_to_gpu(self) -> None:
        import jax.numpy as jnp

        semiring = counting_semiring()
        cpu_alg = algebraic.array(np.array([1.0, 2.0]), semiring=semiring, backend="numpy")
        gpu_arr = jnp.array([3.0, 4.0])
        out_alg, out_raw = to_common_device(cpu_alg, gpu_arr)
        dev_str = str(common_device(out_alg)).lower()
        assert "cuda" in dev_str or "gpu" in dev_str

    def test_multiple_scalars_with_jax_array(self) -> None:
        import jax.numpy as jnp

        arr = jnp.array([1.0])
        results = to_common_device(2.0, arr, 3.0)
        assert len(results) == 3
        for r in results:
            assert is_jax_array(r) or (hasattr(r, "data") and is_jax_array(r.data))


@requires_torch_gpu
class TestToCommonDeviceTorchGPU:
    """to_common_device tests that require a Torch CUDA device."""

    def test_scalar_becomes_cuda_tensor(self) -> None:
        import torch

        arr = torch.tensor([1.0], device="cuda")
        out_arr, out_scalar = to_common_device(arr, 5.0)
        assert is_torch_array(out_arr)
        assert is_torch_array(out_scalar)
        assert str(out_scalar.device).startswith("cuda")

    def test_cpu_tensor_moved_to_gpu(self) -> None:
        import torch

        gpu = torch.tensor([1.0], device="cuda")
        cpu = torch.tensor([2.0], device="cpu")
        out_gpu, out_cpu = to_common_device(gpu, cpu)
        assert is_torch_array(out_gpu) and is_torch_array(out_cpu)
        assert str(out_gpu.device).startswith("cuda")
        assert str(out_cpu.device).startswith("cuda")
        np.testing.assert_allclose(out_cpu.cpu().numpy(), [2.0])

    def test_algebraic_array_on_torch_gpu(self) -> None:
        import torch

        semiring = counting_semiring()
        gpu_tensor = torch.tensor([1.0, 2.0], device="cuda")
        cpu_alg = algebraic.array(torch.tensor([3.0, 4.0]), semiring=semiring, backend="torch")
        out_alg, out_raw = to_common_device(cpu_alg, gpu_tensor)
        assert str(common_device(out_alg)).startswith("cuda")
