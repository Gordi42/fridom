"""Backward-only synthesis entry of the fused distributed contraction.

The synthesis features (single-mode ``mode()`` states, random-phase
channel states) build their coefficient columns host-side and inverse
transform them; on a grid that shards a periodic axis that inverse runs
through ``ContractPlan.synthesize`` -- the contraction's mirrored
backward pipeline (``ifft`` / ``all_to_all`` / ``irfft``) inside one
``jax.shard_map`` region, so every FFT axis is device-local (the
sharded-transform-axis FFT otherwise hits an upstream XLA:GPU
distributed-FFT fault; see ``design/research/multidevice_test_faults.md``).
These tests build the plan on synthetic coefficient columns (no
eigensolve, so they are safe on the forced-4 CPU leg) and compare
against a replicated single-device inverse (plain ``ifft`` / ``irfft``).
``multi_device`` marked tests need the forced-4-device suite
(``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``) to run genuinely sharded.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.distributed_contract import (
    resolve_distributed_contraction,
)

COMPONENTS = ("c0", "c1")
NY = 6


def make_channel_grid(nx, ny, nz, device_ids=None):
    """Return a walled-y channel grid (x, z periodic; y bounded)."""
    meshes = (
        IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(nz, (0.0, 2 * np.pi), periodic=True, name="z"))
    return Grid(meshes, device_ids=device_ids)


def resolve(grid, slices):
    """Resolve the plan for the standard walled-y channel signature."""
    return resolve_distributed_contraction(
        grid, bounded_axis="y", periodic_axis="z",
        components=COMPONENTS, slices=slices)


def coefficient_columns(grid, seed):
    """Synthetic store-frame coefficient columns (a full, b half).

    The engine frame: full spectrum on the sharded axis ``a`` (= x), the
    ``rfft`` half spectrum on the half axis ``b`` (= z), the bounded
    axis nodal. A per-component complex array of that global shape.
    """
    rng = np.random.default_rng(seed)
    coeffs = {}
    for name in COMPONENTS:
        shape = list(grid.create_field(name=name).data.shape)
        shape[2] = shape[2] // 2 + 1  # z (axis 2) -> half spectrum
        coeffs[name] = jnp.asarray(
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    return coeffs


def backward_reference(coeffs, half_n):
    """Replicated single-device inverse of store-frame columns.

    The mirror of ``ContractPlan.synthesize``: inverse ``fft`` on the
    sharded axis (x, axis 0), Hermitian ``irfft`` on the half axis (z,
    axis 2), real part -- the same math the fused region runs per shard.
    """
    result = {}
    for name in COMPONENTS:
        seg = jnp.asarray(coeffs[name])
        seg = jnp.fft.ifft(seg, axis=0, norm="forward")
        seg = jnp.fft.irfft(seg, n=half_n, axis=2, norm="forward")
        result[name] = np.asarray(seg.real)
    return result


def _ceil_mult(n, shards):
    return -(-n // shards) * shards


# ================================================================
#  synthesize matches the replicated backward reference
# ================================================================
@pytest.mark.multi_device
@pytest.mark.parametrize(
    ("nx", "nz", "a_padded"),
    [(16, 8, False), (16, 14, False), (18, 18, True)],
    ids=["divisible", "divisible-half-nopad", "indivisible-padded"])
def test_synthesize_matches_the_backward_reference(
        nx, nz, a_padded, forced_devices):
    # the fused backward-only synthesis reproduces the replicated
    # single-device inverse bit-for-bit (to floating point), lands real,
    # and keeps the grid's own layout (the sharded axis x is never
    # gathered). The three cases exercise: the true field frame; the
    # half axis's no-pad early return (nz=14 -> half extent 8, already a
    # multiple of four); and the indivisible sharded axis's padded-even
    # frame (nx=18 -> pad_a=20).
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    shards = jax.device_count()
    grid = make_channel_grid(nx, NY, nz)
    assert grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    slices = {"c0": slice(0, NY), "c1": slice(NY, 2 * NY)}
    plan = resolve(grid, slices)
    assert plan is not None
    assert plan.a_padded is a_padded
    assert plan._pad_a == _ceil_mult(nx, shards)

    coeffs = coefficient_columns(grid, seed=1)
    templates = {name: grid.create_field(name=name)
                 for name in COMPONENTS}
    out = plan.synthesize(coeffs, templates)
    ref = backward_reference(coeffs, nz)
    for name in COMPONENTS:
        got = np.asarray(out[name].data)
        assert not np.iscomplexobj(got)
        assert out[name]._data.sharding.spec[0] == "devices"
        assert np.allclose(got, ref[name], rtol=1e-11, atol=1e-12)


@pytest.mark.multi_device
def test_synthesize_grad_is_finite(forced_devices):
    # the shard_map / all_to_all VJP of the backward-only region is
    # finite and nonzero (the synthesis features are host-orchestrated,
    # but the inverse pipeline stays reverse-mode differentiable)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = make_channel_grid(16, NY, 8)
    slices = {"c0": slice(0, NY), "c1": slice(NY, 2 * NY)}
    plan = resolve(grid, slices)
    assert plan is not None
    coeffs = coefficient_columns(grid, seed=4)
    templates = {name: grid.create_field(name=name)
                 for name in COMPONENTS}
    c1 = coeffs["c1"]

    def loss(real_part):
        cols = {"c0": real_part.astype(jnp.complex128), "c1": c1}
        out = plan.synthesize(cols, templates)
        return sum(jnp.sum(out[name].data ** 2) for name in COMPONENTS)

    grad = jax.grad(loss)(jnp.real(coeffs["c0"]))
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
