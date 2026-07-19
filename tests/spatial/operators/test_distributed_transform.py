"""The general layout-preserving distributed transform apply.

``DistributedTransform`` runs ``backward(middle(forward(.)))`` inside one
``jax.shard_map`` region so every FFT axis is device-local when its
transform runs (the sharded-transform-axis FFT otherwise hits an upstream
XLA:GPU distributed-FFT fault; see
``design/research/multidevice_test_faults.md``). It generalizes the fused
spectral solve beyond the pressure solve: the coefficient frame stays
internal (``a``-sharded) and the nodal field enters and leaves on its own
negotiated layout -- layout-preserving from the outside. These tests
drive a plain ``Fourier`` transform on a periodic grid and compare
against a replicated single-device reference. ``multi_device`` tests need
the forced-4-device suite (``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.distributed_transform import (
    DistributedTransform,
    resolve_distributed_transform,
)
from fridom.spatial.operators.fourier import Fourier


def periodic_grid(shape, device_ids=None):
    """Return an all-periodic grid of the given per-axis cell counts."""
    names = ("x", "y", "z")[:len(shape)]
    meshes = tuple(
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name=name)
        for n, name in zip(shape, names, strict=True))
    return Grid(meshes, device_ids=device_ids)


def frame_reference(data, geom):
    """Replicated forward in the internal frame (half on the local axis).

    The internal coefficient frame is the half spectrum on the local
    Hermitian axis (the plan's half stage) and the full spectrum on every
    other transformed axis -- deliberately not the single-device
    codomain (half on the first axis).
    """
    half_axis = next(ax for ax, half, _ in geom.local_stages if half)
    ref = jnp.fft.rfft(jnp.asarray(data), axis=half_axis, norm="forward")
    full = tuple(ax for ax in range(data.ndim) if ax != half_axis)
    return np.asarray(jnp.fft.fftn(ref, axes=full, norm="forward"))


def diagonal_reference(data, diag, geom):
    """Replicated ``backward(diag * forward)`` in the internal frame.

    The single-controller reference for ``apply_diagonal``: the internal
    forward (``frame_reference``), the per-mode multiply, and the inverse
    (``ifftn`` on the full axes, ``irfft`` on the half axis, real part).
    """
    half_axis = next(ax for ax, half, _ in geom.local_stages if half)
    coeff = frame_reference(data, geom) * np.asarray(diag)
    full = tuple(ax for ax in range(data.ndim) if ax != half_axis)
    back = jnp.fft.ifftn(jnp.asarray(coeff), axes=full, norm="forward")
    back = jnp.fft.irfft(
        back, n=data.shape[half_axis], axis=half_axis, norm="forward")
    return np.asarray(back.real)


def sharded_axis_diagonal(dt):
    """Return a per-mode diagonal varying along the *sharded* axis ``a``.

    Shaped over the internal coefficient frame; the ``a`` entries differ
    per mode, so the closure ``middle`` of ``apply`` (shard-agnostic)
    could not carry it -- ``apply_diagonal`` threads it sharded on ``a``.
    """
    shape = dt.coeff.bare.shape
    a = dt.geometry.a
    ramp = np.exp(1j * 0.3 * np.arange(shape[a]))
    view = [1] * len(shape)
    view[a] = shape[a]
    return jnp.asarray(np.broadcast_to(ramp.reshape(view), shape))


# ================================================================
#  Resolution and decline conditions
# ================================================================
def test_single_device_declines():
    grid = periodic_grid((12, 8, 12), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    assert resolve_distributed_transform(Fourier(grid), grid, bare) is None


@pytest.mark.multi_device
def test_resolves_on_a_sharded_periodic_grid(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    bare = grid.create_field().function_space.bare
    dt = resolve_distributed_transform(Fourier(grid), grid, bare)
    assert isinstance(dt, DistributedTransform)
    # memoized per (grid, bare)
    assert resolve_distributed_transform(Fourier(grid), grid, bare) is dt


def test_padded_transform_declines():
    grid = periodic_grid((12, 8, 12), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    dealias = Fourier(grid, pad=degree(2))
    assert resolve_distributed_transform(dealias, grid, bare) is None


# ================================================================
#  Layout-preserving fused apply (genuinely sharded)
# ================================================================
@pytest.mark.multi_device
def test_roundtrip_is_the_identity(forced_devices):
    # forward then backward returns the field, layout-preserving (the
    # nodal operand enters and leaves sharded on the same axis)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    (sharded_axis, _), = grid.decomposition.default_layout.device_axes
    rng = np.random.default_rng(1)
    data = rng.standard_normal((12, 8, 12))
    f = grid.create_field(data=jnp.asarray(data))
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    out = dt.apply(f)
    assert not out.function_space.layout.is_local(sharded_axis)
    assert np.abs(np.asarray(out.data) - data).max() <= 1e-12


@pytest.mark.multi_device
def test_scalar_spectral_middle_scales_the_field(forced_devices):
    # a pointwise middle on the internal coefficient frame (scale every
    # mode by two) synthesizes back to twice the field
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(2)
    data = rng.standard_normal((12, 8, 12))
    f = grid.create_field(data=jnp.asarray(data))
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    out = dt.apply(f, lambda c: 2.0 * c)
    assert np.abs(np.asarray(out.data) - 2.0 * data).max() <= 1e-12


@pytest.mark.multi_device
def test_forward_region_matches_the_frame_reference(forced_devices):
    # the raw analysis region reproduces the replicated single-device
    # forward in the internal frame (half on the local Hermitian axis)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(3)
    data = rng.standard_normal((12, 8, 12))
    f = grid.create_field(data=jnp.asarray(data))
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    coeff = dt.forward_region(jnp.asarray(f.data))
    assert coeff.sharding.spec[dt.geometry.a] == dt.geometry.axis_name
    ref = frame_reference(data, dt.geometry)
    assert np.abs(np.asarray(coeff) - ref).max() <= 1e-11


@pytest.mark.multi_device
def test_indivisible_sharded_axis_roundtrips(forced_devices):
    # an indivisible sharded axis rides the padded-even nodal frame
    # (unpad_even / pad_even) -- the true-frame trim would gather it
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((10, 10, 14))
    (sharded_axis, _), = grid.decomposition.default_layout.device_axes
    assert grid.factors[grid.names.index(sharded_axis)].n_cells % 4 != 0
    rng = np.random.default_rng(4)
    shape = tuple(m.n_cells for m in grid.factors)
    data = rng.standard_normal(shape)
    f = grid.create_field(data=jnp.asarray(data))
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    assert dt.geometry.padded
    out = dt.apply(f)
    assert not out.function_space.layout.is_local(sharded_axis)
    assert np.abs(np.asarray(out.data) - data).max() <= 1e-12


# ================================================================
#  Diagonal-middle fused apply (per-mode symbol on the sharded axis)
# ================================================================
@pytest.mark.multi_device
def test_diagonal_middle_varies_along_the_sharded_axis(forced_devices):
    # the diagonal varies per mode along the *sharded* axis a -- the
    # capability the closure ``middle`` cannot carry -- and the fused
    # apply matches the replicated single-controller reference
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(11)
    data = rng.standard_normal((12, 8, 12))
    f = grid.create_field(data=jnp.asarray(data))
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    diag = sharded_axis_diagonal(dt)
    out = dt.apply_diagonal(f, diag)
    # layout-preserving: the sharded axis leaves sharded
    (sharded_axis, _), = grid.decomposition.default_layout.device_axes
    assert not out.function_space.layout.is_local(sharded_axis)
    ref = diagonal_reference(data, diag, dt.geometry)
    assert np.abs(np.asarray(out.data) - ref).max() <= 1e-11


@pytest.mark.multi_device
def test_diagonal_middle_broadcasts_a_constant_diagonal(forced_devices):
    # a diagonal that is size-1 on the sharded axis (constant along a)
    # is broadcast to the full internal frame before sharding, so a
    # scalar-like middle still runs through the diagonal path
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(12)
    data = rng.standard_normal((12, 8, 12))
    f = grid.create_field(data=jnp.asarray(data))
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    diag = jnp.asarray(2.0)
    out = dt.apply_diagonal(f, diag)
    # a constant diagonal of 2 is the scalar-middle round trip: 2 x data
    assert np.abs(np.asarray(out.data) - 2.0 * data).max() <= 1e-12


@pytest.mark.multi_device
def test_diagonal_middle_on_an_indivisible_sharded_axis(forced_devices):
    # an indivisible sharded axis rides the padded-even nodal frame and
    # the diagonal is tail-padded on its coefficient extent; the pad
    # lanes multiply transient pad data the synthesis slices off, so the
    # true-frame result still matches the replicated reference
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((10, 10, 14))
    (sharded_axis, _), = grid.decomposition.default_layout.device_axes
    assert grid.factors[grid.names.index(sharded_axis)].n_cells % 4 != 0
    rng = np.random.default_rng(14)
    shape = tuple(m.n_cells for m in grid.factors)
    data = rng.standard_normal(shape)
    f = grid.create_field(data=jnp.asarray(data))
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    assert dt.geometry.padded
    diag = sharded_axis_diagonal(dt)
    out = dt.apply_diagonal(f, diag)
    assert not out.function_space.layout.is_local(sharded_axis)
    ref = diagonal_reference(data, diag, dt.geometry)
    assert np.abs(np.asarray(out.data) - ref).max() <= 1e-11


@pytest.mark.multi_device
def test_diagonal_middle_grad_is_finite(forced_devices):
    # jax.grad of a quadratic loss through the diagonal apply is finite
    # and matches a central finite difference (the shard_map / all_to_all
    # VJP composes with the per-mode multiply)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(13)
    data = jnp.asarray(rng.standard_normal((12, 8, 12)))
    f = grid.create_field(data=data)
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    diag = sharded_axis_diagonal(dt)

    def loss(arr):
        out = dt.apply_diagonal(f.with_data(arr), diag)
        return jnp.sum(out.data ** 2)

    grad = jax.grad(loss)(data)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
    eps = 1e-4
    pert = jnp.asarray(rng.standard_normal(data.shape))
    num = (loss(data + eps * pert) - loss(data - eps * pert)) / (2 * eps)
    ana = float(jnp.sum(grad * pert))
    assert abs(num - ana) <= 1e-4 * max(1.0, abs(ana))


# ================================================================
#  HLO collective profile
# ================================================================
@pytest.mark.multi_device
def test_hlo_transposes_without_gathers(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(5)
    f = grid.create_field(data=jnp.asarray(rng.standard_normal((12, 8, 12))))
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    text = dt._roundtrip.lower(
        jnp.asarray(f.data)).compile().as_text()
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


# ================================================================
#  Autodiff finiteness through the shard_map / all_to_all VJP
# ================================================================
@pytest.mark.multi_device
def test_grad_is_finite_and_matches_finite_difference(forced_devices):
    # jax.grad of a quadratic loss through the fused apply is finite and
    # matches a central finite difference (a spectral low-pass middle)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(6)
    data = jnp.asarray(rng.standard_normal((12, 8, 12)))
    f = grid.create_field(data=data)
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)

    def middle(c):
        return 0.5 * c

    def loss(arr):
        out = dt.apply(f.with_data(arr), middle)
        return jnp.sum(out.data ** 2)

    grad = jax.grad(loss)(data)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
    eps = 1e-4
    pert = jnp.asarray(rng.standard_normal(data.shape))
    num = (loss(data + eps * pert) - loss(data - eps * pert)) / (2 * eps)
    ana = float(jnp.sum(grad * pert))
    assert abs(num - ana) <= 1e-4 * max(1.0, abs(ana))
