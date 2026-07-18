"""The 2-D channel transpose-pipeline contraction (Channel2DPlan).

The 2-D channel (one periodic axis, one bounded axis) shards its single
periodic axis by default. With no second periodic axis to absorb the
shardedness, ``Channel2DPlan`` transposes **through the bounded axis**
(two ``all_to_all`` moves per direction) so the local ``rfft`` runs and
the per-``kx`` dense contraction stays local -- only ``all_to_all``, no
``all_gather``. These tests build the plan on synthetic ``q`` / weights /
metric (no eigensolve, so they are safe on the forced-4 CPU leg) and
compare against a replicated single-device reference. ``multi_device``
tests need the forced-4-device suite
(``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.distributed_contract import (
    Channel2DPlan,
    resolve_distributed_contraction,
)

COMPONENTS = ("c0", "c1")


def channel_grid(nx, ny, device_ids=None):
    """Return a walled-y 2-D channel grid (x periodic; y bounded)."""
    return Grid((
        IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, 1.0), periodic=False, name="y")),
        device_ids=device_ids)


def make_fields(grid, seed):
    """Two random real (nx, ny) center fields."""
    rng = np.random.default_rng(seed)
    return {name: grid.create_field(
        name=name, data=jnp.asarray(rng.standard_normal(
            grid.create_field(name=name).data.shape)))
        for name in COMPONENTS}


def synthetic_basis(nx, ny, seed):
    """Synthetic (q, weights, metric, slices) on the rfft plane frame."""
    rng = np.random.default_rng(seed)
    dim = 2 * ny
    n_kx = nx // 2 + 1
    slices = {"c0": slice(0, ny), "c1": slice(ny, dim)}

    def cplx(shape):
        return jnp.asarray(rng.standard_normal(shape)
                           + 1j * rng.standard_normal(shape))

    q = cplx((n_kx, dim, dim))
    weights = cplx((n_kx, dim))
    metric = jnp.asarray(rng.uniform(0.5, 1.5, dim))
    return q, weights, metric, slices


def reference(fields, q, weights, metric, slices, nx):
    """Replicated single-device ``Q diag(w) Q^H M z`` reference."""
    coeff = {name: jnp.fft.rfft(jnp.asarray(np.asarray(f.data)),
                                axis=0, norm="forward")
             for name, f in fields.items()}
    z = jnp.concatenate([coeff[name] for name in COMPONENTS], axis=-1)
    amp = jnp.einsum("...dj,d,...d->...j", jnp.conj(q), metric, z)
    out = jnp.einsum("...dj,...j->...d", q, weights * amp)
    result = {}
    for name in COMPONENTS:
        seg = jnp.fft.irfft(out[..., slices[name]], n=nx, axis=0,
                            norm="forward")
        result[name] = np.asarray(seg.real)
    return result


def resolve(grid, slices):
    return resolve_distributed_contraction(
        grid, bounded_axis="y", periodic_axis="x",
        components=COMPONENTS, slices=slices)


def _ceil_mult(n, shards):
    return -(-n // shards) * shards


# ================================================================
#  Resolution and decline conditions
# ================================================================
def test_single_device_declines():
    grid = channel_grid(16, 8, device_ids=(0,))
    _q, _w, _m, slices = synthetic_basis(16, 8, seed=0)
    assert resolve(grid, slices) is None


@pytest.mark.multi_device
def test_two_dimensional_channel_resolves(forced_devices):
    # the 2-D channel now resolves to a Channel2DPlan (the transpose
    # pipeline through the bounded axis) instead of declining
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = channel_grid(16, 8)
    assert grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    _q, _w, _m, slices = synthetic_basis(16, 8, seed=0)
    plan = resolve(grid, slices)
    assert isinstance(plan, Channel2DPlan)
    assert resolve(grid, slices) is plan


@pytest.mark.multi_device
def test_bounded_axis_sharded_declines(forced_devices):
    # the periodic axis (2) is too small to shard, so the layout shards
    # the bounded axis y -- the periodic rfft axis is then local and the
    # plain GSPMD path serves it, so the plan declines
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = channel_grid(2, 16)
    assert grid.decomposition.default_layout.device_axes == (
        ("y", "devices"),)
    slices = {"c0": slice(0, 16), "c1": slice(16, 32)}
    assert resolve(grid, slices) is None


# ================================================================
#  The kernel matches the replicated reference (genuinely sharded)
# ================================================================
@pytest.mark.multi_device
@pytest.mark.parametrize(
    ("nx", "ny"), [(16, 8), (18, 6)],
    ids=["divisible", "indivisible"])
def test_contraction_matches_the_replicated_reference(
        nx, ny, forced_devices):
    # (18, 6): x is indivisible but the bounded y=6 is too short to
    # shard, so x is the sole shardable axis (the padded-even frame);
    # (16, 8): x divisible, the true frame
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    shards = jax.device_count()
    grid = channel_grid(nx, ny)
    assert grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    fields = make_fields(grid, seed=1)
    q, weights, metric, slices = synthetic_basis(nx, ny, seed=2)
    plan = resolve(grid, slices)
    assert plan is not None
    assert plan.a_padded is (nx % shards != 0)

    out = plan.apply(fields, q, weights, metric)
    ref = reference(fields, q, weights, metric, slices, nx)
    for name in COMPONENTS:
        got = np.asarray(out[name].data)
        assert not np.iscomplexobj(got)
        assert out[name]._data.sharding.spec[0] == "devices"
        assert np.allclose(got, ref[name], rtol=1e-11, atol=1e-12)


@pytest.mark.multi_device
def test_small_bounded_axis_transient_shard(forced_devices):
    # the bounded axis (ny=6) is padded transiently to shard it (pad 8
    # over four devices -> a trailing empty pad shard); the zero pad rows
    # stay inert and the result still matches the replicated reference
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny = 16, 6
    grid = channel_grid(nx, ny)
    fields = make_fields(grid, seed=5)
    q, weights, metric, slices = synthetic_basis(nx, ny, seed=6)
    plan = resolve(grid, slices)
    assert plan is not None
    out = plan.apply(fields, q, weights, metric)
    ref = reference(fields, q, weights, metric, slices, nx)
    for name in COMPONENTS:
        assert np.allclose(np.asarray(out[name].data), ref[name],
                           rtol=1e-11, atol=1e-12)


# ================================================================
#  Synthesis (backward-only) matches the replicated inverse
# ================================================================
@pytest.mark.multi_device
@pytest.mark.parametrize(
    ("nx", "ny"), [(16, 8), (18, 6)],
    ids=["divisible", "indivisible"])
def test_synthesize_matches_the_backward_reference(
        nx, ny, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    n_kx = nx // 2 + 1
    grid = channel_grid(nx, ny)
    slices = {"c0": slice(0, ny), "c1": slice(ny, 2 * ny)}
    plan = resolve(grid, slices)
    assert plan is not None
    rng = np.random.default_rng(11)
    coeffs = {name: jnp.asarray(rng.standard_normal((n_kx, ny))
                                + 1j * rng.standard_normal((n_kx, ny)))
              for name in COMPONENTS}
    templates = {name: grid.create_field(name=name)
                 for name in COMPONENTS}
    out = plan.synthesize(coeffs, templates)
    for name in COMPONENTS:
        ref = np.asarray(jnp.fft.irfft(
            coeffs[name], n=nx, axis=0, norm="forward").real)
        got = np.asarray(out[name].data)
        assert not np.iscomplexobj(got)
        assert out[name]._data.sharding.spec[0] == "devices"
        assert np.allclose(got, ref, rtol=1e-11, atol=1e-12)


# ================================================================
#  HLO collective profile, memoization, warm compiles
# ================================================================
@pytest.mark.multi_device
def test_hlo_transposes_without_gathers(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = channel_grid(16, 8)
    fields = make_fields(grid, seed=1)
    q, weights, metric, slices = synthetic_basis(16, 8, seed=2)
    plan = resolve(grid, slices)
    qf = plan._pad_modes(q)
    wf = plan._pad_modes(weights.astype(qf.dtype))
    pieces = {name: jnp.asarray(fields[name].data)
              for name in COMPONENTS}
    text = plan._region.lower(
        pieces, qf, wf, metric).compile().as_text()
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


@pytest.mark.multi_device
def test_warm_contraction_adds_zero_compiles(
        compile_counter, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = channel_grid(16, 8)
    fields = make_fields(grid, seed=1)
    q, weights, metric, slices = synthetic_basis(16, 8, seed=2)
    plan = resolve(grid, slices)

    def apply_once():
        out = plan.apply(fields, q, weights, metric)
        return [np.asarray(out[name].data) for name in COMPONENTS]

    apply_once()
    apply_once()
    compile_counter.reset()
    apply_once()
    assert compile_counter.count == 0


# ================================================================
#  Autodiff finiteness through the shard_map / all_to_all VJP
# ================================================================
@pytest.mark.multi_device
def test_grad_is_finite_and_matches_finite_difference(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny = 16, 8
    grid = channel_grid(nx, ny)
    fields = make_fields(grid, seed=1)
    q, weights, metric, slices = synthetic_basis(nx, ny, seed=2)
    weights = weights.real.astype(q.dtype)  # real weights: real loss
    plan = resolve(grid, slices)
    qf = plan._pad_modes(q)
    wf = plan._pad_modes(weights)
    c0 = jnp.asarray(fields["c0"].data)

    def loss(arr):
        pieces = {"c0": arr, "c1": jnp.asarray(fields["c1"].data)}
        out = plan._region(pieces, qf, wf, metric)
        return sum(jnp.sum(o ** 2) for o in out.values())

    grad = jax.grad(loss)(c0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
    eps = 1e-4
    rng = np.random.default_rng(9)
    pert = jnp.asarray(rng.standard_normal(c0.shape))
    num = (loss(c0 + eps * pert) - loss(c0 - eps * pert)) / (2 * eps)
    ana = float(jnp.sum(grad * pert))
    assert abs(num - ana) <= 1e-4 * max(1.0, abs(ana))
