"""Multi-device channel-eigenbasis projections through the dispatch.

The 3-D channel shards a full periodic axis, so the per-plane Fourier
contraction (``model._eigenbasis._contract_planes``) routes through the
fused ``jax.shard_map`` lowering
(``spatial.operators.distributed_contract``) instead of the plain GSPMD
transform, which hits the upstream XLA:GPU distributed-FFT fault (see
``design/research/multidevice_test_faults.md``).

Two tiers:

- **stub-``em`` unit tests** drive ``_project_masked`` / ``_apply_weighted``
  / ``_contract_planes`` on a *bare* sharded grid with a synthetic
  eigenvector basis (no eigensolve), so they are safe on the forced-4
  CPU leg (the T5b batched-eigh CPU-LAPACK heap corruption on many-core
  hosts never runs). They cover the ``_eigenbasis`` dispatch: the fused
  many-device path matches the replicated one-device path, and the 2-D
  (single-periodic) channel keeps the narrowed taught error.
- **real-eigenbasis end-to-end tests** build a genuine nh channel
  eigenbasis and are GPU-scoped (the fixture skips on the CPU backend):
  the projector / ``function`` application matches the explicit
  one-device reference, is idempotent, lands real, and differentiates
  finitely.

``multi_device`` marked tests need the forced-4-device suite
(``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``) or a genuine multi-GPU node.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.model._eigenbasis import (
    _apply_weighted,
    _contract_planes,
    _project_masked,
)
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

STUB_COMPONENTS = ("u", "v")


# ================================================================
#  Stub-em unit tests (no eigensolve; forced-4 CPU safe)
# ================================================================
def make_bare_channel(nx, ny, nz, device_ids=None):
    """Return a walled-y bare channel grid (shards x, nx>=3 cells)."""
    meshes = (
        IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(nz, (0.0, 2 * np.pi), periodic=True, name="z"))
    return Grid(meshes, device_ids=device_ids)


def stub_em(grid, q, metric, slices, periodic_axis="z"):
    """Return a duck-typed eigenmodes object for the contraction."""
    return SimpleNamespace(
        grid=grid, bounded_axis="y", periodic_axis=periodic_axis,
        components=STUB_COMPONENTS, slices=slices, q=q, metric=metric)


def make_state(grid, data):
    """Return a real (u, v) center-field state carrying ``data``."""
    return VectorField({
        name: grid.create_field(name=name, data=jnp.asarray(data[name]))
        for name in STUB_COMPONENTS})


def synthetic_basis(nx, ny, nz, seed):
    """Synthetic (q, metric, slices) on the rfftn plane frame."""
    rng = np.random.default_rng(seed)
    dim = 2 * ny
    nmodes = nz // 2 + 1
    q = jnp.asarray(rng.standard_normal((nx, nmodes, dim, dim))
                    + 1j * rng.standard_normal((nx, nmodes, dim, dim)))
    metric = jnp.asarray(rng.uniform(0.5, 1.5, dim))
    slices = {"u": slice(0, ny), "v": slice(ny, 2 * ny)}
    return q, metric, slices


def absmax(a, b, components):
    """Max component-wise absolute difference of two states."""
    return max(
        float(np.abs(np.asarray(a[c].data)
                     - np.asarray(b[c].data)).max())
        for c in components)


@pytest.mark.multi_device
def test_project_masked_matches_one_device(forced_devices):
    # the fused many-device _project_masked reproduces the replicated
    # one-device path bit-for-bit (to floating point) and lands real
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny, nz = 12, 8, 12
    rng = np.random.default_rng(1)
    data = {name: rng.standard_normal((nx, ny, nz))
            for name in STUB_COMPONENTS}
    q, metric, slices = synthetic_basis(nx, ny, nz, seed=2)
    mask = jnp.asarray(rng.integers(
        0, 2, (nx, nz // 2 + 1, 2 * ny)).astype(bool))

    g_many = make_bare_channel(nx, ny, nz)
    assert g_many.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    out_many = _project_masked(
        stub_em(g_many, q, metric, slices), mask,
        make_state(g_many, data))
    g_one = make_bare_channel(nx, ny, nz, device_ids=(0,))
    out_one = _project_masked(
        stub_em(g_one, q, metric, slices), mask,
        make_state(g_one, data))
    for name in STUB_COMPONENTS:
        assert not np.iscomplexobj(np.asarray(out_many[name].data))
        assert absmax(out_many, out_one, STUB_COMPONENTS) <= 1e-11


@pytest.mark.multi_device
def test_apply_weighted_matches_one_device(forced_devices):
    # the complex-weight f(L) contraction (the function engine) runs
    # through the same fused path and matches one device
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny, nz = 12, 8, 12
    rng = np.random.default_rng(3)
    data = {name: rng.standard_normal((nx, ny, nz))
            for name in STUB_COMPONENTS}
    q, metric, slices = synthetic_basis(nx, ny, nz, seed=4)
    weights = jnp.asarray(
        rng.standard_normal((nx, nz // 2 + 1, 2 * ny))
        + 1j * rng.standard_normal((nx, nz // 2 + 1, 2 * ny)))

    g_many = make_bare_channel(nx, ny, nz)
    out_many = _apply_weighted(
        stub_em(g_many, q, metric, slices), weights,
        make_state(g_many, data))
    g_one = make_bare_channel(nx, ny, nz, device_ids=(0,))
    out_one = _apply_weighted(
        stub_em(g_one, q, metric, slices), weights,
        make_state(g_one, data))
    assert absmax(out_many, out_one, STUB_COMPONENTS) <= 1e-11


@pytest.mark.multi_device
def test_grad_is_finite_through_contract_planes(forced_devices):
    # the shard_map / all_to_all VJP of the fused contraction is finite
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny, nz = 12, 8, 12
    rng = np.random.default_rng(9)
    data = {name: rng.standard_normal((nx, ny, nz))
            for name in STUB_COMPONENTS}
    q, metric, slices = synthetic_basis(nx, ny, nz, seed=10)
    weights = jnp.asarray(rng.standard_normal((nx, nz // 2 + 1, 2 * ny)))
    grid = make_bare_channel(nx, ny, nz)
    em = stub_em(grid, q, metric, slices)
    u0 = jnp.asarray(data["u"])

    def loss(u):
        state = make_state(grid, {"u": u, "v": data["v"]})
        out = _contract_planes(em, state, lambda amp: weights * amp,
                               weights)
        return sum(jnp.sum(out[c].data ** 2) for c in STUB_COMPONENTS)

    grad = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


@pytest.mark.multi_device
def test_single_periodic_axis_keeps_the_taught_error(forced_devices):
    # a 2-D channel (x periodic, y walled) shards its ONLY periodic axis:
    # the fused lowering has no transpose partner and declines, so the
    # narrowed taught error fires
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = Grid((
        IntervalMesh(16, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")))
    assert grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    q = jnp.zeros((16 // 2 + 1, 16, 16), dtype=complex)
    metric = jnp.ones(16)
    slices = {"u": slice(0, 8), "v": slice(8, 16)}
    em = stub_em(grid, q, metric, slices, periodic_axis="x")
    data = {name: np.zeros((16, 8)) for name in STUB_COMPONENTS}
    mask = jnp.zeros((16 // 2 + 1, 16), dtype=bool)
    with pytest.raises(NotImplementedError, match="shards a periodic axis"):
        _project_masked(em, mask, make_state(grid, data))


# ================================================================
#  Real-eigenbasis end-to-end tests (GPU-scoped: fixture skips CPU)
# ================================================================
NH_COMPONENTS = ("u", "v", "w", "b")


def make_nh_channel(device_ids=None):
    """Return a walled-y nh channel (x=12 shards, z=12, y=8)."""
    meshes = tuple(
        IntervalMesh(n, (0.0, 1.0 if name == "y" else 2 * np.pi),
                     periodic=(name != "y"), name=name)
        for name, n in (("x", 12), ("y", 8), ("z", 12)))
    return nh.Model(
        grid=Grid(meshes, device_ids=device_ids),
        advection=False, dsqr=2.0,
        coriolis=nh.FPlaneCoriolis(f0=1.5),
        stratification=nh.ConstantStratification(n2=3.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def make_nh_channel_last_axis(device_ids=None):
    """Walled-y nh channel whose layout shards the LAST periodic axis.

    x is periodic but its cell count (10) is indivisible by 4 (rank 2),
    so it stays LOCAL; z is periodic and divisible (rank 0), so it is
    the default sharded axis -- the engine's default half (rfft) axis.
    The engine re-designates the half axis to the local x, so the
    shipped fused contraction serves the layout (a = z sharded / full,
    b = x local / rfft half).
    """
    meshes = tuple(
        IntervalMesh(n, (0.0, 1.0 if name == "y" else 2 * np.pi),
                     periodic=(name != "y"), name=name)
        for name, n in (("x", 10), ("y", 8), ("z", 12)))
    return nh.Model(
        grid=Grid(meshes, device_ids=device_ids),
        advection=False, dsqr=2.0,
        coriolis=nh.FPlaneCoriolis(f0=1.5),
        stratification=nh.ConstantStratification(n2=3.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def nh_state(model, fields):
    """Write the shared random fields onto the model's nh state."""
    model.set_fields(**fields)
    return nh.State({c: model.state[c] for c in NH_COMPONENTS})


@pytest.fixture(scope="module")
def nh_pair():
    """Many/one nh channels + eigenbases + shared fields (GPU only).

    Building the channel eigenbasis runs a batched ``eigh`` that
    heap-corrupts jaxlib's CPU LAPACK on many-core hosts (T5b), so this
    real-eigenbasis fixture is GPU-scoped and skips on the CPU backend.
    """
    if jax.default_backend() == "cpu":
        pytest.skip(
            "real channel eigenbasis build runs a batched eigh that "
            "heap-corrupts jaxlib's CPU LAPACK on many-core hosts "
            "(T5b); the real-eigenbasis path is GPU-scoped")
    many = make_nh_channel()
    one = make_nh_channel(device_ids=(0,))
    rng = np.random.default_rng(7)
    fields = {
        c: rng.standard_normal(np.asarray(many.state[c].data).shape)
        for c in NH_COMPONENTS}
    return (many, nh.eigenbasis(many), one, nh.eigenbasis(one), fields)


@pytest.mark.multi_device
def test_real_projection_matches_one_device(nh_pair, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many_m, many_eb, one_m, one_eb, fields = nh_pair
    z_many = nh_state(many_m, fields)
    z_one = nh_state(one_m, fields)
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    for sel in ("vortical", "wave", "kelvin"):
        pm = many_eb.projector(sel)(z_many)
        po = one_eb.projector(sel)(z_one)
        assert not any(
            np.iscomplexobj(np.asarray(pm[c].data))
            for c in NH_COMPONENTS)
        assert absmax(pm, po, NH_COMPONENTS) <= 1e-11


@pytest.mark.multi_device
def test_real_projection_is_idempotent(nh_pair, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many_m, many_eb, _one_m, _one_eb, fields = nh_pair
    proj = many_eb.projector("vortical")
    out = proj(nh_state(many_m, fields))
    assert absmax(proj(out), out, NH_COMPONENTS) <= 1e-11


@pytest.mark.multi_device
def test_real_function_matches_one_device(nh_pair, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many_m, many_eb, one_m, one_eb, fields = nh_pair

    def inverse_l(om):
        return -1.0 / (1j * om)

    fm = many_eb.function(inverse_l, "wave")(nh_state(many_m, fields))
    fo = one_eb.function(inverse_l, "wave")(nh_state(one_m, fields))
    assert not any(
        np.iscomplexobj(np.asarray(fm[c].data)) for c in NH_COMPONENTS)
    assert absmax(fm, fo, NH_COMPONENTS) <= 1e-11


@pytest.mark.multi_device
def test_real_grad_is_finite_through_the_projection(
        nh_pair, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many_m, many_eb, _one_m, _one_eb, fields = nh_pair
    z0 = nh_state(many_m, fields)
    proj = many_eb.projector("vortical")
    u0 = jnp.asarray(z0["u"].data)

    def loss(u):
        z = z0.replace(u=z0["u"].with_data(u))
        out = proj(z)
        return sum(jnp.sum(out[c].data ** 2) for c in NH_COMPONENTS)

    grad = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


@pytest.mark.multi_device
def test_channel_reprojects_when_last_periodic_axis_is_sharded(
        forced_devices):
    # the remainder case: the default layout shards the LAST periodic
    # axis z (the engine's default half/rfft axis), which the fused
    # contraction cannot serve in the fixed frame. The engine
    # re-designates the half axis to the local x, so the shipped kernel
    # serves the layout with roles swapped (a = z, b = x) -- instead of
    # the taught NotImplementedError. GPU-scoped (real eigenbasis eigh).
    if jax.default_backend() == "cpu":
        pytest.skip(
            "real channel eigenbasis build runs a batched eigh that "
            "heap-corrupts jaxlib's CPU LAPACK on many-core hosts "
            "(T5b); the real-eigenbasis path is GPU-scoped")
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many = make_nh_channel_last_axis()
    assert many.grid.decomposition.default_layout.device_axes == (
        ("z", "devices"),)
    eb_many = nh.eigenbasis(many)
    # the half axis moved off the sharded last periodic axis
    assert eb_many.periodic_axis == "x"
    assert bool((np.asarray(eb_many.labels) != -1).all())
    rng = np.random.default_rng(31)
    fields = {c: rng.standard_normal(np.asarray(many.state[c].data).shape)
              for c in NH_COMPONENTS}
    z_many = nh_state(many, fields)
    assert z_many["u"]._data.sharding.spec[2] == "devices"
    proj = eb_many.projector("vortical")
    out_many = proj(z_many)
    assert not any(
        np.iscomplexobj(np.asarray(out_many[c].data))
        for c in NH_COMPONENTS)
    assert absmax(proj(out_many), out_many, NH_COMPONENTS) <= 1e-11
    # the replicated one-device reference (default half axis z)
    one = make_nh_channel_last_axis(device_ids=(0,))
    eb_one = nh.eigenbasis(one)
    assert eb_one.periodic_axis == "z"
    out_one = eb_one.projector("vortical")(nh_state(one, fields))
    assert absmax(out_many, out_one, NH_COMPONENTS) <= 1e-11


@pytest.mark.multi_device
def test_real_two_dimensional_channel_keeps_the_taught_error(
        forced_devices):
    # a real sw 2-D channel (single periodic axis) keeps the taught
    # error on the sharded grid; GPU-scoped (batch-9 eigh, but the setup
    # still runs the batched eigh path guarded above for the nh case)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    if jax.default_backend() == "cpu":
        pytest.skip(
            "real sw channel eigenbasis build is GPU-scoped (T5b)")
    model = sw.Model(
        grid=Grid((
            IntervalMesh(16, (0.0, 2 * np.pi), periodic=True, name="x"),
            IntervalMesh(8, (0.0, 1.0), periodic=False, name="y"))),
        csqr=0.7, rossby_number=0.2, advection=False,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    assert model.grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    eb = sw.eigenbasis(model)
    rng = np.random.default_rng(3)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in eb.components})
    z = sw.State({c: model.state[c] for c in eb.components})
    with pytest.raises(NotImplementedError, match="shards a periodic axis"):
        eb.projector("wave")(z)
