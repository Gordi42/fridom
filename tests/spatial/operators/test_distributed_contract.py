"""The fused distributed channel-eigenbasis contraction (kernel + resolve).

The contraction ``Q diag(w) Q^H M z`` over the periodic Fourier planes
runs inside one ``jax.shard_map`` region so every FFT axis is
device-local when its transform runs (the sharded-transform-axis FFT
otherwise hits an upstream XLA:GPU distributed-FFT fault; see
``design/research/multidevice_test_faults.md``). These tests build the
plan on synthetic ``q`` / weights / metric arrays (no eigensolve, so
they are safe on the forced-4 CPU leg) and compare against a replicated
single-device reference (plain ``rfft`` / ``fft`` + the same einsum).
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
    ContractPlan,
    build_distributed_contraction,
    resolve_distributed_contraction,
)

COMPONENTS = ("c0", "c1")
# the sharded axis needs >= 3 true cells on its shortest shard over four
# devices; nx=16 clears it, the small partner nz stays local.
NX, NY = 16, 6


def make_channel_grid(nx, ny, nz, device_ids=None):
    """Return a walled-y channel grid (x, z periodic; y bounded)."""
    meshes = (
        IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(nz, (0.0, 2 * np.pi), periodic=True, name="z"))
    return Grid(meshes, device_ids=device_ids)


def make_fields(grid, seed):
    """Two random real center fields (both nx, ny, nz)."""
    rng = np.random.default_rng(seed)
    fields = {}
    for name in COMPONENTS:
        shape = grid.create_field(name=name).data.shape
        fields[name] = grid.create_field(
            name=name, data=jnp.asarray(rng.standard_normal(shape)))
    return fields


def synthetic_basis(nx, ny, nz, seed):
    """Synthetic (q, weights, metric, slices) on the rfftn plane frame."""
    rng = np.random.default_rng(seed)
    dim = 2 * ny
    nmodes = nz // 2 + 1
    slices = {"c0": slice(0, ny), "c1": slice(ny, 2 * ny)}

    def cplx(shape):
        return jnp.asarray(rng.standard_normal(shape)
                           + 1j * rng.standard_normal(shape))

    q = cplx((nx, nmodes, dim, dim))
    weights = cplx((nx, nmodes, dim))
    metric = jnp.asarray(rng.uniform(0.5, 1.5, dim))
    return q, weights, metric, slices


def reference(fields, q, weights, metric, slices, grid,
              bounded_axis="y", periodic_axis="z"):
    """Replicated single-device ``Q diag(w) Q^H M z`` reference.

    The engine frame: ``rfft`` on the last periodic axis
    (``periodic_axis``, the half axis), full ``fft`` on the other
    periodic axis; the bounded axis stays nodal and is stacked into the
    column axis. Axis-generic in the bounded position (walled x / y / z).
    """
    names = grid.names
    other = next(n for n in names
                 if n not in (bounded_axis, periodic_axis))
    a_arr = names.index(other)
    b_arr = names.index(periodic_axis)
    bounded_arr = names.index(bounded_axis)
    half_n = next(m for m in grid.factors
                  if periodic_axis in m.names).n_cells
    coeff = {}
    for name, field in fields.items():
        c = jnp.asarray(np.asarray(field.data))
        c = jnp.fft.rfft(c, axis=b_arr, norm="forward")
        c = jnp.fft.fft(c, axis=a_arr, norm="forward")
        coeff[name] = c
    z = jnp.concatenate(
        [jnp.moveaxis(coeff[name], bounded_arr, -1)
         for name in COMPONENTS], axis=-1)
    amp = jnp.einsum("...dj,d,...d->...j", jnp.conj(q), metric, z)
    out = jnp.einsum("...dj,...j->...d", q, weights * amp)
    result = {}
    for name in COMPONENTS:
        seg = jnp.moveaxis(out[..., slices[name]], -1, bounded_arr)
        seg = jnp.fft.ifft(seg, axis=a_arr, norm="forward")
        seg = jnp.fft.irfft(seg, n=half_n, axis=b_arr, norm="forward")
        result[name] = np.asarray(seg.real)
    return result


def resolve(grid, slices):
    """Resolve the plan for the standard walled-y channel signature."""
    return resolve_distributed_contraction(
        grid, bounded_axis="y", periodic_axis="z",
        components=COMPONENTS, slices=slices)


def _ceil_mult(n, shards):
    return -(-n // shards) * shards


# ================================================================
#  Resolution and decline conditions
# ================================================================
def test_single_device_declines():
    grid = make_channel_grid(NX, NY, 8, device_ids=(0,))
    _q, _w, _m, slices = synthetic_basis(NX, NY, 8, seed=0)
    assert resolve(grid, slices) is None


@pytest.mark.multi_device
def test_two_dimensional_channel_declines(forced_devices):
    # a single periodic axis (x periodic, y bounded) has no transpose
    # partner: the fused lowering declines (the 2-D channel stays on the
    # taught error)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = Grid((
        IntervalMesh(NX, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")))
    assert grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    assert resolve_distributed_contraction(
        grid, bounded_axis="y", periodic_axis="x",
        components=COMPONENTS,
        slices={"c0": slice(0, 8), "c1": slice(8, 16)}) is None


@pytest.mark.multi_device
def test_half_axis_sharded_declines(forced_devices):
    # x (18) is indivisible so ranks below the divisible periodic z (16):
    # the default layout shards z, which is the engine's half (rfft)
    # axis -- unreachable under one reshard, so the plan declines
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = make_channel_grid(18, NY, 16)
    assert grid.decomposition.default_layout.device_axes == (
        ("z", "devices"),)
    slices = {"c0": slice(0, NY), "c1": slice(NY, 2 * NY)}
    assert resolve_distributed_contraction(
        grid, bounded_axis="y", periodic_axis="z",
        components=COMPONENTS, slices=slices) is None


@pytest.mark.multi_device
def test_bounded_axis_sharded_declines(forced_devices):
    # the periodic axes (2, 2) are too small to shard over four devices,
    # so the default layout shards the bounded axis y -- the existing
    # GSPMD path already handles that, so the fused lowering declines
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = make_channel_grid(2, 16, 2)
    assert grid.decomposition.default_layout.device_axes == (
        ("y", "devices"),)
    slices = {"c0": slice(0, 16), "c1": slice(16, 32)}
    assert resolve(grid, slices) is None


# ================================================================
#  The kernel matches the replicated reference (genuinely sharded)
# ================================================================
@pytest.mark.multi_device
@pytest.mark.parametrize(
    "nz", [7, 8], ids=["divisible-half", "padded-half"])
def test_contraction_matches_the_replicated_reference(
        nz, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    shards = jax.device_count()
    grid = make_channel_grid(NX, NY, nz)
    assert grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    fields = make_fields(grid, seed=1)
    q, weights, metric, slices = synthetic_basis(NX, NY, nz, seed=2)
    plan = resolve(grid, slices)
    assert plan is not None
    # the sharded axis (x=16) is divisible, so the field frame is the
    # true frame; the padded transpose engages only on the half axis
    assert plan.a_padded is False
    assert plan._pad_b == _ceil_mult(nz // 2 + 1, shards)

    out = plan.apply(fields, q, weights, metric)
    ref = reference(fields, q, weights, metric, slices, grid)
    for name in COMPONENTS:
        got = np.asarray(out[name].data)
        assert not np.iscomplexobj(got)
        assert np.allclose(got, ref[name], rtol=1e-11, atol=1e-12)


@pytest.mark.multi_device
def test_padded_case_has_an_empty_trailing_pad_shard(forced_devices):
    # nz=8 -> half extent 5 over four devices -> pad_b=8 (2 modes per
    # shard); the trailing shard holds only pad lanes (modes 6, 7 >= 5),
    # and the reference match above proves the zero-padded q / w keep
    # them inert
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = make_channel_grid(NX, NY, 8)
    _q, _w, _m, slices = synthetic_basis(NX, NY, 8, seed=0)
    plan = resolve(grid, slices)
    assert plan is not None
    assert plan._n_b == 5
    assert plan._pad_b == 8  # trailing shard (modes 6, 7) is all pad


@pytest.mark.multi_device
def test_indivisible_sharded_axis_uses_the_padded_even_frame(
        forced_devices):
    # x=18 is indivisible over four devices (the small partner z=8 does
    # not shard), so the sharded axis pads to the even frame: apply
    # routes through unpad_even / pad_even instead of the true frame, and
    # still matches the replicated reference
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    shards = jax.device_count()
    nx, ny, nz = 18, NY, 8
    grid = make_channel_grid(nx, ny, nz)
    assert grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    fields = make_fields(grid, seed=7)
    q, weights, metric, slices = synthetic_basis(nx, ny, nz, seed=8)
    plan = resolve(grid, slices)
    assert plan is not None
    assert plan.a_padded is True
    assert plan._pad_a == _ceil_mult(nx, shards)

    out = plan.apply(fields, q, weights, metric)
    ref = reference(fields, q, weights, metric, slices, grid)
    for name in COMPONENTS:
        got = np.asarray(out[name].data)
        assert not np.iscomplexobj(got)
        assert np.allclose(got, ref[name], rtol=1e-11, atol=1e-12)


@pytest.mark.multi_device
def test_walled_z_geometry_matches_the_reference(forced_devices):
    # hydrostatic-like: x, y periodic (y the half axis), z bounded -- the
    # half axis is the middle array axis (b_arr=1) and the bounded axis
    # is last (bounded_arr=2), exercising the axis-generic handling
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny, nz = 16, 8, 6
    grid = Grid((
        IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, 2 * np.pi), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z")))
    assert grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    fields = make_fields(grid, seed=3)
    rng = np.random.default_rng(4)
    dim, nmodes = 2 * nz, ny // 2 + 1

    def cplx(shape):
        return jnp.asarray(rng.standard_normal(shape)
                           + 1j * rng.standard_normal(shape))

    q = cplx((nx, nmodes, dim, dim))
    weights = cplx((nx, nmodes, dim))
    metric = jnp.asarray(rng.uniform(0.5, 1.5, dim))
    slices = {"c0": slice(0, nz), "c1": slice(nz, 2 * nz)}
    plan = resolve_distributed_contraction(
        grid, bounded_axis="z", periodic_axis="y",
        components=COMPONENTS, slices=slices)
    assert plan is not None
    # the half axis (y=8) pads on its coefficient extent (5 -> 8)
    assert plan._pad_b == _ceil_mult(nmodes, jax.device_count())

    out = plan.apply(fields, q, weights, metric)
    ref = reference(fields, q, weights, metric, slices, grid,
                    bounded_axis="z", periodic_axis="y")
    for name in COMPONENTS:
        got = np.asarray(out[name].data)
        assert not np.iscomplexobj(got)
        assert np.allclose(got, ref[name], rtol=1e-11, atol=1e-12)


# ================================================================
#  HLO collective profile, memoization, warm compiles
# ================================================================
@pytest.mark.multi_device
def test_hlo_transposes_without_gathers(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = make_channel_grid(NX, NY, 8)
    fields = make_fields(grid, seed=1)
    q, weights, metric, slices = synthetic_basis(NX, NY, 8, seed=2)
    plan = resolve(grid, slices)
    assert plan is not None
    qf = plan._pad_modes(q)
    wf = plan._pad_modes(weights.astype(qf.dtype))
    pieces = {name: jnp.asarray(fields[name].data)
              for name in COMPONENTS}
    text = plan._region.lower(pieces, qf, wf, metric).compile().as_text()
    # the reshard is an all_to_all; the local FFTs and the plane einsum
    # (no reduction over the sharded mode axis) add no cube gather
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


@pytest.mark.multi_device
def test_plan_is_memoized(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = make_channel_grid(NX, NY, 8)
    _q, _w, _m, slices = synthetic_basis(NX, NY, 8, seed=0)
    plan = resolve(grid, slices)
    assert plan is not None
    assert resolve(grid, slices) is plan
    assert isinstance(plan, ContractPlan)


@pytest.mark.multi_device
def test_warm_contraction_adds_zero_compiles(
        compile_counter, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = make_channel_grid(NX, NY, 8)
    fields = make_fields(grid, seed=1)
    q, weights, metric, slices = synthetic_basis(NX, NY, 8, seed=2)
    plan = resolve(grid, slices)

    def apply_once():
        out = plan.apply(fields, q, weights, metric)
        return [np.asarray(out[name].data) for name in COMPONENTS]

    apply_once()
    apply_once()
    compile_counter.reset()
    apply_once()
    assert compile_counter.count == 0


def test_build_declines_on_a_non_1d_mesh():
    # a genuine 2-D device mesh (pencil) is not the 1-D-slab geometry
    class _PencilMesh:
        axis_names = ("rows", "cols")

    class _PencilDecomp:
        device_count = 4
        device_mesh = _PencilMesh()

    class _PencilGrid:
        decomposition = _PencilDecomp()
        names = ("x", "y", "z")

    assert build_distributed_contraction(
        _PencilGrid(), bounded_axis="y", periodic_axis="z",
        components=COMPONENTS,
        slices={"c0": slice(0, 4), "c1": slice(4, 8)}) is None


# ================================================================
#  Autodiff finiteness through the shard_map / all_to_all VJP
# ================================================================
@pytest.mark.multi_device
def test_grad_is_finite_through_the_region(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = make_channel_grid(NX, NY, 8)
    fields = make_fields(grid, seed=1)
    q, weights, metric, slices = synthetic_basis(NX, NY, 8, seed=2)
    plan = resolve(grid, slices)
    qf = plan._pad_modes(q)
    wf = plan._pad_modes(weights.astype(qf.dtype))
    pieces = {name: jnp.asarray(fields[name].data)
              for name in COMPONENTS}

    def loss(cc):
        out = plan._region(cc, qf, wf, metric)
        return sum(jnp.sum(o ** 2) for o in out.values())

    grads = jax.grad(loss)(pieces)
    for name in COMPONENTS:
        assert bool(jnp.all(jnp.isfinite(grads[name])))
        assert float(jnp.linalg.norm(grads[name])) > 0.0
