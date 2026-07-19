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

import fridom as fr
from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.distributed_transform import (
    DistributedTransform,
    resolve_distributed_transform,
    resolve_walled_vertical_transform,
)
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.transform import axis_slice


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


# ================================================================
#  The fused per-mode matrix apply / project / synthesize
# ================================================================
def matrix_reference(datas, matrix, geom):
    """Replicated ``backward(matrix @ forward)`` in the internal frame."""
    half_axis = next(ax for ax, half, _ in geom.local_stages if half)
    full = tuple(ax for ax in range(datas[0].ndim) if ax != half_axis)
    fwd = np.stack([frame_reference(d, geom) for d in datas], axis=-1)
    out = np.einsum("...jd,...d->...j", np.asarray(matrix), fwd)
    res = []
    for j in range(out.shape[-1]):
        c = np.fft.ifftn(out[..., j], axes=full, norm="forward")
        c = np.fft.irfft(c, n=datas[0].shape[half_axis],
                         axis=half_axis, norm="forward")
        res.append(np.real(c))
    return res


@pytest.mark.multi_device
def test_apply_matrix_matches_replicated_reference(forced_devices):
    # the fused per-mode D x D matrix apply reproduces the replicated
    # single-controller reference to floating point and lands real
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(7)
    names = ("a", "b", "c")
    datas = {n: jnp.asarray(rng.standard_normal((12, 8, 12)))
             for n in names}
    fields = {n: grid.create_field(data=datas[n]) for n in names}
    dt = resolve_distributed_transform(
        Fourier(grid), grid, fields["a"].function_space.bare)
    shape = dt.coeff.bare.shape
    dim = len(names)
    matrix = jnp.asarray(
        rng.standard_normal((*shape, dim, dim))
        + 1j * rng.standard_normal((*shape, dim, dim)))
    out = dt.apply_matrix(fields, matrix)
    ref = matrix_reference([datas[n] for n in names], matrix, dt.geometry)
    for i, name in enumerate(names):
        got = np.asarray(out[name].data)
        assert not np.iscomplexobj(got)
        assert np.allclose(got, ref[i], rtol=1e-10, atol=1e-11)


@pytest.mark.multi_device
def test_apply_matrix_hlo_transposes_without_gathers(forced_devices):
    # the fused matrix region reshards with all-to-all only; the einsum
    # contracts the local component axis (no reduction over the sharded
    # mode axis), so no cube gather / reduce is emitted
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(8)
    names = ("a", "b", "c")
    datas = {n: jnp.asarray(rng.standard_normal((12, 8, 12)))
             for n in names}
    fields = {n: grid.create_field(data=datas[n]) for n in names}
    dt = resolve_distributed_transform(
        Fourier(grid), grid, fields["a"].function_space.bare)
    shape = dt.coeff.bare.shape
    matrix = jnp.asarray(np.zeros((*shape, len(names), len(names)),
                                  dtype=complex))
    region = dt._matrix_region(names)
    pieces = {n: jnp.asarray(datas[n]) for n in names}
    mat = dt._pad_a(matrix)
    text = region.lower(pieces, mat).compile().as_text()
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


@pytest.mark.multi_device
def test_synthesize_round_trips_the_forward(forced_devices):
    # synthesize(forward(f)) == f: the backward-only half inverts the
    # forward-only region on the same internal frame
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(9)
    data = jnp.asarray(rng.standard_normal((12, 8, 12)))
    f = grid.create_field(data=data)
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f.function_space.bare)
    geom = dt.geometry
    coeff = dt.forward_region(jnp.asarray(f.data))
    if geom.pad_a_spec != geom.a_spec_n:
        coeff = axis_slice(coeff, geom.a, 0, geom.a_spec_n)
    out = dt.synthesize({"f": coeff}, {"f": f})
    assert np.allclose(np.asarray(out["f"].data), np.asarray(data),
                       rtol=1e-11, atol=1e-12)


@pytest.mark.multi_device
def test_apply_matrix_grad_is_finite(forced_devices):
    # jax.grad of a quadratic loss through the fused matrix apply is
    # finite and matches a central finite difference (the matrix is a
    # constant of the loss variable; the all_to_all VJP stays finite)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((8, 8, 8))
    rng = np.random.default_rng(10)
    names = ("a", "b")
    base = {n: jnp.asarray(rng.standard_normal((8, 8, 8))) for n in names}
    f0 = grid.create_field(data=base["a"])
    dt = resolve_distributed_transform(
        Fourier(grid), grid, f0.function_space.bare)
    shape = dt.coeff.bare.shape
    matrix = jnp.asarray(
        rng.standard_normal((*shape, 2, 2))
        + 1j * rng.standard_normal((*shape, 2, 2)))

    def loss(arr):
        fields = {"a": f0.with_data(arr),
                  "b": f0.with_data(base["b"])}
        out = dt.apply_matrix(fields, matrix)
        return sum(jnp.sum(out[n].data ** 2) for n in names)

    grad = jax.grad(loss)(base["a"])
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
    eps = 1e-4
    pert = jnp.asarray(rng.standard_normal((8, 8, 8)))
    num = (loss(base["a"] + eps * pert)
           - loss(base["a"] - eps * pert)) / (2 * eps)
    ana = float(jnp.sum(grad * pert))
    assert abs(num - ana) <= 1e-4 * max(1.0, abs(ana))


@pytest.mark.multi_device
def test_project_matches_the_forward_contraction(forced_devices):
    # the forward (analysis + contraction) half: project returns the
    # modal amplitudes rows @ forward(components) on the internal frame,
    # matching the replicated reference (a full synthesize inverts them)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((12, 8, 12))
    rng = np.random.default_rng(11)
    names = ("a", "b")
    datas = {n: jnp.asarray(rng.standard_normal((12, 8, 12)))
             for n in names}
    fields = {n: grid.create_field(data=datas[n]) for n in names}
    dt = resolve_distributed_transform(
        Fourier(grid), grid, fields["a"].function_space.bare)
    shape = dt.coeff.bare.shape
    rows = jnp.asarray(
        rng.standard_normal((*shape, 3, len(names)))
        + 1j * rng.standard_normal((*shape, 3, len(names))))
    amp = dt.project(fields, rows)
    geom = dt.geometry
    # the amplitudes live on the padded internal frame (a sharded)
    assert amp.shape[-1] == 3
    # replicated reference: rows @ stacked forward, sliced back on a
    fwd = np.stack([frame_reference(datas[n], geom) for n in names],
                   axis=-1)
    ref = np.einsum("...jd,...d->...j", np.asarray(rows), fwd)
    got = np.asarray(amp)
    if geom.pad_a_spec != geom.a_spec_n:
        got = np.take(got, range(geom.a_spec_n), axis=geom.a)
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-11)


# ================================================================
#  The walled-vertical fused region (Fourier transpose + local trig)
# ================================================================
def walled_grid(shape=(8, 8, 8), device_ids=None):
    """Return a periodic-x/y, walled-z grid of the given cell counts."""
    nx, ny, nz = shape
    return Grid((
        IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, 2 * np.pi), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z")),
        device_ids=device_ids)


def walled_spaces(grid):
    """Two mixed components with different trig z lattices (Cos/Sin)."""
    return {
        # DCT-II on the Neumann centred z (modes 0..n-1)
        "a": fr.spatial.Collocated(
            wall_bc={"z": BC.NEUMANN}).resolve(grid).bare,
        # DST-I on the Dirichlet inner z face (modes 1..n-1)
        "c": fr.spatial.Staggered(
            "z", wall_bc={"z": BC.DIRICHLET}).resolve(grid).bare,
    }


def walled_matrix_reference(transform, datas, matrix):
    """Replicated Fourier-transpose + local-trig + matrix (numpy)."""
    geom = transform._geom
    chart = transform._chart
    comps = transform._components
    coeff_of = transform._coeff_of
    a, b = geom.a, geom.b
    stacks = []
    for c in comps:
        d = np.asarray(datas[c]) + 0j
        d = np.fft.fft(d, axis=b, norm="forward")
        d = np.fft.fft(d, axis=a, norm="forward")
        d = np.asarray(transform._trig_forward[c](jnp.asarray(d)))
        stacks.append(np.asarray(chart.embed(jnp.asarray(d), coeff_of[c])))
    z = np.stack(stacks, axis=-1)
    out = np.einsum("...jd,...d->...j", np.asarray(matrix), z)
    res = {}
    for i, c in enumerate(comps):
        oc = np.asarray(chart.restrict(
            jnp.asarray(out[..., i]), coeff_of[c]))
        oc = np.asarray(transform._trig_backward[c](jnp.asarray(oc)))
        oc = np.fft.ifft(oc, axis=a, norm="forward")
        oc = np.fft.ifft(oc, axis=b, norm="forward")
        res[c] = np.real(oc)
    return res


@pytest.mark.multi_device
def test_walled_apply_matrix_matches_replicated_reference(forced_devices):
    # the walled-vertical fused region (two periodic axes on the
    # transpose pipeline, the bounded trig axis local + ModeChart) applies
    # a per-mode union-lattice matrix and reproduces the replicated
    # single-controller reference to floating point, landing real
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = walled_grid((8, 8, 8))
    spaces = walled_spaces(grid)
    transform = resolve_walled_vertical_transform(
        grid, spaces, ("a", "c"))
    assert transform is not None
    rng = np.random.default_rng(11)
    datas = {c: rng.standard_normal(spaces[c].shape) for c in spaces}
    fields = {c: grid.create_field(spaces[c], data=datas[c])
              for c in spaces}
    assert fields["a"]._data.sharding.spec[0] == "devices"
    union = transform._coeff_of["a"].shape  # (8, 8, 9)
    union = (*union[:2], grid.factors[-1].n_cells + 1)
    dim = len(spaces)
    matrix = jnp.asarray(
        rng.standard_normal((*union, dim, dim))
        + 1j * rng.standard_normal((*union, dim, dim)))
    out = transform.apply_matrix(fields, matrix)
    ref = walled_matrix_reference(transform, datas, matrix)
    for c in spaces:
        got = np.asarray(out[c].data)
        assert not np.iscomplexobj(got)
        assert np.allclose(got, ref[c], rtol=1e-10, atol=1e-11)


@pytest.mark.multi_device
def test_walled_apply_matrix_hlo_transposes_without_gathers(
        forced_devices):
    # the walled fused region reshards with all-to-all only: the two
    # periodic axes transpose, the local trig + the einsum over the
    # local component axis emit no cube gather / reduce
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = walled_grid((8, 8, 8))
    spaces = walled_spaces(grid)
    transform = resolve_walled_vertical_transform(
        grid, spaces, ("a", "c"))
    rng = np.random.default_rng(12)
    datas = {c: rng.standard_normal(spaces[c].shape) for c in spaces}
    pieces = {c: jnp.asarray(datas[c]) for c in spaces}
    union = (*transform._coeff_of["a"].shape[:2],
             grid.factors[-1].n_cells + 1)
    matrix = transform._pad_a(jnp.zeros((*union, 2, 2), dtype=complex))
    text = transform._region.lower(pieces, matrix).compile().as_text()
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


def test_resolve_walled_vertical_transform_declines_single_device():
    # single device: no reshard is needed, so the walled region declines
    grid = walled_grid((8, 8, 8), device_ids=(0,))
    spaces = walled_spaces(grid)
    assert resolve_walled_vertical_transform(
        grid, spaces, ("a", "c")) is None


@pytest.mark.multi_device
def test_resolve_walled_vertical_transform_declines_periodic(
        forced_devices):
    # a fully periodic grid has no trig (ComposedTransform) component,
    # so the walled builder declines (the periodic route serves it)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid = periodic_grid((8, 8, 8))
    spaces = {"a": grid.create_field(
        data=np.zeros((8, 8, 8))).function_space.bare}
    assert resolve_walled_vertical_transform(
        grid, spaces, ("a",)) is None
