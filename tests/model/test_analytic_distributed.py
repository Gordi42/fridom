"""The analytic-eigenmode distributed router (``resolve_route``).

The analytic sibling of the numeric channel's
``resolve_distributed_contraction``: resolves a per-component
``DistributedTransform`` sharing one transpose geometry for the fully
periodic (plain-Fourier) analytic eigenmode consumers, and declines
(``None`` -- the caller keeps its bit-identical path or taught error) on
a single device, a walled-vertical ``ComposedTransform`` (Wave B), or a
replicated operand.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.analytic_distributed import (
    AnalyticDistributedRoute,
    analytic_route,
    hermitian_reframe,
    resolve_route,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth


def _eigenmodes(device_ids, *, walled=None, n=8):
    """Analytic nonhydro eigenmodes at the given device layout."""
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            n, (0.0, 2 * np.pi), periodic=(name != walled), name=name)
        for name in ("x", "y", "z"))
    model = nh.Model(
        grid=fr.spatial.Grid(meshes, device_ids=device_ids),
        core=nh.Core(aspect_ratio=1.0),
        time_stepper=AdamBashforth(5e-3, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=3.0),
        advection=False)
    return nh.eigenmodes.from_model(model)


def test_resolve_route_declines_a_single_device_grid():
    em = _eigenmodes((0,))
    assert resolve_route(em.grid, em._analysis, em._components) is None


@pytest.mark.multi_device
def test_resolve_route_serves_a_periodic_sharded_grid(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    em = _eigenmodes(None)
    route = resolve_route(em.grid, em._analysis, em._components)
    assert isinstance(route, AnalyticDistributedRoute)
    # the plain-Fourier route serves the fused backward-only synthesis
    # (the random-state / mode() consumers ride it via hermitian_reframe)
    assert route.can_synthesize is True
    # every prognostic + auxiliary analysis component resolves to the
    # same transpose geometry (staggered u/v/w and collocated b/p)
    for name in em._analysis:
        assert route.coeff_of(name) is not None


def test_resolve_route_declines_a_single_device_walled_grid():
    # single device: the walled route also declines (the eager
    # per-component round-trip stays bit-identical off the sharded path)
    em = _eigenmodes((0,), walled="z")
    assert resolve_route(em.grid, em._analysis, em._components) is None


@pytest.mark.multi_device
def test_resolve_route_serves_a_walled_vertical_grid(forced_devices):
    # Wave B: the walled-vertical mixed ComposedTransform
    # (Fourier x Fourier x trig) now resolves through the fused
    # WalledVerticalTransform region -- the two periodic axes ride the
    # transpose pipeline and the bounded trig axis rides local inside the
    # column. The router serves it instead of keeping the Tier-1 taught
    # error.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    em = _eigenmodes(None, walled="z")
    route = resolve_route(em.grid, em._analysis, em._components)
    assert isinstance(route, AnalyticDistributedRoute)
    # the walled route serves apply_matrix (projections / f(L) / balance)
    # but NOT the fused backward-only synthesis: its internal frame runs
    # both periodic axes fully complex, so it never coincides with the
    # single-device random-phase frame -- synthesis-only consumers keep
    # the replicated (device-invariant) backward (synthesize_columns
    # reads can_synthesize to fall back instead of crashing).
    assert route.can_synthesize is False
    # every prognostic + auxiliary component resolves to an internal
    # coefficient frame (the trig z factor on its own lattice)
    for name in em._analysis:
        assert route.coeff_of(name) is not None
    # the periodic router declines the walled grid (the ComposedTransform
    # has no plain-Fourier DistributedTransform)
    from fridom.model.analytic_distributed import (  # noqa: PLC0415
        _resolve_periodic_route,
    )
    assert _resolve_periodic_route(
        em.grid, em._analysis, em._components) is None


@pytest.mark.multi_device
def test_analytic_route_reroutes_a_sharded_operand(forced_devices):
    # a route resolves on the sharded grid and a genuinely sharded
    # operand (the default nodal layout shards a transform axis) is
    # rerouted; a single-device operand keeps the plain path (None)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    em = _eigenmodes(None)
    route = resolve_route(em.grid, em._analysis, em._components)
    assert route is not None
    rng = np.random.default_rng(1)
    sharded = em.grid.create_field(
        em._analysis["u"], data=rng.standard_normal((8, 8, 8)))
    assert sharded._data.sharding.spec[0] == "devices"
    assert route.shards(sharded) is True
    assert analytic_route(em, nh.State({"u": sharded})) is not None
    one = _eigenmodes((0,))
    assert analytic_route(one, nh.State({"u": one.q(0)["u"]})) is None


@pytest.mark.multi_device
def test_walled_route_apply_matrix_is_device_count_invariant(
        forced_devices):
    # the walled route's fused apply_matrix (Fourier transpose + local
    # trig + union-lattice per-mode matrix) reproduces the eager
    # single-device projector to floating point and lands real
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    components = ("u", "v", "w", "b")
    n = 8
    rng = np.random.default_rng(2)
    fields = {c: rng.standard_normal((8, 8, 7 if c == "w" else 8))
              for c in components}
    many = _eigenmodes(None, walled="z", n=n)
    one = _eigenmodes((0,), walled="z", n=n)

    def state(em):
        return nh.State({
            c: em.grid.create_field(
                em.physical_space(c), data=fields[c])
            for c in components})

    z_many, z_one = state(many), state(one)
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    route = resolve_route(many.grid, many._analysis, many._components)
    matrix = many.operator_matrix(route.coeff_of, branches=(0,))
    out_many = route.apply_matrix(
        {c: z_many[c] for c in components}, matrix)
    # the one-device reference: the eager per-component projector round
    # trip (the plain path the single-device grid keeps)
    ref = nh.transforms.VorticalProjection(one)(z_one)
    for c in components:
        got = np.asarray(out_many[c].data)
        assert not np.iscomplexobj(got)
        assert np.abs(got - np.asarray(ref[c].data)).max() < 1e-12


# ================================================================
#  Hermitian half-axis re-expression (hermitian_reframe)
# ================================================================
def _mirror(plane):
    """Conjugate reflection ``k -> -k`` of a 2-D full-spectrum plane."""
    return np.conj(np.roll(np.flip(np.flip(plane, 0), 1), (1, 1), (0, 1)))


@pytest.mark.parametrize("nx", [8, 9], ids=["even", "odd"])
def test_hermitian_reframe_completes_the_full_spectrum(nx):
    # the re-expression of a single-device half-axis (rfft) column onto a
    # full-complex target frame reproduces the numpy full DFT of the real
    # field exactly: interior half-axis planes are kept, the missing half
    # is the conjugate reflection, and the self-conjugate DC plane (with,
    # at even extents, the Nyquist plane) is averaged to its Hermitian part
    # -- so the field the target frame synthesizes is real. Covers both the
    # even (DC + Nyquist) and odd (DC only, no Nyquist) source parity.
    em = _eigenmodes((0,), n=nx)
    source = em.kit.coeff("b").bare
    target = source.as_complex()
    rng = np.random.default_rng(0)
    field = rng.standard_normal((nx, nx, nx))
    full = np.fft.fftn(field, norm="forward")
    half = jnp.asarray(full[: nx // 2 + 1])
    out = np.asarray(hermitian_reframe(half, source, target))
    assert out.shape == (nx, nx, nx)
    assert np.abs(out - full).max() < 1e-12


def test_hermitian_reframe_projects_self_conjugate_planes():
    # a column that is *not* Hermitian on the DC / Nyquist planes of the
    # half axis (as the analytic gains generally are not): the single-device
    # backward (irfft) keeps only the Hermitian part of those planes, and
    # the re-expression reproduces that projection (the (stored + reflected
    # conjugate) / 2 averaging), so the completed spectrum has Hermitian DC
    # and Nyquist planes -- while interior planes are kept verbatim and the
    # missing half is their conjugate reflection.
    nx = 8
    em = _eigenmodes((0,), n=nx)
    source = em.kit.coeff("b").bare
    target = source.as_complex()
    rng = np.random.default_rng(1)
    half = (rng.standard_normal((nx // 2 + 1, nx, nx))
            + 1j * rng.standard_normal((nx // 2 + 1, nx, nx)))
    out = np.asarray(hermitian_reframe(jnp.asarray(half), source, target))
    # DC (kx = 0) and Nyquist (kx = 4) planes: Hermitian after averaging
    for plane in (0, nx // 2):
        assert np.abs(out[plane] - _mirror(out[plane])).max() < 1e-12
    # interior plane kept verbatim; its reflection fills the missing half
    assert np.abs(out[1] - half[1]).max() < 1e-12
    assert np.abs(out[nx - 1] - _mirror(half[1])).max() < 1e-12


@pytest.mark.multi_device
def test_reframed_synthesis_has_no_all_gather(forced_devices):
    # the fused IC path -- half-axis re-expression onto the re-designated
    # internal frame, then the fused shard_map backward -- keeps every
    # transform axis device-local: the HLO transposes (all-to-all) and
    # never gathers the sharded axis (all-gather / all-reduce absent).
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    em = _eigenmodes(None)
    kit = em.kit
    route = resolve_route(em.grid, em._analysis, em._components)
    # the sharded axis is the single-device half axis, so the internal
    # frame re-designates it -- the frames differ and reframe is exercised
    assert any(route.coeff_of(c) != kit.coeff(c) for c in em._components)
    templates = {
        c: em.grid.create_field(kit.backward(c).codomain, name=c)
        for c in em._components}

    def synth(cols):
        coeffs = {
            c: hermitian_reframe(cols[c], kit.coeff(c), route.coeff_of(c))
            for c in em._components}
        return {c: v.data
                for c, v in route.synthesize(coeffs, templates).items()}

    rng = np.random.default_rng(2)
    cols = {
        c: jnp.asarray(rng.standard_normal(kit.coeff(c).shape)
                       + 1j * rng.standard_normal(kit.coeff(c).shape))
        for c in em._components}
    text = jax.jit(synth).lower(cols).compile().as_text()
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text
