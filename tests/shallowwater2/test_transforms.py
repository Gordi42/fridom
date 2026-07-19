"""Shallow-water eigenmode projections as StateTransforms (wave 7 C).

Validates the staggered ``sw.transforms`` projections and the shared
``EigenProjection`` / ``ProjectionFactory`` base: idempotency, the
``WaveProjection = P(+1) + P(-1)`` algebra identity, and partition of
unity ``Vortical + Wave + Divergence == Identity`` on the model's
staggered state. The discrete ``{vortical, +gravity, -gravity}``
basis is complete per wavenumber (three modes span the three
components; the patched ``k = 0`` inertial triple and the even-grid
interpolation-Nyquist steady strata included) — so
``DivergenceProjection`` is the structural zero map on any periodic
state (previously it captured a "Nyquist-vortical residual" on
even grids; that residual no longer exists).

The channel (engine) path: the same factories on a walled model
route to the labeled ``sw.ChannelEigenmodes`` families — agreement
with an in-test port of the reference SpectralProjection pipeline,
partition of unity of the named families (vortical + wave + kelvin),
idempotency, mutual annihilation, the energy partition, and the
sharded multi-device application.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.terms import term
from fridom.model.transforms.errors import SignatureMismatchError
from fridom.model.transforms.projection import EigenProjection

from .conftest import N, make_grid, make_model

COMPONENTS = ("u", "v", "p")


def _pinned_grid(*, periodic_y=True):
    # device_ids=(0,) twin of the conftest make_grid: the analytic
    # projections synthesize through the naive (GSPMD) transform, a
    # Tier-1 taught error on a sharded transform axis (see transform.py).
    # (The numeric 2-D-channel eigenbasis projection is served on a
    # sharded axis by Channel2DPlan; the analytic path is not.) Pinning
    # to one device tests the math at any device count (single-device
    # suite unchanged).
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=periodic_y, name="y")
    return fr.spatial.Grid((mx, my), device_ids=(0,))


def _eig(f0=1.0, csqr=1.0):
    model = make_model(_pinned_grid(), csqr=csqr, f0=f0)
    return sw.eigenmodes.from_model(model), model


def _state(model, *, seed=None):
    """Build a staggered ``(u, v, p)`` probe state on the model.

    Band-limited (Nyquist-free) trig fields by default so the
    three-mode basis is complete on the probe; ``seed`` switches to
    random data (which carries Nyquist-vortical content).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        shape = np.asarray(model.state["u"].data).shape
        model.set_fields(**{
            c: rng.standard_normal(shape) for c in COMPONENTS})
    else:
        x = (np.arange(N) + 0.5) / N
        gx, gy = np.meshgrid(x, x, indexing="ij")
        model.set_fields(
            u=np.sin(2 * np.pi * gx) * np.cos(2 * np.pi * gy),
            v=0.3 * np.cos(2 * np.pi * gx)
            - 0.2 * np.sin(2 * np.pi * gy),
            p=0.1 * np.sin(2 * np.pi * (gx + gy)))
    return sw.State({c: model.state[c] for c in COMPONENTS})


def _absmax(a, b):
    return max(
        float(np.abs(np.asarray(a[c].data) - np.asarray(b[c].data)).max())
        for c in COMPONENTS)


# ================================================================
#  Idempotency
# ================================================================
def test_vortical_and_wave_are_idempotent():
    em, model = _eig()
    z = _state(model, seed=1)
    fr.model.transforms.assert_idempotent(
        sw.transforms.VorticalProjection(em), z)
    fr.model.transforms.assert_idempotent(sw.transforms.WaveProjection(em), z)


def test_divergence_is_the_zero_map_on_band_limited_states():
    # off the Nyquist planes the three-mode basis is complete, so the
    # residual vanishes on a band-limited probe.
    em, model = _eig()
    z = _state(model)
    div = sw.transforms.DivergenceProjection(em)
    once = div(z)
    assert max(float(np.abs(np.asarray(once[c].data)).max())
               for c in COMPONENTS) < 1e-10
    # trivially idempotent (relative_l2 is ill-defined at zero, so an
    # absolute distance is used here)
    fr.model.transforms.assert_idempotent(div, z, norm=_absmax)


def test_divergence_is_the_structural_zero_map_on_random_states():
    # SEMANTICS (Nyquist completion): the interpolation-Nyquist
    # steady strata joined the vortical family, so the mode family
    # is complete on the even grid and the residual is the
    # structural zero map even on random (Nyquist-carrying) states.
    em, model = _eig()
    z = _state(model, seed=7)
    div = sw.transforms.DivergenceProjection(em)
    once = div(z)
    assert max(float(np.abs(np.asarray(once[c].data)).max())
               for c in COMPONENTS) < 1e-12
    fr.model.transforms.assert_idempotent(div, z, norm=_absmax)


# ================================================================
#  The algebra: Wave == P(+1) + P(-1); partition of unity
# ================================================================
def test_wave_equals_sum_of_single_mode_projections():
    em, model = _eig()
    z = _state(model, seed=2)
    wave = sw.transforms.WaveProjection(em)
    manual = (sw.transforms.mode_projection(em, 1)
              + sw.transforms.mode_projection(em, -1))
    assert _absmax(wave(z), manual(z)) < 1e-12


def test_partition_of_unity_reconstructs_the_state():
    # V + W + D == I EXACTLY on any state (D is the complement), the
    # patched k = 0 triple and the Nyquist planes included.
    em, model = _eig(f0=0.7, csqr=2.0)
    z = _state(model, seed=4)
    partition = (sw.transforms.VorticalProjection(em)
                 + sw.transforms.WaveProjection(em)
                 + sw.transforms.DivergenceProjection(em))
    assert _absmax(partition(z), z) < 1e-12


def test_wave_projection_merges_into_one_idempotent_projection():
    # P(+1) + P(-1) merges (same eigenmodes) into a single idempotent
    # EigenProjection over both modes, not a generic Sum node.
    em, _ = _eig()
    wave = sw.transforms.WaveProjection(em)
    assert isinstance(wave, EigenProjection)
    assert wave.modes == (-1, 1)
    assert wave.idempotent


# ================================================================
#  The shared base: signatures, dual constructors, repr
# ================================================================
def test_projection_has_a_concrete_staggered_signature():
    em, model = _eig()
    proj = sw.transforms.VorticalProjection(em)
    assert proj.domain is proj.codomain
    assert proj.domain.grid is model.grid
    assert proj.domain.names == ("u", "v", "p")
    assert proj.eigenmodes is em
    # the signature accepts the model's own staggered state
    proj.domain.validate_input(_state(model))


def test_call_rejects_a_state_missing_a_mapped_component():
    em, model = _eig()
    partial = sw.State({
        "u": model.state["u"],
        "v": model.state["v"]})
    with pytest.raises(SignatureMismatchError, match="p"):
        sw.transforms.VorticalProjection(em)(partial)


def test_from_model_builds_the_same_projection():
    _, model = _eig()
    proj = sw.transforms.WaveProjection.from_model(model)
    assert isinstance(proj, EigenProjection)
    assert proj.modes == (-1, 1)


def test_from_model_rejects_a_beta_plane():
    grid = make_grid()
    model = sw.Model(
        grid=grid, csqr=1.0,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"coriolis\.f0"):
        sw.transforms.VorticalProjection.from_model(model)


def test_reprs_are_informative():
    em, _ = _eig()
    assert "VorticalProjection" in repr(sw.transforms.VorticalProjection(em))
    assert repr(sw.transforms.WaveProjection) == "WaveProjection"
    assert "modes=(0,)" in repr(sw.transforms.mode_projection(em, 0))


def test_projection_is_tier_one_and_costless():
    em, _ = _eig()
    proj = sw.transforms.WaveProjection(em)
    assert proj.traceable
    assert proj.cost().model_steps == 0


# ================================================================
#  The channel (engine) path: labeled family projections
# ================================================================
@pytest.fixture(scope="module")
def channel():
    """One walled channel model + labeled eigenbasis (shared)."""
    model = make_model(_pinned_grid(periodic_y=False), advection=False)
    return model, sw.eigenbasis(model)


def _channel_state(model, seed):
    """Write random data onto the channel's staggered components."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in COMPONENTS})
    return sw.State({c: model.state[c] for c in COMPONENTS})


def _reference_projection(em, codes, state, csqr=1.0):
    """In-test port of the reference SpectralProjection pipeline.

    The shape of Adiabatic-Coriolis-Ramping's
    ``SpectralProjection.__call__``: full ``fft`` in x, per selected
    mode the Galerkin coefficient under the ``(c^2, c^2, 1)`` weights
    (inner product summed over y, per-mode normalization), accumulate
    the eigenfunction, inverse ``fft``, real part. The engine's
    labeled columns drive the mode iterator; the negative-``kx``
    planes of the full spectrum use the conjugate columns of the
    stored half spectrum (the operator is real).
    """
    n = np.asarray(state["u"].data).shape[0]
    arrs = {c: np.fft.fft(np.asarray(state[c].data), axis=0)
            for c in COMPONENTS}
    weights = {"u": csqr, "v": csqr, "p": 1.0}
    out = {c: np.zeros_like(arrs[c]) for c in COMPONENTS}
    q = np.asarray(em.q)
    labels = np.asarray(em.labels)
    n_kx = q.shape[0]
    for ikx in range(n):
        half = ikx if ikx < n_kx else n - ikx
        plane_q = q[half] if ikx < n_kx else np.conj(q[half])
        z = {c: arrs[c][ikx] for c in COMPONENTS}
        for col in np.flatnonzero(np.isin(labels[half], codes)):
            vec = {c: plane_q[em.slices[c], col] for c in COMPONENTS}
            norm = sum(weights[c] * np.sum(np.conj(vec[c]) * vec[c])
                       for c in COMPONENTS)
            inner = sum(weights[c] * np.sum(np.conj(vec[c]) * z[c])
                        for c in COMPONENTS)
            for c in COMPONENTS:
                out[c][ikx] += (inner / norm) * vec[c]
    return {c: np.real(np.fft.ifft(out[c], axis=0))
            for c in COMPONENTS}


def _m_energy(em, state, csqr=1.0):
    """Measure-weighted physical energy under ``diag(1, 1, 1/c^2)``."""
    weights = {"u": 1.0, "v": 1.0, "p": 1.0 / csqr}
    total = 0.0
    for c in COMPONENTS:
        mu = np.asarray(
            state[c].measure(em.bounded_axis).data).ravel()
        total += weights[c] * float(
            np.sum(np.asarray(state[c].data) ** 2 * mu[None, :]))
    return total


@pytest.mark.parametrize("family", ["vortical", "kelvin"])
def test_channel_projection_agrees_with_the_reference_pipeline(
        channel, family):
    model, eb = channel
    factory = {"vortical": sw.transforms.VorticalProjection,
               "kelvin": sw.transforms.KelvinProjection}[family]
    codes = tuple(code for name, code in eb.families.items()
                  if name.startswith(family))
    z = _channel_state(model, seed=9)
    got = factory(eb)(z)
    ref = _reference_projection(eb, codes, z)
    assert max(
        float(np.abs(ref[c] - np.asarray(got[c].data)).max())
        for c in COMPONENTS) < 1e-11


def test_channel_named_families_partition_unity(channel):
    # V + W + K == I at floating point on the f-plane channel (the
    # labeler resolves every column), so the divergence complement is
    # the zero map up to fp
    model, eb = channel
    z = _channel_state(model, seed=4)
    v = sw.transforms.VorticalProjection(eb)(z)
    w = sw.transforms.WaveProjection(eb)(z)
    k = sw.transforms.KelvinProjection(eb)(z)
    assert _absmax(
        sw.State({c: v[c] + w[c] + k[c] for c in COMPONENTS}),
        z) < 1e-12
    d = sw.transforms.DivergenceProjection(eb)(z)
    assert max(float(np.abs(np.asarray(d[c].data)).max())
               for c in COMPONENTS) < 1e-12


def test_channel_projections_are_idempotent_and_annihilating(channel):
    model, eb = channel
    z = _channel_state(model, seed=2)
    projections = {
        "vortical": sw.transforms.VorticalProjection(eb),
        "wave": sw.transforms.WaveProjection(eb),
        "kelvin": sw.transforms.KelvinProjection(eb)}
    for proj in projections.values():
        fr.model.transforms.assert_idempotent(proj, z)
    parts = {name: proj(z) for name, proj in projections.items()}
    for a, proj in projections.items():
        for b, part in parts.items():
            if a == b:
                continue
            crossed = proj(part)
            assert max(
                float(np.abs(np.asarray(crossed[c].data)).max())
                for c in COMPONENTS) < 1e-12, (a, b)


def test_channel_energy_partition(channel):
    # the families are M-orthogonal per plane and the x-FFT is
    # unitary up to a constant, so the measure-weighted physical
    # energies of the named projections sum to the total
    model, eb = channel
    z = _channel_state(model, seed=6)
    parts = [factory(eb)(z) for factory in (
        sw.transforms.VorticalProjection, sw.transforms.WaveProjection,
        sw.transforms.KelvinProjection)]
    total = _m_energy(eb, z)
    assert abs(sum(_m_energy(eb, p) for p in parts)
               - total) < 1e-12 * total


def test_channel_projection_has_the_tagged_signature(channel):
    model, eb = channel
    proj = sw.transforms.VorticalProjection(eb)
    assert isinstance(proj, EigenProjection)
    assert proj.idempotent
    assert proj.domain is proj.codomain
    assert proj.domain.grid is model.grid
    proj.domain.validate_input(_channel_state(model, seed=1))


def test_channel_from_model_routes_to_the_engine_path(channel):
    model, eb = channel
    z = _channel_state(model, seed=3)
    proj = sw.transforms.WaveProjection.from_model(model)
    assert isinstance(proj.eigenmodes, sw.ChannelEigenmodes)
    assert _absmax(proj(z), sw.transforms.WaveProjection(eb)(z)) == 0.0


def test_kelvin_projection_needs_walls():
    # the fully periodic path raises the taught error on both the
    # explicit-eigenmodes and the from_model routes
    em, model = _eig()
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        sw.transforms.KelvinProjection(em)
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        sw.transforms.KelvinProjection.from_model(model)


def test_periodic_analytic_path_is_unchanged_by_the_dispatch():
    # the engine dispatch must leave the fully periodic projections
    # on the original analytic path, bitwise
    em, model = _eig()
    z = _state(model, seed=8)
    assert _absmax(sw.transforms.VorticalProjection(em)(z),
                   sw.transforms.mode_projection(em, 0)(z)) == 0.0
    manual = (sw.transforms.mode_projection(em, 1)
              + sw.transforms.mode_projection(em, -1))
    assert _absmax(sw.transforms.WaveProjection(em)(z),
                   manual(z)) == 0.0


# ================================================================
#  The rest policy on a tracer-carrying state
# ================================================================
class _PassiveTracer(fr.model.Module):

    """A module declaring one prognostic passive tracer ``c``."""

    field_declarations = (fr.model.FieldDeclaration.tracer("c"),)

    @term(name="c_hold", advances=("c",))
    def hold(self, state, _ctx):
        return {"c": state["c"] * 0.0}


def test_projection_rest_zero_completes_a_passive_tracer():
    # a state extended by a prognostic passive tracer: the vortical
    # projection (rest="zero") returns the tracer as a zero field on
    # its own space, and the residual carries it fully (§10.7.2)
    model = make_model(_pinned_grid(), advection=False,
                       modules_extra=(_PassiveTracer(),))
    _state(model, seed=11)
    rng = np.random.default_rng(12)
    shape = np.asarray(model.state["c"].data).shape
    model.set_fields(c=rng.standard_normal(shape))
    z = sw.State({c: model.state[c] for c in (*COMPONENTS, "c")})
    proj = sw.transforms.VorticalProjection.from_model(model)
    out = proj(z)
    assert out.component_names == (*COMPONENTS, "c")
    assert np.all(np.asarray(out["c"].data) == 0.0)
    assert (out["c"].function_space.bare
            == z["c"].function_space.bare)
    residual = z - out
    assert np.allclose(np.asarray(residual["c"].data),
                       np.asarray(z["c"].data))
    # the complement transform carries the tracer through unchanged
    assert np.allclose(np.asarray(proj.complement(z)["c"].data),
                       np.asarray(z["c"].data))


# ================================================================
#  The sharded multi-device application (forced-devices gate)
# ================================================================
def _sharded_channel_model(device_ids=None):
    """Return a walled-y sw 2-D channel model (x periodic; y bounded)."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=False, name="y")
    return make_model(fr.spatial.Grid((mx, my), device_ids=device_ids),
                      advection=False)


@pytest.mark.multi_device
def test_channel_projection_on_a_sharded_grid_matches_one_device(
        forced_devices):
    # the 2-D channel (single periodic axis) is now served by the
    # transpose pipeline (Channel2DPlan): the sharded projection matches
    # the replicated one-device reference and lands real
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many = _sharded_channel_model()
    one = _sharded_channel_model(device_ids=(0,))
    assert many.grid.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    rng = np.random.default_rng(12)
    fields = {"u": rng.standard_normal((N, N)),
              "v": rng.standard_normal((N, N - 1)),
              "p": rng.standard_normal((N, N))}
    many.set_fields(**fields)
    one.set_fields(**fields)
    z_many = sw.State({c: many.state[c] for c in COMPONENTS})
    z_one = sw.State({c: one.state[c] for c in COMPONENTS})
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    pm = sw.transforms.VorticalProjection(sw.eigenbasis(many))(z_many)
    po = sw.transforms.VorticalProjection(sw.eigenbasis(one))(z_one)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(pm[c].data))
    absmax = max(float(np.abs(np.asarray(pm[c].data)
                              - np.asarray(po[c].data)).max())
                 for c in COMPONENTS)
    assert absmax <= 1e-11


# ================================================================
#  The fully periodic (analytic) sharded application (forced-4)
# ================================================================
def _periodic_grid(device_ids):
    """Return a fully periodic 2-D grid at a device layout."""
    return fr.spatial.Grid(tuple(
        fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                       name=name)
        for name in ("x", "y")), device_ids=device_ids)


@pytest.mark.multi_device
def test_analytic_projections_run_on_a_sharded_axis(forced_devices):
    # the fully periodic analytic vortical / wave / divergence
    # projections route through the fused per-mode 3x3 matrix apply on a
    # grid that shards a transform axis (the 2-D internal frame is fully
    # complex), matching the replicated one-device reference to floating
    # point, landing real and staying idempotent projectors.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    rng = np.random.default_rng(4)
    fields = {c: rng.standard_normal((N, N)) for c in COMPONENTS}
    many = make_model(_periodic_grid(None), csqr=1.0, f0=1.0)
    many.set_fields(**fields)
    z_many = sw.State({c: many.state[c] for c in COMPONENTS})
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    em_many = sw.eigenmodes.from_model(many)
    one = make_model(_periodic_grid((0,)), csqr=1.0, f0=1.0)
    one.set_fields(**fields)
    z_one = sw.State({c: one.state[c] for c in COMPONENTS})
    em_one = sw.eigenmodes.from_model(one)
    for factory in (sw.transforms.VorticalProjection,
                    sw.transforms.WaveProjection,
                    sw.transforms.DivergenceProjection):
        out_many = factory(em_many)(z_many)
        out_one = factory(em_one)(z_one)
        assert not any(
            np.iscomplexobj(np.asarray(out_many[c].data))
            for c in COMPONENTS)
        assert _absmax(out_many, out_one) < 1e-11
        assert _absmax(factory(em_many)(out_many), out_many) < 1e-10


@pytest.mark.multi_device
def test_grad_through_analytic_projection_is_finite(forced_devices):
    # jax.grad of a quadratic loss through the fused projection is finite
    # and matches a central finite difference
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    model = make_model(_periodic_grid(None), csqr=1.0, f0=1.0)
    rng = np.random.default_rng(5)
    model.set_fields(**{c: rng.standard_normal((N, N))
                        for c in COMPONENTS})
    base = sw.State({c: model.state[c] for c in COMPONENTS})
    proj = sw.transforms.VorticalProjection(
        sw.eigenmodes.from_model(model))
    u0 = jnp.asarray(base["u"].data)

    def loss(u):
        z = sw.State({
            c: (base[c].with_data(u) if c == "u" else base[c])
            for c in COMPONENTS})
        out = proj(z)
        return sum(jnp.sum(out[c].data ** 2) for c in COMPONENTS)

    grad = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
    eps = 1e-4
    pert = jnp.asarray(rng.standard_normal((N, N)))
    num = (loss(u0 + eps * pert) - loss(u0 - eps * pert)) / (2 * eps)
    ana = float(jnp.sum(grad * pert))
    assert abs(num - ana) <= 1e-4 * max(1.0, abs(ana))
