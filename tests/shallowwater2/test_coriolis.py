"""The exactly-conserving shallow-water Coriolis (both routes).

The linear Coriolis term is skew under the LINEARIZED metric, not under
the thickness-weighted energy the nonlinear scheme conserves, so it
produces ``etot_full`` at O(Ro). The conserving discrete f-term is the
f-part of the vector-invariant PV flux, ``(f/h_bar)(h v)_bar`` — a
ratio of averages, nonlinear in h. Two routes ship it:

- route A: the linear module + the optional ``CoriolisEnergyCorrection``
  (``linear=False``), so the linear operator L is untouched;
- route B: the ``Nonlinear*Coriolis`` modules, which carry the whole
  term instead of the linear one — cheaper, but L loses rotation, so
  the eigenmode / projection / balance machinery refuses the model.

The gates here: the invariant closes to machine zero for BOTH routes on
flat, channel and sphere; A and B agree to rounding; B refuses to
co-exist with A (double-counted rotation) and refuses to hand out L;
the correction is exactly zero at a rest state and leaves L (and every
other term) bitwise unchanged.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.errors import LinearOperatorGapError
from fridom.model.model import _chunk_body
from fridom.model.params import CORIOLIS_F0
from fridom.shallowwater2.modules.coriolis import _safe_pv_divide

CSQR = 0.7
RO = 0.4
F0 = 1.0
OMEGA = 1.5
LAT_MAX = float(np.deg2rad(80.0))
NAMES = ("u", "v", "p")

#: everything but the correction term (the off-switch probe)
NOT_CORRECTION = ~fr.model.term_predicates.named(
    "CoriolisEnergyCorrection/correction")


# ================================================================
#  Models: flat (periodic / channel) and the lat-lon sphere chart
# ================================================================
def flat_grid(*, periodic_y=True):
    """Return a tiny square grid, walled in y when asked."""
    mx = fr.spatial.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(
        16, (0.0, 1.0), periodic=periodic_y, name="y")
    return fr.spatial.Grid((mx, my), device_ids=(0,))


def sphere_grid(nlon=16, nlat=8, radius=1.0):
    """Return the documented lat-lon sphere chart grid."""
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=radius, lat_extent=(-LAT_MAX, LAT_MAX),
        device_ids=(0,))


def stepper():
    """Return the shared (unused: no run) time stepper."""
    return fr.model.time_steppers.AdamBashforth(2e-3, order=3)


def flat_model(route, *, periodic_y=True, csqr=CSQR, advection=True):
    """Assemble a flat model on the given Coriolis route.

    ``route`` is ``"linear"`` (today's behaviour), ``"A"`` (linear +
    correction) or ``"B"`` (the conserving module alone).
    """
    extra = ()
    if route == "B":
        coriolis = sw.modules.NonlinearFPlaneCoriolis(f0=F0)
    else:
        coriolis = sw.modules.FPlaneCoriolis(f0=F0)
        if route == "A":
            extra = (sw.modules.CoriolisEnergyCorrection(),)
    return sw.Model(
        grid=flat_grid(periodic_y=periodic_y), csqr=csqr,
        rossby_number=RO, coriolis=coriolis, advection=advection,
        modules_extra=extra, time_stepper=stepper())


def sphere_model(route):
    """Assemble the spherical preset on the given Coriolis route."""
    extra = ()
    if route == "B":
        coriolis = sw.modules.NonlinearRotationCoriolis(
            omega=(0.0, 0.0, OMEGA), coords=("lon", "lat"))
    else:
        coriolis = sw.modules.RotationCoriolis(
            omega=(0.0, 0.0, OMEGA), coords=("lon", "lat"),
            metric_weight="csqr")
        if route == "A":
            extra = (sw.modules.CoriolisEnergyCorrection(
                coords=("lon", "lat")),)
    return sw.Model(
        grid=sphere_grid(), coords=("lon", "lat"), csqr=CSQR,
        rossby_number=RO, coriolis=coriolis, modules_extra=extra,
        time_stepper=stepper())


def channel_model(route):
    """Assemble the walled (channel) flat model."""
    return flat_model(route, periodic_y=False)


MODELS = {
    "periodic": flat_model,
    "channel": channel_model,
    "sphere": sphere_model,
}


@pytest.fixture(params=sorted(MODELS), ids=sorted(MODELS))
def builder(request):
    """Return a ``route -> model`` builder, once per grid family."""
    return MODELS[request.param]


class _ForeignCoriolisField(fr.model.Module):

    """A module that owns ``f_coriolis`` but carries no rotation term."""

    @property
    def field_declarations(self):
        """The one-DOF Coriolis field, with no term to go with it."""
        return (fr.model.FieldDeclaration(
            "f_coriolis", space=fr.spatial.Profile(),
            lifecycle=fr.model.Lifecycle.AUXILIARY,
            default=lambda grid, space: grid.create_field(
                space, data=jnp.full(space.shape, F0),
                name="f_coriolis")),)


def set_random(model, seed=11):
    """Fill the prognostics (walls are structural: u.n = 0)."""
    rng = np.random.default_rng(seed)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        p=0.3 * rng.standard_normal(model.state["p"].shape))
    return model


def energy_rate(model, diagnostic):
    """Semi-discrete d/dt of a diagnostic under the FULL tendency.

    The exact directional derivative of the discrete functional along
    the discrete tendency (a jvp — no stepping, no differencing error),
    with **every** term included: the Coriolis production is what this
    gate is about.
    """
    z = model.state
    dz = model.tendency(z)

    def total(data):
        state = z.replace(**{name: z[name].with_data(data[name])
                             for name in NAMES})
        return diagnostic(state, model.parameters).integrate(
        ).data.ravel()[0]

    primal = {name: z[name].data for name in NAMES}
    tangent = {name: dz[name].data for name in NAMES}
    return float(jax.jvp(total, (primal,), (tangent,))[1])


def relative_production(model):
    """Return the etot_full production rate, scaled by its parts."""
    rate = energy_rate(model, sw.diagnostics.etot_full)
    scale = sum(
        abs(energy_rate(model, diagnostic))
        for diagnostic in (sw.diagnostics.ekin_full,
                           sw.diagnostics.epot_full))
    return abs(rate) / scale


# ================================================================
#  THE gate: the invariant closes, for both routes
# ================================================================
@pytest.mark.parametrize("route", ["A", "B"])
def test_the_conserving_coriolis_closes_the_invariant(builder, route):
    # the whole point: the semi-discrete production of the
    # thickness-weighted energy INCLUDING Coriolis is machine zero —
    # on flat, channel AND sphere (measured ~1e-16, matching the
    # gravity + Sadourny gates of test_diagnostics)
    assert relative_production(set_random(builder(route))) < 1e-13


def test_the_linear_coriolis_produces_the_invariant_at_o_ro(builder):
    # the error being removed: the split (linear) Coriolis term is skew
    # under the LINEARIZED metric only, and produces the nonlinear
    # invariant at O(Ro)
    assert relative_production(set_random(builder("linear"))) > 1e-4


def test_route_b_conserves_with_a_variable_depth():
    # the conserving form weights the rotation by the THICKNESS, so it
    # is exact for any depth profile — no metric_weight needed (and the
    # preset does not demand one)
    model = flat_model(
        "B", csqr=lambda y: 1.0 + 0.5 * jnp.tanh(4.0 * (y - 0.5)))
    assert relative_production(set_random(model)) < 1e-13


# ================================================================
#  A and B are the same physics
# ================================================================
def test_routes_a_and_b_assemble_the_same_tendency(builder):
    # linear + correction == the conserving module, to rounding: this
    # is what pins the two routes as one discrete term
    model_a = set_random(builder("A"))
    model_b = set_random(builder("B"))
    dz_a = model_a.tendency(model_a.state)
    dz_b = model_b.tendency(model_b.state)
    for name in NAMES:
        got = np.asarray(dz_a[name].data)
        expected = np.asarray(dz_b[name].data)
        scale = float(np.max(np.abs(expected)))
        assert np.max(np.abs(got - expected)) / scale < 1e-13


# ================================================================
#  Route A leaves the linear operator alone
# ================================================================
def test_the_correction_is_exactly_zero_on_a_rest_state(builder):
    # both parts of the correction are proportional to the velocity, so
    # it contributes nothing at u = v = 0 (h = c^2): the linear model
    # is untouched, exactly
    model = builder("A")  # rest state: no set_random
    dz = model.tendency(
        model.state,
        filter=fr.model.term_predicates.named(
            "CoriolisEnergyCorrection/correction"))
    for name in ("u", "v"):
        assert np.array_equal(np.asarray(dz[name].data),
                              np.zeros(dz[name].shape))


def test_the_correction_leaves_every_other_term_bitwise(builder):
    # the off switch: with the correction term filtered out, the
    # assembled tendency is bit-for-bit the tendency of a model that
    # never carried the module
    model_a = set_random(builder("A"))
    model_base = set_random(builder("linear"))
    dz_a = model_a.tendency(model_a.state, filter=NOT_CORRECTION)
    dz_base = model_base.tendency(model_base.state)
    for name in NAMES:
        assert np.array_equal(np.asarray(dz_a[name].data),
                              np.asarray(dz_base[name].data))


def test_the_correction_leaves_the_linear_model_bitwise(builder):
    # L itself: the correction is declared linear=False, so
    # fr.model.linearize drops it — the linear variant's tendency is
    # bit-for-bit that of the uncorrected model
    model_a = set_random(builder("A"))
    model_base = set_random(builder("linear"))
    dz_a = fr.model.linearize(model_a).tendency(model_a.state)
    dz_base = fr.model.linearize(model_base).tendency(
        model_base.state)
    for name in NAMES:
        assert np.array_equal(np.asarray(dz_a[name].data),
                              np.asarray(dz_base[name].data))


def test_the_correction_leaves_the_eigenmodes_bitwise():
    # the analytic spectrum and a projection are BITWISE identical with
    # and without the correction module (L is untouched)
    model_a = set_random(flat_model("A"))
    model_base = set_random(flat_model("linear"))
    em_a = sw.eigenmodes.from_model(model_a)
    em_base = sw.eigenmodes.from_model(model_base)
    for s in (-1, 0, 1):
        assert np.array_equal(np.asarray(em_a.omega(s).data),
                              np.asarray(em_base.omega(s).data))
    got = sw.transforms.VorticalProjection.from_model(model_a)(
        model_a.state)
    expected = sw.transforms.VorticalProjection.from_model(
        model_base)(model_base.state)
    for name in NAMES:
        assert np.array_equal(np.asarray(got[name].data),
                              np.asarray(expected[name].data))


# ================================================================
#  Route B: honest about L
# ================================================================
def test_route_b_refuses_to_linearize():
    model = flat_model("B")
    with pytest.raises(LinearOperatorGapError, match="NO rotation"):
        fr.model.linearize(model)


def test_route_b_refuses_the_analytic_eigenmodes():
    model = flat_model("B")
    with pytest.raises(LinearOperatorGapError,
                       match=r"sw\.eigenmodes\.from_model"):
        sw.eigenmodes.from_model(model)


def test_route_b_refuses_the_channel_eigenbasis():
    model = flat_model("B", periodic_y=False)
    with pytest.raises(LinearOperatorGapError, match=r"sw\.eigenbasis"):
        sw.eigenmodes.eigenbasis(model)


def test_route_b_refuses_a_projection():
    model = flat_model("B")
    with pytest.raises(LinearOperatorGapError, match="invalid"):
        sw.transforms.VorticalProjection.from_model(model)


def test_route_b_provides_no_f0():
    # provides-implies-constancy: publishing coriolis.f0 would claim
    # that L rotates at f0, which is exactly what it does not do
    assert CORIOLIS_F0 not in flat_model("B").parameters
    assert CORIOLIS_F0 in flat_model("linear").parameters


def test_route_a_keeps_the_model_linearizable():
    # the contrast: with the correction the model still hands out L
    assert fr.model.linear_operator_gaps(flat_model("A")) == ()
    assert fr.model.linearize(flat_model("A")) is not None


# ================================================================
#  Route B refuses to co-exist with route A (double-counted rotation)
# ================================================================
@pytest.mark.parametrize("other", ["linear", "correction",
                                   "conserving"])
def test_route_b_refuses_a_second_rotation_module(other):
    # assembling the conserving module next to ANY other rotation
    # module double-counts the rotation: a taught error
    modules = {
        "linear": sw.modules.FPlaneCoriolis(f0=F0),
        "correction": sw.modules.CoriolisEnergyCorrection(),
        "conserving": sw.modules.NonlinearBetaPlaneCoriolis(f0=F0),
    }
    with pytest.raises(ValueError, match="counted twice"):
        sw.Model(
            grid=flat_grid(), csqr=CSQR, rossby_number=RO,
            coriolis=sw.modules.NonlinearFPlaneCoriolis(f0=F0),
            modules_extra=(modules[other],), time_stepper=stepper())


def test_route_b_refuses_the_correction_at_bind():
    # the same rule through explicit assembly (no preset to pre-check):
    # the conserving module's bind is the backstop
    modules = (
        sw.modules.DynamicalCore(csqr=CSQR, rossby_number=RO),
        sw.modules.NonlinearFPlaneCoriolis(f0=F0),
        sw.modules.CoriolisEnergyCorrection())
    with pytest.raises(ValueError, match="counted twice"):
        fr.model.Model(grid=flat_grid(), modules=modules,
                       time_stepper=stepper())


def test_two_f_coriolis_owners_collide_at_assembly():
    # explicit assembly of route B next to a linear module never
    # reaches bind: two modules declaring f_coriolis is a structural
    # collision (which names both of them)
    modules = (
        sw.modules.DynamicalCore(csqr=CSQR, rossby_number=RO),
        sw.modules.NonlinearFPlaneCoriolis(f0=F0),
        sw.modules.FPlaneCoriolis(f0=F0))
    with pytest.raises(ValueError, match="declared twice"):
        fr.model.Model(grid=flat_grid(), modules=modules,
                       time_stepper=stepper())


def test_the_correction_needs_a_linear_coriolis_module():
    # rotation is opt-in, so a model may carry none at all — and then
    # there is nothing to correct
    with pytest.raises(ValueError, match="no module contributing"):
        sw.Model(
            grid=flat_grid(), csqr=CSQR, rossby_number=RO,
            modules_extra=(sw.modules.CoriolisEnergyCorrection(),),
            time_stepper=stepper())


def test_the_correction_needs_the_linear_rotation_term():
    # a foreign module may declare f_coriolis without contributing the
    # linear rotation term — then there is nothing to correct, and the
    # correction would subtract a term nobody added
    modules = (
        sw.modules.DynamicalCore(csqr=CSQR, rossby_number=RO),
        _ForeignCoriolisField(),
        sw.modules.CoriolisEnergyCorrection())
    with pytest.raises(ValueError, match="no module contributing"):
        fr.model.Model(grid=flat_grid(), modules=modules,
                       time_stepper=stepper())


def test_the_correction_refuses_two_linear_coriolis_modules():
    with pytest.raises(ValueError, match="exactly one"):
        sw.Model(
            grid=flat_grid(), csqr=CSQR, rossby_number=RO,
            coriolis=sw.modules.FPlaneCoriolis(f0=F0),
            modules_extra=(sw.modules.BetaPlaneCoriolis(f0=F0),
                           sw.modules.CoriolisEnergyCorrection()),
            time_stepper=stepper())


def test_the_correction_refuses_a_foreign_metric_weight():
    with pytest.raises(ValueError, match="energy weight"):
        sw.Model(
            grid=flat_grid(), csqr=CSQR, rossby_number=RO,
            coriolis=sw.modules.FPlaneCoriolis(f0=F0,
                                               metric_weight="p"),
            modules_extra=(sw.modules.CoriolisEnergyCorrection(),),
            time_stepper=stepper())


# ================================================================
#  Declarations, knobs, and the shared predicate
# ================================================================
def test_the_correction_adopts_the_paired_weight():
    weighted = flat_model("linear").module(sw.modules.FPlaneCoriolis)
    assert weighted.metric_weight is None
    unweighted = flat_model("A").module(
        sw.modules.CoriolisEnergyCorrection)
    assert unweighted.metric_weight is None
    model = sw.Model(
        grid=flat_grid(), csqr=CSQR, rossby_number=RO,
        coriolis=sw.modules.FPlaneCoriolis(f0=F0,
                                           metric_weight="csqr"),
        modules_extra=(sw.modules.CoriolisEnergyCorrection(),),
        time_stepper=stepper())
    assert model.module(
        sw.modules.CoriolisEnergyCorrection).metric_weight == "csqr"


def test_the_conserving_modules_take_no_metric_weight():
    # the thickness IS the weight
    assert sw.modules.NonlinearFPlaneCoriolis().metric_weight is None
    assert sw.modules.NonlinearRotationCoriolis().metric_weight is None


def test_the_correction_declares_its_coords_and_halo():
    module = sw.modules.CoriolisEnergyCorrection(coords=("lon", "lat"))
    assert module.coords == ("lon", "lat")
    assert dict(module.extra_halo.widths) == {"lon": 2, "lat": 2}
    conserving = sw.modules.NonlinearBetaPlaneCoriolis(
        beta=0.5, coords=("a", "b"))
    assert conserving.coords == ("a", "b")
    assert dict(conserving.extra_halo.widths) == {"a": 2, "b": 2}


@pytest.mark.parametrize("bad", [("x",), ("x", "x"), ("x", 3)])
def test_bad_coords_are_rejected(bad):
    with pytest.raises(TypeError, match="zonal"):
        sw.modules.CoriolisEnergyCorrection(coords=bad)


def test_carries_linear_rotation_separates_the_families():
    assert sw.modules.carries_linear_rotation(
        sw.modules.FPlaneCoriolis())
    assert sw.modules.carries_linear_rotation(
        sw.modules.RotationCoriolis())
    assert not sw.modules.carries_linear_rotation(
        sw.modules.NonlinearFPlaneCoriolis())
    assert not sw.modules.carries_linear_rotation(
        sw.modules.CoriolisEnergyCorrection())
    assert not sw.modules.carries_linear_rotation(
        sw.modules.SadournyAdvection())


def test_the_conserving_beta_plane_varies_with_y():
    # the f(y) declaration is inherited from BetaPlaneCoriolis; only
    # the term changes
    model = sw.Model(
        grid=flat_grid(), csqr=CSQR, rossby_number=RO,
        coriolis=sw.modules.NonlinearBetaPlaneCoriolis(f0=F0,
                                                       beta=0.5),
        time_stepper=stepper())
    f = np.asarray(model.state["f_coriolis"].data).ravel()
    assert f.size > 1
    assert np.all(np.diff(f) > 0.0)
    assert relative_production(set_random(model)) < 1e-13


def test_the_conserving_f_plane_rejects_a_chart_grid():
    # inherited grid validation: the metric-blind modules stay
    # metric-blind (the conserving term does not fix that)
    with pytest.raises(ValueError, match="metric-blind"):
        sw.Model(
            grid=sphere_grid(), coords=("lon", "lat"), csqr=CSQR,
            rossby_number=RO,
            coriolis=sw.modules.NonlinearFPlaneCoriolis(
                f0=F0, coords=("lon", "lat")),
            time_stepper=stepper())


# ================================================================
#  R2: the ramped beta-plane FieldBlend end to end (AR-D2)
# ================================================================
# The conserving rotation reads f its own way (state["f_coriolis"]);
# route B (NonlinearBetaPlaneCoriolis) inherits the beta-plane blend, so
# the conserving channel supports a ramped beta end to end. Route A (the
# correction) under a ramped f, and the conserving f-plane under a ramped
# f0, are taught follow-ups (no field blend to honor them).
RAMP_DT = 2e-3


def _conserving_beta_channel(beta, f0=F0):
    """Return a conserving (route B) beta channel; float/Ramp beta."""
    return sw.Model(
        grid=flat_grid(periodic_y=False), csqr=CSQR, rossby_number=RO,
        coriolis=sw.modules.NonlinearBetaPlaneCoriolis(f0=f0, beta=beta),
        advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(RAMP_DT))


@pytest.mark.parametrize("t", [0.0, 0.02, 0.05])
def test_conserving_ramped_beta_tendency_equals_static_at_stage(t):
    """Route-B conserving rotation reads the stage-time f(y) blend.

    ramped.tendency(z, t) == static conserving model at beta = ramp(t):
    the conserving (f/h_bar)(h v)_bar term consumes the fresh blend, so
    the sw2 channel supports a ramped beta end to end.
    """
    ramp = fr.model.Ramp(0.0, 2.0, period=0.05, curve="exp")
    ramped = _conserving_beta_channel(ramp)
    set_random(ramped)
    z = sw.State({c: ramped.state[c] for c in NAMES})
    got = ramped.tendency(z, t=t)

    const = _conserving_beta_channel(float(ramp.at_time(t)))
    const.set_fields(**{c: np.asarray(z[c].data) for c in NAMES})
    z_const = sw.State({c: const.state[c] for c in NAMES})
    want = const.tendency(z_const)
    for c in NAMES:
        np.testing.assert_allclose(
            np.asarray(got[c].data), np.asarray(want[c].data),
            rtol=1e-12, atol=1e-13)


def test_conserving_ramped_beta_declares_the_blend_ingredients():
    model = _conserving_beta_channel(fr.model.Ramp(0.0, 2.0, period=1.0))
    for name in ("f_coriolis", "f_coriolis_const", "f_coriolis_grad"):
        assert name in model.state
    # route B carries the rotation in N, so it reports no time-dependent
    # LINEAR parameter (unlike the linear beta module)
    module = model.module(sw.modules.NonlinearBetaPlaneCoriolis)
    assert module.time_dependent_linear_parameters() == ()


def test_conserving_f_plane_rejects_a_ramped_f0():
    # the conserving f-plane carries no field blend, so a ramped f0
    # would silently freeze -- a taught error instead
    with pytest.raises(TypeError, match="conserving f-plane"):
        sw.modules.NonlinearFPlaneCoriolis(
            f0=fr.model.Ramp(0.0, 1.0, period=1.0))


def test_correction_rejects_a_ramped_beta_linear_module():
    # route A subtracts the frozen f_coriolis snapshot; pairing it with a
    # ramped (blend-active) linear module double-counts the ramp
    with pytest.raises(ValueError, match="ramped f\\(y\\) FieldBlend"):
        sw.Model(
            grid=flat_grid(periodic_y=False), csqr=CSQR, rossby_number=RO,
            coriolis=sw.modules.BetaPlaneCoriolis(
                f0=F0, beta=fr.model.Ramp(0.0, 2.0, period=1.0)),
            modules_extra=(sw.modules.CoriolisEnergyCorrection(),),
            advection=True, time_stepper=stepper())


@pytest.mark.multi_device
def test_conserving_ramped_beta_is_device_count_invariant(forced_devices):
    """Gate (d): the conserving f(y) blend is halo-neutral (forced-4)."""
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    ramp = fr.model.Ramp(0.0, 2.0, period=5e-2, curve="exp")

    def build(device_ids):
        mx = fr.spatial.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
        my = fr.spatial.meshes.IntervalMesh(
            16, (0.0, 1.0), periodic=False, name="y")
        return sw.Model(
            grid=fr.spatial.Grid((mx, my), device_ids=device_ids),
            csqr=CSQR, rossby_number=RO,
            coriolis=sw.modules.NonlinearBetaPlaneCoriolis(f0=F0, beta=ramp),
            advection=True,
            time_stepper=fr.model.time_steppers.AdamBashforth(RAMP_DT))

    rng = np.random.default_rng(4)
    fields = {"u": rng.standard_normal((16, 16)),
              "v": rng.standard_normal((16, 15)),
              "p": rng.standard_normal((16, 16))}
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = build(device_ids)
        model.set_fields(**fields)
        model.advance(4)
        results[tag] = {c: np.asarray(model.state[c].data) for c in NAMES}
        if tag == "many":
            assert model.state["u"]._data.sharding.spec[0] == "devices"
    assert max(
        float(np.abs(results["many"][c] - results["one"][c]).max())
        for c in NAMES) < 1e-11


# ================================================================
#  R2 physics gate: adiabatic beta ramp -> exponentially small leakage
# ================================================================
# The acceptance demo of the FieldBlend on the linearized sw2 channel.
# A vortical (slow) state of the reference system stays slow as beta is
# ramped adiabatically to the target, so the relative imbalance eta
# = ||(I - P_slow) z|| / ||z|| (a volume-weighted energy norm) at the
# far end DECREASES as the ramp period tau lengthens. Parameters:
# f0=1, c^2=1, beta=2 (well below the 2k^2 Rossby/Kelvin edge, k>=2pi
# so 2k^2 >~ 79); dt=1e-2 stays inside the AB3 stability limit for the
# high-k gravity waves (omega_max ~ 25); tau in {0.5, 1, 2} spans the
# adiabatic onset while keeping the step counts (50/100/200) small.
# Both ramp directions are tested. The slow subspace is the labeled
# VORTICAL (geostrophic/Rossby) family. The reduction is finite (a
# small-grid / finite-tau leakage floor, plan risk 5.1), so the gate
# asserts a monotone decrease and a conservative minimum reduction.
_AD_F0 = 1.0
_AD_CSQR = 1.0
_AD_BETA = 2.0
_AD_DT = 1e-2
_AD_TAUS = (0.5, 1.0, 2.0)


def _adiabatic_norm(eb, state):
    """Volume-weighted physical energy under diag(1, 1, 1/c^2)."""
    weights = {"u": 1.0, "v": 1.0, "p": 1.0 / _AD_CSQR}
    total = 0.0
    for c in NAMES:
        mu = np.asarray(state[c].measure(eb.bounded_axis).data).ravel()
        total += weights[c] * float(
            np.sum(np.asarray(state[c].data) ** 2 * mu[None, :]))
    return total


def _imbalance(eb, projection, state):
    """Eta = ||(I - P) z|| / ||z|| under the volume-weighted norm."""
    residual = state - projection(state)
    return float(np.sqrt(_adiabatic_norm(eb, residual)
                         / _adiabatic_norm(eb, state)))


@pytest.fixture(scope="module")
def adiabatic_channel():
    """Build reference (beta=0) + target channels and slow projectors.

    Built on ONE shared grid (the reference is a variant of the target)
    so the eigenmode projections and the ramped Propagator legs all
    carry the same grid identity.
    """
    mx = fr.spatial.meshes.IntervalMesh(8, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(8, (0.0, 1.0), periodic=False,
                                        name="y")
    target = sw.Model(
        grid=fr.spatial.Grid((mx, my), device_ids=(0,)),
        csqr=_AD_CSQR, rossby_number=0.2,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=_AD_F0, beta=_AD_BETA),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(_AD_DT, order=3))
    reference = target.variant(updates={"coriolis.beta": 0.0})
    eb_ref = sw.eigenbasis(reference)
    eb_tgt = sw.eigenbasis(target)
    return {
        "target": target, "reference": reference,
        "eb_ref": eb_ref, "eb_tgt": eb_tgt,
        "P_ref": sw.transforms.VorticalProjection(eb_ref),
        "P_tgt": sw.transforms.VorticalProjection(eb_tgt)}


def _slow_state(model, projection, seed):
    """Return a random state projected onto the labeled slow subspace."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in NAMES})
    return projection(sw.State({c: model.state[c] for c in NAMES}))


@pytest.mark.parametrize("direction", ["up", "down"])
def test_ramped_beta_leakage_decays_with_tau(adiabatic_channel, direction):
    ch = adiabatic_channel
    target = ch["target"]
    if direction == "up":
        # slow state of the reference; ramp 0 -> beta; project at target
        z0 = _slow_state(ch["reference"], ch["P_ref"], seed=3)
        endpoints = (0.0, _AD_BETA)
        eb_far, p_far = ch["eb_tgt"], ch["P_tgt"]
    else:
        # slow state of the target; ramp beta -> 0; project at reference
        z0 = _slow_state(target, ch["P_tgt"], seed=5)
        endpoints = (_AD_BETA, 0.0)
        eb_far, p_far = ch["eb_ref"], ch["P_ref"]

    # the initial state is (numerically) purely slow
    eb_near = ch["eb_ref"] if direction == "up" else ch["eb_tgt"]
    p_near = ch["P_ref"] if direction == "up" else ch["P_tgt"]
    assert _imbalance(eb_near, p_near, z0) < 1e-10

    etas = []
    for tau in _AD_TAUS:
        prop = fr.model.Propagator(
            target, steps=round(tau / _AD_DT),
            updates={"coriolis.beta": fr.model.Ramp(
                *endpoints, period=tau, curve="exp")},
            term_filter=fr.model.term_predicates.linear)
        etas.append(_imbalance(eb_far, p_far, prop(z0)))

    # monotone decrease across the doubling tau sequence, and a
    # conservative net reduction (tolerance set away from the classifi-
    # cation edge and above the small-grid leakage floor)
    assert etas[0] > etas[1] > etas[2], etas
    assert etas[-1] < 0.75 * etas[0], etas
    # the leakage is real but small: the slow state stays mostly slow
    assert etas[0] < 0.2, etas


# ================================================================
#  Reverse-mode autodiff: the conserving f/h potential-vorticity divide
# ================================================================
# The conserving Coriolis carries the ``f``-part of the PV, ``f / h_bar``.
# On a walled grid (the lat-lon sphere's polar caps, a closed basin) the
# corner thickness ``h`` is an exact zero in the never-valid corner/halo
# padding, where ``f`` vanishes too, so the bare quotient is a masked
# ``0/0`` whose reverse-mode VJP (``-f/h^2``, ``h = 0``) is a NaN poison
# — the same masked singularity ``SadournyAdvection._potential_vorticity``
# cures for the advective PV divide. ``_safe_pv_divide`` guards it; these
# pin the guard (padding NaN removed, valid interior bitwise identical)
# and a finite, FD-matched gradient through a route-B run.
def ic_grad_loss(model, n_steps):
    """Quadratic loss in the initial-pressure storage via _chunk_body."""
    record = model._artifacts.record
    stepper = model._stepper
    p_leaf = model._carry.state["p"].storage
    leaves, treedef = jax.tree_util.tree_flatten(model._carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is p_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        carry = jax.tree_util.tree_unflatten(treedef, new)
        state = _chunk_body(record, n_steps, carry, stepper).state
        return (jnp.sum(state["p"].data ** 2)
                + jnp.sum(state["u"].data ** 2)
                + jnp.sum(state["v"].data ** 2))

    return loss, p_leaf


def test_conserving_pv_divide_removes_padding_nan():
    """The f/h guard eliminates the ghost 0/0 without touching cells."""
    model = set_random(sphere_model("B"))
    state = model._carry.state
    u, v, p, c = state["u"], state["v"], state["p"], state["csqr"]
    f = state["f_coriolis"]
    h = c.to(p) + RO * p
    corner = u.function_space.bare.replace(
        lat=v.function_space.bare.factor("lat"))
    num, den = f.to(corner), h.to(corner)

    bare = num / den                           # the un-guarded quotient
    guarded = _safe_pv_divide(num, den)

    # the bug: exact zeros in the padded denominator -> NaN in storage
    assert int((np.asarray(den.storage) == 0.0).sum()) > 0
    assert bool(np.isnan(np.asarray(bare.storage)).any())
    # the fix: no NaN anywhere in the guarded storage
    assert not bool(np.isnan(np.asarray(guarded.storage)).any())
    # and the valid interior is bitwise identical (only padding changed)
    assert np.array_equal(np.asarray(bare.data),
                          np.asarray(guarded.data))


def test_conserving_rotation_ic_grad_is_finite_and_matches_fd():
    """Grad through a route-B walled-sphere run w.r.t. the IC: finite, FD."""
    model = set_random(sphere_model("B"))
    loss, p_leaf = ic_grad_loss(model, n_steps=6)

    grad = np.asarray(jax.grad(loss)(p_leaf))
    # the pre-seal bug NaNed every entry with a data path through the
    # conserving rotation; the f/h seal keeps them finite
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(p_leaf.shape),
                            dtype=p_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(p_leaf + eps * direction))
          - float(loss(p_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
