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
from fridom.model.params import CORIOLIS_F0

CSQR = 0.7
RO = 0.4
F0 = 1.0
OMEGA = 1.5
TWO_PI = float(2.0 * np.pi)
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
    return fr.spatial.Grid((mx, my))


def sphere_grid(nlon=16, nlat=8, radius=1.0):
    """Return the documented lat-lon sphere chart grid."""
    mlon = fr.spatial.meshes.IntervalMesh(nlon, (0.0, TWO_PI),
                                          name="lon")
    mlat = fr.spatial.meshes.IntervalMesh(
        nlat, (-LAT_MAX, LAT_MAX), periodic=False, name="lat")
    mapping = fr.spatial.CoordinateMapping(chart={
        "X": lambda lon, lat: (
            radius * jnp.cos(lat) * jnp.cos(lon),
            radius * jnp.cos(lat) * jnp.sin(lon),
            radius * jnp.sin(lat))}, orthogonal=True)
    return fr.spatial.Grid((mlon, mlat), mapping=mapping)


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
