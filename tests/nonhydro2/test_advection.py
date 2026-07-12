"""Tests for the nonhydro2 upwind-biased and WENO advection modules.

The load-bearing checks are the OLD-stack parity tests: the new
modules reproduce ``fr.model.modules.advection.UpwindAdvection`` /
``fr.model.modules.advection.WENO`` tendencies to machine precision on
both C-grid flux paths (tracer and velocity self-advection), both
bias sides, and both grounded orders. Physics sanity (zero tendency,
convergence order, ENO step transport, Rossby scaling, divergence
form) rides on top.
"""
from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.framework as frold
import fridom.nonhydro as nhold
from fridom.model.eigen import numeric_eigenpairs
from fridom.model.model import Model as FrModel
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.model.modules.moving_geometry import MovingGeometry
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
    _BiasedFaceReconstruction,
    _CenteredFaceInterpolation,
    _linear_row,
)
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.composed import Divergence
from fridom.spatial.operators.graded import biased_rows
from fridom.spatial.operators.movement import Sync
from fridom.spatial.spaces.nodal import NodeSet

L = 2 * np.pi
NY = 8
DT = 0.01


# ================================================================
#  Helpers
# ================================================================
def make_grid(nx, lx=L, ny=NY):
    return Grid((
        IntervalMesh(nx, (0.0, lx), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        IntervalMesh(ny, (0.0, L), name="z"),
    ))


def make_model(nx, advection, *, lx=L, stratified=True, ro=1.0):
    modules = [DynamicalCore(rossby_number=ro)]
    if stratified:
        modules.append(ConstantStratification(n2=1.0))
    modules.append(advection)
    return FrModel(grid=make_grid(nx, lx=lx),
                   modules=tuple(modules),
                   time_stepper=AdamBashforth(DT, order=3))


def centers(nx, lx=L):
    return (np.arange(nx) + 0.5) * (lx / nx)


def faces(nx, lx=L):
    return (np.arange(nx) + 1.0) * (lx / nx)


def broadcast(profile, nx):
    return profile[:, None, None] * np.ones((nx, NY, NY))


def advection_tendency(model, cls):
    return model.tendency(model.state, constraints=False,
                          filter=fr.model.term_predicates.owned_by(cls))


# ================================================================
#  Construction and validation
# ================================================================
def test_order_property_and_defaults():
    assert UpwindAdvection().order == 3
    assert WENOAdvection().order == 3
    assert UpwindAdvection(5).order == 5
    assert WENOAdvection(5).order == 5


@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
def test_unsupported_orders_are_rejected(cls):
    with pytest.raises(ValueError, match=r"orders \(3, 5\)"):
        cls(4)
    with pytest.raises(ValueError, match=r"orders \(3, 5\)"):
        cls(7)


def test_biased_operator_rejects_unknown_weighting():
    with pytest.raises(ValueError, match="weighting"):
        _BiasedFaceReconstruction(3, "left", "quadratic")


def test_biased_operator_properties_and_interning():
    op = _BiasedFaceReconstruction(5, "left", "weno")
    assert (op.order, op.bias, op.weighting) == (5, "left", "weno")
    assert op is _BiasedFaceReconstruction(5, "left", "weno")
    assert op is not _BiasedFaceReconstruction(5, "left", "linear")


def test_centered_interp_size_validation():
    assert _CenteredFaceInterpolation(4).size == 4
    with pytest.raises(ValueError, match=r"sizes \(2, 4\)"):
        _CenteredFaceInterpolation(3)


def test_face_codomain_signature():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")
    for op in (_BiasedFaceReconstruction(3, "left", "linear"),
               _CenteredFaceInterpolation(2)):
        assert op.codomain(mx.center) is mx.right
        assert op.codomain(mx.right) is mx.center
        with pytest.raises(SpaceMismatchError, match="nodal"):
            op.codomain(mx.cell_avg)
        with pytest.raises(SpaceMismatchError, match="nodal"):
            op.codomain(mx.center.as_complex())
        with pytest.raises(SpaceMismatchError, match="periodic-only"):
            op.codomain(my.center)


def test_face_codomain_rejects_other_node_sets():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    op = _BiasedFaceReconstruction(3, "left", "linear")
    # a periodic nodal factor outside the C-grid Center/Right pair
    with pytest.raises(SpaceMismatchError, match="flux positions"):
        op.codomain(mx.left)
    # average-family factors are not nodal at all
    with pytest.raises(SpaceMismatchError, match="nodal"):
        op.codomain(mx.face_avg)


@pytest.mark.parametrize(("order", "halo"), [(3, 2), (5, 3)])
def test_biased_operator_requirements(order, halo):
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    op = _BiasedFaceReconstruction(order, "left", "weno")
    assert op.requirements(mx.center).halo == halo
    interp = _CenteredFaceInterpolation(order - 1)
    assert interp.requirements(mx.center).halo == (order - 1) // 2


# ================================================================
#  OLD-stack numerics parity (the port contract)
# ================================================================
def old_advection_tendency(profile, u_profile, order, scheme,
                           n, lx, staggered=False):
    """Old-framework 1D advection tendency (true cells only)."""
    grid = frold.grid.cartesian.Grid(shape=(n,), domain_size=(lx,))
    mset = frold.ModelSettingsBase(grid=grid)
    mset.halo = 4
    mset = mset.setup()
    if scheme == "upwind":
        adv = frold.modules.advection.UpwindAdvection(order=order)
    else:
        adv = frold.modules.advection.WENO(order=order)
    adv = adv.setup(mset=mset)
    x = mset.grid.x_mesh[0]
    dx = lx / n
    if staggered:
        base = frold.ScalarField(mset, name="tmp")
        q = frold.ScalarField(mset, name="q",
                              position=base.position.shift(0))
        q.arr = profile(x + 0.5 * dx)
        u = frold.ScalarField(mset, name="u", position=q.position)
        u.arr = u_profile(x + 0.5 * dx)
    else:
        q = frold.ScalarField(mset, name="q")
        q.arr = profile(x)
        u = frold.ScalarField(mset, name="u",
                              position=q.position.shift(0))
        u.arr = u_profile(x + 0.5 * dx)
    velocity = frold.VectorField(mset, field_list=[u])
    return np.asarray(adv.advection(velocity, q).arr)[4:-4]


def test_old_stack_coefficient_parity():
    # the linear upwind rows are the old stack's biased
    # reconstruction coefficients (cell-average method)
    for order in (3, 5):
        coeffs = np.asarray(
            frold.grid.cartesian.compute_polynomial_coefficients(
                stencil_size=order, method="cell_average"))
        np.testing.assert_allclose(
            _linear_row(order, "left"), coeffs[order // 2 + 1],
            rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            _linear_row(order, "right"), coeffs[order // 2],
            rtol=0, atol=1e-14)


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize(
    ("scheme", "cls"),
    [pytest.param("upwind", UpwindAdvection, id="upwind"),
     pytest.param("weno", WENOAdvection, id="weno")])
def test_old_stack_tendency_parity(scheme, cls, order):
    # one new model; three old-stack references: tracer advection at
    # u = +-1.5 (both bias branches) and mixed-sign velocity
    # self-advection (the staggered Right -> Center flux path)
    n, lx = 16, 16.0
    xc, xf = centers(n, lx), faces(n, lx)
    ones = np.ones((n, NY, NY))
    prof = lambda x: (np.sin(2 * np.pi * x / lx)  # noqa: E731
                      + 0.3 * np.cos(4 * np.pi * x / lx + 0.7))
    model = make_model(n, cls(order), lx=lx)
    for u0 in (1.5, -1.5):
        model.set_fields(b=prof(xc)[:, None, None] * ones,
                         u=u0 * ones)
        new = np.asarray(
            advection_tendency(model, cls)["b"].data)[:, 0, 0]
        old = old_advection_tendency(
            prof, lambda x, u0=u0: u0 * np.ones_like(x),
            order, scheme, n, lx)
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-13)
    sprof = lambda x: (0.3  # noqa: E731
                       + 0.9 * np.sin(2 * np.pi * x / lx))
    model.set_fields(u=sprof(xf)[:, None, None] * ones,
                     b=np.zeros((n, NY, NY)))
    new = np.asarray(
        advection_tendency(model, cls)["u"].data)[:, 0, 0]
    old = old_advection_tendency(sprof, sprof, order, scheme,
                                 n, lx, staggered=True)
    np.testing.assert_allclose(new, old, rtol=0, atol=1e-13)


# ================================================================
#  Physics sanity
# ================================================================
@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
def test_constant_state_has_zero_tendency(cls):
    n = 8
    ones = np.ones((n, NY, NY))
    model = make_model(n, cls(3))
    model.set_fields(u=1.5 * ones, v=-0.5 * ones, w=0.25 * ones,
                     b=3.3 * ones)
    tau = advection_tendency(model, cls)
    for name in ("u", "v", "w", "b"):
        assert np.abs(np.asarray(tau[name].data)).max() < 1e-13


@pytest.mark.parametrize(
    ("cls", "order", "min_slope"),
    [pytest.param(UpwindAdvection, 3, 2.5, id="upwind3"),
     pytest.param(UpwindAdvection, 5, 4.4, id="upwind5"),
     pytest.param(WENOAdvection, 5, 4.4, id="weno5")])
def test_smooth_advection_converges_at_design_order(cls, order,
                                                    min_slope):
    # uniform velocity, b = sin(x): the advection tendency
    # approximates -U cos(x) at the scheme's design order (max
    # error away from the critical points of sin, where WENO-JS
    # is known to degrade)
    errors = []
    for n in (16, 32):
        model = make_model(n, cls(order))
        xc = centers(n)
        model.set_fields(u=np.ones((n, NY, NY)),
                         b=broadcast(np.sin(xc), n))
        got = np.asarray(
            advection_tendency(model, cls)["b"].data)[:, 0, 0]
        err = np.abs(got + np.cos(xc))
        mask = np.abs(np.cos(xc)) > 0.3
        errors.append(err[mask].max())
    assert np.log2(errors[0] / errors[1]) > min_slope


def test_weno3_converges_and_stays_essentially_third_order():
    # WENO3-JS with eps = 1e-10 degrades near critical points (a
    # known property, old-stack parity): loose convergence sanity
    errors = []
    for n in (16, 32):
        model = make_model(n, WENOAdvection(3))
        xc = centers(n)
        model.set_fields(u=np.ones((n, NY, NY)),
                         b=broadcast(np.sin(xc), n))
        got = np.asarray(advection_tendency(
            model, WENOAdvection)["b"].data)[:, 0, 0]
        err = np.abs(got + np.cos(xc))
        mask = np.abs(np.cos(xc)) > 0.3
        errors.append(err[mask].max())
    # measured ratio ~2.6 at (16, 32): converging, though below the
    # design order at these resolutions (WENO3-JS critical-point
    # degradation, identical in the old stack)
    assert errors[1] < errors[0] / 2.2


def test_weno_reduces_to_the_linear_upwind_row_on_smooth_data():
    # on smooth data the nonlinear weights approach the optimal
    # ones: the WENO tendency matches the linear upwind tendency
    # of the same order far more closely than the scheme error
    n = 32
    xc = centers(n)
    tendencies = {}
    for cls in (UpwindAdvection, WENOAdvection):
        model = make_model(n, cls(5))
        model.set_fields(u=np.ones((n, NY, NY)),
                         b=broadcast(np.sin(xc), n))
        tendencies[cls] = np.asarray(
            advection_tendency(model, cls)["b"].data)[:, 0, 0]
    scheme_err = np.abs(
        tendencies[UpwindAdvection] + np.cos(xc)).max()
    gap = np.abs(tendencies[WENOAdvection]
                 - tendencies[UpwindAdvection])
    mask = np.abs(np.cos(xc)) > 0.3
    assert gap[mask].max() < 40.0 * scheme_err  # same order of
    assert gap[mask].max() < 2e-3               # magnitude, small


def test_step_transport_no_amplified_overshoot():
    # advecting a step with uniform velocity: centered rings
    # (Gibbs), the upwind-biased scheme damps the ringing, WENO is
    # essentially non-oscillatory (measured: centered ~0.35,
    # upwind3 ~0.075, weno5 ~1e-5)
    nx = 64

    def overshoot(advection):
        model = FrModel(
            grid=make_grid(nx),
            modules=(DynamicalCore(), advection),
            time_stepper=AdamBashforth(0.02, order=3))
        xc = centers(nx)
        step = np.where((xc > L / 4) & (xc < 3 * L / 4), 1.0, 0.0)
        model.set_fields(u=np.ones((nx, NY, NY)),
                         w=broadcast(step, nx))
        model.advance(100)
        w = np.asarray(model.state["w"].data)
        assert np.isfinite(w).all()
        return max(w.max() - 1.0, -w.min())

    over_centered = overshoot(CenteredAdvection())
    over_upwind = overshoot(UpwindAdvection(3))
    over_weno = overshoot(WENOAdvection(5))
    assert over_centered > 0.1
    assert over_upwind < 0.5 * over_centered
    assert over_weno < 0.05 * over_centered
    assert over_weno < 1e-3


def test_rossby_scaling_and_divergence_form():
    n = 16
    xc = centers(n)
    taus = {}
    for ro in (1.0, 0.5):
        model = make_model(n, UpwindAdvection(3), ro=ro)
        model.set_fields(
            u=broadcast(1.0 + 0.5 * np.sin(xc), n),
            b=broadcast(np.sin(2 * xc), n))
        taus[ro] = advection_tendency(model, UpwindAdvection)
    for name in ("u", "b"):
        # the Rossby number scales the tendency exactly
        np.testing.assert_allclose(
            np.asarray(taus[0.5][name].data),
            0.5 * np.asarray(taus[1.0][name].data),
            rtol=0, atol=1e-15)
        # divergence form: the flux difference telescopes, so the
        # domain sum vanishes on the periodic grid
        data = np.asarray(taus[1.0][name].data)
        scale = max(np.abs(data).max(), 1.0) * data.size
        assert np.abs(data.sum()) < 1e-12 * scale


def test_second_advance_compiles_nothing(compile_counter):
    model = make_model(8, WENOAdvection(5))
    xc = centers(8)
    model.set_fields(u=np.ones((8, NY, NY)),
                     b=broadcast(0.1 * np.sin(xc), 8))
    model.advance(4)
    compile_counter.reset()
    model.advance(4)
    assert compile_counter.count == 0


# ================================================================
#  Walled grids: the biased schemes install their graded closure
# ================================================================
@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
@pytest.mark.parametrize("order", [3, 5])
def test_walled_grid_installs_the_graded_kernels(cls, order):
    # binding on a walled grid swaps every face kernel for its
    # boundary="graded" variant (the near-wall closure); a fully
    # periodic grid keeps the plain periodic-only kernels
    grid = Grid((
        IntervalMesh(8, (0.0, L), name="x"),
        IntervalMesh(8, (0.0, L), name="y"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="z"),
    ))
    module = cls(order)
    FrModel(grid=grid, modules=(DynamicalCore(), module),
            time_stepper=AdamBashforth(DT, order=3))
    assert module._walled == ("z",)
    for op in (module._left, module._right, module._lin_left,
               module._lin_right, module._interp):
        assert op.boundary == "graded"

    plain = cls(order)
    FrModel(grid=make_grid(8), modules=(DynamicalCore(), plain),
            time_stepper=AdamBashforth(DT, order=3))
    assert plain._walled == ()
    for op in (plain._left, plain._right, plain._lin_left,
               plain._lin_right, plain._interp):
        assert op.boundary == "none"


@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
def test_walled_axis_too_short_for_the_ladder_is_taught(cls):
    # the graded ladder needs order + 1 cells between the two walls
    grid = Grid((
        IntervalMesh(8, (0.0, L), name="x"),
        IntervalMesh(8, (0.0, L), name="y"),
        IntervalMesh(5, (0.0, 1.0), periodic=False, name="z"),
    ))
    with pytest.raises(NotImplementedError,
                       match=r"at least 6 cells on every walled axis"):
        FrModel(grid=grid,
                modules=(DynamicalCore(), cls(5)),
                time_stepper=AdamBashforth(DT, order=3))


# ================================================================
#  Walled grids: the centered scheme (structural-zero wall flux)
# ================================================================
WALLED_TOPOLOGIES = [
    pytest.param(("y",), id="channel-y"),
    pytest.param(("z",), id="rigid-lid"),
    pytest.param(("y", "z"), id="channel-and-lid"),
]

#: relative machine-precision level of the centered scheme's
#: quadratic invariants (measured ~1e-16 on all topologies)
CONSERVATION_TOL = 1e-13


def make_walled_model(walled, advection, n=8):
    """Build a small nonhydro model with walls on the given axes."""
    grid = Grid(tuple(
        IntervalMesh(n, (0.0, 1.0 if name in walled else L),
                     periodic=(name not in walled), name=name)
        for name in ("x", "y", "z")))
    return FrModel(
        grid=grid,
        modules=(DynamicalCore(), ConstantStratification(n2=1.0),
                 advection),
        time_stepper=AdamBashforth(DT, order=3))


def set_random_state(model, seed):
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(model.state[c].data.shape)
        for c in ("u", "v", "w", "b")})


def relative_energy_rates(model):
    """Per-component |<q, A(q)>| / sum|q A(q)| on a projected state."""
    state = model.constrain(model.state)
    tau = model.tendency(state, constraints=False,
                         filter=fr.model.term_predicates.owned_by(CenteredAdvection))
    rates = {}
    for c in ("u", "v", "w", "b"):
        product = (np.asarray(state[c].data)
                   * np.asarray(tau[c].data))
        rates[c] = abs(float(np.sum(product))) / float(
            np.sum(np.abs(product)))
    return state, tau, rates


@pytest.mark.parametrize("walled", WALLED_TOPOLOGIES)
def test_walled_energy_conservation_matches_the_periodic_level(
        walled):
    # the discrete quadratic invariant of the centered flux form on
    # a discretely divergence-free (projected) state: the PERIODIC
    # scheme conserves each component's <q, A(q)> to machine
    # precision, and the walled cases must sit at the same level
    # (per-component zeros make the statement independent of the
    # diagonal M-weighting) — the structural-zero wall flux adds no
    # boundary source
    periodic = make_walled_model((), CenteredAdvection())
    set_random_state(periodic, seed=11)
    _, _, base = relative_energy_rates(periodic)
    assert all(rate < CONSERVATION_TOL for rate in base.values())

    model = make_walled_model(walled, CenteredAdvection())
    set_random_state(model, seed=11)
    _, _, rates = relative_energy_rates(model)
    assert all(rate < CONSERVATION_TOL for rate in rates.values())


@pytest.mark.parametrize("walled", WALLED_TOPOLOGIES)
def test_walled_total_buoyancy_is_conserved(walled):
    # flux form with an exact-zero wall flux: the b tendency sums to
    # zero over the (uniform) cells to machine precision
    model = make_walled_model(walled, CenteredAdvection())
    set_random_state(model, seed=12)
    state = model.constrain(model.state)
    tau = model.tendency(state, constraints=False,
                         filter=fr.model.term_predicates.owned_by(CenteredAdvection))
    db = np.asarray(tau["b"].data)
    assert abs(float(np.sum(db))) < 1e-12 * float(
        np.sum(np.abs(db)))


@pytest.mark.parametrize("walled", WALLED_TOPOLOGIES)
def test_walled_projected_tendency_stays_divergence_free(walled):
    # impermeability: the advective tendency does not push flow
    # through the walls — after the CONSTRAINT stage the discrete
    # divergence sits at the walled-solver level everywhere
    model = make_walled_model(walled, CenteredAdvection())
    set_random_state(model, seed=13)
    state = model.constrain(model.state)
    tau = model.tendency(state, constraints=True)
    div = Divergence()(VectorField(
        {c: tau[c] for c in ("u", "v", "w")}))
    assert float(np.abs(np.asarray(div.data)).max()) < 1e-13


@pytest.mark.parametrize("walled", WALLED_TOPOLOGIES)
def test_walled_tendency_is_finite_on_a_random_state(walled):
    # NaN safety: no beyond-wall read survives into the tendency
    # (the only wall value consumed is the Dirichlet zero)
    model = make_walled_model(walled, CenteredAdvection())
    set_random_state(model, seed=14)
    tau = model.tendency(model.state, constraints=False,
                         filter=fr.model.term_predicates.owned_by(CenteredAdvection))
    assert all(np.isfinite(np.asarray(tau[c].data)).all()
               for c in ("u", "v", "w", "b"))


def test_periodic_tendency_is_bitwise_unchanged():
    # the walled support must not touch the periodic code path: the
    # module tendency equals the pre-walls flux loop (verbatim
    # below) BITWISE — the flux-space substitution and the retags
    # are interned identities on periodic axes
    module = CenteredAdvection()
    model = make_model(8, module)
    set_random_state(model, seed=15)
    state = model.state

    class Ctx:
        params: dict = {fr.model.params.SCALING_ROSSBY: 1.0}  # noqa: RUF012

    got = module._advect(state, Ctx)
    for qname in ("u", "v", "w", "b"):
        q = state[qname]
        res = None
        for axis, vname in module._axis_velocity:
            v = state[vname]
            flux_space = q.diff(axis).function_space
            v_face = v.to(flux_space)
            flux = v_face * q.to(flux_space)
            divergence = flux.diff(axis)
            res = -divergence if res is None else res - divergence
        want = 1.0 * res
        assert np.array_equal(np.asarray(got[qname].data),
                              np.asarray(want.data))


# ================================================================
#  Walled grids: the biased schemes (graded near-wall closure)
# ================================================================
#: (module class, order) of every walled biased configuration
BIASED = [
    pytest.param(UpwindAdvection, 3, id="upwind3"),
    pytest.param(UpwindAdvection, 5, id="upwind5"),
    pytest.param(WENOAdvection, 3, id="weno3"),
    pytest.param(WENOAdvection, 5, id="weno5"),
]


def walled_flux(module, model, qname, axis):
    """Rebuild the module's own flux for one advected component."""
    state = model.state
    q = state[qname]
    v = state[dict(module._axis_velocity)[axis]]
    flux_space = module._flux_space(q, v, axis)
    v_face = module._velocity_face(v, flux_space)
    flux = v_face * module._face_value(q, v_face, axis, flux_space)
    return flux, flux_space


@pytest.mark.parametrize(("cls", "order"), BIASED)
@pytest.mark.parametrize("walled", WALLED_TOPOLOGIES)
def test_walled_biased_tendency_is_finite(cls, order, walled):
    model = make_walled_model(walled, cls(order))
    set_random_state(model, seed=14)
    tau = model.tendency(model.state, constraints=False,
                         filter=fr.model.term_predicates.owned_by(cls))
    assert all(np.isfinite(np.asarray(tau[c].data)).all()
               for c in ("u", "v", "w", "b"))


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_walled_biased_wall_flux_is_exactly_zero(cls, order):
    # impermeability, EXACTLY (not to truncation): the flux space
    # adopts the wall-normal velocity's Dirichlet tag, so the wall
    # face is a boundary condition and not a DOF — its structured
    # fill is the exact zero the divergence closes on. The graded
    # rows only ever produce the INTERIOR faces, so no reduced-order
    # near-wall value can leak through the wall.
    module = cls(order)
    model = make_walled_model(("y",), module)
    set_random_state(model, seed=17)
    flux, flux_space = walled_flux(module, model, "b", "y")
    factor = flux_space.bare.factor("y")
    assert factor.node_set is NodeSet.INNER
    assert factor.bc.components == (BC.DIRICHLET, BC.DIRICHLET)

    synced = Sync()(flux)
    width = model.grid.decomposition.halo["y"]
    data = np.asarray(synced._data)
    left = data[:, width - 1, :]              # the y = 0 wall face
    right = data[:, width + factor.shape[0], :]   # the y = 1 wall
    assert (left == 0.0).all()
    assert (right == 0.0).all()


@pytest.mark.parametrize(("cls", "order"), BIASED)
@pytest.mark.parametrize("walled", WALLED_TOPOLOGIES)
def test_walled_biased_total_buoyancy_is_conserved(cls, order,
                                                   walled):
    # the flux form telescopes and the wall flux is an exact zero, so
    # the b tendency sums to zero over the (uniform) cells — at least
    # as well as the centered scheme manages on the same topology
    def drift(module):
        model = make_walled_model(walled, module)
        set_random_state(model, seed=12)
        state = model.constrain(model.state)
        tau = model.tendency(
            state, constraints=False,
            filter=fr.model.term_predicates.owned_by(type(module)))
        db = np.asarray(tau["b"].data)
        return abs(float(np.sum(db))) / float(np.sum(np.abs(db)))

    centered = drift(CenteredAdvection())
    biased = drift(cls(order))
    assert centered < CONSERVATION_TOL
    assert biased < CONSERVATION_TOL
    # "at least as good as CenteredAdvection achieves there"
    assert biased <= max(centered, 1e-15)


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_walled_biased_preserves_a_uniform_tracer(cls, order):
    # free-stream preservation on a walled grid: with a discretely
    # divergence-free (projected) velocity, a CONSTANT tracer has
    # zero tendency — every graded rung is exact on constants (the
    # rows sum to one; the wall cells the ladder synthesizes are
    # never multiplied into a tracer flux, because the wall-normal
    # velocity there is the impermeability zero)
    model = make_walled_model(("y", "z"), cls(order))
    set_random_state(model, seed=5)
    state = model.constrain(model.state)
    model.set_fields(**{c: np.asarray(state[c].data)
                        for c in ("u", "v", "w")})
    model.set_fields(b=np.full(model.state["b"].data.shape, 3.0))
    tau = model.tendency(model.state, constraints=False,
                         filter=fr.model.term_predicates.owned_by(cls))
    assert float(np.abs(np.asarray(tau["b"].data)).max()) < 1e-13


@pytest.mark.parametrize(("cls", "order"), BIASED)
@pytest.mark.parametrize("walled", WALLED_TOPOLOGIES)
def test_walled_biased_projected_tendency_stays_divergence_free(
        cls, order, walled):
    model = make_walled_model(walled, cls(order))
    set_random_state(model, seed=13)
    state = model.constrain(model.state)
    tau = model.tendency(state, constraints=True)
    div = Divergence()(VectorField(
        {c: tau[c] for c in ("u", "v", "w")}))
    assert float(np.abs(np.asarray(div.data)).max()) < 1e-13


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_walled_biased_model_run_stays_finite(cls, order):
    # stability smoke: a short nonlinear run on a doubly walled grid
    model = make_walled_model(("y", "z"), cls(order))
    set_random_state(model, seed=9)
    model.advance(20)
    for c in ("u", "v", "w", "b"):
        data = np.asarray(model.state[c].data)
        assert np.isfinite(data).all()
        # the upwind dissipation damps; nothing grows without bound
        assert np.abs(data).max() < 10.0


# ---- convergence on a walled axis -------------------------------
#: v(y) = sin(pi y) vanishes at both walls (impermeability); b is a
#: smooth tracer, so the y advection tendency of b is -d/dy (v b)
def _wall_profiles(y):
    return np.sin(np.pi * y)


def _tracer(y):
    return np.sin(2 * np.pi * y + 0.7)


def _flux_derivative(y):
    return -(np.pi * np.cos(np.pi * y) * _tracer(y)
             + np.sin(np.pi * y) * 2 * np.pi
             * np.cos(2 * np.pi * y + 0.7))


def _walled_tendency_error(cls, order, n):
    """Max |tendency - analytic| at the interior / all b cells."""
    grid = Grid((
        IntervalMesh(4, (0.0, L), name="x"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(4, (0.0, L), name="z"),
    ))
    module = cls(order)
    model = FrModel(
        grid=grid,
        modules=(DynamicalCore(), ConstantStratification(n2=1.0),
                 module),
        time_stepper=AdamBashforth(DT, order=3))
    yv = np.asarray(grid.evaluation_nodes(
        model.state["v"].function_space, "y").data).ravel()
    yb = np.asarray(grid.evaluation_nodes(
        model.state["b"].function_space, "y").data).ravel()
    model.set_fields(
        v=np.broadcast_to(_wall_profiles(yv)[None, :, None],
                          model.state["v"].data.shape).copy(),
        b=np.broadcast_to(_tracer(yb)[None, :, None],
                          model.state["b"].data.shape).copy())
    tau = model.tendency(model.state, constraints=False,
                         filter=fr.model.term_predicates.owned_by(cls))
    err = np.abs(np.asarray(tau["b"].data)[0, :, 0]
                 - _flux_derivative(yb))
    k = order // 2 + 2          # drop every cell a reduced rung feeds
    return err[k:err.size - k].max(), err.max()


#: minimum interior tendency rate per configuration. The ceiling here
#: is 2, NOT the design order, and that is a property of the nh flux
#: form, not of the closure: it multiplies an INTERPOLATED velocity by
#: a RECONSTRUCTED tracer, which is only 2nd-order accurate once the
#: velocity varies in space — the periodic scheme measures exactly the
#: same 2.0 on this problem. The design order of the graded rows
#: themselves is pinned on the FD-flux difference they actually feed,
#: in test_graded_rows_keep_the_design_order_in_the_interior.
#: WENO3-JS degrades near critical points (old-stack parity, see
#: test_weno3_converges_and_stays_essentially_third_order), so it
#: reaches the plateau later: measured 1.01 -> 1.81 over (64, 128, 256).
INTERIOR_RATE = {
    (UpwindAdvection, 3): 1.8,   # measured 1.99
    (UpwindAdvection, 5): 1.8,   # measured 2.00
    (WENOAdvection, 3): 0.9,     # measured 1.01 (critical points)
    (WENOAdvection, 5): 1.8,     # measured 2.00
}


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_walled_biased_interior_tendency_converges(cls, order):
    # the graded closure costs nothing in the interior: away from the
    # K reduced faces the walled tendency converges at the same rate
    # the periodic scheme reaches on this problem (see INTERIOR_RATE)
    sizes = (64, 128, 256)
    errs = [_walled_tendency_error(cls, order, n)[0] for n in sizes]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert min(rates) > INTERIOR_RATE[(cls, order)]


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_walled_biased_global_error_converges_at_first_order(
        cls, order):
    # the documented price of the BC-free graded closure: the K
    # near-wall faces per side drop to their reduced rungs, and in the
    # FD-flux form an O(h^p) flux error becomes an O(h^(p-1))
    # tendency error there. The wall-adjacent rung is 1st-order
    # upwind, and the wall-normal velocity vanishes linearly at the
    # wall (impermeability), so the max-norm tendency error over the
    # WHOLE walled axis converges at ~1 — measured 1.01-1.12. It
    # converges monotonically; it neither stalls nor blows up.
    sizes = (64, 128, 256)
    errs = [_walled_tendency_error(cls, order, n)[1] for n in sizes]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(fine < coarse for coarse, fine in pairwise(errs))
    assert min(rates) > 0.9


# ---- the graded rows in isolation (design order, no exterior read)
def _graded_flux_difference_rates(order, bias, weighting, node_set):
    """Interior convergence of d/dy of the graded reconstruction."""
    op = _BiasedFaceReconstruction(order, bias, weighting, "graded")
    shift = 1 if node_set is NodeSet.INNER else 0
    k = biased_rows(order, shift)
    errs = []
    sizes = (64, 128, 256)
    for n in sizes:
        mesh = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
        grid = Grid((mesh,), device_ids=(0,))
        grid.negotiate(halo=HaloSpec({"y": order // 2 + 1}))
        src = (mesh.center if node_set is NodeSet.CENTER
               else mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
        f = grid.create_field(src, init=_smooth)
        face = np.asarray(op["y"](f).data)
        dy = 1.0 / n
        deriv = (face[1:] - face[:-1]) / dy
        y = np.asarray(grid.evaluation_nodes(
            op["y"](f).function_space).data)
        mid = 0.5 * (y[1:] + y[:-1])
        err = np.abs(deriv - _smooth_prime(mid))
        keep = np.zeros_like(mid, dtype=bool)
        keep[k: mid.size - k] = True
        # mask the critical points of the derivative (WENO-JS is
        # known to degrade there — old-stack parity, see the periodic
        # weno3 test above)
        keep &= (np.abs(_smooth_prime(mid))
                 > 0.3 * np.abs(_smooth_prime(mid)).max())
        errs.append(err[keep].max())
    return [np.log2(errs[i] / errs[i + 1]) for i in range(2)]


def _smooth(y):
    # vanishes at both walls, so the Inner (Dirichlet) direction is a
    # legal operand; jnp so it also serves as a field initializer
    return jnp.sin(jnp.pi * y) * (1.0 + 0.4 * jnp.cos(3 * jnp.pi * y))


def _smooth_prime(y):
    return (np.pi * np.cos(np.pi * y)
            * (1.0 + 0.4 * np.cos(3 * np.pi * y))
            - 1.2 * np.pi * np.sin(np.pi * y)
            * np.sin(3 * np.pi * y))


@pytest.mark.parametrize("node_set", [NodeSet.CENTER, NodeSet.INNER])
@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("order", [3, 5])
def test_graded_rows_keep_the_design_order_in_the_interior(
        order, bias, node_set):
    # the load-bearing accuracy gate of the closure: the FD-flux
    # quantity the scheme consumes — the DIFFERENCE of two graded
    # face values — converges at the design order at the interior
    # faces of a WALLED axis, on both C-grid directions
    # (Center -> Inner and the dual Inner -> Center) and both biases.
    # Measured: 3.00 and 5.00 for the linear rows.
    rates = _graded_flux_difference_rates(order, bias, "linear",
                                          node_set)
    assert min(rates) > order - 0.2


@pytest.mark.parametrize("node_set", [NodeSet.CENTER, NodeSet.INNER])
@pytest.mark.parametrize("bias", ["left", "right"])
def test_graded_weno5_rows_keep_the_design_order_in_the_interior(
        bias, node_set):
    rates = _graded_flux_difference_rates(5, bias, "weno", node_set)
    assert min(rates) > 4.5


@pytest.mark.parametrize("node_set", [NodeSet.CENTER, NodeSet.INNER])
@pytest.mark.parametrize("bias", ["left", "right"])
def test_graded_weno3_rows_converge_essentially_third_order(
        bias, node_set):
    # WENO3-JS degrades near critical points (a known property, old-
    # stack parity — see test_weno3_converges_and_stays_essentially_
    # third_order): the graded rows inherit exactly that, nothing more
    rates = _graded_flux_difference_rates(3, bias, "weno", node_set)
    assert min(rates) > 2.0


@pytest.mark.parametrize("weighting", ["linear", "weno"])
@pytest.mark.parametrize("node_set", [NodeSet.CENTER, NodeSet.INNER])
@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("order", [3, 5])
def test_graded_rows_read_no_exterior_value(order, bias, weighting,
                                            node_set):
    # NaN-poison gate (the Fallback idiom): poison EVERY ghost slot —
    # including, on the Dirichlet Inner operand, the wall slot itself
    # — and claim the ghosts valid so the consumption-side sync leaves
    # them. The graded output must stay finite: the ladder reads only
    # true DOFs and SYNTHESIZES the exact-zero wall values, so no
    # exterior (or even ghost) read survives. The plain kernel would
    # poison the whole near-wall output.
    n = 16
    mesh = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
    grid = Grid((mesh,), device_ids=(0,))
    width = order // 2 + 1
    grid.negotiate(halo=HaloSpec({"y": width}))
    src = (mesh.center if node_set is NodeSet.CENTER
           else mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    f = grid.create_field(src, init=_smooth)
    storage = f._data
    poisoned = storage.at[:width].set(jnp.nan)
    poisoned = poisoned.at[storage.shape[0] - width:].set(jnp.nan)
    f._data = poisoned
    f._halo_valid = HaloSpec({"y": width})

    op = _BiasedFaceReconstruction(order, bias, weighting, "graded")
    result = op["y"](f)
    assert result.function_space.bare is (
        mesh.inner if node_set is NodeSet.CENTER else mesh.center)
    assert bool(jnp.all(jnp.isfinite(result.data)))


@pytest.mark.parametrize("node_set", [NodeSet.CENTER, NodeSet.INNER])
@pytest.mark.parametrize("size", [2, 4])
def test_graded_velocity_interpolation_reads_no_exterior_value(
        size, node_set):
    # the same gate for the order-coupled velocity interpolation (the
    # other kernel the walled biased schemes apply on a walled axis)
    n = 16
    mesh = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
    grid = Grid((mesh,), device_ids=(0,))
    width = max(size // 2, 1)
    grid.negotiate(halo=HaloSpec({"y": width}))
    src = (mesh.center if node_set is NodeSet.CENTER
           else mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    f = grid.create_field(src, init=_smooth)
    storage = f._data
    poisoned = storage.at[:width].set(jnp.nan)
    poisoned = poisoned.at[storage.shape[0] - width:].set(jnp.nan)
    f._data = poisoned
    f._halo_valid = HaloSpec({"y": width})

    op = _CenteredFaceInterpolation(size, "graded")
    assert bool(jnp.all(jnp.isfinite(op["y"](f).data)))


# ---- the boundary knob ------------------------------------------
@pytest.mark.parametrize(
    "make", [lambda b: _BiasedFaceReconstruction(3, "left",
                                                 "linear", b),
             lambda b: _CenteredFaceInterpolation(2, b)])
def test_unknown_boundary_variant_is_taught(make):
    with pytest.raises(ValueError, match=r"boundary must be one of"):
        make("one_sided")


@pytest.mark.parametrize(
    "make", [lambda b: _BiasedFaceReconstruction(3, "left",
                                                 "linear", b),
             lambda b: _CenteredFaceInterpolation(2, b)])
def test_boundary_variant_is_part_of_the_intern_key(make):
    assert make("none") is make("none")
    assert make("graded") is make("graded")
    assert make("none") is not make("graded")
    assert make("none").boundary == "none"
    assert make("graded").boundary == "graded"


def test_plain_kernel_has_no_bounded_signature():
    mesh = IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")
    op = _BiasedFaceReconstruction(3, "left", "linear")
    with pytest.raises(SpaceMismatchError,
                       match=r"periodic-only in its boundary='none'"):
        op.codomain(mesh.center)


def test_graded_kernel_needs_a_dirichlet_wall_on_the_inner_operand():
    # the Inner -> Center direction reaches the wall face and reads
    # its exact zero: only a homogeneous-Dirichlet tag defines one
    mesh = IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")
    op = _BiasedFaceReconstruction(3, "left", "linear", "graded")
    assert op.codomain(mesh.center) is mesh.inner
    assert op.codomain(
        mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)) is mesh.center
    with pytest.raises(SpaceMismatchError,
                       match=r"homogeneous-Dirichlet boundary values"):
        op.codomain(mesh.inner)


def test_graded_kernel_rejects_an_outer_operand():
    mesh = IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")
    op = _BiasedFaceReconstruction(3, "left", "linear", "graded")
    with pytest.raises(SpaceMismatchError,
                       match=r"Center -> Inner and Inner -> Center"):
        op.codomain(mesh.outer)


def test_walled_background_terms_run_and_telescope():
    # a tangential background on the walled channel: both terms
    # evaluate finite and their sum telescopes to the single-pass
    # full-velocity scheme through the module's own hooks
    module = CenteredAdvection(
        background={"u": lambda y: 1.0 + 0.5 * np.sin(np.pi * y),
                    "w": 0.0})
    model = make_walled_model(("y",), module)
    set_random_state(model, seed=16)
    state = model.state
    total = model.tendency(
        state, constraints=False,
        filter=fr.model.term_predicates.owned_by(CenteredAdvection))

    for qname in ("u", "v", "w", "b"):
        q = state[qname]
        res = None
        for axis, vname in module._axis_velocity:
            v = 1.0 * state[vname]
            sample = module._background_by_axis.get(axis)
            if sample is not None:
                v = v + state[sample]
            flux_space = module._flux_space(q, v, axis)
            v_face = module._velocity_face(v, flux_space)
            flux = v_face * module._face_value(
                q, v_face, axis, flux_space)
            divergence = flux.diff(axis).retag(q)
            res = -divergence if res is None else res - divergence
        assert np.isfinite(np.asarray(res.data)).all()
        np.testing.assert_allclose(
            np.asarray(total[qname].data), np.asarray(res.data),
            rtol=0, atol=1e-13)


def test_walled_background_wall_normal_must_vanish():
    # impermeability is a taught bind error on the user's input: a
    # nonzero wall-normal component (constant or callable) cannot
    # ride the structurally impermeable sample
    with pytest.raises(ValueError,
                       match=r"background\['v'\] does not vanish "
                             r"at the 'y' wall"):
        make_walled_model(
            ("y",), CenteredAdvection(background={"v": 0.3}))
    with pytest.raises(ValueError,
                       match=r"background\['w'\] does not vanish "
                             r"at the 'z' wall"):
        make_walled_model(
            ("z",), CenteredAdvection(
                background={"w": lambda z: np.cos(2.0 * z)}))
    # a callable that does not name the wall coordinate cannot
    # vanish there either (it is constant along the wall normal)
    with pytest.raises(ValueError,
                       match=r"background\['v'\] does not vanish "
                             r"at the 'y' wall"):
        make_walled_model(
            ("y",), CenteredAdvection(
                background={"v": lambda x: np.sin(x)}))  # noqa: PLW0108 — must name a coordinate
    # a wall-normal profile that vanishes on the walls is accepted
    make_walled_model(
        ("z",), CenteredAdvection(
            background={"w": lambda z: np.sin(np.pi * z)}))


# ================================================================
#  Prescribed background flow
# ================================================================
def u_shear(y):
    """Mixed-sign background shear profile along the periodic y."""
    return 1.0 + 1.4 * np.sin(y)


W_BG = -0.7


def make_background(cls, order):
    """One module of the family with the shared test background."""
    background = {"u": u_shear, "w": W_BG}
    if cls is CenteredAdvection:
        return cls(background=background)
    return cls(order, background=background)


def set_perturbation(model, nx, lx=L):
    """Smooth x-varying perturbation on every prognostic field."""
    ones = np.ones((nx, NY, NY))
    xc, xf = centers(nx, lx), faces(nx, lx)
    k = 2 * np.pi / lx
    model.set_fields(
        u=0.4 * np.sin(k * xf)[:, None, None] * ones,
        v=0.2 * np.cos(k * xc)[:, None, None] * ones,
        w=0.1 * np.sin(2 * k * xc)[:, None, None] * ones,
        b=(np.sin(k * xc)
           + 0.3 * np.cos(2 * k * xc + 0.7))[:, None, None] * ones)


def linear_term_tendency(model, cls):
    return model.tendency(
        model.state, constraints=False,
        filter=fr.model.term_predicates.named(f"{cls.__name__}/background_advection"))


# ----------------------------------------------------------------
#  Construction, declarations and sampling
# ----------------------------------------------------------------
def test_background_rejects_unknown_components():
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['q'\]"):
        CenteredAdvection(background={"q": 1.0})


def test_background_rejects_non_numeric_values():
    with pytest.raises(TypeError, match="coordinate callable"):
        UpwindAdvection(3, background={"u": "fast"})


def test_background_term_registration():
    # background=None (and the empty mapping): the single term,
    # exactly as before; with a background: the difference split
    # with a genuinely separate linear term (V-S3)
    for module in (WENOAdvection(3), WENOAdvection(3, background={})):
        terms = module.tendency_terms()
        assert [t.name for t in terms] == ["advection"]
        assert not terms[0].linear
        assert module.field_declarations == ()
        assert module.field_references == ()
    terms = WENOAdvection(3, background={"u": 1.0}).tendency_terms()
    assert [t.name for t in terms] == ["advection",
                                       "background_advection"]
    assert [t.linear for t in terms] == [False, True]
    assert terms[0].advances == terms[1].advances
    assert terms[0].transports == terms[1].transports


def test_background_samples_at_each_components_own_nodes():
    n = 8
    module = CenteredAdvection(background={"u": u_shear, "w": 0.25})
    model = make_model(n, module)
    state = model.state
    # u's background profile u_shear(y) sampled at u's own nodes:
    # staggered along x, cell centers along y
    yc = centers(NY)
    got = np.asarray(state["background_u"].data)
    np.testing.assert_allclose(got[0, :, 0], u_shear(yc),
                               rtol=0, atol=1e-15)
    assert (state["background_u"].function_space
            is state["u"].function_space)
    # the constant fills w's own space; no sample for unmapped v
    assert np.asarray(state["background_w"].data) == pytest.approx(
        0.25)
    assert (state["background_w"].function_space
            is state["w"].function_space)
    declared = {d.name for d in module.field_declarations}
    assert declared == {"background_u", "background_w"}


def test_background_profile_names_unknown_coordinate():
    with pytest.raises(ValueError,
                       match=r"coordinate\(s\) \['r'\], which the "
                             r"grid does not have"):
        make_model(8, CenteredAdvection(
            background={"u": lambda r: r}))


def test_background_rides_the_walled_biased_scheme():
    # the background split works on a walled grid too: the linear
    # (optimal-weight) row of the background term picks up the same
    # graded closure, and the tangential background needs no wall
    # value (the wall-normal one is still checked for impermeability)
    grid = Grid((
        IntervalMesh(8, (0.0, L), name="x"),
        IntervalMesh(8, (0.0, L), name="y"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="z"),
    ))
    module = UpwindAdvection(3, background={"u": 1.0})
    model = FrModel(grid=grid,
                    modules=(DynamicalCore(),
                             ConstantStratification(n2=1.0), module),
                    time_stepper=AdamBashforth(DT, order=3))
    assert module._lin_left.boundary == "graded"
    assert {t.name for t in module.tendency_terms()} == {
        "advection", "background_advection"}
    tau = model.tendency(model.state, constraints=False,
                         filter=fr.model.term_predicates.owned_by(
                             UpwindAdvection))
    assert all(np.isfinite(np.asarray(tau[c].data)).all()
               for c in ("u", "v", "w", "b"))


def test_wall_normal_background_must_still_vanish_on_the_wall():
    # impermeability is unchanged by the graded closure
    grid = Grid((
        IntervalMesh(8, (0.0, L), name="x"),
        IntervalMesh(8, (0.0, L), name="y"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="z"),
    ))
    with pytest.raises(ValueError, match=r"does not vanish at the"):
        FrModel(grid=grid,
                modules=(DynamicalCore(),
                         UpwindAdvection(3, background={"w": 1.0})),
                time_stepper=AdamBashforth(DT, order=3))


def test_background_off_c_grid_staggering_is_taught():
    # a core declaring "u" as the y-velocity: the background sample
    # (declared on the x-staggered nh pattern) cannot ride it
    class _SwappedCore(fr.model.Module):
        field_declarations = (
            fr.model.FieldDeclaration.velocity(
                "u", "y", space=fr.spatial.Staggered("y"), units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "v", "x", space=fr.spatial.Staggered("x"), units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "w", "z", space=fr.spatial.Staggered("z"), units="m/s"),
        )

    with pytest.raises(ValueError, match="nh C-grid staggering"):
        FrModel(grid=make_grid(8),
                modules=(_SwappedCore(),
                         CenteredAdvection(background={"u": 1.0})),
                time_stepper=AdamBashforth(DT, order=3))


# ----------------------------------------------------------------
#  The difference-form split: telescoping, linearity, reduction
# ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("cls", "order"),
    [pytest.param(CenteredAdvection, None, id="centered"),
     pytest.param(UpwindAdvection, 3, id="upwind3"),
     pytest.param(UpwindAdvection, 5, id="upwind5"),
     pytest.param(WENOAdvection, 3, id="weno3"),
     pytest.param(WENOAdvection, 5, id="weno5")])
def test_background_split_telescopes_to_the_full_scheme(cls, order):
    # the two terms sum to S_full(U + Ro u', q): checked against an
    # unsplit single-pass evaluation through the module's own hooks
    # (no subtraction), and — for the tracer row, where an honest
    # independent reference exists — against a background=None
    # module at Ro=1 whose velocity data is U + Ro u'
    n, ro = 16, 0.5
    module = make_background(cls, order)
    model = make_model(n, module, ro=ro)
    set_perturbation(model, n)
    state = model.state
    total = advection_tendency(model, cls)

    for qname in ("u", "v", "w", "b"):
        q = state[qname]
        res = None
        for axis, vname in module._axis_velocity:
            v = ro * state[vname]
            sample = module._background_by_axis.get(axis)
            if sample is not None:
                v = v + state[sample]
            flux_space = q.diff(axis).function_space
            v_face = module._velocity_face(v, flux_space)
            flux = v_face * module._face_value(
                q, v_face, axis, flux_space)
            divergence = flux.diff(axis)
            res = -divergence if res is None else res - divergence
        np.testing.assert_allclose(
            np.asarray(total[qname].data), np.asarray(res.data),
            rtol=0, atol=1e-13)

    # independent tracer-row reference: an unsplit background=None
    # module at Ro=1 with the full velocity U + ro u' as data
    reference = make_model(
        n, cls() if order is None else cls(order), ro=1.0)
    full = {name: ro * np.asarray(state[name].data) for name
            in ("u", "v", "w")}
    full["u"] = full["u"] + np.asarray(state["background_u"].data)
    full["w"] = full["w"] + np.asarray(state["background_w"].data)
    reference.set_fields(b=np.asarray(state["b"].data), **full)
    ref = advection_tendency(reference, cls)
    np.testing.assert_allclose(
        np.asarray(total["b"].data), np.asarray(ref["b"].data),
        rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    ("cls", "order", "background"),
    [pytest.param(CenteredAdvection, None, {"u": u_shear, "w": W_BG},
                  id="centered-shear"),
     pytest.param(UpwindAdvection, 3, {"u": u_shear, "w": W_BG},
                  id="upwind3-shear"),
     pytest.param(WENOAdvection, 5, {"u": u_shear, "w": W_BG},
                  id="weno5-shear"),
     pytest.param(WENOAdvection, 3, {"u": 1.3}, id="weno3-const")])
def test_background_term_is_exactly_linear(cls, order, background):
    # L(a q1 + b q2) = a L(q1) + b L(q2) to machine precision — for
    # the WENO module too (its linear term applies the linear
    # optimal-weight row, never the state-dependent WENO weights),
    # and for a background sheared along the periodic y (test 5 of
    # the split design; eigen analysis is deliberately N/A there:
    # a sheared U makes the operator non-normal, so the channel
    # engine's Hermiticity guard refuses such models by design)
    n = 16
    module = (cls(background=background) if order is None
              else cls(order, background=background))
    model = make_model(n, module)
    xc, xf = centers(n), faces(n)
    ones = np.ones((n, NY, NY))
    z1 = {"u": 0.4 * np.sin(xf)[:, None, None] * ones,
          "b": np.sin(xc)[:, None, None] * ones}
    z2 = {"u": 0.3 * np.cos(2 * xf)[:, None, None] * ones,
          "b": np.cos(3 * xc + 0.4)[:, None, None] * ones}
    a, b = 1.7, -0.6
    z3 = {name: a * z1[name] + b * z2[name] for name in z1}

    tendencies = []
    for fields in (z1, z2, z3):
        model.set_fields(**fields)
        tendencies.append(linear_term_tendency(model, cls))
    for name in ("u", "b"):
        combined = (a * np.asarray(tendencies[0][name].data)
                    + b * np.asarray(tendencies[1][name].data))
        np.testing.assert_allclose(
            np.asarray(tendencies[2][name].data), combined,
            rtol=0, atol=1e-13)


def test_background_term_ignores_the_perturbation_velocity():
    # the advecting velocity of L is the static background alone:
    # changing u' must not move L's tracer row (bitwise)
    n = 16
    model = make_model(n, UpwindAdvection(
        3, background={"u": u_shear, "w": W_BG}))
    xc = centers(n)
    ones = np.ones((n, NY, NY))
    b0 = np.sin(xc)[:, None, None] * ones
    model.set_fields(u=0.0 * ones, b=b0)
    first = linear_term_tendency(model, UpwindAdvection)
    model.set_fields(u=-3.7 * ones, v=1.1 * ones, b=b0)
    second = linear_term_tendency(model, UpwindAdvection)
    assert np.array_equal(np.asarray(first["b"].data),
                          np.asarray(second["b"].data))


def test_background_none_reduction_is_bitwise():
    # background=None takes the literally unchanged single-term code
    # path (same jit compilation), and a zero background telescopes
    # to it exactly: S_full(0 + Ro u') == Ro S_full(u') bitwise here
    # (the velocity enters the flux linearly, upwind selection is
    # invariant under positive scaling, and the zero linear term
    # vanishes identically)
    n = 16
    models = {
        "default": make_model(n, UpwindAdvection(3), ro=0.5),
        "explicit": make_model(
            n, UpwindAdvection(3, background=None), ro=0.5),
        "zero": make_model(n, UpwindAdvection(
            3, background={"u": 0.0, "w": 0.0}), ro=0.5),
    }
    taus = {}
    for key, model in models.items():
        set_perturbation(model, n)
        taus[key] = advection_tendency(model, UpwindAdvection)
    for name in ("u", "v", "w", "b"):
        expected = np.asarray(taus["default"][name].data)
        assert np.array_equal(
            np.asarray(taus["explicit"][name].data), expected)
        assert np.array_equal(
            np.asarray(taus["zero"][name].data), expected)


# ----------------------------------------------------------------
#  Linearization and the Doppler-shifted spectrum
# ----------------------------------------------------------------
def test_linearize_keeps_l_and_drops_n():
    n = 16
    model = make_model(n, WENOAdvection(
        3, background={"u": u_shear, "w": W_BG}))
    set_perturbation(model, n)
    state = model.state
    linear = fr.model.linearize(model)

    # the linear variant's whole advection contribution is exactly
    # the parent's background_advection term ...
    kept = linear.tendency(state, constraints=False,
                           filter=fr.model.term_predicates.owned_by(WENOAdvection))
    parent_l = linear_term_tendency(model, WENOAdvection)
    dropped = linear.tendency(
        state, constraints=False,
        filter=fr.model.term_predicates.named("WENOAdvection/advection"))
    total = advection_tendency(model, WENOAdvection)
    for name in ("u", "v", "w", "b"):
        assert np.array_equal(np.asarray(kept[name].data),
                              np.asarray(parent_l[name].data))
        # ... the nonlinear term is dropped from its schedule ...
        assert np.abs(np.asarray(dropped[name].data)).max() == 0.0
    # ... and the dropped piece is genuinely nonzero in the parent
    assert any(
        np.abs(np.asarray(total[name].data)
               - np.asarray(parent_l[name].data)).max() > 1e-8
        for name in ("u", "v", "w", "b"))


def test_constant_background_doppler_shifts_the_spectrum():
    # a constant background U on the fully periodic grid adds the
    # discrete advection symbol to every physical branch: probe the
    # linearized eigenvalues and check the frequency shift against
    # the symbol computed from the module's own linear term (an
    # impulse response, never a hardcoded formula)
    n = 8

    def build(background):
        grid = Grid(tuple(
            IntervalMesh(n, (0.0, L), name=name)
            for name in ("x", "y", "z")))
        return FrModel(
            grid=grid,
            modules=(DynamicalCore(dsqr=1.0, rossby_number=1.0),
                     FPlaneCoriolis(f0=1.0),
                     ConstantStratification(n2=1.0),
                     CenteredAdvection(background=background)),
            time_stepper=AdamBashforth(0.02, order=3))

    omega0 = np.asarray(numeric_eigenpairs(build(None)).omega)
    model = build({"u": 1.3})
    omega_bg = np.asarray(numeric_eigenpairs(model).omega)

    # the discrete symbol of L from its own impulse response:
    # sigma(k) = FFT(L delta)(k), L q = -i omega q -> the frequency
    # shift is -Im sigma
    delta = np.zeros((n, n, n))
    delta[0, 0, 0] = 1.0
    model.set_fields(b=delta)
    response = linear_term_tendency(model, CenteredAdvection)
    symbol = np.fft.fftn(np.asarray(response["b"].data))
    assert np.abs(symbol.real).max() < 1e-12
    shift = symbol.imag

    # per mode: the three physical branches shift by Im sigma(k),
    # the constraint (divergence-free complement) zero stays exact;
    # the k = 0 mean carries no constraint zero (inertial +-f pairs)
    # and its shift vanishes anyway — excluded, the eigen-test
    # convention
    flat0 = omega0.reshape(-1, 4)
    flat_bg = omega_bg.reshape(-1, 4)
    flat_shift = shift.reshape(-1)
    errors = []
    for row0, sigma, row_bg in zip(flat0, flat_shift, flat_bg,
                                   strict=True):
        physical = np.delete(row0, np.argmin(np.abs(row0)))
        expected = np.sort(np.append(physical - sigma, 0.0))
        errors.append(np.abs(np.sort(row_bg) - expected).max())
    errors = np.asarray(errors).reshape(n, n, n)
    errors[0, 0, 0] = 0.0  # the k = 0 mean (see above)
    assert errors.max() < 1e-9


# ----------------------------------------------------------------
#  OLD-stack background parity (Ro = 1 sidesteps the convention)
# ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("scheme", "cls", "order"),
    [pytest.param("upwind", UpwindAdvection, 3, id="upwind3"),
     pytest.param("weno", WENOAdvection, 5, id="weno5")])
def test_old_stack_background_parity_at_ro_one(scheme, cls, order):
    # the old stack computed Ro * S(u' + U, q) (its scaling factor
    # multiplied the background too); at Ro = 1 both conventions
    # coincide with S_full(U + u', q), so the old advect_state with
    # `background` set is an exact reference for the two-term sum
    n, lx = 16, 16.0
    k = 2 * np.pi / lx

    def u_prof(x):
        return 0.4 * np.sin(k * x)

    def v_prof(x):
        return 0.2 * np.cos(k * x)

    def w_prof(x):
        return 0.1 * np.sin(2 * k * x)

    def b_prof(x):
        return np.sin(k * x) + 0.3 * np.cos(2 * k * x + 0.7)

    # --- the old stack: nh ModelSettings + advect_state ---
    grid_old = nhold.grid.cartesian.Grid(
        shape=(n, NY, NY), domain_size=(lx, L, L))
    mset = nhold.ModelSettings(grid=grid_old, rossby_number=1.0)
    if scheme == "upwind":
        mset.tendencies.advection = (
            frold.modules.advection.UpwindAdvection(order=order))
    else:
        mset.tendencies.advection = (
            frold.modules.advection.WENO(order=order))
    mset.halo = 4
    mset = mset.setup()
    adv = mset.tendencies.advection
    assert adv.scaling == 1.0
    x_mesh, y_mesh, z_mesh = mset.grid.x_mesh
    dx = lx / n
    z = nhold.State(mset)
    z.u.arr = u_prof(x_mesh + 0.5 * dx)
    z.v.arr = v_prof(x_mesh)
    z.w.arr = w_prof(x_mesh)
    z.b.arr = b_prof(x_mesh)
    z.sync()
    background = nhold.State(mset)
    background.u.arr = u_shear(y_mesh)
    background.w.arr = W_BG * np.ones_like(z_mesh)
    background.sync()
    adv.background = background
    dz = adv.advect_state(z, nhold.State(mset))
    interior = (slice(4, -4),) * 3

    # --- the new stack: the two-term sum at Ro = 1 ---
    module = make_background(cls, order)
    model = make_model(n, module, lx=lx, ro=1.0)
    ones = np.ones((n, NY, NY))
    xc, xf = centers(n, lx), faces(n, lx)
    model.set_fields(u=u_prof(xf)[:, None, None] * ones,
                     v=v_prof(xc)[:, None, None] * ones,
                     w=w_prof(xc)[:, None, None] * ones,
                     b=b_prof(xc)[:, None, None] * ones)
    tau = advection_tendency(model, cls)
    for name in ("u", "v", "w", "b"):
        np.testing.assert_allclose(
            np.asarray(tau[name].data),
            np.asarray(dz[name].arr)[interior],
            rtol=0, atol=1e-13)


# ================================================================
#  Mapped grids: the physical flux divergence (stage C4)
# ================================================================
H0 = 0.7


def depth(x):
    """Smooth periodic water depth H(x) (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(x)


def make_mapped_grid(n, init=depth, ny=4, periodic_column=False):
    """Terrain-following grid ``zp = z * H(x)`` (z in [0, 1])."""
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": init})
    return Grid((
        IntervalMesh(n, (0.0, L), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=periodic_column,
                     name="z"),
    ), mapping=mapping)


def make_mapped_model(n, advection, *, init=depth, ny=4):
    return FrModel(
        grid=make_mapped_grid(n, init=init, ny=ny),
        modules=(DynamicalCore(),
                 ConstantStratification(n2=0.0),
                 advection),
        time_stepper=AdamBashforth(DT, order=3))


def test_mapped_flat_advection_matches_the_flat_grid():
    # constant H: every slope metric is exactly zero and the column
    # scaling folds to 1/H0, so the mapped flux divergence equals
    # the flat grid's advection tendency to rounding (measured: u,
    # v, w bitwise; b one ulp)
    n = 8
    flat = FrModel(
        grid=Grid((
            IntervalMesh(n, (0.0, L), name="x"),
            IntervalMesh(n, (0.0, L), name="y"),
            IntervalMesh(n, (0.0, H0), periodic=False, name="z"),
        )),
        modules=(DynamicalCore(), ConstantStratification(n2=0.0),
                 CenteredAdvection()),
        time_stepper=AdamBashforth(DT, order=3))
    mapped = make_mapped_model(
        n, CenteredAdvection(), init=lambda x: H0 + 0.0 * x, ny=n)
    hor = centers(n)
    ver = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    fields = {
        "u": 0.3 + 0.1 * np.sin(y),
        "v": 0.2 * np.cos(x),
        "b": np.sin(x) * np.cos(np.pi * z),
    }
    flat.set_fields(**fields)
    mapped.set_fields(**fields)
    tf = advection_tendency(flat, CenteredAdvection)
    tm = advection_tendency(mapped, CenteredAdvection)
    for c in ("u", "v", "w", "b"):
        a = np.asarray(tf[c].data)
        b = np.asarray(tm[c].data)
        scale = max(np.abs(a).max(), 1e-30)
        assert np.abs(a - b).max() <= 1e-14 * scale, c


def test_mapped_transport_converges_at_second_order():
    # uniform physical flow over the sloped column: b = cos(x) zp
    # depends on the PHYSICAL height, so the honest transport is
    # -U db/dx|_zp = U sin(x) zp; the computational derivative
    # alone would be off at O(1). Measured errors 1.43e-2, 3.75e-3,
    # 9.54e-4 at n = 16, 32, 64 — orders 1.93, 1.98.
    U = 0.4
    errors = []
    for n in (16, 32, 64):
        model = make_mapped_model(n, CenteredAdvection())
        hor = centers(n)
        hory = centers(4)
        ver = (np.arange(n) + 0.5) / n
        x, _, z = np.meshgrid(hor, hory, ver, indexing="ij")
        zp = z * depth(x)
        model.set_fields(u=U + 0 * x, b=np.cos(x) * zp)
        tau = advection_tendency(model, CenteredAdvection)
        exact = U * np.sin(x) * zp
        errors.append(
            np.abs(np.asarray(tau["b"].data) - exact).max())
    orders = np.log2(np.asarray(errors[:-1])
                     / np.asarray(errors[1:]))
    assert np.all(orders > 1.8)


@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
def test_mapped_grid_is_a_taught_error_for_biased_schemes(cls):
    # a PERIODIC mapped column isolates the mapped rejection from
    # the walled one: the biased reconstructions are computational-
    # coordinate rows — future work
    grid = make_mapped_grid(8, periodic_column=True)
    with pytest.raises(NotImplementedError,
                       match=r"does not support mapped grids"
                             r".*CenteredAdvection"):
        FrModel(grid=grid,
                modules=(DynamicalCore(), cls(3)),
                time_stepper=AdamBashforth(DT, order=3))


# ================================================================
#  Stretched meshes: taught rejection of the biased schemes at bind
# ================================================================
def wavy_map(s):
    """Smooth wavy stretching of the unit computational interval."""
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


def make_stretched_grid(n=8, ny=NY):
    """Periodic grid whose z factor is a stretched (mapped) mesh."""
    return Grid((
        IntervalMesh(n, (0.0, L), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        MappedIntervalMesh(n, (0.0, 1.0), wavy_map, periodic=True,
                           name="z"),
    ))


@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
@pytest.mark.parametrize("order", [3, 5])
def test_stretched_mesh_is_a_taught_error_for_biased_schemes(
        cls, order):
    # a plain MappedIntervalMesh declares NO CoordinateMapping, so
    # column_corrections is empty: before the mapped_factor() guard
    # the biased schemes bound happily here and silently dropped to
    # 2nd order (measured: upwind-5 and weno-5 both 5.0 -> 2.0)
    grid = make_stretched_grid()
    with pytest.raises(
            NotImplementedError,
            match=r"does not support stretched \(mapped\) meshes"
                  r".*'z'.*uniform-offset.*silently drop to 2nd "
                  r"order.*CenteredAdvection"):
        FrModel(grid=grid,
                modules=(DynamicalCore(), cls(order)),
                time_stepper=AdamBashforth(DT, order=3))


def test_stretched_mesh_binds_the_centered_scheme():
    # the guard must not over-fire: the centered scheme's two-point
    # stencils divide by the codomain measure field (order 2) and
    # stay grounded on a stretched mesh. (A full nh Model on a plain
    # stretched axis is a separate deferral — the pressure solver
    # wants the spectral transform MappedIntervalMesh refuses (C2) —
    # so this exercises the module's own bind seam, as the
    # two-mapped-columns test does.)
    module = CenteredAdvection()
    module._bind_mapping(make_stretched_grid())  # no raise
    assert module.extra_halo is None  # no mapped column here


@pytest.mark.parametrize("op", [
    pytest.param(_BiasedFaceReconstruction(3, "left", "weno"),
                 id="biased"),
    pytest.param(_CenteredFaceInterpolation(4), id="centered-face"),
])
def test_biased_face_kernels_reject_a_stretched_factor(op):
    # the operator-level twin of the bind guard (direct misuse)
    mesh = MappedIntervalMesh(8, (0.0, 1.0), wavy_map,
                              periodic=True, name="z")
    with pytest.raises(SpaceMismatchError,
                       match=r"uniform-mesh only.*uniform-offset"):
        op.codomain(mesh.center)


def test_two_mapped_columns_are_a_taught_error():
    # two single-base analytic maps (parameter-free H keeps their
    # coupled coordinate sets disjoint) exceed the stage-C4 support
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H,
              "yp": lambda y, YN: y * YN},
        params={"H": lambda: 0.8, "YN": lambda: 0.9})
    grid = Grid((
        IntervalMesh(8, (0.0, L), name="x"),
        IntervalMesh(8, (0.0, 1.0), name="y"),
        IntervalMesh(8, (0.0, 1.0), name="z"),
    ), mapping=mapping)
    module = CenteredAdvection()
    with pytest.raises(NotImplementedError,
                       match="exactly one mapped column"):
        module._bind_mapping(grid)


def test_mapped_halo_substitute_and_flat_none():
    # the mapped flux divergence multiplies grid.metric coefficients
    # the halo tracer cannot follow: two cells per coordinate; the
    # flat path declares nothing (fully halo-traced, pre-C4)
    module = CenteredAdvection()
    make_mapped_model(8, module)
    assert module.extra_halo is not None
    assert dict(module.extra_halo.widths) == {
        "x": 2, "y": 2, "z": 2}
    flat_module = CenteredAdvection()
    make_model(8, flat_module)
    assert flat_module.extra_halo is None


def test_mapped_background_split_telescopes():
    # both background call sites on the mapped grid: the two-term
    # sum at Ro = 1 equals the no-background module on the combined
    # advecting velocity (the centered hooks are linear in the
    # velocity). The identity holds for the components whose
    # ADVECTED field carries no background (v, w, b): the split
    # transports the perturbation u itself (old-stack convention),
    # so the u rows advect different fields by design — asserted
    # finite only.
    n = 8

    def u_bg(y):
        return 1.0 + 0.5 * np.sin(y)

    module = CenteredAdvection(background={"u": u_bg})
    model = make_mapped_model(n, module, ny=n)
    set_random_state(model, seed=21)
    total = advection_tendency(model, CenteredAdvection)

    combined = make_mapped_model(n, CenteredAdvection(), ny=n)
    hor = centers(n)
    y = np.meshgrid(faces(n), hor, (np.arange(n) + 0.5) / n,
                    indexing="ij")[1]
    state = model.state
    combined.set_fields(
        u=np.asarray(state["u"].data) + u_bg(y),
        v=np.asarray(state["v"].data),
        w=np.asarray(state["w"].data),
        b=np.asarray(state["b"].data))
    want = advection_tendency(combined, CenteredAdvection)
    for c in ("v", "w", "b"):
        scale = max(np.abs(np.asarray(want[c].data)).max(), 1e-30)
        np.testing.assert_allclose(
            np.asarray(total[c].data), np.asarray(want[c].data),
            rtol=0, atol=1e-13 * scale)
    assert np.isfinite(np.asarray(total["u"].data)).all()


def test_mapped_advection_reads_the_current_geometry():
    # the dynamic-params seam: a MovingGeometry frozen at a depth
    # DIFFERENT from the grid's static default drives the advection
    # metrics — the tendency matches a static grid built at that
    # depth, and differs from the static-default tendency
    def other(x):
        return 1.0 + 0.1 * jnp.cos(2.0 * x)

    n = 8
    moving = FrModel(
        grid=make_mapped_grid(n),
        modules=(DynamicalCore(), ConstantStratification(n2=0.0),
                 CenteredAdvection(),
                 MovingGeometry(
                     {"H": lambda x, t: other(x) + 0.0 * t})),
        time_stepper=AdamBashforth(DT, order=3))
    static = make_mapped_model(n, CenteredAdvection(), init=other)
    default = make_mapped_model(n, CenteredAdvection())
    hor = centers(n)
    hory = centers(4)
    ver = (np.arange(n) + 0.5) / n
    x, _, z = np.meshgrid(hor, hory, ver, indexing="ij")
    fields = {"u": 0.4 + 0 * x, "b": np.cos(x) * z}
    for model in (moving, static, default):
        model.set_fields(**fields)
    got = advection_tendency(moving, CenteredAdvection)
    want = advection_tendency(static, CenteredAdvection)
    other_t = advection_tendency(default, CenteredAdvection)
    np.testing.assert_allclose(
        np.asarray(got["b"].data), np.asarray(want["b"].data),
        rtol=0, atol=1e-14)
    assert not np.allclose(np.asarray(got["b"].data),
                           np.asarray(other_t["b"].data))
