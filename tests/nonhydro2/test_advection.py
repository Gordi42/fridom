"""Tests for the nonhydro2 upwind-biased and WENO advection modules.

The load-bearing checks are the OLD-stack parity tests: the new
modules reproduce ``fr.model.modules.advection.UpwindAdvection`` /
``fr.model.modules.advection.WENO`` tendencies to machine precision on
both C-grid flux paths (tracer and velocity self-advection), both
bias sides, and both grounded orders. Physics sanity (zero tendency,
convergence order, ENO step transport, Rossby scaling, divergence
form) rides on top.
"""

import numpy as np
import pytest

import fridom as fr
import fridom.framework as frold
from fridom.model.model import Model as FrModel
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
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

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
    # UNIFORM velocity, b = sin(x): the advection tendency
    # approximates -U cos(x) at the scheme's design order (max
    # error away from the critical points of sin, where WENO-JS
    # is known to degrade).
    #
    # The constant u is LOAD-BEARING, not incidental: it is the only
    # regime in which the design order of the composite tendency is
    # observable at all. The flux form differences v_face * R(q),
    # while the high-order face quantity of an FV reconstruction (a
    # deconvolution) is the deconvolved FLUX R(v q); the mismatch is
    # a cross term ~ (h^2/24) * 2 u' q', which vanishes identically
    # for u' = 0 and otherwise caps the tendency at 2nd order — see
    # test_varying_velocity_tendency_is_only_second_order, which pins
    # exactly that. Do not "generalize" this test to a varying u: it
    # would then measure 2, and the reconstruction order would go
    # unpinned.
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


#: the honest rate of the composite tendency once the advecting
#: velocity varies along the flux axis: 2, for every biased scheme and
#: every design order. Band, not a floor — the point of the test is
#: that the rate is NEITHER below 2 (the scheme is still consistent)
#: NOR at the design order (it is not, and no amount of reconstruction
#: order makes it so). weno3 is excluded: its critical-point
#: degradation (see the test above) dominates the error on this
#: problem at any resolution we can afford, so the plateau is not
#: cleanly measurable there.
SECOND_ORDER = [
    pytest.param(UpwindAdvection, 3, id="upwind3"),
    pytest.param(UpwindAdvection, 5, id="upwind5"),
    pytest.param(WENOAdvection, 5, id="weno5"),
]


@pytest.mark.parametrize(("cls", "order"), SECOND_ORDER)
def test_varying_velocity_tendency_is_only_second_order(cls, order):
    # The sibling of test_smooth_advection_converges_at_design_order,
    # and the honest one: a periodic, uniform grid, a smooth tracer,
    # and an advecting velocity that VARIES along the flux axis
    # (u = 1 + 0.5 sin x, exactly the regime of any real flow). The
    # composite tendency then converges at 2, not at the design order
    # 3 / 5 — the product-rule / deconvolution mismatch of the C-grid
    # flux form (the scheme differences u_face * R(b), but only the
    # deconvolved flux R(u b) is high-order; the leftover cross term
    # ~ (h^2/24) * 2 u' b' is what is measured here).
    #
    # This is a property of the flux form shared with Oceananigans,
    # MITgcm, MOM6 and ROMS, and it is PINNED, not tolerated: the
    # route that would restore the design order (reconstruct the flux
    # u*b — Mishra, Pares-Pulido & Pressel, arXiv:1905.13665) costs
    # the exact-zero wall flux and constancy preservation, both of
    # which this suite tests elsewhere. Measured rates (32/64/128):
    # upwind3 2.15, 2.07; upwind5 1.98, 1.99; weno5 2.01, 2.00.
    sizes = (32, 64, 128)
    errors = []
    for n in sizes:
        model = make_model(n, cls(order))
        xc, xf = centers(n), faces(n)
        model.set_fields(
            u=broadcast(1.0 + 0.5 * np.sin(xf), n),
            b=broadcast(np.sin(xc), n))
        got = np.asarray(
            advection_tendency(model, cls)["b"].data)[:, 0, 0]
        # -d/dx (u b) with u = 1 + 0.5 sin x, b = sin x
        exact = -(np.cos(xc) + 0.5 * np.sin(2 * xc))
        errors.append(np.abs(got - exact).max())
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(rates) > 1.8   # still consistent (and not first order)
    assert max(rates) < 2.3   # and NOT the design order 3 / 5


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
