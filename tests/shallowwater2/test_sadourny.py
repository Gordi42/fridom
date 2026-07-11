"""Sadourny advection: the csqr-field behaviour and conservation."""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater as swold
import fridom.shallowwater2 as sw
from fridom.shallowwater2 import params as sw_params

from .conftest import (
    N,
    gaussian_bump,
    make_grid,
    make_model,
    total_energy,
)

CSQR = 0.7

BG_TERM = fr.model.term_predicates.named(
    "SadournyAdvection/background_advection")
NL_TERM = fr.model.term_predicates.named("SadournyAdvection/advect")


# ================================================================
#  Walled-grid helpers: the exactly-conserved discrete energy
# ================================================================
def walled_model(*, periodic_x=True, f0=0.0, ro=0.4, dt=2e-3):
    """Return a nonlinear walled model (y walls; x optional)."""
    grid = make_grid(periodic_x=periodic_x, periodic_y=False)
    return make_model(grid, csqr=CSQR, rossby_number=ro, f0=f0,
                      advection=True, dt=dt)


def set_random(model, seed=11, amp=1.0):
    """Fill the prognostics with random data.

    Walls are structural: v carries interior faces only, so any
    data satisfies u.n = 0.
    """
    rng = np.random.default_rng(seed)
    model.set_fields(
        u=amp * rng.standard_normal(model.state["u"].shape),
        v=amp * rng.standard_normal(model.state["v"].shape),
        p=0.3 * amp * rng.standard_normal(model.state["p"].shape))


def h_energy_terms(model):
    """Per-term chain rule of the conserved discrete energy.

    ``E = sum 1/2 hbar^x u^2 + 1/2 hbar^y v^2 + 1/2 p^2`` with
    ``h = c^2 + Ro p`` (measure-weighted sums); the exact invariant
    of gravity + Sadourny advection on periodic AND walled grids
    (see the module docstring of shallowwater2/modules/sadourny.py).
    Returns the five dE/dt contributions as floats.
    """
    z = model.state
    dz = model.tendency(z)
    ro = float(model.parameters[fr.model.params.SCALING_ROSSBY])
    u, v, p = z["u"], z["v"], z["p"]
    du, dv, dp = dz["u"], dz["v"], dz["p"]
    h = z["csqr"].to(p) + ro * p
    terms = (
        u * du * h.to(u),
        0.5 * ro * (u * u) * dp.to(u),
        v * dv * h.to(v),
        0.5 * ro * (v * v) * dp.to(v),
        p * dp,
    )
    return [float(t.integrate().data.ravel()[0]) for t in terms]


def h_energy(model):
    """Evaluate the conserved discrete energy functional."""
    z = model.state
    ro = float(model.parameters[fr.model.params.SCALING_ROSSBY])
    u, v, p = z["u"], z["v"], z["p"]
    h = z["csqr"].to(p) + ro * p
    parts = (0.5 * u * u * h.to(u), 0.5 * v * v * h.to(v),
             0.5 * p * p)
    return sum(float(t.integrate().data.ravel()[0]) for t in parts)


# ================================================================
#  The csqr-FIELD delta (§8.8): Sadourny reads state["csqr"]
# ================================================================
def test_csqr_is_a_state_field_not_a_scalar():
    model = make_model(csqr=1.5)
    c = model.state["csqr"]
    # a one-DOF Profile() field (constant depth) that broadcasts to the
    # nodal join in the tendency terms — still a field, not a scalar
    assert isinstance(c, fr.spatial.ScalarField)
    assert c.function_space.bare is (
        fr.spatial.Profile().resolve(model.grid))
    np.testing.assert_allclose(np.asarray(c.data), 1.5)


def test_csqr_scalar_is_published_for_host_reads():
    model = make_model(csqr=2.25)
    assert float(model.parameters[sw_params.CSQR]) == 2.25


def test_advection_changes_the_solution_vs_linear():
    grid = make_grid()
    nonlin = make_model(grid, rossby_number=0.5, advection=True)
    linear = make_model(grid, rossby_number=0.5, advection=False)
    nonlin.set_fields(p=gaussian_bump(amp=0.2),
                      u=lambda x, y: 0.2 * np.sin(2 * np.pi * y) + 0.0 * x)
    linear.set_fields(p=gaussian_bump(amp=0.2),
                      u=lambda x, y: 0.2 * np.sin(2 * np.pi * y) + 0.0 * x)
    nonlin.advance(40)
    linear.advance(40)
    # the nonlinear advection leaves a measurable imprint
    diff = np.abs(np.asarray(nonlin.state["p"].data)
                  - np.asarray(linear.state["p"].data)).max()
    assert diff > 1e-4


# ================================================================
#  Conservation: mass exact, energy/enstrophy bounded
# ================================================================
def test_mass_exact_under_full_dynamics():
    model = make_model(rossby_number=0.5)
    model.set_fields(p=gaussian_bump(amp=0.15),
                     v=lambda x, y: 0.1 * np.cos(2 * np.pi * x) + 0.0 * y)
    mass0 = float(model.state["p"].integrate().data.ravel()[0])
    model.advance(50)
    mass1 = float(model.state["p"].integrate().data.ravel()[0])
    assert abs(mass1 - mass0) < 1e-12


def test_energy_bounded_with_a_balanced_start():
    model = make_model(rossby_number=0.3, dt=2e-3)
    model.set_fields(p=gaussian_bump(amp=0.08),
                     u=lambda x, y: 0.05 * np.sin(2 * np.pi * y) + 0.0 * x,
                     v=lambda x, y: 0.05 * np.sin(2 * np.pi * x) + 0.0 * y)
    e0 = total_energy(model)
    peak = 0.0
    for _ in range(20):
        model.advance(5)
        peak = max(peak, abs(total_energy(model) - e0) / e0)
    assert peak < 5e-3


def test_potential_enstrophy_stays_bounded():
    model = make_model(rossby_number=0.4, dt=2e-3)
    model.set_fields(p=gaussian_bump(amp=0.1),
                     u=lambda x, y: 0.1 * np.sin(2 * np.pi * y) + 0.0 * x)

    def enstrophy(m):
        # 0.5 * integral zeta^2 (NE corner) — a bounded invariant
        z = m.state.rel_vort
        return float((0.5 * z * z).integrate().data.ravel()[0])

    q0 = enstrophy(model)
    model.advance(80)
    q1 = enstrophy(model)
    assert np.isfinite(q1)
    assert q1 < 5.0 * (q0 + 1e-9)   # no runaway enstrophy growth


# ================================================================
#  Walled grids: free-slip Sadourny (channel and doubly-walled)
# ================================================================
@pytest.mark.parametrize("periodic_x", [
    pytest.param(True, id="channel"),
    pytest.param(False, id="double-walled"),
])
@pytest.mark.parametrize("seed", [3, 11])
def test_walled_semi_discrete_energy_rate_is_machine_zero(
        periodic_x, seed):
    # gravity + Sadourny advection conserve E = sum 1/2 hbar^x u^2
    # + 1/2 hbar^y v^2 + 1/2 p^2 exactly (semi-discrete) for ANY
    # state honouring u.n = 0 (structural on the walled spaces);
    # the vorticity-flux exchange is antisymmetric for any finite q
    # and the wall values consumed by the fills are exact zeros.
    model = walled_model(periodic_x=periodic_x, f0=0.0)
    set_random(model, seed=seed)
    terms = h_energy_terms(model)
    scale = sum(abs(t) for t in terms)
    assert abs(sum(terms)) / scale < 1e-13


@pytest.mark.parametrize("periodic_x", [
    pytest.param(True, id="channel"),
    pytest.param(False, id="double-walled"),
])
def test_walled_mass_rate_is_machine_zero(periodic_x):
    # impermeability closes the thickness flux at the walls exactly
    model = walled_model(periodic_x=periodic_x, f0=1.0)
    set_random(model, seed=5)
    dp = model.tendency(model.state)["p"]
    rate = float(dp.integrate().data.ravel()[0])
    scale = float(abs(dp).integrate().data.ravel()[0])
    assert abs(rate) / scale < 1e-14


def test_walled_interior_tendency_matches_periodic_stencils():
    # far from the walls (> stencil reach of 2 cells) the walled
    # tendencies are the periodic stencil values bit for bit
    rng = np.random.default_rng(9)
    u = rng.standard_normal((N, N))
    v = rng.standard_normal((N, N))
    v[:, -1] = 0.0                    # the wall face of the torus
    p = 0.3 * rng.standard_normal((N, N))

    periodic = make_model(make_grid(), csqr=CSQR, rossby_number=0.4,
                          f0=1.0, advection=True)
    walled = walled_model(f0=1.0)
    periodic.set_fields(u=u, v=v, p=p)
    walled.set_fields(u=u, v=v[:, :-1], p=p)

    dzp = periodic.tendency(periodic.state)
    dzw = walled.tendency(walled.state)
    inner = slice(3, N - 3)
    for name in ("u", "p"):
        np.testing.assert_allclose(
            np.asarray(dzw[name].data)[:, inner],
            np.asarray(dzp[name].data)[:, inner],
            rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(dzw["v"].data)[:, inner],
        np.asarray(dzp["v"].data)[:, inner],
        rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("periodic_x", [
    pytest.param(True, id="channel"),
    pytest.param(False, id="double-walled"),
])
def test_walled_nonlinear_run_is_stable(periodic_x):
    # O(600) steps of the full nonlinear walled model: no NaN and a
    # bounded drift of the exactly-conserved discrete energy (AB3
    # time error + the split-Coriolis commutator, both small)
    model = walled_model(periodic_x=periodic_x, f0=1.0, ro=0.3,
                         dt=2e-3)
    model.set_fields(
        p=gaussian_bump(amp=0.08),
        u=lambda x, y: 0.05 * np.sin(2 * np.pi * y) + 0.0 * x)
    e0 = h_energy(model)
    peak = 0.0
    for _ in range(20):
        model.advance(30)             # 600 steps
        for name in ("u", "v", "p"):
            assert not bool(model.state[name].has_nan())
        peak = max(peak, abs(h_energy(model) - e0) / e0)
    assert peak < 5e-3


# ================================================================
#  Variable depth: Sadourny reads the csqr(y) FIELD (verified)
# ================================================================
def varying_walled_model(*, f0=0.0, ro=0.4):
    """Build a variable-depth walled channel (csqr(y) field)."""
    coriolis = sw.modules.FPlaneCoriolis(
        f0=f0, metric_weight="csqr")
    return sw.Model(
        grid=make_grid(periodic_y=False),
        csqr=lambda y: 1.0 + 0.5 * np.sin(np.pi * y),
        rossby_number=ro, coriolis=coriolis, advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(2e-3, order=3))


@pytest.mark.parametrize("seed", [3, 11])
def test_varying_depth_energy_rate_is_machine_zero(seed):
    # the h-weighted invariant (h = c^2(y) + Ro p) holds for the
    # variable depth too: the scheme reads the csqr FIELD (the §8.8
    # fix), so gravity + advection telescope exactly
    model = varying_walled_model(f0=0.0)
    set_random(model, seed=seed)
    terms = h_energy_terms(model)
    scale = sum(abs(t) for t in terms)
    assert abs(sum(terms)) / scale < 1e-13


def test_varying_depth_mass_rate_is_machine_zero():
    model = varying_walled_model(f0=1.0)
    set_random(model, seed=5)
    dp = model.tendency(model.state)["p"]
    rate = float(dp.integrate().data.ravel()[0])
    scale = float(abs(dp).integrate().data.ravel()[0])
    assert abs(rate) / scale < 1e-14


# ================================================================
#  Prescribed background flow (the V-S3 linear term)
# ================================================================
def background_model(background, *, grid=None, ro=0.4, f0=1.0,
                     dt=2e-3):
    """Build a model carrying a Sadourny background flow."""
    if grid is None:
        grid = make_grid()
    return sw.Model(
        grid=grid, csqr=CSQR, rossby_number=ro,
        coriolis=sw.modules.FPlaneCoriolis(f0=f0),
        advection=False,
        modules_extra=(
            sw.modules.SadournyAdvection(background=background),),
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))


def streamfunction_background(amp=0.3, n=N):
    """Discretely divergence-free background from a streamfunction.

    ``u_b = -delta_y psi``, ``v_b = +delta_x psi`` with the discrete
    differences of a corner streamfunction: the discrete divergence
    cancels exactly (commuting finite differences).
    """
    dx = 1.0 / n

    def psi(x, y):
        return (amp * np.sin(2 * np.pi * x)
                * np.cos(2 * np.pi * y) / (2 * np.pi))

    return {
        "u": lambda x, y: -(psi(x, y + 0.5 * dx)
                            - psi(x, y - 0.5 * dx)) / dx,
        "v": lambda x, y: (psi(x + 0.5 * dx, y)
                           - psi(x - 0.5 * dx, y)) / dx,
    }


def channel_background(amp=0.3, n=N):
    """Walled-channel background: psi constant along the y walls."""
    dx = 1.0 / n

    def psi(x, y):
        return (amp * np.sin(2 * np.pi * x)
                * np.sin(np.pi * y) / (2 * np.pi))

    return {
        "u": lambda x, y: -(psi(x, y + 0.5 * dx)
                            - psi(x, y - 0.5 * dx)) / dx,
        "v": lambda x, y: (psi(x + 0.5 * dx, y)
                           - psi(x - 0.5 * dx, y)) / dx,
    }


def test_background_samples_at_the_staggered_nodes():
    # a callable naming a coordinate subset is sampled at the
    # component's own nodes (u: east faces -> y at the centres)
    model = background_model(
        {"u": lambda y: 0.3 * np.sin(2 * np.pi * y)})
    ub = model.state["u_background"]
    yc = (np.arange(N) + 0.5) / N
    np.testing.assert_allclose(
        np.asarray(ub.data), np.broadcast_to(
            0.3 * np.sin(2 * np.pi * yc), (N, N)),
        rtol=0.0, atol=1e-14)
    # the missing component is zero
    assert np.asarray(model.state["v_background"].data).max() == 0.0


def test_background_constant_component_fills():
    model = background_model({"u": 0.37})
    np.testing.assert_allclose(
        np.asarray(model.state["u_background"].data), 0.37)


def test_background_property_round_trips():
    assert sw.modules.SadournyAdvection().background is None
    module = sw.modules.SadournyAdvection(background={"u": 0.37})
    assert module.background == {"u": 0.37, "v": 0.0}


def test_background_pressure_split_is_exact():
    # background dp + nonlinear dp == the single bilinear flux
    # -div((Ro u' + u_b) p) evaluated directly (machine precision)
    model = background_model(streamfunction_background(), ro=0.4)
    set_random(model, seed=2)
    z = model.state
    dp_bg = model.tendency(z, filter=BG_TERM)["p"]
    dp_nl = model.tendency(z, filter=NL_TERM)["p"]
    u, v, p = z["u"], z["v"], z["p"]
    full_u = 0.4 * u + z["u_background"]
    full_v = 0.4 * v + z["v_background"]
    direct = -((full_u * p.to(u)).diff("x")
               + (full_v * p.to(v)).diff("y"))
    err = np.abs(np.asarray((dp_bg + dp_nl - direct).data)).max()
    scale = np.abs(np.asarray(direct.data)).max()
    assert err / scale < 1e-13


def test_background_term_is_exactly_linear():
    # additivity and homogeneity in the state; the operator itself
    # is state-independent (transport by the prescribed u_b only)
    model = background_model(streamfunction_background())
    set_random(model, seed=7)
    z1 = model.state
    rng = np.random.default_rng(13)
    z2 = z1.replace(
        u=z1["u"].with_data(
            jnp.asarray(rng.standard_normal(z1["u"].shape))),
        v=z1["v"].with_data(
            jnp.asarray(rng.standard_normal(z1["v"].shape))),
        p=z1["p"].with_data(
            jnp.asarray(0.3 * rng.standard_normal(z1["p"].shape))))
    zsum = z1.replace(u=z1["u"] + z2["u"], v=z1["v"] + z2["v"],
                      p=z1["p"] + z2["p"])
    t1 = model.tendency(z1, filter=BG_TERM)
    t2 = model.tendency(z2, filter=BG_TERM)
    tsum = model.tendency(zsum, filter=BG_TERM)
    zdouble = z1.replace(u=z1["u"] * 2.0, v=z1["v"] * 2.0,
                         p=z1["p"] * 2.0)
    tdouble = model.tendency(zdouble, filter=BG_TERM)
    for name in ("u", "v", "p"):
        additivity = np.abs(np.asarray(
            (tsum[name] - t1[name] - t2[name]).data)).max()
        homogeneity = np.abs(np.asarray(
            (tdouble[name] - t1[name] * 2.0).data)).max()
        scale = np.abs(np.asarray(t1[name].data)).max()
        assert additivity / scale < 1e-13
        assert homogeneity == 0.0


def test_background_none_is_bitwise_identical():
    # SadournyAdvection(background=None) == the preset default
    grid = make_grid()
    preset = make_model(grid, csqr=CSQR, rossby_number=0.4,
                        advection=True, dt=2e-3)
    explicit = background_model(None, grid=grid)
    rng = np.random.default_rng(4)
    u = rng.standard_normal((N, N))
    v = rng.standard_normal((N, N))
    p = 0.3 * rng.standard_normal((N, N))
    preset.set_fields(u=u, v=v, p=p)
    explicit.set_fields(u=u, v=v, p=p)
    d1 = preset.tendency(preset.state)
    d2 = explicit.tendency(explicit.state)
    for name in ("u", "v", "p"):
        assert np.array_equal(np.asarray(d1[name].data),
                              np.asarray(d2[name].data))


def test_linearize_keeps_background_drops_nonlinear():
    grid = make_grid()
    model = background_model(streamfunction_background(), grid=grid)
    set_random(model, seed=5)
    z = model.state
    lin = fr.model.linearize(model)
    # the linear variant is the linear-filtered tendency (bitwise)
    tl = lin.tendency(z)
    tf = model.tendency(z, filter=fr.model.term_predicates.linear)
    for name in ("u", "v", "p"):
        assert np.array_equal(np.asarray(tl[name].data),
                              np.asarray(tf[name].data))
    # the nonlinear term is dropped: the variant is exactly linear
    zdouble = z.replace(u=z["u"] * 2.0, v=z["v"] * 2.0,
                        p=z["p"] * 2.0)
    tl2 = lin.tendency(zdouble)
    for name in ("u", "v", "p"):
        assert np.abs(np.asarray(
            (tl2[name] - tl[name] * 2.0).data)).max() == 0.0
    # the background term IS kept: linear variant = (gravity +
    # coriolis, from the background-free linear terms) + bg term
    tb = model.tendency(z, filter=BG_TERM)
    t0 = model.tendency(
        z, filter=fr.model.term_predicates.linear & ~BG_TERM)
    for name in ("u", "v", "p"):
        err = np.abs(np.asarray(
            (tl[name] - t0[name] - tb[name]).data)).max()
        scale = np.abs(np.asarray(tl[name].data)).max()
        assert err / scale < 1e-14


def discrete_advection_symbol(model, mode):
    """Probe ``U * sin(k dx)/dx`` from the term's own stencil.

    Applies the background pressure-flux stencil to plane waves on
    the p space and projects: ``A(cos kx) = U ktilde sin(kx)``.
    """
    z = model.state
    u, p = z["u"], z["p"]
    ub = z["u_background"]
    k = 2 * np.pi * mode
    grid = model.grid
    fc = grid.create_field(
        p.function_space,
        init=lambda x, y: np.cos(k * x) + 0.0 * y)
    fs = grid.create_field(
        p.function_space,
        init=lambda x, y: np.sin(k * x) + 0.0 * y)
    response = -(ub * fc.to(u)).diff("x")
    num = float((response * fs).integrate().data.ravel()[0])
    den = float((fs * fs).integrate().data.ravel()[0])
    return num / den


def test_background_doppler_shifts_the_eigenvalues():
    # a constant background U on a periodic grid Doppler-shifts the
    # whole spectrum by the discrete advection symbol: omega_bg(k) =
    # omega_0(k) - U sin(kx dx)/dx, uniformly over the branches
    u0 = 0.37
    with_bg = background_model({"u": u0})
    without = make_model(make_grid(), csqr=CSQR, rossby_number=0.4,
                         advection=True, dt=2e-3)
    omega_bg = np.asarray(fr.model.numeric_eigenpairs(with_bg).omega)
    omega_0 = np.asarray(fr.model.numeric_eigenpairs(without).omega)
    dx = 1.0 / N
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ktilde = np.sin(kx * dx) / dx
    # the analytic symbol IS the term's own stencil (probed)
    for mode in (1, 2, 3):
        probed = discrete_advection_symbol(with_bg, mode)
        assert abs(probed - u0 * ktilde[mode]) < 1e-12
    expected = omega_0 - u0 * ktilde[:, None, None]
    np.testing.assert_allclose(omega_bg, expected,
                               rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("background_factory", "periodic_y"),
    [pytest.param(streamfunction_background, True, id="periodic"),
     pytest.param(channel_background, False, id="channel")])
def test_background_term_conserves_quadratic_energy(
        background_factory, periodic_y):
    # the background term alone conserves the PLAIN quadratic energy
    # sum (u^2 + v^2 + p^2)/2 exactly (semi-discrete): flux-form
    # transport by a discretely solenoidal field telescopes. The
    # h-weighted invariant of gravity + self-advection is NOT
    # conserved with a background (module docstring).
    model = background_model(
        background_factory(),
        grid=make_grid(periodic_y=periodic_y))
    set_random(model, seed=8)
    z = model.state
    dz = model.tendency(z, filter=BG_TERM)
    rate = sum(
        float((z[name] * dz[name]).integrate().data.ravel()[0])
        for name in ("u", "v", "p"))
    scale = sum(
        float(abs(z[name] * dz[name]).integrate().data.ravel()[0])
        for name in ("u", "v", "p"))
    assert abs(rate) / scale < 1e-14


def test_background_walled_interior_matches_periodic_stencils():
    # far from the walls the background-term tendencies are the
    # periodic stencil values bit for bit (same retag machinery as
    # the nonlinear term; identity on periodic axes)
    background = {"u": lambda y: 0.3 + 0.2 * np.sin(2 * np.pi * y)}
    rng = np.random.default_rng(9)
    u = rng.standard_normal((N, N))
    v = rng.standard_normal((N, N))
    v[:, -1] = 0.0
    p = 0.3 * rng.standard_normal((N, N))
    periodic = background_model(background)
    walled = background_model(
        background, grid=make_grid(periodic_y=False))
    periodic.set_fields(u=u, v=v, p=p)
    walled.set_fields(u=u, v=v[:, :-1], p=p)
    dzp = periodic.tendency(periodic.state, filter=BG_TERM)
    dzw = walled.tendency(walled.state, filter=BG_TERM)
    inner = slice(3, N - 3)
    for name in ("u", "v", "p"):
        np.testing.assert_allclose(
            np.asarray(dzw[name].data)[:, inner],
            np.asarray(dzp[name].data)[:, inner],
            rtol=0.0, atol=1e-13)


def test_background_run_is_stable():
    # a short full nonlinear run with a background stays finite
    model = background_model(channel_background(amp=0.1),
                             grid=make_grid(periodic_y=False),
                             ro=0.3)
    model.set_fields(p=gaussian_bump(amp=0.05))
    model.advance(200)
    for name in ("u", "v", "p"):
        assert not bool(model.state[name].has_nan())


def test_old_stack_background_parity():
    # the old module with disable_nonlinear=True and a background at
    # scaling=1 is exactly the new background term (the new term
    # carries no Rossby factor; the old stack multiplied the
    # background by scaling=Ro, so old users passed pre-scaled
    # backgrounds)
    n, lx = N, 2 * np.pi
    dx = lx / n

    def psi(x, y):
        return 0.4 * np.sin(2 * np.pi * x / lx) * np.cos(
            4 * np.pi * y / lx)

    ub_fn = lambda x, y: -(psi(x, y + 0.5 * dx)  # noqa: E731
                           - psi(x, y - 0.5 * dx)) / dx
    vb_fn = lambda x, y: (psi(x + 0.5 * dx, y)  # noqa: E731
                          - psi(x - 0.5 * dx, y)) / dx
    u_fn = lambda x, y: (np.sin(2 * np.pi * x / lx)  # noqa: E731
                         * np.cos(2 * np.pi * y / lx))
    v_fn = lambda x, y: (np.cos(4 * np.pi * x / lx)  # noqa: E731
                         + 0.1 * np.sin(2 * np.pi * y / lx))
    p_fn = lambda x, y: (0.3 * np.sin(2 * np.pi * x / lx)  # noqa: E731
                         * np.sin(2 * np.pi * y / lx))

    # old stack: background transport isolated at scaling = 1
    grid_old = swold.grid.cartesian.Grid(
        shape=(n, n), domain_size=(lx, lx))
    mset = swold.ModelSettings(grid_old, f0=1.0, csqr=CSQR,
                               rossby_number=1.0)
    mset.time_stepper.dt = np.timedelta64(10, "ms")
    mset = mset.setup()
    adv = mset.tendencies.advection
    x, y = mset.grid.x_mesh
    xf, yf = x + 0.5 * dx, y + 0.5 * dx
    bg_state = swold.State(mset)
    bg_state.u.arr = jnp.asarray(ub_fn(xf, y))
    bg_state.v.arr = jnp.asarray(vb_fn(x, yf))
    bg_state.sync()
    adv.background = bg_state
    adv.disable_nonlinear = True
    z_old = swold.State(mset)
    z_old.u.arr = jnp.asarray(u_fn(xf, y))
    z_old.v.arr = jnp.asarray(v_fn(x, yf))
    z_old.p.arr = jnp.asarray(p_fn(x, y))
    z_old.sync()
    dz_old = adv.advect_state(z_old, swold.State(mset))

    # new stack: the background term (any Rossby number)
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, lx), periodic=True,
                                     name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, lx), periodic=True,
                                     name="y")
    model = background_model({"u": ub_fn, "v": vb_fn},
                             grid=fr.spatial.Grid((mx, my)))
    model.set_fields(u=u_fn, v=v_fn, p=p_fn)
    dz_new = model.tendency(model.state, filter=BG_TERM)

    halo = (np.asarray(dz_old.u.arr).shape[0] - n) // 2
    trim = slice(halo, -halo) if halo else slice(None)
    for name in ("u", "v", "p"):
        old = np.asarray(getattr(dz_old, name).arr)[trim, trim]
        np.testing.assert_allclose(
            old, np.asarray(dz_new[name].data),
            rtol=0.0, atol=1e-13)


# ----------------------------------------------------------------
#  Background validation (taught errors)
# ----------------------------------------------------------------
def test_background_rejects_non_divergence_free():
    # analytically smooth but not DISCRETELY solenoidal -> taught
    # error pointing at a streamfunction-derived background
    bad = {"u": lambda x, y: 0.1 * np.sin(2 * np.pi * x)
           * np.cos(2 * np.pi * y)}
    with pytest.raises(ValueError,
                       match="divergence-free background"):
        background_model(bad)


def test_background_rejects_wall_normal_constant():
    with pytest.raises(ValueError, match="impermeable"):
        background_model({"v": 0.5},
                         grid=make_grid(periodic_y=False))


def test_background_rejects_wall_normal_callable():
    bad = {"v": lambda y: 0.1 + 0.0 * y}
    with pytest.raises(ValueError, match="impermeable"):
        background_model(bad, grid=make_grid(periodic_y=False))


def test_background_rejects_unknown_component():
    with pytest.raises(TypeError, match="component names"):
        sw.modules.SadournyAdvection(background={"w": 1.0})


def test_background_rejects_empty_mapping():
    with pytest.raises(ValueError, match="at least one"):
        sw.modules.SadournyAdvection(background={})


def test_background_rejects_bad_value():
    with pytest.raises(TypeError, match="constant or a"):
        sw.modules.SadournyAdvection(background={"u": "fast"})


def test_background_rejects_unknown_coordinate():
    bad = {"u": lambda z: 0.0 * z}
    with pytest.raises(ValueError, match="coordinate"):
        background_model(bad)
