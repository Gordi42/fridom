r"""The thermal-wind background: Eady mean-flow terms and instability.

``hy.ThermalWindBackground`` supplies the two linearized mean-flow
interaction terms of a zonal thermal-wind state ``U(z) = shear*(z-z0)``,
``B(y) = -f0*shear*y`` that the shared advection ``background=`` does
not: the momentum tilting ``du/dt += -shear*w'`` and the baroclinic
conversion ``db/dt += +f0*shear*v'`` (the Eady energy source). Paired
with ``CenteredAdvection(background=tw.background_velocity())`` (the
Doppler part), it unblocks the Eady baroclinic-instability gate H5
found structurally missing.

This suite pins the term signs (derived from thermal wind), the energy
honesty (the pair is the *sole* non-conservative source — no perturbation
norm is conserved, the mean flow feeds the instability), and the growth
rate: the time-integrated growth matches the discrete linear operator's
most-unstable eigenvalue tightly, the operator eigenvalue sits below the
QG continuous ``0.31*f0*shear/N`` by the primitive-equation (Stone 1966)
Richardson-number correction, and it trends toward the QG value as
``Ri = N^2/shear^2`` grows.

Runtime-budgeted; the operator assembly (``6*nz`` tendency probes) is the
cost and is kept at modest resolution. Self-contained per the AGENTS
oversized-module rule (small builders duplicated).
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
import fridom.model.term_predicates as tp
from fridom.model.errors import MissingParameterError
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh


# ================================================================
#  Builders (duplicated per the self-contained-shard rule)
# ================================================================
def make_grid(nx, ny, nz, lx=1.0, depth=1.0):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, lx), periodic=True, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def eady_model(*, nx=32, ny=4, nz=12, lx=1.0, depth=1.0, csqr=30.0,
               f0=1.0, n2=1.0, shear=0.5, dt=0.02, epsilon=0.0,
               free_surface=None, advection=True):
    """Build a hydrostatic model with a thermal-wind mean flow.

    Returns ``(model, tw)``: the thermal-wind module rides
    ``modules_extra``; its matching background velocity feeds the shared
    advection's Doppler ``background=`` (unless ``advection`` is a plain
    module or False).
    """
    tw = hy.ThermalWindBackground(shear=shear, reference_height=depth / 2)
    if advection is True:
        advection = fr.model.modules.CenteredAdvection(
            background={"u": tw.background_velocity()})
    if free_surface is None:
        free_surface = hy.ImplicitFreeSurface(epsilon=epsilon)
    model = hy.Model(
        grid=make_grid(nx, ny, nz, lx=lx, depth=depth),
        core=hy.Core(gravity=csqr / depth),
        time_stepper=AdamBashforth(dt, order=2, eps=0.1),
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=free_surface,
        advection=advection,
        modules_extra=(tw,))
    return model, tw


def trig_layer(model, name, kx, lx, phase):
    """cos/sin(2 pi kx x / lx), depth-uniform, at name's nodes."""
    fs = model.state[name].function_space
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)
    trig = np.cos if phase == "cos" else np.sin

    def init(**c):
        return trig(2 * np.pi * kx * c.get("x", 0) / lx) \
            + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return np.asarray(model.grid.create_field(fs, init=init).data)


def eady_spectrum(model, kx, lx):
    """Return (eigvals, eigvecs, cols) of the linear operator L.

    The full per-z-layer ``(u, v, b)`` linear tendency restricted to
    horizontal wavenumber ``kx`` (ky = 0), with the rigid-lid CONSTRAINT
    projection applied (``constraints=True``): its most-unstable
    eigenpair is the discrete Eady mode. Only the ``linear=True`` terms
    (Doppler, Coriolis, pressure, stratification, thermal wind) enter, so
    the pinned nonlinear advection never contaminates the operator.
    """
    fields = ["u", "v", "b"]
    nz = model.state["u"].data.shape[2]
    trig = {(f, p): trig_layer(model, f, kx, lx, p)
            for f in fields for p in ("cos", "sin")}
    cols = [(f, j, p) for f in fields for j in range(nz)
            for p in ("cos", "sin")]
    zero = {g: 0.0 * np.asarray(model.state[g].data) for g in fields}

    def make(col):
        f, j, p = col
        data = {g: model.grid.create_field(
            model.state[g].function_space, data=zero[g]) for g in fields}
        arr = np.array(zero[f])
        arr[:, :, j] = trig[(f, p)][:, :, j]
        data[f] = model.grid.create_field(
            model.state[f].function_space, data=arr)
        return model.state.replace(**data)

    n = len(cols)
    mat = np.zeros((n, n))
    for i, col in enumerate(cols):
        d_x = model.tendency(make(col), filter=tp.linear, constraints=True)
        for r, (g, jr, pr) in enumerate(cols):
            d = np.asarray(d_x[g].data)
            num = float((d[:, :, jr] * trig[(g, pr)][:, :, jr]).sum())
            den = float((trig[(g, pr)][:, :, jr] ** 2).sum())
            mat[r, i] = num / den
    ev, vecs = np.linalg.eig(mat)
    return ev, vecs, cols, trig


def qg_eady_sigma(k, f0, n2, depth, shear):
    """Continuous QG Eady growth rate (Stone Ri -> inf limit)."""
    n = np.sqrt(n2)
    alpha = k * n * depth / f0
    val = ((alpha / 2 - np.tanh(alpha / 2))
           * (1.0 / np.tanh(alpha / 2) - alpha / 2))
    return (shear * f0 / n) * np.sqrt(val) if val > 0 else 0.0


# ================================================================
#  Declarations, references, wiring
# ================================================================
def test_declares_shear_and_references_the_constant_coriolis():
    tw = hy.ThermalWindBackground(shear=0.7)
    decls = {d.name: d for d in tw.parameter_declarations}
    assert str(hy.params.SHEAR) in {str(n) for n in decls}
    # reads the constant f0 provided by the f-plane Coriolis
    refs = {str(r.name) for r in tw.parameter_references}
    assert str(hy.params.CORIOLIS_F0) in refs
    # references the fields it couples: u/v/w and the advanced b
    fields = {r.name for r in tw.field_references}
    assert fields == {"u", "v", "w", "b"}


def test_requires_a_constant_coriolis_parameter():
    # the thermal wind d_y B = -f0 shear needs a constant f0: without
    # hy.FPlaneCoriolis the coriolis.f0 provide is missing
    with pytest.raises(MissingParameterError, match=r"coriolis\.f0"):
        hy.Model(
            grid=make_grid(8, 8, 6),
            core=hy.Core(gravity=1.0),
            time_stepper=AdamBashforth(1e-2),
            coriolis=None,
            stratification=hy.ConstantStratification(n2=1.0),
            free_surface=hy.ExplicitFreeSurface(),
            advection=False,
            modules_extra=(hy.ThermalWindBackground(shear=0.5),))


def test_background_velocity_is_the_matching_shear_profile():
    tw = hy.ThermalWindBackground(shear=0.7, reference_height=0.5)
    profile = tw.background_velocity()
    zs = np.linspace(0.0, 1.0, 5)
    assert np.allclose([profile(z) for z in zs], 0.7 * (zs - 0.5))


def test_time_dependent_shear_reports_the_ramped_linear_parameter():
    # a ramped shear feeds the linear term -> reported for frozen-L
    # steppers (AR-D7); a plain float reports nothing
    ramped = hy.ThermalWindBackground(
        shear=fr.model.Ramp(0.0, 1.0, period=1.0))
    assert str(hy.params.SHEAR) in ramped.time_dependent_linear_parameters()
    assert hy.ThermalWindBackground(shear=0.5) \
        .time_dependent_linear_parameters() == ()


# ================================================================
#  Term signs (derived from thermal wind, verified exactly)
# ================================================================
def test_tilting_term_is_minus_shear_times_diagnosed_w():
    f0, n2, shear, depth = 1.3, 1.0, 0.7, 2.0
    model, _ = eady_model(nx=8, ny=8, nz=6, lx=3.3, depth=depth,
                          csqr=3.0, f0=f0, n2=n2, shear=shear,
                          free_surface=hy.ExplicitFreeSurface())
    rng = np.random.default_rng(1)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape),
                     b=rng.standard_normal(model.state["b"].shape))
    st = model.state
    d_x = model.tendency(
        st, filter=tp.owned_by(hy.ThermalWindBackground),
        constraints=False)
    # reconstruct the diagnosed w exactly as the core does
    div = st["u"].diff("x") + st["v"].diff("y")
    w = -CumulativeIntegral(direction="up", target="face")["z"](div)
    tilting = -(shear * w.to(st["u"]))
    assert np.allclose(np.asarray(d_x["u"].data),
                       np.asarray(tilting.data), atol=1e-12)


def test_conversion_term_is_plus_f0_shear_times_v():
    f0, n2, shear, depth = 1.3, 1.0, 0.7, 2.0
    model, _ = eady_model(nx=8, ny=8, nz=6, lx=3.3, depth=depth,
                          csqr=3.0, f0=f0, n2=n2, shear=shear,
                          free_surface=hy.ExplicitFreeSurface())
    rng = np.random.default_rng(2)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape),
                     b=rng.standard_normal(model.state["b"].shape))
    st = model.state
    d_x = model.tendency(
        st, filter=tp.owned_by(hy.ThermalWindBackground),
        constraints=False)
    conversion = (f0 * shear) * st["v"].to(st["b"])
    assert np.allclose(np.asarray(d_x["b"].data),
                       np.asarray(conversion.data), atol=1e-12)
    # the meridional momentum carries no thermal-wind term
    assert float(np.max(np.abs(np.asarray(d_x["v"].data)))) == 0.0


def test_the_conversion_term_is_visible_to_the_linear_operator():
    # linear=True: fr.model.linearize / the eigenmode consumers keep it.
    # a v-only, divergence-free mode (w == 0) has NO carrier for db/dt
    # except the +f0 shear v conversion; under the linear filter it is
    # nonzero (the term H5 pinned as absent is now present)
    f0, shear = 1.0, 0.5
    model, _ = eady_model(nx=8, ny=8, nz=6, lx=1.0, f0=f0, shear=shear,
                          advection=False)
    tw = model.module(hy.ThermalWindBackground)
    model2 = eady_model(nx=8, ny=8, nz=6, lx=1.0, f0=f0, shear=shear,
                        advection=fr.model.modules.CenteredAdvection(
                            background={"u": tw.background_velocity()}))[0]
    st = model2.state.replace(v=trig_v(model2, 1))
    d_x = model2.tendency(st, filter=tp.linear, constraints=False)
    # v-only, varying in x only: the diagnosed w is exactly zero ...
    assert float(np.max(np.abs(np.asarray(st["v"].data)))) > 0.1
    # ... so a nonzero db/dt is the +f0 shear v conversion alone
    assert float(np.max(np.abs(np.asarray(d_x["b"].data)))) > 1e-3


def trig_v(model, kx):
    """Return a v-mode varying only in x (divergence-free, w == 0)."""
    fs = model.state["v"].function_space

    def init(**c):
        return np.cos(2 * np.pi * kx * c.get("x", 0)) + 0.0 * sum(c.values())
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return model.grid.create_field(fs, init=init)


# ================================================================
#  Energy honesty: the pair is the SOLE non-conservative source
# ================================================================
def test_thermal_wind_is_the_only_nonconservative_energy_source():
    r"""Everything except the thermal wind is exactly M-skew.

    Under ``M = diag(1, 1, 1/N^2, 1/c^2)`` (``hy.energy``), the internal
    KE<->PE conversion (pressure/stratification, the ``<w'b'>`` flux) and
    the Doppler advection by the divergence-free ``U`` are machine-exact
    skew, so ``<X, M dX/dt>`` vanishes to round-off when the shear is
    zero, and with a nonzero shear the *entire* energy source is the two
    thermal-wind terms. Hence no perturbation quadratic form is conserved
    (the Eady mean flow is a genuine source) — the honest gate.
    """
    f0, n2, shear, csqr, depth = 1.3, 1.0, 0.7, 3.0, 2.0
    model, _ = eady_model(nx=8, ny=8, nz=6, lx=3.3, depth=depth,
                          csqr=csqr, f0=f0, n2=n2, shear=shear,
                          free_surface=hy.ExplicitFreeSurface())
    rng = np.random.default_rng(3)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape),
                     b=rng.standard_normal(model.state["b"].shape),
                     ps=rng.standard_normal(model.state["ps"].shape))
    st = model.state
    p3 = st["b"].function_space

    def integral(field):
        return float(field.integrate().data.ravel()[0])

    def m_rate(d_x):
        return (integral(st["u"] * d_x["u"])
                + integral(st["v"] * d_x["v"])
                + integral((st["b"] / n2) * d_x["b"])
                + integral((st["ps"].to(p3) / csqr) * d_x["ps"].to(p3)))

    full = model.tendency(st, filter=tp.linear, constraints=True)
    rest = model.tendency(
        st, filter=tp.linear & ~tp.owned_by(hy.ThermalWindBackground),
        constraints=True)
    scale = (abs(integral(st["u"] * full["u"]))
             + abs(integral(st["v"] * full["v"]))
             + abs(integral((st["b"] / n2) * full["b"]))
             + abs(integral((st["ps"].to(p3) / csqr) * full["ps"].to(p3))))
    # every non-thermal-wind term is exactly M-skew (round-off)
    assert abs(m_rate(rest)) < 1e-12 * scale
    # the shear drives a genuinely nonzero energy source (not conserved)
    assert abs(m_rate(full)) > 1e-3 * scale


def test_shear_zero_recovers_machine_exact_energy_conservation():
    # with shear = 0 the module is inert and the H2 skew identity holds
    # (a linear model, mirroring the H2 energy gate exactly)
    f0, n2, csqr = 1.3, 1.0, 3.0
    model, _ = eady_model(nx=8, ny=8, nz=6, lx=3.3, depth=2.0, csqr=csqr,
                          f0=f0, n2=n2, shear=0.0, advection=False,
                          free_surface=hy.ExplicitFreeSurface())
    rng = np.random.default_rng(4)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape),
                     b=rng.standard_normal(model.state["b"].shape),
                     ps=rng.standard_normal(model.state["ps"].shape))
    st = model.state
    p3 = st["b"].function_space

    def integral(field):
        return float(field.integrate().data.ravel()[0])

    terms = [integral(st["u"] * (d := model.tendency(st))["u"]),
             integral(st["v"] * d["v"]),
             integral((st["b"] / n2) * d["b"]),
             integral((st["ps"].to(p3) / csqr) * d["ps"].to(p3))]
    assert abs(sum(terms)) < 1e-12 * sum(abs(t) for t in terms)


# ================================================================
#  Eady baroclinic instability: growth at the linear-operator rate
# ================================================================
def test_eady_mode_grows_at_the_linear_operator_rate():
    r"""The seeded most-unstable mode grows at the operator eigenvalue.

    Assemble the discrete linear operator on the ``(u, v, b)`` reduced
    mode space, take its most-unstable eigenpair (the discrete Eady
    mode), seed the model with the eigenvector, and integrate. The
    perturbation energy grows exponentially at exactly the operator's
    growth rate. The operator eigenvalue sits *below* the QG continuous
    ``0.31 f0 shear/N`` by the primitive-equation (Stone 1966)
    Richardson-number correction (here ``Ri = N^2/shear^2 = 4``), so the
    QG value is context, and the operator eigenvalue is the target.
    """
    f0, n2, shear, depth = 1.0, 1.0, 0.5, 1.0     # Ri = 4
    n = np.sqrt(n2)
    alpha, dt = 0.9, 0.02
    k = alpha * f0 / (n * depth)
    lx, nx, nz = 2 * np.pi / k, 32, 12
    model, _ = eady_model(nx=nx, ny=4, nz=nz, lx=lx, depth=depth,
                          csqr=30.0, f0=f0, n2=n2, shear=shear, dt=dt)

    ev, vecs, cols, trig = eady_spectrum(model, 1, lx)
    i_pos = int(np.argmax(ev.real))
    sigma_op = float(ev.real[i_pos])
    vec = vecs[:, i_pos]
    qg = qg_eady_sigma(k, f0, n2, depth, shear)

    # a genuine instability, below QG by the PE (Stone) correction
    assert sigma_op > 0.05
    assert 0.90 < sigma_op / qg < 1.0

    # seed the eigenvector (tiny, so the pinned nonlinear advection is
    # negligible) and integrate
    def seed(name):
        arr = np.zeros(np.asarray(model.state[name].data).shape,
                       dtype=complex)
        h_c, h_s = trig[(name, "cos")], trig[(name, "sin")]
        for j in range(nz):
            amp = (vec[cols.index((name, j, "cos"))]
                   - 1j * vec[cols.index((name, j, "sin"))])
            arr[:, :, j] = amp * (h_c[:, :, j] + 1j * h_s[:, :, j])
        return 1e-4 * np.real(arr)

    model.set_state(model.state.replace(**{
        nme: model.grid.create_field(
            model.state[nme].function_space, data=seed(nme))
        for nme in ("u", "v", "b")}))

    def energy():
        u = np.asarray(model.state["u"].data)
        v = np.asarray(model.state["v"].data)
        b = np.asarray(model.state["b"].data)
        return 0.5 * ((u ** 2).sum() + (v ** 2).sum()
                      + (b ** 2).sum() / n2)

    times, energies, t = [0.0], [energy()], 0.0
    for _ in range(14):
        model.advance(20)
        t += 20 * dt
        times.append(t)
        energies.append(energy())
    # windowed fit of ln(E)/2 vs t (skip the first window as transient)
    sigma_meas = 0.5 * np.polyfit(
        times[2:], np.log(energies[2:]), 1)[0]

    # the time-integrated growth is the operator eigenvalue (the residual
    # is the AB2 time-discretization, O(sigma^2 dt^2))
    assert abs(sigma_meas - sigma_op) / sigma_op < 0.02
    # the state actually grew exponentially (a real, resolved instability)
    assert energies[-1] / energies[0] > 2.0
    # ... monotonically, once past the initial adjustment
    assert all(energies[i + 1] > energies[i] for i in range(2, 14))


def test_eady_growth_rate_rises_toward_qg_as_richardson_increases():
    r"""The operator growth approaches QG as ``Ri = N^2/shear^2`` grows.

    The gap to the QG continuous ``0.31 f0 shear/N`` is the
    primitive-equation (non-geostrophic, Stone 1966) correction, not a
    discretization error: it is grid-converged (see the growth test's
    ``sigma_op`` stability) and closes along the *physical* Richardson
    axis, not under mesh refinement. Loose, documented.
    """
    f0, depth, alpha = 1.0, 1.0, 0.9
    ratios = []
    for n2, shear in [(1.0, 1.0), (1.0, 0.25)]:   # Ri = 1, then 16
        n = np.sqrt(n2)
        k = alpha * f0 / (n * depth)
        lx = 2 * np.pi / k
        model, _ = eady_model(nx=16, ny=4, nz=10, lx=lx, depth=depth,
                              csqr=30.0, f0=f0, n2=n2, shear=shear)
        ev, _, _, _ = eady_spectrum(model, 1, lx)
        sigma_op = float(ev.real.max())
        qg = qg_eady_sigma(k, f0, n2, depth, shear)
        ratios.append(sigma_op / qg)
    # low Ri (=1) is well below QG; high Ri (=16) is close to it
    assert ratios[0] < 0.92
    assert ratios[1] > 0.97
    assert ratios[1] > ratios[0]


# ================================================================
#  Rest state: a zero perturbation stays zero (no spurious forcing)
# ================================================================
def test_rest_state_stays_zero_over_a_short_run():
    r"""A zero perturbation is a fixed point to round-off.

    The module's terms are ``-shear w'`` and ``+f0 shear v'`` — both
    vanish identically at a zero perturbation — so the balanced state
    (the thermal-wind mean flow rides ``background_u`` / the module, not
    the perturbation state) does not spin up a spurious tendency: every
    prognostic/diagnosed perturbation field stays at round-off over a
    short run (the implicit free-surface solve on a zero RHS returns
    zero). The mean-flow AUXILIARY fields are excluded (they carry the
    balanced ``U(z)``/``f0`` and are legitimately nonzero).
    """
    model, _ = eady_model(nx=8, ny=4, nz=6, shear=0.7, dt=2e-3)
    # the carry allocates a zero perturbation; leave it and integrate
    model.advance(5)
    for name in ("u", "v", "w", "b", "ps"):
        peak = float(np.max(np.abs(np.asarray(model.state[name].data))))
        assert peak < 1e-13


# ================================================================
#  Differentiability: grad through a short run w.r.t. the shear
# ================================================================
def test_thermal_wind_grad_wrt_shear_matches_fd():
    r"""``jax.grad`` w.r.t. the shear matches a central FD (rtol 1e-4).

    Differentiability policy (AGENTS.md): the thermal-wind term is
    step-path tendency code, so ``jax.grad`` of a quadratic loss through
    a short ``Model.propagator`` run w.r.t. the shear
    :math:`\Lambda = \partial_z U` (equivalently the meridional gradient
    :math:`M^2 = f_0\Lambda`) is finite and FD-matched. The two terms are
    linear interpolation and scalar scaling of the state (no masked
    singularity), so the gradient is clean; and the ``shear`` leaf is
    spliceable — its owner materializes no AUXILIARY coefficient field
    and ``AdamBashforth`` does not freeze the linear operator, so the
    propagator accepts it as a ``wrt`` target.
    """
    model, _ = eady_model(nx=8, ny=4, nz=6, shear=0.5, dt=2e-3)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        c: 0.1 * rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in ("u", "v", "b")})
    run = model.propagator(wrt=(str(hy.params.SHEAR),), steps=6)
    lam0 = jnp.asarray(0.5)

    def loss(lam):
        out = run((lam,))
        return sum(jnp.sum(f.data ** 2) for f in out.state)

    grad = float(jax.grad(loss)(lam0))
    assert np.isfinite(grad)
    eps = 1e-4
    fd = (float(loss(lam0 + eps)) - float(loss(lam0 - eps))) / (2.0 * eps)
    assert grad == pytest.approx(fd, rel=1e-4)
