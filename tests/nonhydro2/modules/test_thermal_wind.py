r"""The thermal-wind background: front terms and symmetric instability.

``nh.ThermalWindBackground`` supplies the two linearized mean-flow
interaction terms of a lateral thermal-wind state ``V(z) = M^2 z/f0``,
``B(x, z) = N^2 z + M^2 x``: the mean-shear tilting
``dv/dt += -(M^2/f0) w`` and the horizontal buoyancy advection
``db/dt += -M^2 u``. The remaining ``-N^2 w`` is the
``ConstantStratification`` restoring, and there is deliberately no
Doppler term (the mean flow is meridional, the state y-independent, so
``V d_y`` vanishes identically).

The load-bearing gate is the **growth rate**. Eliminating pressure from
the y-independent linearized equations with ``u = m s``, ``w = -k s``
(incompressibility) for a plane wave ``exp(i(kx + mz))`` gives

    sigma^2 (k^2 + m^2) = -N^2 k^2 + 2 M^2 k m - f0^2 m^2 ,

i.e. ``sigma^2`` is the Rayleigh quotient of ``[[-N^2, M^2],
[M^2, -f0^2]]`` at the wavevector ``(k, m)``. So the fastest-growing
mode has

    sigma_max = sqrt( ( sqrt((N^2 - f0^2)^2 + 4 M^4) - (N^2 + f0^2) )/2 )

along the slope ``m/k = (N^2 + sigma_max^2)/M^2``, and ``sigma_max > 0``
exactly when ``M^4 > N^2 f0^2``, i.e. when the balanced Richardson
number ``Ri = N^2 f0^2/M^4`` drops below one — the sign of the Ertel PV
``q = f0 N^2 (1 - 1/Ri)`` of the balanced state. The suite pins the
measured rate against that prediction at the three Richardson numbers
of the symmetric-instability example (0.25, 0.5, 0.75), shows the
1/2-percent-level deficit at 32^2 is second-order discretization (it
falls by 4 under refinement), and confirms the ``Ri > 1`` state does not
grow.

Runtime-budgeted: the seeded growth runs are 32^2 x 1 cells and a few
hundred linear steps (no advection, no CFL — the nonhydrostatic wave
frequency is bounded by ``N``, so ``dt`` is resolution-independent).
Self-contained per the AGENTS oversized-module rule (small builders
duplicated rather than imported across test files).
"""
import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.model.term_predicates as tp
import fridom.nonhydro2 as nh
from fridom.model.errors import MissingParameterError
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.operators.composed import Divergence
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh

#: the symmetric-instability example's front (Stamper & Taylor 2016)
F0 = 1e-4
M2 = -1e-7
LZ = 200.0


# ================================================================
#  Builders (duplicated per the self-contained-shard rule)
# ================================================================
def make_grid(nx, ny, nz, lx=1.0, ly=1.3, lz=0.7, periodic_z=True):
    """Return a horizontally periodic grid (walled z on request)."""
    return fr.spatial.Grid((
        IM(nx, (0.0, lx), periodic=True, name="x"),
        IM(ny, (0.0, ly), periodic=True, name="y"),
        IM(nz, (0.0, lz), periodic=periodic_z, name="z")))


def front_model(*, nx=8, ny=8, nz=8, lx=1.0, ly=1.3, lz=0.7, f0=1.3,
                n2=2.0, m2=-0.7, dt=1e-3, aspect_ratio=1.0,
                advection=None, coriolis=True, buoyancy=True,
                periodic_z=True):
    """Build a nonhydrostatic model with a thermal-wind front.

    Returns ``(model, tw)``: the thermal-wind module rides
    ``modules_extra`` and reads ``f0`` off the f-plane Coriolis, so the
    balance ``f0 dV/dz = dB/dx`` holds by construction.
    """
    tw = nh.ThermalWindBackground(m2=m2, reference_height=lz / 2)
    model = nh.Model(
        grid=make_grid(nx, ny, nz, lx=lx, ly=ly, lz=lz,
                       periodic_z=periodic_z),
        core=nh.Core(aspect_ratio=aspect_ratio),
        time_stepper=AdamBashforth(dt, order=2),
        coriolis=nh.FPlaneCoriolis(f0=f0) if coriolis else None,
        buoyancy=(nh.ConstantStratification(n2=n2) if buoyancy
                  else nh.BuoyancyTracer()),
        advection=advection,
        modules_extra=(tw,))
    return model, tw


def fill(model, name, fn):
    """Return ``fn(x, z)`` sampled at the ``name`` field's own nodes."""
    space = model.state[name].function_space
    varying = tuple(n for f in space.bare.factors
                    if not isinstance(f, ConstantSpace) for n in f.names)

    def init(**coords):
        zero = sum(0.0 * np.asarray(c) for c in coords.values())
        return fn(coords.get("x", 0.0), coords.get("z", 0.0)) + zero

    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in varying])
    return np.asarray(model.grid.create_field(space, init=init).data)


# ================================================================
#  The analytic symmetric mode (see the module docstring)
# ================================================================
def si_growth_rate(f0, m2, n2):
    """Max growth rate of the y-independent (symmetric) mode."""
    f_sq = f0 * f0
    root = math.sqrt((n2 - f_sq) ** 2 + 4.0 * m2 * m2)
    lam = 0.5 * (root - (n2 + f_sq))
    return math.sqrt(lam) if lam > 0.0 else 0.0


def si_slope(f0, m2, n2):
    """Wavevector slope ``m/k`` of the fastest-growing mode."""
    sigma = si_growth_rate(f0, m2, n2)
    return (n2 + sigma * sigma) / m2


def n2_for(ri):
    """Return the example's stratification at Richardson number ``ri``."""
    return ri * M2 * M2 / (F0 * F0)


def energy(model, n2, dsqr=1.0):
    """Perturbation energy under ``diag(1, 1, dsqr, 1/N^2)``."""
    data = {c: np.asarray(model.state[c].data)
            for c in ("u", "v", "w", "b")}
    return 0.5 * ((data["u"] ** 2).sum() + (data["v"] ** 2).sum()
                  + dsqr * (data["w"] ** 2).sum()
                  + (data["b"] ** 2).sum() / n2)


def growth_history(model, n2, *, chunks, per_chunk, dt):
    """Advance in chunks; return ``(times, energies)``."""
    times, energies = [0.0], [energy(model, n2)]
    for i in range(chunks):
        model.advance(per_chunk)
        times.append((i + 1) * per_chunk * dt)
        energies.append(energy(model, n2))
    return np.asarray(times), np.asarray(energies)


def tail_growth_rate(times, energies):
    """Fit ``d/dt ln(E)/2`` over the second half of the record."""
    half = len(times) // 2
    return 0.5 * np.polyfit(times[half:], np.log(energies[half:]), 1)[0]


def seeded_model(ri, *, n=32, dt=50.0, amplitude=1e-6):
    """Return a model seeded with the analytic fastest-growing mode.

    The domain carries exactly one wavelength of the optimal wavevector
    (``lx/lz = |m/k|``, the mode taken with ``m < 0``), and each field is
    sampled at its own C-grid nodes from the analytic eigenvector
    ``(u, v, w, b) = (m, (-f0 m + k M^2/f0)/sigma, -k,
    (-M^2 m + N^2 k)/sigma) cos(kx + mz)``.
    """
    n2 = n2_for(ri)
    sigma = si_growth_rate(F0, M2, n2)
    lx = abs(si_slope(F0, M2, n2)) * LZ
    k = 2.0 * math.pi / lx
    m = -2.0 * math.pi / LZ
    model, tw = front_model(nx=n, ny=1, nz=n, lx=lx, ly=1.0, lz=LZ,
                            f0=F0, n2=n2, m2=M2, dt=dt)
    amps = {"u": m, "w": -k,
            "v": (-F0 * m + k * M2 / F0) / sigma,
            "b": (-M2 * m + n2 * k) / sigma}
    scale = amplitude / max(abs(amps["u"]), abs(amps["w"]))

    def mode(amp):
        return lambda x, z: amp * scale * np.cos(k * x + m * z)

    model.set_fields(**{c: fill(model, c, mode(a))
                        for c, a in amps.items()})
    return model, tw, sigma, n2


# ================================================================
#  Declarations, references, wiring
# ================================================================
def test_declares_m2_and_references_the_constant_coriolis():
    tw = nh.ThermalWindBackground(m2=-1e-7)
    decls = {str(d.name) for d in tw.parameter_declarations}
    assert str(nh.params.THERMAL_WIND_M2) in decls
    # reads the constant f0 provided by the f-plane Coriolis: the
    # balance f0 dV/dz = dB/dx is never the caller's to keep
    refs = {str(r.name) for r in tw.parameter_references}
    assert str(nh.params.CORIOLIS_F0) in refs
    fields = {r.name for r in tw.field_references}
    assert fields == {"u", "v", "w", "b"}


def test_requires_a_constant_coriolis_parameter():
    # without nh.FPlaneCoriolis there is no coriolis.f0 provide, and
    # V(z) = M^2 z/f0 is not defined
    with pytest.raises(MissingParameterError, match=r"coriolis\.f0"):
        front_model(coriolis=False)


def test_ramped_m2_reports_the_time_dependent_linear_parameter():
    # a ramped front feeds the linear term -> reported for frozen-L
    # steppers (AR-D7); a plain float reports nothing
    ramped = nh.ThermalWindBackground(
        m2=fr.model.Ramp(0.0, -1e-7, period=1.0))
    assert str(nh.params.THERMAL_WIND_M2) in \
        ramped.time_dependent_linear_parameters()
    assert nh.ThermalWindBackground(m2=-1e-7) \
        .time_dependent_linear_parameters() == ()


def test_the_linear_operator_declares_both_parameters():
    # the term's L depends on m2 AND the consumed f0, so a propagator
    # gradient w.r.t. either is refused under a frozen-L stepper
    tw = nh.ThermalWindBackground(m2=-1e-7)
    assert set(tw.linear_operator_parameters()) == {
        str(nh.params.THERMAL_WIND_M2), str(nh.params.CORIOLIS_F0)}


def test_background_velocity_is_the_balanced_shear_profile():
    tw = nh.ThermalWindBackground(m2=-1e-7, reference_height=100.0)
    profile = tw.background_velocity(F0)
    zs = np.linspace(0.0, 200.0, 5)
    assert np.allclose([profile(z) for z in zs],
                       -1e-7 * (zs - 100.0) / F0)


def test_stratification_n2_realizes_the_requested_richardson_number():
    tw = nh.ThermalWindBackground(m2=M2)
    for ri in (0.25, 0.5, 0.75, 2.0):
        n2 = tw.stratification_n2(F0, richardson_number=ri)
        # Ri = N^2 f0^2 / M^4, recovered exactly
        assert n2 * F0 ** 2 / M2 ** 2 == pytest.approx(ri, rel=1e-12)
        # ... and it is the Ri = 1 threshold that the growth rate flips
        # sign at (positive below, exactly zero above)
        assert (si_growth_rate(F0, M2, n2) > 0.0) is (ri < 1.0)


# ================================================================
#  Thermal wind holds by construction
# ================================================================
@pytest.mark.parametrize(("f0", "m2"), [(1.3, -0.7), (-2.0, 0.5),
                                        (1e-4, -1e-7)])
def test_thermal_wind_balance_holds_by_construction(f0, m2):
    r"""``f0 dV/dz == dB/dx`` for whatever the caller supplies.

    The module takes ONE number (``m2``) and reads ``f0`` off the
    assembly, so the two carriers of the mean state cannot disagree.
    Read them back out of the step path itself: on a uniform ``u = 1``,
    ``w = 1`` perturbation the interpolations are exact, so the tilting
    tendency is ``-dV/dz`` and the buoyancy tendency is ``-dB/dx``.
    """
    model, _ = front_model(f0=f0, m2=m2, n2=2.0)
    ones = {c: np.ones(model.state[c].shape) for c in ("u", "w")}
    model.set_fields(**ones)
    d_x = model.tendency(
        model.state, filter=tp.owned_by(nh.ThermalWindBackground),
        constraints=False)
    dv_dz = -float(np.asarray(d_x["v"].data).ravel()[0])
    db_dx = -float(np.asarray(d_x["b"].data).ravel()[0])
    assert f0 * dv_dz == pytest.approx(db_dx, rel=1e-14)
    assert db_dx == pytest.approx(m2, rel=1e-14)


# ================================================================
#  The two terms in isolation
# ================================================================
def test_the_two_terms_hit_exactly_their_own_fields():
    f0, m2, n2 = 1.3, -0.7, 2.0
    model, _ = front_model(f0=f0, m2=m2, n2=n2)
    rng = np.random.default_rng(1)
    model.set_fields(**{c: rng.standard_normal(model.state[c].shape)
                        for c in ("u", "v", "w", "b")})
    st = model.state
    d_x = model.tendency(
        st, filter=tp.owned_by(nh.ThermalWindBackground),
        constraints=False)
    # the mean-shear tilting: -(M^2/f0) w, on the v faces
    tilting = -((m2 / f0) * st["w"].to(st["v"]))
    assert np.allclose(np.asarray(d_x["v"].data),
                       np.asarray(tilting.data), atol=1e-14)
    # the horizontal buoyancy advection: -M^2 u, on the b cells
    advection = -(m2 * st["u"].to(st["b"]))
    assert np.allclose(np.asarray(d_x["b"].data),
                       np.asarray(advection.data), atol=1e-14)
    # ... and nothing else: no zonal, no vertical momentum term
    assert float(np.max(np.abs(np.asarray(d_x["u"].data)))) == 0.0
    assert float(np.max(np.abs(np.asarray(d_x["w"].data)))) == 0.0


def test_the_terms_are_visible_to_the_linear_operator():
    # linear=True: fr.model.linearize / the eigenmode consumers keep
    # them. A u-only state has no other carrier for db/dt (the
    # stratification restoring reads w), so a nonzero db/dt under the
    # linear filter is the -M^2 u advection alone.
    model, _ = front_model(m2=-0.7)
    model.set_fields(u=np.ones(model.state["u"].shape))
    d_x = model.tendency(model.state, filter=tp.linear,
                         constraints=False)
    assert float(np.max(np.abs(np.asarray(d_x["b"].data)))) \
        == pytest.approx(0.7, rel=1e-12)


def test_a_bare_buoyancy_tracer_also_carries_the_front():
    # the module needs a b field, not a background stratification:
    # nh.BuoyancyTracer() (no N^2) serves it too
    model, _ = front_model(m2=-0.7, buoyancy=False)
    model.set_fields(u=np.ones(model.state["u"].shape))
    d_x = model.tendency(
        model.state, filter=tp.owned_by(nh.ThermalWindBackground),
        constraints=False)
    assert float(np.max(np.abs(np.asarray(d_x["b"].data)))) \
        == pytest.approx(0.7, rel=1e-12)


def test_the_terms_serve_the_walled_thin_channel():
    r"""The example's geometry: walled ``z``, thin periodic ``y``.

    The symmetric-instability setup is a rigid-lid channel with a
    single y cell and the full nonlinear model on top (advection plus
    the Smagorinsky closure). The staggered ``.to`` averages cross the
    walled ``z`` axis and the one-cell ``y`` axis here, so this pins
    that the terms still evaluate exactly and the model steps.
    """
    f0, m2, n2 = 1e-4, -1e-7, 2.5e-7
    model, _ = front_model(nx=16, ny=1, nz=16, lx=552.0, ly=1.0,
                           lz=200.0, f0=f0, n2=n2, m2=m2, dt=3.0,
                           periodic_z=False, advection=nh.CenteredAdvection())
    rng = np.random.default_rng(5)
    model.set_fields(**{
        c: 1e-4 * rng.standard_normal(model.state[c].shape)
        for c in ("u", "v", "w", "b")})
    st = model.state
    d_x = model.tendency(
        st, filter=tp.owned_by(nh.ThermalWindBackground),
        constraints=False)
    assert np.allclose(np.asarray(d_x["v"].data),
                       np.asarray((-(m2 / f0) * st["w"].to(st["v"])).data),
                       atol=1e-18)
    assert np.allclose(np.asarray(d_x["b"].data),
                       np.asarray((-(m2 * st["u"].to(st["b"]))).data),
                       atol=1e-18)
    model.advance(5)
    assert np.isfinite(np.asarray(model.state["b"].data)).all()


# ================================================================
#  Energy honesty: the front is the SOLE non-conservative source
# ================================================================
def test_the_front_is_the_only_nonconservative_energy_source():
    r"""Everything except the thermal wind is exactly M-skew.

    Under ``M = diag(1, 1, delta^2, 1/N^2)`` (``nh.energy``) the
    Coriolis pair and the buoyancy coupling are machine-exact skew on a
    divergence-free state, so the *entire* energy source of the linear
    operator is the two thermal-wind terms. Hence no perturbation
    quadratic form is conserved — the balanced mean state is a genuine
    reservoir, which is the instability, and the honest gates are the
    term identities and the growth rate, never a conserved norm.
    """
    f0, m2, n2 = 1.3, -0.7, 2.0
    model, _ = front_model(f0=f0, m2=m2, n2=n2)
    rng = np.random.default_rng(3)
    model.set_fields(**{c: rng.standard_normal(model.state[c].shape)
                        for c in ("u", "v", "w", "b")})
    model.advance(1)          # the projection makes the state div-free
    st = model.state
    div = Divergence()(VectorField(
        {c: st[c] for c in ("u", "v", "w")}))
    assert float(np.max(np.abs(np.asarray(div.data)))) < 1e-12

    def rate(filter_):
        d_x = model.tendency(st, filter=filter_, constraints=True)
        weights = {"u": 1.0, "v": 1.0, "w": 1.0, "b": 1.0 / n2}
        parts = [weights[c]
                 * float((st[c] * d_x[c]).integrate().data.ravel()[0])
                 for c in ("u", "v", "w", "b")]
        return sum(parts), sum(abs(p) for p in parts)

    full, scale = rate(tp.linear)
    rest, _ = rate(tp.linear & ~tp.owned_by(nh.ThermalWindBackground))
    assert abs(rest) < 1e-12 * scale
    assert abs(full) > 1e-3 * scale


def test_rest_state_stays_zero_over_a_short_run():
    # both terms are linear in the perturbation, so a zero perturbation
    # is a fixed point to round-off: the balanced mean state rides the
    # module's parameters, never the state
    model, _ = front_model(m2=-0.7, dt=2e-3)
    model.advance(5)
    for name in ("u", "v", "w", "b"):
        peak = float(np.max(np.abs(np.asarray(model.state[name].data))))
        assert peak < 1e-14


# ================================================================
#  Symmetric instability: growth exactly below Ri = 1
# ================================================================
@pytest.mark.parametrize(("ri", "unstable"),
                         [(0.25, True), (2.0, False)],
                         ids=["ri_0.25", "ri_2.0"])
def test_random_perturbation_grows_only_below_unit_richardson(ri,
                                                              unstable):
    r"""A small random state grows for ``Ri < 1`` and not for ``Ri > 1``.

    The qualitative gate, on the same domain for both cases: the linear
    model (``advection=None``) from a tiny random perturbation. Below
    ``Ri = 1`` the balanced state has PV of the wrong sign and the
    perturbation energy grows exponentially at the fastest-growing
    slantwise rate; above it every mode is oscillatory and the energy
    stays bounded.
    """
    dt, chunks, per_chunk = 50.0, 8, 100
    n2 = n2_for(ri)
    lx = abs(si_slope(F0, M2, n2_for(0.25))) * LZ
    model, _ = front_model(nx=32, ny=1, nz=32, lx=lx, ly=1.0, lz=LZ,
                           f0=F0, n2=n2, m2=M2, dt=dt)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        c: 1e-6 * rng.standard_normal(model.state[c].shape)
        for c in ("u", "v", "w", "b")})
    times, energies = growth_history(model, n2, chunks=chunks,
                                     per_chunk=per_chunk, dt=dt)
    measured = tail_growth_rate(times, energies)
    reference = si_growth_rate(F0, M2, n2_for(0.25))
    if unstable:
        # several e-foldings of energy growth over the record, at
        # (nearly) the fastest-growing rate — a random state still
        # carries the slower slants, so this end is loose on purpose
        assert energies[-1] / energies[0] > 1e3
        assert 0.8 * reference < measured < 1.02 * reference
    else:
        assert energies[-1] / energies[0] < 2.0
        assert abs(measured) < 0.02 * reference
        # ... and flat, not merely slow: the tail is level
        assert energies[-1] / energies[chunks // 2] < 1.1


@pytest.mark.parametrize("ri", [0.25, 0.5, 0.75])
def test_growth_rate_matches_the_analytic_symmetric_mode(ri):
    r"""The seeded optimal mode grows at the analytic rate (0.5%).

    Seed the analytic fastest-growing eigenvector on a domain holding
    exactly one of its wavelengths and integrate the linear model: the
    measured energy growth matches

        sigma = sqrt(( sqrt((N^2-f0^2)^2 + 4 M^4) - (N^2+f0^2) )/2)

    at all three Richardson numbers of the symmetric-instability
    example. The residual is a *discretization* deficit of the same
    relative size at every ``Ri`` (the wavevector is the same), pinned
    to second order by the companion test below.

    This gates BOTH term signs, not merely their product: flipping
    either one alone turns the cross term of the Rayleigh quotient off,
    leaving ``sigma^2 (k^2+m^2) = -(N^2 k^2 + f0^2 m^2) < 0`` — no
    instability at any slope — while flipping both mirrors the optimal
    slope, which this seeded (negative-slope) mode does not carry.
    """
    dt, chunks, per_chunk = 50.0, 8, 50
    model, _, sigma, n2 = seeded_model(ri, n=32, dt=dt)
    times, energies = growth_history(model, n2, chunks=chunks,
                                     per_chunk=per_chunk, dt=dt)
    measured = tail_growth_rate(times, energies)
    assert measured == pytest.approx(sigma, rel=6e-3)
    assert energies[-1] / energies[0] > 5.0


def test_growth_rate_deficit_is_second_order_in_the_mesh():
    r"""The gap to the analytic rate is discretization, and converges.

    The measured rate sits ~0.5% below the continuous ``sigma`` at
    32^2 because the C-grid differences and the staggered ``.to``
    averages carry the mode's ``k dx`` truncation, not because a term
    is wrong: halving the mesh spacing cuts the deficit by ~4.
    """
    dt, chunks, per_chunk = 50.0, 8, 50
    deficits = []
    for n in (16, 32):
        model, _, sigma, n2 = seeded_model(0.25, n=n, dt=dt)
        times, energies = growth_history(model, n2, chunks=chunks,
                                         per_chunk=per_chunk, dt=dt)
        deficits.append(1.0 - tail_growth_rate(times, energies) / sigma)
    assert deficits[0] > deficits[1] > 0.0
    assert 3.0 < deficits[0] / deficits[1] < 5.0
    assert deficits[1] < 1e-2


# ================================================================
#  Differentiability: grad through a short run w.r.t. the front
# ================================================================
def test_thermal_wind_grad_wrt_m2_matches_fd():
    r"""``jax.grad`` w.r.t. ``m2`` matches a central FD (rtol 1e-4).

    Differentiability policy (AGENTS.md): the thermal-wind terms are
    step-path tendency code, so ``jax.grad`` of a quadratic loss through
    a short ``Model.propagator`` run w.r.t. the front's ``m2`` is finite
    and FD-matched. Both terms are a scalar scaling of a linear
    interpolation of the state — the one division is by the assembly's
    constant ``f0``, never by a state value, so there is no masked
    singularity to poison the VJP.
    """
    m2_0 = -0.7
    model, _ = front_model(nx=8, ny=8, nz=8, m2=m2_0, dt=2e-3)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        c: 0.1 * rng.standard_normal(model.state[c].shape)
        for c in ("u", "v", "w", "b")})
    run = model.propagator(wrt=(str(nh.params.THERMAL_WIND_M2),),
                           steps=6)

    def loss(m2):
        out = run((m2,))
        return sum(jnp.sum(f.data ** 2) for f in out.state)

    grad = float(jax.grad(loss)(jnp.asarray(m2_0)))
    assert np.isfinite(grad)
    eps = 1e-5
    fd = (float(loss(jnp.asarray(m2_0 + eps)))
          - float(loss(jnp.asarray(m2_0 - eps)))) / (2.0 * eps)
    assert grad == pytest.approx(fd, rel=1e-4)
