"""The two shallow-water energy families.

``ekin`` / ``epot`` are the linearized quadratics (the invariant of
the *linear* model, the eigenmode metric); ``ekin_full`` /
``epot_full`` / ``etot_full`` are the thickness-weighted energy the
Sadourny scheme plus the core's gravity term conserve exactly. The
gates here:

- the full diagnostic **reproduces the validated per-space invariant**
  (the hand-written ``h_energy`` form of ``test_sadourny`` /
  ``tests/validation/test_spherical_shallowwater``) to rounding, on
  periodic, walled and chart grids;
- its semi-discrete production rate under gravity + Sadourny is
  **machine zero** (flat AND sphere) — the diagnostic and the scheme
  agree on what is conserved;
- the linearized pair is *not* that invariant (it is produced at
  O(Ro) by the same terms), and its values are the physical-velocity
  quadratics (the chart branch collapses to the flat spelling now that
  the state components are physical, D4).
"""
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

from .conftest import gaussian_bump, make_grid, make_model

CSQR = 0.7
RO = 0.4
LAT_MAX = float(np.deg2rad(80.0))

#: the exactly-conserving pair (the Coriolis term is skew under the
#: LINEARIZED metric instead — sadourny.py module docstring)
SCHEME = (fr.model.term_predicates.named("Core/gravity")
          | fr.model.term_predicates.named("SadournyAdvection/advect"))

NAMES = ("u", "v", "p")


# ================================================================
#  Models: flat (periodic / walled) and the lat-lon sphere chart
# ================================================================
def sphere_grid(nlon=16, nlat=8, radius=1.0):
    """Lat-lon sphere chart grid (the documented sw.Model recipe)."""
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=radius, lat_extent=(-LAT_MAX, LAT_MAX))


def sphere_model(*, ro=RO, omega=1.5):
    """Assemble the spherical shallow-water preset."""
    return sw.Model(
        grid=sphere_grid(),
        core=sw.Core(froude_number=ro, depth=CSQR,
                     coords=("lon", "lat")),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.RotationCoriolis(
            omega=(0.0, 0.0, omega), coords=("lon", "lat"),
            metric_weight="csqr"),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            2e-3, order=3))


def flat_model(*, periodic_y=True, ro=RO):
    """Assemble a flat Cartesian model (walled y when asked)."""
    grid = make_grid(periodic_y=periodic_y)
    return make_model(grid, csqr=CSQR, rossby_number=ro, f0=1.0,
                      advection=True)


def set_random(model, seed=11):
    """Fill the prognostics (walls are structural: u.n = 0)."""
    rng = np.random.default_rng(seed)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        p=0.3 * rng.standard_normal(model.state["p"].shape))


MODELS = {
    "periodic": flat_model,
    "channel": lambda: flat_model(periodic_y=False),
    "sphere": sphere_model,
}


@pytest.fixture(params=sorted(MODELS), ids=sorted(MODELS))
def model(request):
    """Return a random-state model on each grid family."""
    m = MODELS[request.param]()
    set_random(m)
    return m


# ================================================================
#  The reference: the hand-written per-space invariant (validated)
# ================================================================
def reference_energy(model):
    """``E = sum hbar U^2 / 2 + p^2 / 2``, per-space sums (physical).

    The form the C2 work validated (the ``h_energy`` helpers of
    ``test_sadourny`` and ``tests/validation``): each velocity
    quadratic integrated on its OWN staggered space, so
    ``integrate`` applies that space's ``sqrt(g)`` and measure. The
    velocities are the **physical** components on every grid
    (``physical_state_components.md``), so the chart form carries no
    ``g_ii`` factor (it is folded into the physical ``U^2``).
    """
    z = model.state
    ro = float(model.parameters[fr.model.params.SCALING_NONLINEARITY])
    u, v, p = z["u"], z["v"], z["p"]
    h = z["csqr"].to(p) + ro * p
    ku = 0.5 * u * u * h.to(u)
    kv = 0.5 * v * v * h.to(v)
    return sum(float(part.integrate().data.ravel()[0])
               for part in (ku, kv, 0.5 * p * p))


def energy_rate(model, diagnostic):
    """Semi-discrete d/dt of a diagnostic under gravity + Sadourny.

    The exact directional derivative of the discrete functional along
    the discrete tendency (a jvp — no time stepping, no differencing
    error), so a nonzero result is the scheme's production, not the
    stepper's.
    """
    z = model.state
    dz = model.tendency(z, filter=SCHEME)

    def total(data):
        state = z.replace(**{name: z[name].with_data(data[name])
                             for name in NAMES})
        return diagnostic(state, model.parameters).integrate(
        ).data.ravel()[0]

    primal = {name: z[name].data for name in NAMES}
    tangent = {name: dz[name].data for name in NAMES}
    return float(jax.jvp(total, (primal,), (tangent,))[1])


def linearized(state, params):
    """Return the linearized total ``ekin + epot`` density."""
    return (sw.diagnostics.ekin(state, params)
            + sw.diagnostics.epot(state, params))


# ================================================================
#  The full (thickness-weighted) energy IS the scheme's invariant
# ================================================================
def test_full_energy_matches_the_validated_per_space_invariant(model):
    got = model.diagnostics.etot_full().integrate().item()
    expected = reference_energy(model)
    assert abs(got - expected) / abs(expected) < 1e-14


def test_full_energy_splits_into_kinetic_and_potential(model):
    kin = model.diagnostics.ekin_full().integrate().item()
    pot = model.diagnostics.epot_full().integrate().item()
    total = model.diagnostics.etot_full().integrate().item()
    assert kin > 0.0
    assert pot > 0.0
    assert abs(kin + pot - total) / abs(total) < 1e-14


def test_semi_discrete_energy_rate_is_machine_zero(model):
    # THE gate: gravity + Sadourny produce no etot_full — the
    # diagnostic and the scheme agree on the invariant (flat AND
    # sphere; measured ~1e-16 relative)
    rate = energy_rate(model, sw.diagnostics.etot_full)
    scale = sum(
        abs(energy_rate(model, diagnostic))
        for diagnostic in (sw.diagnostics.ekin_full,
                           sw.diagnostics.epot_full))
    assert abs(rate) / scale < 1e-13


def test_the_linearized_energy_is_not_the_nonlinear_invariant(model):
    # the whole point of the two families: the same terms that leave
    # etot_full at machine zero produce the linearized quadratic at
    # O(Ro) — a user monitoring ekin + epot sees a spurious drift
    rate = energy_rate(model, linearized)
    scale = abs(energy_rate(model, sw.diagnostics.ekin_full))
    assert abs(rate) / scale > 1e-3


# ================================================================
#  The individual densities (values, spaces, params)
# ================================================================
def test_thickness_is_the_full_geopotential():
    # the thickness() diagnostic function is retired: the full
    # geopotential is now the core's DIAGNOSE-stage state field,
    # rewritten every substage. Pin the stage's spelling —
    # h = csqr + eps * p (nondim parity: the (Fr/eps)^2 ratio is an
    # exact 1, so the coefficient collapses to eps = Ro bitwise).
    model = flat_model()
    set_random(model)
    z = model.state
    assert "thickness" in z.component_names
    ctx = SimpleNamespace(params=model.parameters, clock=0.0)
    h = model.module(sw.Core)._update_thickness(z, ctx)["thickness"]
    expected = z["csqr"].to(z["p"]) + RO * z["p"]
    assert np.array_equal(np.asarray(h.data),
                          np.asarray(expected.data))


def test_epot_full_drops_the_inverse_csqr_weight():
    # epot = 0.5 p^2 / c^2 (the M weight), epot_full = 0.5 p^2
    model = flat_model()
    set_random(model)
    pot = np.asarray(model.diagnostics.epot_full().data)
    lin = np.asarray(model.diagnostics.epot().data)
    np.testing.assert_allclose(pot, CSQR * lin, rtol=1e-15)


def test_full_kinetic_energy_carries_the_metric_on_the_sphere():
    # the physical KE carries only the sqrt_g Jacobian on each
    # velocity's own space (no g_ii — the velocities are physical);
    # integrate re-applies the centre sqrt_g (D4)
    model = sphere_model()
    set_random(model)
    grid = model.grid
    z = model.state
    u, v, p = z["u"], z["v"], z["p"]
    h = z["csqr"].to(p) + RO * p
    u_bare, v_bare = (u.function_space.bare, v.function_space.bare)
    e_u = grid.metric(u_bare, "sqrt_g") * u * u * h.to(u)
    e_v = grid.metric(v_bare, "sqrt_g") * v * v * h.to(v)
    sqrt_g = grid.metric(p.function_space.bare, "sqrt_g")
    expected = 0.5 * (e_u.to(p) + e_v.to(p)) / sqrt_g
    got = model.diagnostics.ekin_full()
    assert got.function_space is p.function_space
    np.testing.assert_allclose(np.asarray(got.data),
                               np.asarray(expected.data), atol=1e-14)


def test_full_kinetic_energy_is_the_flat_quadratic_on_a_flat_grid():
    model = flat_model()
    set_random(model)
    z = model.state
    u, v, p = z["u"], z["v"], z["p"]
    h = z["csqr"].to(p) + RO * p
    expected = 0.5 * ((u * u * h.to(u)).to(p)
                      + (v * v * h.to(v)).to(p))
    assert np.array_equal(
        np.asarray(model.diagnostics.ekin_full().data),
        np.asarray(expected.data))


# ================================================================
#  The linearized family: unchanged (the M-metric quadratics)
# ================================================================
def test_linearized_diagnostics_keep_the_cartesian_formulas():
    model = flat_model()
    set_random(model)
    z = model.state
    center = z["p"].function_space
    u_c, v_c = z["u"].to(center), z["v"].to(center)
    assert np.array_equal(
        np.asarray(model.diagnostics.ekin().data),
        np.asarray(0.5 * (u_c.data**2 + v_c.data**2)))
    assert np.array_equal(
        np.asarray(model.diagnostics.epot().data),
        np.asarray(0.5 * z["p"].data**2 / CSQR))


def test_the_linear_model_conserves_the_m_norm_exactly():
    # the counterpart of the machine-zero gate above: with the
    # advection off, the exact invariant is the M-norm — the
    # quadratics summed on the fields' OWN staggered spaces. The
    # ekin / epot DENSITIES square the centre-interpolated
    # velocities, so their integral is a centre-sampled proxy of
    # that norm (docstring); the norm itself is conserved to machine
    # precision.
    model = make_model(make_grid(), csqr=CSQR, rossby_number=RO,
                       f0=1.0, advection=False)
    set_random(model)
    z = model.state
    dz = model.tendency(z)
    weights = {"u": 1.0, "v": 1.0, "p": 1.0 / CSQR}
    terms = [
        weight * float((z[name] * dz[name]).integrate(
        ).data.ravel()[0])
        for name, weight in weights.items()]
    scale = sum(abs(term) for term in terms)
    assert abs(sum(terms)) / scale < 1e-13


def test_the_diagnostics_namespace_exposes_both_families():
    model = flat_model()
    # the thickness FUNCTION is retired (the core's DIAGNOSE-stage
    # state field replaces it), so the mapping no longer names it
    names = set(sw.diagnostics.DIAGNOSTICS)
    assert names == {"ekin", "epot", "ekin_full", "epot_full",
                     "etot_full", "pot_vort"}
    for name in names:
        assert isinstance(getattr(model.diagnostics, name)(),
                          fr.spatial.ScalarField)


# ================================================================
#  Potential vorticity: q = (f + Ro zeta) / h on the vorticity corner
# ================================================================
def test_pot_vort_is_the_scaled_vector_invariant():
    # the spelling gate: q = (f + Ro zeta) / h, on the SAME corner as
    # rel_vort, with f read from the carried f_coriolis field and h
    # the full geopotential thickness (bitwise vs a hand assembly)
    model = flat_model()
    set_random(model)
    z = model.state
    zeta = z.rel_vort
    corner = zeta.function_space
    h = (z["csqr"].to(z["p"]) + RO * z["p"]).to(corner)
    expected = (z["f_coriolis"].to(corner) + RO * zeta) / h
    got = model.diagnostics.pot_vort()
    assert got.function_space is corner
    assert np.array_equal(np.asarray(got.data),
                          np.asarray(expected.data))
    assert got.metadata.name == "pot_vort"
    assert got.metadata.units == "s/m^2"


def test_pot_vort_of_a_rest_state_is_f_over_h():
    # the analytic anchor (gate i): at rest zeta = 0 exactly, so the
    # PV collapses to f / h everywhere
    model = flat_model()          # rest state (no set_random)
    z = model.state
    corner = z.rel_vort.function_space
    expected = z["f_coriolis"].to(corner) / (
        z["csqr"].to(z["p"]) + RO * z["p"]).to(corner)
    got = model.diagnostics.pot_vort()
    assert np.array_equal(np.asarray(got.data),
                          np.asarray(expected.data))


def test_pot_vort_rest_state_on_the_sphere_is_f_over_h():
    # the metric discriminator (gate ii): on the lat-lon chart a rest
    # state has zero metric-curl vorticity, so the PV is exactly the
    # latitude-varying f divided by the corner thickness — to roundoff
    model = sphere_model()        # rest state
    z = model.state
    corner = z.rel_vort.function_space
    h = (z["csqr"].to(z["p"]) + RO * z["p"]).to(corner)
    expected = z["f_coriolis"].to(corner) / h
    got = model.diagnostics.pot_vort()
    # f genuinely varies with latitude (this is not the trivial
    # f = const case), and zeta is machine zero at rest
    assert np.ptp(np.asarray(z["f_coriolis"].data)) > 1.0
    assert np.max(np.abs(np.asarray(z.rel_vort.data))) == 0.0
    np.testing.assert_allclose(np.asarray(got.data),
                               np.asarray(expected.data),
                               rtol=0.0, atol=0.0)


def test_pot_vort_extrema_are_materially_conserved():
    # the material-conservation sanity (gate iii): the vector-invariant
    # PV is the tracer the Sadourny + conserving-Coriolis flux
    # transports, so a short nonlinear f-plane run advects its extrema
    # to advective-scheme tolerance (the OLD-stack (zeta+f)/h would
    # drift at O(Ro-1) ~ 1e-2 over the same run — see the module note)
    model = make_model(make_grid(32), csqr=1.0, rossby_number=RO,
                       f0=1.0, advection=True, dt=1.5e-3)
    model.set_fields(p=gaussian_bump(amp=0.5, sigma=0.12))
    model.advance(100)                          # spin up real vorticity
    q0 = np.asarray(model.diagnostics.pot_vort().data)
    lo0, hi0 = q0.min(), q0.max()
    assert hi0 - lo0 > 0.1                       # a non-trivial PV range
    model.advance(400)
    q1 = np.asarray(model.diagnostics.pot_vort().data)
    span = hi0 - lo0
    # extrema drift is small relative to the PV span (advective
    # tolerance; measured ~1e-4 of the span, vs ~0.1 for (zeta+f)/h)
    assert abs(q1.min() - lo0) / span < 5e-3
    assert abs(q1.max() - hi0) / span < 5e-3
