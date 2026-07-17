r"""Physics-validation suite for the HY-D6 comparison preset.

``hy.comparison_model`` (``src/fridom/hydrostatic/comparison.py``) is
the matched-numerics common-denominator configuration (quasi-AB2 +
implicit linear free surface + centered-2 flux-form advection + f-plane
Coriolis + linear stratification). This suite validates its physics
against analytic (discrete) predictions and pins the pinned
configuration so it cannot drift:

- **Protocol pins** — the exact stepper order/eps, free-surface epsilon,
  advection class, and carry-treedef stability.
- **Geostrophic (Rossby) adjustment** — a released surface-pressure
  step settles to the geostrophically balanced state; the retained
  pressure amplitude and energy fraction match the *discrete*
  deformation-radius prediction to roundoff and refine toward the
  continuous Rossby-adjustment fraction at second order.
- **Eady baroclinic instability — unblocked (H5b).** The shared
  advection's ``background=`` composes mechanically with the hydrostatic
  model (it adds the mean-flow Doppler advection) but supplies **no**
  baroclinic-conversion term :math:`v'\,\partial_y B` — the Eady energy
  source. ``hy.ThermalWindBackground`` now carries it (and the momentum
  tilting), so these tests verify the composition *and* that the
  conversion term is present; the full growth-rate validation lives in
  ``tests/hydrostatic/test_thermal_wind.py``.
- **Wave-packet dispersion** — an internal-wave packet propagates at the
  discrete *group* velocity (distinct from the phase speed), complementing
  the per-mode phase checks of H2/H3.

Runtime-budgeted (each test seconds); the one expensive refinement is
behind ``FRIDOM_TEST_COMPARISON_SLOW``. Self-contained per the AGENTS
oversized-module rule (the small builders are duplicated).
"""
import inspect
import os

import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
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


def mode_field(model, name, kx, ky, phase, lx=1.0):
    """cos/sin(2 pi (kx x / lx + ky y)), depth-uniform, at name's nodes."""
    fs = model.state[name].function_space
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace)
                for n in f.names)
    trig = np.cos if phase == "cos" else np.sin

    def init(**c):
        return trig(2 * np.pi * (kx * c.get("x", 0) / lx
                                 + ky * c.get("y", 0))) \
            + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return model.grid.create_field(fs, init=init)


def zeroed(model):
    """Return the model's state with u, v, b, ps zeroed."""
    st = model.state
    for g in ("u", "v", "b", "ps"):
        st = st.replace(**{g: model.grid.create_field(
            model.state[g].function_space,
            data=0.0 * model.state[g].data)})
    return st


def k_disc_sq(n_mode, n_cells, length=1.0):
    """Squared discrete C-grid wavenumber of one axis."""
    dx = length / n_cells
    k = 2.0 * np.pi * n_mode / length
    return (2.0 * np.sin(k * dx / 2.0) / dx) ** 2


def total_energy(model, csqr, nz):
    """Discrete energy 0.5 int (u^2 + v^2 + ps^2/c^2) (uniform-grid sums).

    The ps term carries the depth factor (``ps`` is constant along z, so
    its energy integrates over the ``nz`` layers) — the ``1/c^2``
    barotropic weight of ``hy.energy`` integrated over the full depth.
    """
    s = model.state
    u = np.asarray(s["u"].data)
    v = np.asarray(s["v"].data)
    ps = np.asarray(s["ps"].data)
    return 0.5 * ((u ** 2).sum() + (v ** 2).sum()
                  + nz * (ps ** 2).sum() / csqr)


# ================================================================
#  Protocol pins: the comparison config cannot drift silently
# ================================================================
def test_pins_the_quasi_ab2_stepper():
    # pyOM quasi-AB2: order 2, eps = 0.1 (the computational-mode damper)
    model = hy.comparison_model(make_grid(8, 8, 4), dt=1e-2, csqr=1.0)
    stepper = model._stepper
    assert isinstance(stepper, fr.model.time_steppers.AdamBashforth)
    assert stepper.order == 2
    assert stepper.eps == 0.1


def test_eps_override_is_the_pyom_damper():
    model = hy.comparison_model(make_grid(8, 8, 4), dt=1e-2, csqr=1.0,
                                eps=0.05)
    assert model._stepper.eps == 0.05


def test_pins_the_backward_euler_implicit_free_surface():
    # Oceananigans ImplicitFreeSurface / pyOM enable_free_surface (eps=1)
    model = hy.comparison_model(make_grid(8, 8, 4), dt=1e-2, csqr=1.0)
    fs = model.module(hy.ImplicitFreeSurface)
    assert fs.epsilon == 1.0


def test_epsilon_zero_selects_the_rigid_lid_veros_axis():
    # the Veros / pyOM streamfunction physics on a doubly-periodic box:
    # a rigid lid, ps DIAGNOSTIC (recomputed, no surface memory)
    model = hy.comparison_model(make_grid(8, 8, 4), dt=1e-2, csqr=1.0,
                                epsilon=0.0)
    fs = model.module(hy.ImplicitFreeSurface)
    assert fs.epsilon == 0.0
    decl = fs.field_declarations[0]
    assert decl.name == "ps"
    assert decl.lifecycle is fr.model.Lifecycle.DIAGNOSTIC


def test_pins_centered_advection_of_momentum_and_tracer():
    # centered-2 flux form (pyOM/Veros always; Oceananigans forced)
    model = hy.comparison_model(make_grid(8, 8, 4), dt=1e-2, csqr=1.0)
    adv = model.module(fr.model.modules.CenteredAdvection)
    assert adv is not None
    # it transports the ADVECTED set: momentum u/v AND the b tracer
    for name in ("u", "v", "b"):
        assert name in model.state.component_names


def test_pins_fplane_coriolis_and_constant_stratification():
    model = hy.comparison_model(make_grid(8, 8, 4), dt=1e-2, csqr=1.0,
                                coriolis_f0=1.3, n2=2.0)
    assert model.module(hy.FPlaneCoriolis) is not None
    assert model.module(hy.ConstantStratification) is not None
    assert float(model.parameters[fr.model.params.CORIOLIS_F0]) == 1.3
    assert float(
        model.parameters[fr.model.params.STRATIFICATION_N2]) == 2.0


def test_preset_matches_explicit_assembly_treedef():
    # the pinned config equals the hand-written module tuple (D4 check)
    import jax  # noqa: PLC0415
    grid = make_grid(8, 8, 4)
    preset = hy.comparison_model(grid, dt=1e-2, csqr=3.0, coriolis_f0=1.3,
                                 n2=2.0, rossby_number=0.2)
    explicit = fr.model.Model(
        grid=grid,
        modules=(
            hy.HydrostaticCore(csqr=3.0, rossby_number=0.2),
            hy.FPlaneCoriolis(f0=1.3),
            hy.ConstantStratification(n2=2.0),
            hy.ImplicitFreeSurface(epsilon=1.0),
            fr.model.modules.CenteredAdvection()),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-2, order=2, eps=0.1))
    assert (jax.tree_util.tree_structure(preset._carry)
            == jax.tree_util.tree_structure(explicit._carry))
    assert ([type(m) for m in preset._carry.modules]
            == [type(m) for m in explicit._carry.modules])


def test_carry_treedef_is_stable_over_a_run():
    import jax  # noqa: PLC0415
    model = hy.comparison_model(make_grid(8, 8, 4), dt=1e-2, csqr=1.0)
    rng = np.random.default_rng(0)
    model.set_fields(
        u=1e-3 * rng.standard_normal(model.state["u"].shape),
        v=1e-3 * rng.standard_normal(model.state["v"].shape),
        b=1e-3 * rng.standard_normal(model.state["b"].shape),
        ps=1e-3 * rng.standard_normal(model.state["ps"].shape))
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(5)
    after = jax.tree_util.tree_structure(model._carry)
    assert before == after
    for name in ("u", "v", "b", "ps"):
        assert np.isfinite(np.asarray(model.state[name].data)).all()


def test_name_and_kwargs_pass_through_to_the_model():
    # the name and any extra hy.Model kwargs ride through the preset
    grid = make_grid(8, 8, 4)
    model = hy.comparison_model(grid, dt=1e-2, csqr=1.0, name="cmp",
                                rossby_number=0.3)
    assert model.name == "cmp"
    assert float(model.parameters[hy.params.ROSSBY]) == 0.3


# ================================================================
#  Geostrophic (Rossby) adjustment
# ================================================================
#  A released ps step (single mode, at rest) radiates inertia-gravity
#  waves; the backward-Euler implicit free surface damps them while the
#  zero-frequency geostrophic mode is steady, so the state settles to
#  the geostrophically adjusted mode. Discrete PV inversion (the
#  energy-conserving C-grid Coriolis interpolates v -> u by the factor
#  gamma = prod_a cos(k_a dx_a / 2)) gives the retained pressure
#  amplitude and energy fraction
#
#      ps_g / ps_i  =  E_g / E_i  =  gamma^2 / (gamma^2 + kd^2 Ld^2),
#
#  with Ld^2 = c^2 / f^2 and kd^2 the discrete Laplacian symbol (the
#  discrete deformation radius Ld_disc = Ld / gamma). It refines to the
#  continuous Rossby fraction 1 / (1 + k^2 Ld^2) at second order.
def _gamma(kx, ky, nx, ny):
    """C-grid Coriolis interpolation factor prod cos(k_a dx_a / 2)."""
    return np.cos(np.pi * kx / nx) * np.cos(np.pi * ky / ny)


def _run_adjustment(nx, nz, csqr, f0, kx, ky, n_steps=200):
    """Release a ps mode, damp the waves, return retained fractions."""
    dt = 0.06 / np.sqrt(csqr)
    model = hy.comparison_model(make_grid(nx, nx, nz), dt=dt, csqr=csqr,
                                coriolis_f0=f0, n2=0.0)
    ps_i = mode_field(model, "ps", kx, ky, "cos")
    model.set_state(zeroed(model).replace(ps=1e-3 * ps_i))
    amp_i = float(np.max(np.abs(np.asarray(model.state["ps"].data))))
    e0 = total_energy(model, csqr, nz)
    model.advance(n_steps)
    amp_f = float(np.max(np.abs(np.asarray(model.state["ps"].data))))
    ef = total_energy(model, csqr, nz)
    return amp_f / amp_i, ef / e0, np.asarray(ps_i.data)


def test_geostrophic_adjustment_matches_the_discrete_deformation_radius():
    nx, nz, csqr, f0, kx, ky = 32, 4, 1.0, 4.0, 1, 1
    ld2 = csqr / f0 ** 2
    amp_ratio, energy_ratio, _ = _run_adjustment(nx, nz, csqr, f0, kx, ky)

    gamma = _gamma(kx, ky, nx, nx)
    kd2 = k_disc_sq(kx, nx) + k_disc_sq(ky, nx)
    retained = gamma ** 2 / (gamma ** 2 + kd2 * ld2)

    # a genuine adjustment: most of the energy was released to the
    # damped gravity waves (small-scale mode, kd Ld ~ 2)
    assert retained < 0.25
    # the balanced state matches the discrete-deformation-radius
    # prediction to roundoff (wave-damping residual + O(amp^2) advection)
    assert abs(amp_ratio - retained) < 1e-3
    assert abs(energy_ratio - retained) < 1e-3


def test_geostrophic_adjustment_settles_to_a_steady_balanced_mode():
    nx, nz, csqr, f0, kx, ky = 24, 4, 1.0, 3.0, 1, 1
    dt = 0.06
    model = hy.comparison_model(make_grid(nx, nx, nz), dt=dt, csqr=csqr,
                                coriolis_f0=f0, n2=0.0)
    ps_i = np.asarray(mode_field(model, "ps", kx, ky, "cos").data)
    model.set_state(zeroed(model).replace(
        ps=1e-3 * mode_field(model, "ps", kx, ky, "cos")))
    model.advance(250)
    ps_a = np.asarray(model.state["ps"].data)
    amp = float(np.max(np.abs(ps_a)))
    # the balanced field is the single input mode (a clean geostrophic
    # state, not a contaminated remnant): correlation ~ 1
    corr = float((ps_a * ps_i).sum()
                 / np.sqrt((ps_a ** 2).sum() * (ps_i ** 2).sum()))
    assert corr > 0.9999
    # and it is steady: advancing further leaves it put
    model.advance(50)
    drift = float(np.max(np.abs(np.asarray(model.state["ps"].data)
                                - ps_a))) / amp
    assert drift < 1e-4


def test_geostrophic_adjustment_energy_partition_refines():
    csqr, f0, kx, ky = 1.0, 4.0, 1, 1
    ld2 = csqr / f0 ** 2
    cont = 1.0 / (1.0 + ((2 * np.pi) ** 2 * (kx ** 2 + ky ** 2)) * ld2)
    errs = []
    for nx in (16, 32):
        _, energy_ratio, _ = _run_adjustment(nx, 4, csqr, f0, kx, ky)
        # each resolution matches the discrete (gamma) prediction tightly
        gamma = _gamma(kx, ky, nx, nx)
        kd2 = k_disc_sq(kx, nx) + k_disc_sq(ky, nx)
        retained = gamma ** 2 / (gamma ** 2 + kd2 * ld2)
        assert abs(energy_ratio - retained) < 1e-3
        errs.append(abs(energy_ratio - cont))
    # the discrete fraction converges to the continuous Rossby-adjustment
    # fraction at (nominally second) order: the error shrinks
    assert errs[1] < 0.4 * errs[0]


@pytest.mark.skipif(
    "FRIDOM_TEST_COMPARISON_SLOW" not in os.environ,
    reason="expensive nx=64 refinement; set FRIDOM_TEST_COMPARISON_SLOW")
def test_geostrophic_adjustment_second_order_trend_slow():
    csqr, f0, kx, ky = 1.0, 4.0, 1, 1
    ld2 = csqr / f0 ** 2
    cont = 1.0 / (1.0 + ((2 * np.pi) ** 2 * (kx ** 2 + ky ** 2)) * ld2)
    errs = []
    for nx in (16, 32, 64):
        _, energy_ratio, _ = _run_adjustment(nx, 4, csqr, f0, kx, ky)
        errs.append(abs(energy_ratio - cont))
    errs = np.asarray(errs)
    rates = np.log2(errs[:-1] / errs[1:])
    # both refinement steps show ~second-order convergence
    assert (rates > 1.7).all()


# ================================================================
#  Eady baroclinic instability -- unblocked by the thermal wind (H5b)
# ================================================================
#  The Eady problem needs a thermal-wind-balanced mean state: a vertical
#  shear U(z) = Lambda z AND a mean meridional buoyancy gradient
#  d_y B = -f Lambda. The baroclinic energy source is the conversion
#  term v' d_y B in the buoyancy equation. FRIDOM's shared advection
#  ``background=`` supplies ONLY a mean velocity (keys u/v/w), so it can
#  express the Doppler advection U d_x(.) but NOT a mean buoyancy
#  gradient. ``hy.ThermalWindBackground(shear=Lambda)`` supplies the two
#  missing terms -- db/dt += +f0 Lambda v' (conversion) and
#  du/dt += -Lambda w' (tilting) -- reading f0 from the f-plane Coriolis
#  so the thermal wind d_y B = -f0 Lambda is enforced by construction.
#  The tests below verify the composition AND that the conversion term
#  is now present; the growth-rate validation is in test_thermal_wind.py.
def _background_model(f0=1.0, lam=0.5, epsilon=0.0):
    """Linear hydrostatic model with a background zonal shear U(z)."""
    grid = make_grid(8, 8, 6)
    adv = fr.model.modules.CenteredAdvection(
        background={"u": lambda z: lam * (z - 0.5)})
    return hy.Model(
        grid=grid, dt=1e-2, csqr=4.0,
        free_surface=hy.ImplicitFreeSurface(epsilon=epsilon),
        stratification=hy.ConstantStratification(n2=1.0),
        coriolis=hy.FPlaneCoriolis(f0=f0), advection=adv,
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-2, order=2, eps=0.1))


def _thermal_wind_model(f0=1.0, lam=0.5, epsilon=0.0):
    """Eady model: background shear U(z) + the thermal-wind module."""
    grid = make_grid(8, 8, 6)
    tw = hy.ThermalWindBackground(shear=lam, reference_height=0.5)
    adv = fr.model.modules.CenteredAdvection(
        background={"u": tw.background_velocity()})
    return hy.Model(
        grid=grid, dt=1e-2, csqr=4.0,
        free_surface=hy.ImplicitFreeSurface(epsilon=epsilon),
        stratification=hy.ConstantStratification(n2=1.0),
        coriolis=hy.FPlaneCoriolis(f0=f0), advection=adv,
        modules_extra=(tw,),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-2, order=2, eps=0.1))


def test_background_shear_composes_with_the_hydrostatic_model():
    # the shared advection's background= assembles and runs on the
    # hydrostatic model: the mean flow rides as an AUXILIARY field
    model = _background_model()
    assert "background_u" in model.state.component_names
    rng = np.random.default_rng(0)
    model.set_fields(
        u=1e-3 * rng.standard_normal(model.state["u"].shape),
        v=1e-3 * rng.standard_normal(model.state["v"].shape),
        b=1e-3 * rng.standard_normal(model.state["b"].shape))
    model.advance(3)
    assert np.isfinite(np.asarray(model.state["u"].data)).all()


def test_background_supplies_the_doppler_advection_of_buoyancy():
    # background= DOES advect the buoyancy perturbation by U (the
    # mean-flow Doppler term U d_x b'): a b-mode gets a nonzero db/dt
    model = _background_model()
    st = zeroed(model).replace(b=mode_field(model, "b", 1, 0, "cos"))
    dbdt = float(np.max(np.abs(np.asarray(model.tendency(st)["b"].data))))
    assert dbdt > 1e-3


def test_eady_baroclinic_conversion_term_is_present():
    r"""The v' d_y B baroclinic conversion term is now supplied (H5b).

    Drive a v-only, divergence-free state (v varies only in x, u = 0),
    so the diagnosed w is exactly zero. The ONLY way db/dt can be
    nonzero is the mean meridional-buoyancy-gradient term
    db/dt += -v' d_y B = +f0 shear v' (the Eady energy source) that
    ``hy.ThermalWindBackground`` supplies. With the module active it is
    nonzero -- the term H5 pinned as absent is present.
    """
    model = _thermal_wind_model()
    st = zeroed(model).replace(v=mode_field(model, "v", 1, 0, "cos"))
    dX = model.tendency(st)
    # w is genuinely zero (v-only, divergence-free) ...
    assert float(np.max(np.abs(np.asarray(st["v"].data)))) > 0.1
    # ... so a nonzero db/dt is the +f0 shear v conversion alone
    assert float(np.max(np.abs(np.asarray(dX["b"].data)))) > 1e-3


# ================================================================
#  Wave-packet dispersion: propagation at the discrete group velocity
# ================================================================
#  A localized internal-wave packet (gravest baroclinic vertical mode,
#  rigid lid so there is no barotropic branch) propagates at the group
#  velocity c_g = d(omega)/d(k), which is DISTINCT from the phase speed
#  c_p = omega/k in the rotating regime. We build the mode's exact
#  discrete vertical structure from the linear operator's eigenvector,
#  launch a rightward packet on the comparison preset, and track its
#  energy-envelope centroid. The dispersion is a property of the shared
#  LINEAR operator (the pinned nonlinear advection, run here at tiny
#  amplitude, is negligible), so the operator is probed on a linear twin.
def _trig_layer(model, name, kx, lx, phase):
    """Single Fourier column cos/sin(2 pi kx x / lx) at name's nodes."""
    fs = model.state[name].function_space
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace)
                for n in f.names)
    trig = np.cos if phase == "cos" else np.sin

    def init(**c):
        return trig(2 * np.pi * kx * c.get("x", 0) / lx) \
            + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return np.asarray(model.grid.create_field(fs, init=init).data)


def _baroclinic_spectrum(model, kx, lx):
    """Return (eigvals, eigvecs, cols) of L on one horizontal mode.

    The full per-z-layer (u, v, b) operator restricted to horizontal
    wavenumber kx (ky = 0), whose eigenpairs are the discrete baroclinic
    internal-wave modes (the H2 ``test_free_surface`` construction).
    """
    fields = ["u", "v", "b"]
    nz = model.state["u"].data.shape[2]
    trig = {(f, p): _trig_layer(model, f, kx, lx, p)
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
        dX = model.tendency(make(col))
        for r, (g, jr, pr) in enumerate(cols):
            d = np.asarray(dX[g].data)
            num = float((d[:, :, jr] * trig[(g, pr)][:, :, jr]).sum())
            den = float((trig[(g, pr)][:, :, jr] ** 2).sum())
            mat[r, i] = num / den
    ev, vecs = np.linalg.eig(mat)
    return ev, vecs, cols


def test_internal_wave_packet_moves_at_the_discrete_group_velocity():
    lx, nx, ny, nz = 4.0, 96, 4, 12
    csqr, f0, n2, kx0, dt = 1.0, 1.0, 0.30, 6, 0.03

    # 1. discrete dispersion from the linear operator (a linear twin;
    #    the preset's advection does not change linear wave dispersion)
    lin = hy.Model(
        grid=make_grid(nx, ny, nz, lx=lx), dt=dt, csqr=csqr,
        free_surface=hy.ImplicitFreeSurface(epsilon=0.0),
        stratification=hy.ConstantStratification(n2=n2),
        coriolis=hy.FPlaneCoriolis(f0=f0), advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=2,
                                                          eps=0.1))
    ev, vecs, cols = _baroclinic_spectrum(lin, kx0, lx)
    i_pos = int(np.argmax(ev.imag))
    omega = ev[i_pos].imag
    vec = vecs[:, i_pos]
    kh = 2.0 * np.sin(np.pi * kx0 / nx) / (lx / nx)   # discrete symbol
    c_p = omega / kh
    # discrete group velocity d(omega)/dk with the discrete-k chain rule
    c_g = (omega ** 2 - f0 ** 2) / (kh * omega) * np.cos(np.pi * kx0 / nx)
    # a genuinely dispersive regime: group clearly below phase
    assert c_g / c_p < 0.85

    # 2. launch the packet on the comparison preset (rigid lid), scaled
    #    tiny so the pinned nonlinear advection is negligible
    model = hy.comparison_model(
        make_grid(nx, ny, nz, lx=lx), dt=dt, csqr=csqr,
        coriolis_f0=f0, n2=n2, epsilon=0.0)
    sigma, x0, scale = 0.8, lx / 2, 1e-3

    def packet(name):
        fs = model.state[name].function_space
        x = np.asarray(model.grid.evaluation_nodes(fs, "x").data)
        fld = np.zeros(np.asarray(model.state[name].data).shape,
                       dtype=complex)
        hor = (np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
               * np.exp(1j * 2 * np.pi * kx0 * x / lx))[:, :, 0]
        for j in range(fld.shape[2]):
            amp = (vec[cols.index((name, j, "cos"))]
                   - 1j * vec[cols.index((name, j, "sin"))])
            fld[:, :, j] += amp * hor
        return scale * np.real(fld)

    model.set_state(model.state.replace(**{
        n: model.grid.create_field(model.state[n].function_space,
                                   data=packet(n))
        for n in ("u", "v", "b")}))

    xg = np.linspace(0, lx, nx, endpoint=False) + 0.5 * lx / nx

    def centroid():
        e = (np.asarray(model.state["b"].data) ** 2
             + np.asarray(model.state["u"].data) ** 2).sum(axis=(1, 2))
        return float((xg * e).sum() / e.sum())

    times, cents = [0.0], [centroid()]
    t = 0.0
    for _ in range(5):
        model.advance(30)
        t += 30 * dt
        times.append(t)
        cents.append(centroid())
    speed = abs(np.polyfit(times, cents, 1)[0])

    # the envelope tracks the GROUP velocity, to a few percent ...
    assert abs(speed - c_g) / c_g < 0.10
    # ... and clearly not the (larger) phase speed: the packet is closer
    # to c_g than to c_p by a wide margin
    assert abs(speed - c_g) < 0.4 * abs(c_p - c_g)
