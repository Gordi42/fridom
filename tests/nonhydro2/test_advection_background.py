"""Advection with a prescribed background flow: split, spectrum."""

import jax
import numpy as np
import pytest

import fridom as fr
import fridom.framework as frold
import fridom.nonhydro as nhold
from fridom.model.eigen import numeric_eigenpairs
from fridom.model.model import Model as FrModel
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
)
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

L = 2 * np.pi
NY = 8
DT = 0.01


# ================================================================
#  Helpers
# ================================================================
def invariant(a, b):
    """Device-count invariance for the result of a *sharded reduction*.

    Bitwise on real backends. Under the forced-host-device CPU
    emulation (``XLA_FLAGS=--xla_force_host_platform_device_count=N``,
    the forced-4 CI backend) XLA reassociates the multi-device FP
    reductions relative to the single-device program, so the tendency
    matches only to a tight absolute tolerance, not bit-for-bit. A real
    device-count bug is O(1) or NaN, far above the tolerance.
    """
    a, b = np.asarray(a), np.asarray(b)
    if jax.default_backend() == "cpu":
        return np.allclose(a, b, rtol=0.0, atol=1e-12)
    return np.array_equal(a, b)


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


def advection_tendency(model, cls):
    return model.tendency(model.state, constraints=False,
                          filter=fr.model.term_predicates.owned_by(cls))


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
        # forced-CPU multi-device reassociates the reduction (see the
        # invariant helper); bitwise on real backends.
        assert invariant(taus["explicit"][name].data, expected)
        assert invariant(taus["zero"][name].data, expected)


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
# the old stack shards axis 0 over all visible devices, so its
# reference tendency is not device-count invariant; pin this parity
# check to a single device where the reference is meaningful.
@pytest.mark.single_device
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
