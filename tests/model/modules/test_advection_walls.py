"""Advection on walled grids: closures, invariants, convergence."""
from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.model import Model as FrModel
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
    _BiasedFaceReconstruction,
    _CenteredFaceInterpolation,
)
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
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


def faces(nx, lx=L):
    return (np.arange(nx) + 1.0) * (lx / nx)


def advection_tendency(model, cls):
    return model.tendency(model.state, constraints=False,
                          filter=fr.model.term_predicates.owned_by(cls))


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
        params: dict = {fr.model.params.SCALING_NONLINEARITY: 1.0}  # noqa: RUF012

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
    # the projection zeroes the divergence to the FP roundoff of the
    # walled (DCT-II) solve, which scales with the tendency magnitude
    # (~30 here on channel-and-lid) and drifts by ULPs across FFT code
    # paths (rfft/Makhoul, 55b0b866): bound it relative to the
    # tendency, not absolutely (measured <= ~5e-15 relative)
    tau_scale = max(float(np.abs(np.asarray(tau[c].data)).max())
                    for c in ("u", "v", "w"))
    assert float(np.abs(np.asarray(div.data)).max()) < 1e-13 * tau_scale


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


def _walled_tendency_error(cls, order, n, wall="upwind1"):
    """Max |tendency - analytic| at the interior / all b cells."""
    grid = Grid((
        IntervalMesh(4, (0.0, L), name="x"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(4, (0.0, L), name="z"),
    ))
    module = cls(order, wall=wall)
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
#: is 2, NOT the design order, and that is a property of the C-grid
#: flux form, not of the graded closure: the scheme differences
#: v_face * R(b), while the high-order face quantity of an FV
#: reconstruction is the deconvolved FLUX R(v b) — the product-rule
#: mismatch leaves a cross term ~ (h^2/24) * 2 v' b' that survives
#: wherever the advecting velocity varies along the flux axis. The
#: periodic scheme measures exactly the same 2.0 (see
#: test_varying_velocity_tendency_is_only_second_order), so nothing
#: here is charged to the walls. The design order of the graded rows
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


# ================================================================
#  The wall-adjacent rung (``wall=``): accuracy vs monotonicity
# ================================================================
def test_wall_rung_defaults_to_upwind_one():
    # the default is the historical behavior; nothing changes silently
    assert UpwindAdvection().wall == "upwind1"
    assert WENOAdvection(5).wall == "upwind1"
    assert UpwindAdvection(5, wall="centered2").wall == "centered2"
    assert _BiasedFaceReconstruction(3, "left", "linear").wall == (
        "upwind1")


@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
def test_unknown_wall_rung_is_taught(cls):
    with pytest.raises(ValueError,
                       match=r"near-wall rungs \('upwind1', "
                             r"'centered2'\)"):
        cls(3, wall="quick")
    with pytest.raises(ValueError, match=r"wall must be one of"):
        _BiasedFaceReconstruction(3, "left", "linear", "graded",
                                  "quick")


def test_wall_rung_is_part_of_the_intern_key():
    def make(wall):
        return _BiasedFaceReconstruction(5, "left", "weno", "graded",
                                         wall)

    assert make("upwind1") is make("upwind1")
    assert make("upwind1") is not make("centered2")
    assert make("centered2").wall == "centered2"


def test_walled_bind_installs_the_chosen_bottom_rung():
    module = WENOAdvection(5, wall="centered2")
    assert module._left.boundary == "none"     # pre-bind
    make_walled_model(("y",), module)
    for kernel in (module._left, module._right, module._lin_left,
                   module._lin_right):
        assert kernel.boundary == "graded"
        assert kernel.wall == "centered2"


def test_periodic_path_ignores_the_wall_rung():
    # a fully periodic grid keeps the plain kernels — the same INTERNED
    # objects — and the tendency is bitwise the default's: the wall rung
    # cannot change the periodic numerics
    default = UpwindAdvection(5)
    centered = UpwindAdvection(5, wall="centered2")
    assert centered._left is default._left
    assert centered._right is default._right

    taus = []
    for module in (default, centered):
        model = make_model(8, module)
        set_random_state(model, seed=21)
        taus.append(advection_tendency(model, type(module)))
    for name in ("u", "v", "w", "b"):
        assert np.array_equal(np.asarray(taus[0][name].data),
                              np.asarray(taus[1][name].data))


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_centered2_wall_flux_is_exactly_zero(cls, order):
    # the blocking gate: the centered bottom rung must not leak through
    # the wall. It cannot — it only ever writes the INTERIOR faces, and
    # the wall face is a Dirichlet boundary value of the flux space, not
    # a DOF (the same structural argument as the upwind1 bottom)
    module = cls(order, wall="centered2")
    model = make_walled_model(("y",), module)
    set_random_state(model, seed=17)
    flux, flux_space = walled_flux(module, model, "b", "y")
    factor = flux_space.bare.factor("y")
    assert factor.bc.components == (BC.DIRICHLET, BC.DIRICHLET)

    synced = Sync()(flux)
    width = model.grid.decomposition.halo["y"]
    data = np.asarray(synced._data)
    assert (data[:, width - 1, :] == 0.0).all()          # y = 0 wall
    assert (data[:, width + factor.shape[0], :] == 0.0).all()


@pytest.mark.parametrize(("cls", "order"), BIASED)
@pytest.mark.parametrize("walled", WALLED_TOPOLOGIES)
def test_centered2_conserves_total_buoyancy(cls, order, walled):
    # the flux form still telescopes onto an exact-zero wall flux
    module = cls(order, wall="centered2")
    model = make_walled_model(walled, module)
    set_random_state(model, seed=12)
    state = model.constrain(model.state)
    tau = model.tendency(
        state, constraints=False,
        filter=fr.model.term_predicates.owned_by(cls))
    db = np.asarray(tau["b"].data)
    assert np.isfinite(db).all()
    assert abs(float(np.sum(db))) / float(
        np.sum(np.abs(db))) < CONSERVATION_TOL


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_centered2_preserves_a_uniform_tracer(cls, order):
    # free-stream preservation: the two-point mean row sums to one, so
    # the centered bottom rung is exact on constants exactly as the
    # upwind cell is
    model = make_walled_model(("y", "z"),
                              cls(order, wall="centered2"))
    set_random_state(model, seed=5)
    state = model.constrain(model.state)
    model.set_fields(**{c: np.asarray(state[c].data)
                        for c in ("u", "v", "w")})
    model.set_fields(b=np.full(model.state["b"].data.shape, 3.0))
    tau = model.tendency(model.state, constraints=False,
                         filter=fr.model.term_predicates.owned_by(cls))
    assert float(np.abs(np.asarray(tau["b"].data)).max()) < 1e-13


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_centered2_leaves_the_interior_bitwise_unchanged(cls, order):
    # the option buys its order at the wall ONLY: away from the K
    # reduced faces per side the two rungs produce the same numbers,
    # bit for bit (the ladder above the bottom rung is identical)
    taus = []
    for wall in ("upwind1", "centered2"):
        model = make_walled_model(("y",), cls(order, wall=wall), n=16)
        set_random_state(model, seed=23)
        taus.append(advection_tendency(model, cls))
    k = order // 2 + 2
    for name in ("u", "v", "w", "b"):
        got = [np.asarray(tau[name].data)[:, k:-k, :] for tau in taus]
        assert np.array_equal(got[0], got[1])
        # ... and the near-wall rows genuinely differ
        near = [np.asarray(tau[name].data)[:, :1, :] for tau in taus]
        assert not np.array_equal(near[0], near[1])


@pytest.mark.parametrize(("cls", "order"), BIASED)
def test_centered2_lifts_the_global_error_to_second_order(cls, order):
    # the payoff, measured on the same walled tracer problem the
    # first-order gate above uses: the
    # O(h^2) wall-adjacent face value lifts the max-norm tendency rate
    # over the WHOLE walled axis from ~1 to ~2 (measured 2.03-2.11 for
    # every configuration but weno3, whose global error is capped by
    # its own interior critical-point degradation, not by the wall)
    sizes = (64, 128, 256)
    errs = [_walled_tendency_error(cls, order, n, wall="centered2")[1]
            for n in sizes]
    base = [_walled_tendency_error(cls, order, n)[1] for n in sizes]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(fine < coarse for coarse, fine in pairwise(errs))
    # weno3's global max error sits in the interior (critical points),
    # so the wall rung cannot lift it; everywhere else it reaches ~2
    floor = 0.9 if (cls, order) == (WENOAdvection, 3) else 1.8
    assert min(rates) > floor
    # never worse than the upwind1 bottom, at any resolution
    assert all(new <= old for new, old in zip(errs, base, strict=True))


def _wall_front(module, n=32, t_end=0.5):
    """Drive a step front INTO the y = 1 wall; return the profile.

    ``v = sin(pi y)`` is impermeable and compressive at the top wall,
    so the discontinuity (started two cells out) is pressed onto the
    wall-adjacent face — exactly where the two rungs differ. The exact
    solution is monotone with an exact minimum of 0 below the front,
    so any ``b < 0`` is a spurious oscillation.
    """
    grid = Grid((
        IntervalMesh(4, (0.0, L), name="x"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(4, (0.0, L), name="z"),
    ))
    model = FrModel(
        grid=grid,
        modules=(DynamicalCore(), ConstantStratification(n2=1.0),
                 module),
        time_stepper=AdamBashforth(DT, order=3))
    yv = np.asarray(grid.evaluation_nodes(
        model.state["v"].function_space, "y").data).ravel()
    yb = np.asarray(grid.evaluation_nodes(
        model.state["b"].function_space, "y").data).ravel()
    vfield = np.broadcast_to(np.sin(np.pi * yv)[None, :, None],
                             model.state["v"].data.shape).copy()
    b = np.broadcast_to(
        (yb > 1.0 - 2.0 / n).astype(float)[None, :, None],
        model.state["b"].data.shape).copy()

    def rhs(data):
        model.set_fields(v=vfield, b=data)
        tau = model.tendency(
            model.state, constraints=False,
            filter=fr.model.term_predicates.owned_by(type(module)))
        return np.asarray(tau["b"].data)

    dt = 0.2 / n                       # CFL 0.2 (max |v| = 1)
    for _ in range(round(t_end / dt)):          # SSP-RK3
        b1 = b + dt * rhs(b)
        b2 = 0.75 * b + 0.25 * (b1 + dt * rhs(b1))
        b = (b + 2.0 * (b2 + dt * rhs(b2))) / 3.0
    return b[0, :, 0]


@pytest.mark.parametrize(
    ("cls", "order"),
    [pytest.param(UpwindAdvection, 3, id="upwind3"),
     pytest.param(WENOAdvection, 5, id="weno5")])
def test_centered2_rings_on_a_wall_adjacent_front(cls, order):
    # the documented PRICE, measured: with the centered bottom rung the
    # wall-adjacent face has no upwind bias (both members of the upwind
    # pair return the same value there), so a front pressed against the
    # wall oscillates. Measured at n=32, t=0.5: undershoot 0.00 ->0.57
    # (weno5, whose ENO property no longer applies on that face) and
    # 0.13 -> 0.72 (upwind3); the total variation roughly doubles. The
    # oscillation stays BOUNDED (a trapped 2-cell wiggle) — it is a
    # monotonicity failure, not an instability.
    profiles = {
        wall: _wall_front(cls(order, wall=wall))
        for wall in ("upwind1", "centered2")}
    under = {wall: max(0.0, -float(p.min()))
             for wall, p in profiles.items()}
    variation = {wall: float(np.abs(np.diff(p)).sum())
                 for wall, p in profiles.items()}

    assert all(np.isfinite(p).all() for p in profiles.values())
    assert under["upwind1"] < 0.2
    assert under["centered2"] > 0.4
    assert variation["centered2"] > 1.4 * variation["upwind1"]
    # bounded, not blowing up (the exact solution's range is [0, ~7])
    assert float(np.abs(profiles["centered2"]).max()) < 10.0


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


@pytest.mark.parametrize("wall", ["upwind1", "centered2"])
@pytest.mark.parametrize("weighting", ["linear", "weno"])
@pytest.mark.parametrize("node_set", [NodeSet.CENTER, NodeSet.INNER])
@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("order", [3, 5])
def test_graded_rows_read_no_exterior_value(order, bias, weighting,
                                            node_set, wall):
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

    op = _BiasedFaceReconstruction(order, bias, weighting, "graded",
                                   wall)
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
