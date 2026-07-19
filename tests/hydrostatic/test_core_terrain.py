r"""The hydrostatic core on a terrain-following (sigma) grid.

Terrain items H1/H2 (research record ``stretched_terrain_combined.md``
§6): the ``p_hyd`` column integral carries the column Jacobian
(``jacobian=(mapped,)``); the diagnosed ``w`` is the contravariant
vertical volume flux ``J\omega`` from the flux-form horizontal
divergence (a machine-exact fundamental theorem, ``w = 0`` at the
terrain bottom); and the baroclinic pressure gradient is the
slope-corrected (constant-physical-height) derivative, so a stratified
fluid at rest over topography stays at rest to the scheme's truncation
order (the sigma pressure-gradient-error gate). A flat grid is byte-
identical to before. Self-contained builders (AGENTS oversized-module
rule).
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh
MIM = fr.spatial.meshes.MappedIntervalMesh
N2, CSQR, F0 = 2.0, 3.0, 1.3
ORDER_FLOOR = 1.7


def _depth(x, y):
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _mapping():
    return CoordinateMapping(maps={"zp": lambda z, H: z * H},
                             params={"H": _depth})


def _terrain_grid(n, *, stretched=False):
    """Sigma grid, flat surface (z=0), sloped bottom (z=-1 -> zp=-H)."""
    mx = IM(n, (0.0, 1.0), periodic=True, name="x")
    my = IM(n, (0.0, 1.0), periodic=True, name="y")
    if stretched:
        mz = MIM(n, (-1.0, 0.0),
                 lambda s: -1.0 + s - 0.3 * jnp.sin(jnp.pi * s), name="z")
    else:
        mz = IM(n, (-1.0, 0.0), periodic=False, name="z")
    return fr.spatial.Grid((mx, my, mz), mapping=_mapping())


def _flat_grid(n):
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(n, (-1.0, 0.0), periodic=False, name="z")))


def _model(grid, *, coriolis=None, advection=False, dt=1e-3,
           free_surface=None):
    return hy.Model(
        grid=grid, dt=dt, csqr=CSQR,
        stratification=hy.ConstantStratification(n2=N2),
        coriolis=coriolis, advection=advection,
        free_surface=free_surface or hy.ExplicitFreeSurface(),
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))


def _orders(errs):
    errs = np.asarray(errs)
    return np.log2(errs[:-1] / errs[1:])


def _smooth(grid, space, fn):
    var = tuple(n for f in space.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)

    def init(**c):
        return fn(**c) + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return grid.create_field(space, init=init)


def _nodes(grid, space, name):
    return grid.evaluation_nodes(space.bare, name).data


# ================================================================
#  H1: p_hyd carries the column Jacobian (converges to the physical
#      integral -int b dz_p at second order)
# ================================================================
@pytest.mark.parametrize("stretched", [False, True],
                         ids=["uniform-sigma", "stretched-sigma"])
def test_p_hyd_converges_to_the_physical_integral(stretched):
    # b = sin(zp);  -int_zp^0 b dzp' = 1 - cos(zp)  (surface at zp = 0)
    errs = []
    for n in (8, 16, 32):
        grid = _terrain_grid(n, stretched=stretched)
        coll = fr.spatial.Collocated().resolve(grid)
        zp = (_nodes(grid, coll, "z")
              * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
        b = grid.create_field(coll, data=jnp.sin(zp))
        p = -CumulativeIntegral(
            direction="down", target="center",
            jacobian=("zp",))["z"](b)
        errs.append(float(jnp.abs(p.data - (1.0 - jnp.cos(zp))).max()))
    assert np.all(_orders(errs) > ORDER_FLOOR)


def test_p_hyd_terrain_differs_from_the_plain_integral():
    # the old (unweighted) diagnosis omits J: a real, O(1) error the
    # record measured at 27%. Here it is clearly non-negligible.
    grid = _terrain_grid(16)
    coll = fr.spatial.Collocated().resolve(grid)
    zp = (_nodes(grid, coll, "z")
          * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
    b = grid.create_field(coll, data=jnp.sin(zp))
    weighted = -CumulativeIntegral(
        direction="down", target="center", jacobian=("zp",))["z"](b)
    plain = -CumulativeIntegral(direction="down", target="center")["z"](b)
    rel = float(jnp.abs(weighted.data - plain.data).max()
                / jnp.abs(weighted.data).max())
    assert rel > 0.1


# ================================================================
#  H2a: w is the contravariant volume flux J*omega (flux form),
#       FTC exact, w == 0 at the terrain bottom
# ================================================================
def test_diagnosed_w_is_the_flux_form_with_exact_ftc():
    grid = _terrain_grid(16)
    model = _model(grid)
    su, sv = (model.state["u"].function_space,
              model.state["v"].function_space)
    u = _smooth(grid, su, lambda **c: jnp.sin(2 * jnp.pi * c["x"]))
    v = _smooth(grid, sv, lambda **c: jnp.cos(2 * jnp.pi * c["y"]))
    core = model.module(hy.HydrostaticCore)
    w = core._diagnose_w(model.state.replace(u=u, v=v), None)["w"]

    jname = "dzp_dz"
    ju = u * grid.metric(su.bare, jname)
    jv = v * grid.metric(sv.bare, jname)
    dh = ju.diff("x") + jv.diff("y")
    # fundamental theorem in J-weighted (flux) form: d_z w == -Dh
    ftc = np.asarray(w.diff("z").data) + np.asarray(dh.data)
    assert np.abs(ftc).max() < 1e-11
    # w == 0 at the terrain bottom (zero normal flow on the sigma column)
    wd = np.asarray(w.data)
    zaxis = next(i for i, f in enumerate(w.function_space.bare.factors)
                 if "z" in f.names)
    assert np.abs(np.take(wd, 0, axis=zaxis)).max() < 1e-13


def test_flat_w_is_byte_identical_to_the_cartesian_form():
    # on a flat grid the terrain path is off: w is the plain divergence
    # cumint, byte-for-byte.
    grid = _flat_grid(8)
    model = _model(grid)
    su, sv = (model.state["u"].function_space,
              model.state["v"].function_space)
    rng = np.random.default_rng(2)
    u = grid.create_field(su, data=rng.standard_normal(su.shape))
    v = grid.create_field(sv, data=rng.standard_normal(sv.shape))
    core = model.module(hy.HydrostaticCore)
    assert core._column is None
    w = core._diagnose_w(model.state.replace(u=u, v=v), None)["w"]
    expect = -CumulativeIntegral(direction="up", target="face")["z"](
        u.diff("x") + v.diff("y"))
    assert np.array_equal(np.asarray(w.data), np.asarray(expect.data))


# ================================================================
#  H2b: rest state over topography (the sigma PG-error gate)
# ================================================================
def _rest_tendency(n, *, kind):
    grid = _terrain_grid(n)
    model = _model(grid)
    coll = model.state["b"].function_space
    zp = (_nodes(grid, coll, "z")
          * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
    bd = -2.0 * zp if kind == "linear" else -2.0 * zp - 0.7 * zp ** 2
    model.set_fields(
        u=np.zeros(model.state["u"].shape),
        v=np.zeros(model.state["v"].shape),
        b=np.asarray(bd), ps=np.zeros(model.state["ps"].shape))
    dX = model.tendency(model.state)
    return max(float(jnp.abs(dX["u"].data).max()),
               float(jnp.abs(dX["v"].data).max()))


@pytest.mark.parametrize("kind", ["linear", "nonlinear"])
def test_rest_state_pressure_gradient_error_converges(kind):
    # a stratified fluid at rest over a seamount: the slope-corrected
    # pressure gradient leaves only a truncation-order residual current
    # that vanishes at ~2nd order (the plain gradient leaves O(1)).
    errs = [_rest_tendency(n, kind=kind) for n in (16, 32, 64)]
    assert np.all(_orders(errs) > ORDER_FLOOR)


def test_rest_state_stays_near_rest_over_a_short_run():
    grid = _terrain_grid(16)
    model = _model(grid, dt=2e-3)
    coll = model.state["b"].function_space
    zp = (_nodes(grid, coll, "z")
          * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
    model.set_fields(
        u=np.zeros(model.state["u"].shape),
        v=np.zeros(model.state["v"].shape),
        b=np.asarray(-2.0 * zp), ps=np.zeros(model.state["ps"].shape))
    model.run(10, progress=False)
    # the spurious current stays at the truncation-order floor (small),
    # not growing to O(1)
    assert float(jnp.abs(model.state["u"].data).max()) < 1e-2
    assert float(jnp.abs(model.state["v"].data).max()) < 1e-2


def test_flat_pressure_gradient_is_the_plain_difference():
    grid = _flat_grid(8)
    model = _model(grid)
    core = model.module(hy.HydrostaticCore)
    rng = np.random.default_rng(4)
    model.set_fields(
        u=np.zeros(model.state["u"].shape),
        v=np.zeros(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape),
        ps=np.zeros(model.state["ps"].shape))
    st = model.state
    p_hyd = core._diagnose_p_hyd(st, None)["p_hyd"]
    out = core.pressure_gradient(st.replace(p_hyd=p_hyd), None)
    expect_u = (-p_hyd.diff("x")).retag(st["u"])
    assert np.array_equal(np.asarray(out["u"].data),
                          np.asarray(expect_u.data))


def test_flat_restoring_is_the_plain_minus_n2_w():
    # on a flat grid the stratification's terrain branch is off
    # (column is None): db/dt is the plain -N^2 w.to(b), byte-for-byte,
    # with no slope-advection term.
    grid = _flat_grid(8)
    model = _model(grid)
    strat = model.module(hy.ConstantStratification)
    assert strat._column is None
    assert strat.extra_halo is None
    rng = np.random.default_rng(5)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape),
        ps=np.zeros(model.state["ps"].shape))
    st = model.state
    dX = model.tendency(st)
    core = model.module(hy.HydrostaticCore)
    w = core._diagnose_w(st, None)["w"]
    expect = -(N2 * w.to(st["b"]))
    assert np.array_equal(np.asarray(dX["b"].data),
                          np.asarray(expect.data))


# ================================================================
#  H4 (baroclinic leg): the KE<->PE conversion is energy-consistent
#      under the PHYSICAL metric to second order once the buoyancy
#      couples to the physical vertical velocity (the missing
#      slope-advection term; energy_metric_asymmetry.md)
# ================================================================
def _broadband(grid, space, seed):
    """Return a resolved band-limited random field sampled on `space`."""
    xs = {a: np.asarray(_nodes(grid, space, a)) for a in ("x", "y", "z")}
    rng = np.random.default_rng(seed)
    out = 0.0
    for kx in range(1, 5):
        for kz in range(1, 5):
            amp = rng.standard_normal(2) / (kx * kz)
            out = (out
                   + amp[0] * np.sin(2 * np.pi * kx * xs["x"])
                   * np.cos(kz * np.pi * xs["z"])
                   + amp[1] * np.cos(2 * np.pi * kx * xs["y"])
                   * np.sin(kz * np.pi * (xs["z"] + 1.0)))
    return np.broadcast_to(
        out, np.broadcast_shapes(*(v.shape for v in xs.values())))


def _broadband2d(grid, space, seed):
    """Return a resolved band-limited random surface field (the ps leg)."""
    xs = {a: np.asarray(_nodes(grid, space, a)) for a in ("x", "y")}
    rng = np.random.default_rng(seed)
    out = 0.0
    for kx in range(1, 5):
        amp = rng.standard_normal(2) / kx
        out = (out + amp[0] * np.sin(2 * np.pi * kx * xs["x"])
               + amp[1] * np.cos(2 * np.pi * kx * xs["y"]))
    return np.broadcast_to(
        out, np.broadcast_shapes(*(v.shape for v in xs.values())))


def _phys_skew(model, seeds_x, seeds_y):
    """Bilinear physical-metric skew of the linear tendency operator L.

    ``<X, L Y>_M + <Y, L X>_M`` (relative to the exchange scale) under
    the hand-built PHYSICAL metric M: the u/v/b legs J-weighted at their
    faces / cell (the seeded ``integrate``), the ps leg lifted to the
    3D b space so the plain J-weighted volume integral supplies the
    per-column physical depth ``H/c^2`` (NOT ``EnergyMetric``, whose ps
    weight is fixed elsewhere). Independent broadband states X, Y (all
    components, fixed seeds) — robust to the state-selection accident of
    the old single-mode gate.
    """
    grid = model.state["b"].grid
    fields = ("u", "v", "b", "ps")

    def state_and_tendency(seeds):
        model.set_fields(
            u=_broadband(grid, model.state["u"].function_space, seeds[0]),
            v=_broadband(grid, model.state["v"].function_space, seeds[1]),
            b=_broadband(grid, model.state["b"].function_space, seeds[2]),
            ps=_broadband2d(grid, model.state["ps"].function_space,
                            seeds[3]))
        snap = {n: model.state[n].with_data(jnp.asarray(model.state[n].data))
                for n in fields}
        return snap, model.tendency(model.state)

    x_state, lx = state_and_tendency(seeds_x)
    y_state, ly = state_and_tendency(seeds_y)
    p3 = model.state["b"].function_space

    def jint(f):
        return float(f.integrate().data.ravel()[0])

    def pairing(a, db):
        return (jint(a["u"] * db["u"]) + jint(a["v"] * db["v"])
                + jint((a["b"] / N2) * db["b"])
                + jint((a["ps"].to(p3) / CSQR) * db["ps"].to(p3)))

    # bilinear cross terms <X, L Y>_M + <Y, L X>_M (not the diagonal)
    xy, yx = pairing(x_state, ly), pairing(y_state, lx)
    return abs(xy + yx) / (abs(xy) + abs(yx))


def test_baroclinic_energy_conversion_collapses_under_physical_metric():
    # The KE<->PE exchange (baroclinic pressure gradient <-> buoyancy
    # restoring) is skew under the PHYSICAL (Jacobian-weighted) metric
    # only once the buoyancy couples to the *physical* vertical velocity
    # w_true = J*omega + u Zx + v Zy -- the slope-advection term this
    # module adds. The pre-fix operator leaks O(slope) here,
    # RESOLUTION-INDEPENDENT (the old single-mode gate passed by
    # state-selection accident: single modes sit in the leak's null
    # set). With the analytic slope term the physical-metric skew
    # converges at ~second order (the exact discrete adjoint would make
    # it machine-zero, but it bakes the grid quadrature into the
    # buoyancy tendency and fights the C-grid staggering -- see the
    # record addendum in energy_metric_asymmetry.md; the shipped physics
    # is the local w_true, invariant O(h^2), not roundoff).
    models = [_model(_terrain_grid(n)) for n in (16, 32, 64)]
    skews = [_phys_skew(m, (1, 2, 3, 4), (5, 6, 7, 8)) for m in models]
    assert np.all(_orders(skews) > ORDER_FLOOR)


# ================================================================
#  Model assembles + runs on terrain (linear and advective)
# ================================================================
def test_terrain_model_assembles_and_runs():
    grid = _terrain_grid(8)
    model = _model(grid, coriolis=hy.FPlaneCoriolis(f0=F0),
                   advection=True)
    rng = np.random.default_rng(0)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        b=0.1 * rng.standard_normal(model.state["b"].shape),
        ps=np.zeros(model.state["ps"].shape))
    dX = model.tendency(model.state)
    assert all(bool(jnp.isfinite(dX[k].data).all())
               for k in ("u", "v", "b", "ps"))
    model.run(3, progress=False)
    assert bool(jnp.isfinite(model.state["u"].data).all())


def test_terrain_core_derives_the_stencil_halo():
    # the terrain DIAGNOSE stages and slope-corrected pressure gradient
    # multiply metric fields the halo trace cannot follow, so the core
    # declares its own width -- DERIVED (not a literal) from the order-2
    # rows the stages apply: 1 on each horizontal coordinate. The
    # vertical is 2: the slope gradient composes the column derivative
    # (centre->face diff) with the face->centre re-alignment interp, and
    # each staggered row publishes its per-shard footprint 1 (the fix in
    # b57e3e78: a bounded stencil's exterior reach cancels to 0 at the
    # wall, but a sharded interior slot reads a neighbour), so the pair
    # sums to 2.
    model = _model(_terrain_grid(8))
    core = model.module(hy.HydrostaticCore)
    assert dict(core.extra_halo.widths) == {"x": 1, "y": 1, "z": 2}


# ================================================================
#  Taught errors (H0): unsupported terrain combinations
# ================================================================
def test_non_base_vertical_is_a_taught_error():
    # a column mapped from x (horizontal base): the hydrostatic vertical
    # axis 'z' is not the base -> taught error at bind.
    grid = fr.spatial.Grid(
        (IM(8, (0.1, 0.9), periodic=False, name="x"),
         IM(8, (0.0, 1.0), periodic=True, name="y"),
         IM(6, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"xp": lambda x, W: x * W},
                                  params={"W": lambda x: 1.0 + 0.0 * x}))
    with pytest.raises(NotImplementedError, match="not the base"):
        _model(grid)


def _terrain_immersed_grid(*, order, min_fraction=0.0):
    return fr.spatial.Grid(
        (IM(8, (0.0, 1.0), periodic=True, name="x"),
         IM(8, (0.0, 1.0), periodic=True, name="y"),
         IM(6, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mapping(),
        immersed=ImmersedDomain(lambda x, y, z: 1.0 + 0.0 * (x + y + z),
                                order=order, min_fraction=min_fraction))


def test_terrain_immersed_assembles_with_quadrature_fractions():
    # stage M5: a terrain + immersed grid with genuine chart quadrature
    # (order >= 2) is the composed masked contravariant continuity — it
    # assembles and the masked/terrain DIAGNOSE stages compose.
    model = _model(_terrain_immersed_grid(order=4))
    core = model.module(hy.HydrostaticCore)
    assert core._column == ("zp", "z")
    assert core._immersed is not None


def test_terrain_immersed_collocation_mask_is_a_taught_error():
    # a collocation-order mask (order=None/1) on a chart mis-places the
    # geometry, so the composition needs order >= 2 (a taught error).
    grid = _terrain_immersed_grid(order=None)
    with pytest.raises(NotImplementedError,
                       match="genuine per-cell quadrature"):
        _model(grid)


# ================================================================
#  Differentiability policy: grad through a short terrain run
# ================================================================
@pytest.mark.parametrize("advection", [False, True],
                         ids=["linear", "advective"])
def test_grad_wrt_initial_buoyancy_is_finite_and_matches_fd(advection):
    # the terrain step path crosses the guarded terrain singularities:
    # the slope coefficient Z/J of the baroclinic pressure gradient
    # (core) and the reciprocal physical depth 1/H of the free-surface
    # depth mean, both sealed on the never-valid padding by the double-
    # `where`. With advection=True the differentiated data additionally
    # crosses the shared nodal mapped divergence's Z/J slope factor,
    # sealed by advection._safe_ratio -- unguarded it is forward-finite
    # but reverse-NaN-poisons the whole gradient here.
    grid = _terrain_grid(8)
    model = _model(grid, coriolis=hy.FPlaneCoriolis(f0=1.0),
                   advection=advection, dt=1e-2)
    rng = np.random.default_rng(11)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        b=0.1 * rng.standard_normal(model.state["b"].shape),
        ps=np.zeros(model.state["ps"].shape))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    leaf = carry.state["b"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 6, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(leaf))
    assert bool(np.all(np.isfinite(grad)))
    direction = jnp.asarray(rng.standard_normal(leaf.shape),
                            dtype=leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(leaf + eps * direction))
          - float(loss(leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


def test_grad_wrt_initial_velocity_via_propagator_matches_fd():
    # the slope-advection term feeds the initial velocity into the
    # buoyancy tendency (-N^2 (u Zx + v Zy)); grad of a quadratic loss
    # w.r.t. the initial zonal velocity through a short terrain run --
    # via the PUBLIC Model.propagator surface (AGENTS differentiability
    # policy) -- is finite and matches a central finite difference. The
    # slope metrics are finite (no 1/J), so the new term adds no VJP
    # singularity of its own.
    grid = _terrain_grid(8)
    model = _model(grid, coriolis=hy.FPlaneCoriolis(f0=1.0), dt=1e-2)
    rng = np.random.default_rng(7)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        b=0.1 * rng.standard_normal(model.state["b"].shape),
        ps=np.zeros(model.state["ps"].shape))
    run = model.propagator(wrt=("u",), steps=6)
    u0 = model._carry.state["u"].storage

    def loss(field):
        return sum(jnp.sum(f.data ** 2) for f in run((field,)).state)

    grad = np.asarray(jax.grad(loss)(u0))
    assert bool(np.all(np.isfinite(grad)))
    direction = jnp.asarray(rng.standard_normal(u0.shape), dtype=u0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u0 + eps * direction))
          - float(loss(u0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
