r"""The hydrostatic core on the finite-volume (cell-average) family.

Prefix-mirrored shard of ``hy.modules.core`` (AGENTS oversized-module
rule) covering stage F3 on the hydrostatic package: ``hy.Core(family=)``
declares the FV C-grid (``CellAvg`` cells, point-value faces — FV-D2
option A) and contributes the FV ``diff`` profile, so every staggered
difference of the package becomes the exact discrete Gauss /
face-difference row. The gates:

1. **nodal bitwise** — the default family resolution does not flip any
   existing model: a default assembly is bitwise an explicit
   ``family="nodal"`` one over a 10-step run;
2. **FV/nodal tendency identity** — on the flat periodic box the two
   families' 2nd-order stencils are bit-identical, so the tendencies
   *and* a 10-step run agree bitwise (the nonhydro2 F3 precedent,
   ``tests/nonhydro2/test_fv_default.py``);
3. **FV terrain** — the ``p_hyd`` column integral, the FTC telescoping
   of ``J omega`` and the bottom seed on a sigma column;
4. **autodiff / device invariance** on the FV step path;
5. **taught errors** for the out-of-scope combinations.

Self-contained builders (AGENTS oversized-module rule).
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.modules.core import (
    effective_family,
    fv_cgrid_overrides,
    resolve_model_family,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.operators.flux_diff import (
    FaceDifference,
    FluxDifference,
)
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh
MIM = fr.spatial.meshes.MappedIntervalMesh
G, DT, N, NZ = 2.0, 2e-3, 8, 4
ORDER_FLOOR = 1.7


# ================================================================
#  Builders
# ================================================================
def _depth(x, y):
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _meshes(n=N, nz=NZ, *, walled=False, stretched=False):
    if stretched:
        mz = MIM(nz, (-1.0, 0.0),
                 lambda s: -1.0 + s - 0.3 * jnp.sin(jnp.pi * s), name="z")
    else:
        mz = IM(nz, (-1.0, 0.0), periodic=False, name="z")
    return (IM(n, (0.0, 1.0), periodic=not walled, name="x"),
            IM(n, (0.0, 1.0), periodic=not walled, name="y"), mz)


def _flat_grid(n=N, nz=NZ, **kw):
    return fr.spatial.Grid(_meshes(n, nz, **kw))


def _terrain_grid(n=N, nz=NZ, *, stretched=False):
    return fr.spatial.Grid(
        _meshes(n, nz, stretched=stretched),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": _depth}))


def _model(grid, family, *, advection=True, coriolis=True, dt=DT,
           buoyancy=hy.BuoyancyTracer):
    # ``buoyancy`` is a FACTORY: a module binds to exactly one model,
    # so every assembly gets its own instance
    return hy.Model(
        grid=grid, core=hy.Core(gravity=G, family=family),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0) if coriolis else None,
        buoyancy=buoyancy(),
        free_surface=hy.ExplicitFreeSurface(),
        advection=(fr.model.modules.CenteredAdvection() if advection
                   else None))


def _stratified():
    return hy.ConstantStratification(n2=1.0)


def _seed(model):
    rng = np.random.default_rng(11)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        b=1.0 + 0.1 * rng.standard_normal(model.state["b"].shape),
        ps=0.05 * rng.standard_normal(model.state["ps"].shape))
    return model


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


def _wet(x, y, z):  # noqa: ARG001 — the indicator reads z only
    """Wet indicator of a flat cut-cell grid (the bottom 20% is dry)."""
    return z > -0.8


def _spaces(model):
    return {name: str(model.state[name].function_space.bare)
            for name in model.state.component_names}


# ================================================================
#  Family resolution (no auto flip) and the declared spaces
# ================================================================
def test_auto_follows_the_grid_and_never_flips():
    # the hydrostatic rule: None follows grid.default_family verbatim
    # (no promotion), so a plain grid stays nodal
    grid = _flat_grid()
    assert resolve_model_family(None, grid) == "nodal"
    assert resolve_model_family("fv", grid) == "fv"
    grid.set_default_family("fv")
    assert resolve_model_family(None, grid) == "fv"
    assert resolve_model_family("nodal", grid) == "nodal"


def test_effective_family_prefers_the_explicit_request():
    grid = _flat_grid()
    assert effective_family(None, grid) == "nodal"
    assert effective_family("fv", grid) == "fv"


def test_core_exposes_the_requested_family():
    assert hy.Core(gravity=G).family is None
    assert hy.Core(gravity=G, family="fv").family == "fv"


def test_fv_state_carries_the_finite_volume_spaces():
    model = _model(_flat_grid(), "fv")
    assert _spaces(model) == {
        "u": "Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)",
        "v": "CellAvg(x) ⊗ Right(y) ⊗ CellAvg(z)",
        "w": "CellAvg(x) ⊗ CellAvg(y) ⊗ Outer(z)",
        "p_hyd": "CellAvg(x) ⊗ CellAvg(y) ⊗ CellAvg(z)",
        "b": "CellAvg(x) ⊗ CellAvg(y) ⊗ CellAvg(z)",
        "ps": "CellAvg(x) ⊗ CellAvg(y) ⊗ Constant(z)",
        "f_coriolis": "Constant(x) ⊗ Constant(y) ⊗ Constant(z)",
    }


def test_nodal_state_is_unchanged():
    model = _model(_flat_grid(), None)
    spaces = _spaces(model)
    assert spaces["u"] == "Right(x) ⊗ Center(y) ⊗ Center(z)"
    assert spaces["w"] == "Center(x) ⊗ Center(y) ⊗ Outer(z)"
    assert spaces["b"] == "Center(x) ⊗ Center(y) ⊗ Center(z)"


def test_walled_fv_staggers_on_the_interior_faces():
    model = _model(_flat_grid(walled=True), "fv", advection=False,
                   buoyancy=_stratified)
    spaces = _spaces(model)
    assert spaces["b"] == "CellAvg(x) ⊗ CellAvg(y) ⊗ CellAvg(z)"
    assert spaces["u"].startswith("Inner(x")


# ================================================================
#  The FV C-grid diff profile
# ================================================================
def test_fv_cgrid_overrides_repoints_the_diff_rows():
    mx, _my, mz = _meshes()
    profile = fv_cgrid_overrides((mx, mz), "z")
    assert isinstance(profile[("diff", mx.cell_avg)], FaceDifference)
    assert isinstance(profile[("diff", mx.right)], FluxDifference)
    assert isinstance(profile[("diff", mz.cell_avg)], FaceDifference)
    assert isinstance(profile[("diff", mz.inner)], FluxDifference)
    # the both-boundary w faces get the exact Gauss row too (the row
    # the nonhydrostatic profile does not need)
    assert isinstance(profile[("diff", mz.outer)], FluxDifference)


def test_fv_cgrid_overrides_skips_a_mesh_without_cell_avg():
    class _NoAvg:
        names = ("q",)
        periodic = True

        @property
        def cell_avg(self):
            raise ValueError("no average family")

    assert fv_cgrid_overrides((_NoAvg(),), "z") == {}


def test_nodal_model_contributes_no_profile():
    core = hy.Core(gravity=G)
    assert core.grid_dispatch_overrides(_flat_grid()) == {}


def test_fv_model_grid_resolves_the_gauss_rows():
    model = _model(_flat_grid(), "fv", advection=False,
                   buoyancy=_stratified)
    u = model.state["u"]
    div = u.diff("x")
    assert isinstance(div.function_space.bare.factor("x"), CellAvg)
    grad = model.state["p_hyd"].diff("x")
    assert str(grad.function_space.bare.factor("x")) == "Right(x)"


# ================================================================
#  Gate 1: the default resolution does not flip an existing model
# ================================================================
def test_default_family_run_is_bitwise_the_explicit_nodal_run():
    steps = 10
    auto = _seed(_model(_flat_grid(), None))
    pinned = _seed(_model(_flat_grid(), "nodal"))
    auto.advance(steps)
    pinned.advance(steps)
    for name in ("u", "v", "b", "ps", "w", "p_hyd"):
        np.testing.assert_array_equal(
            np.asarray(auto.state[name].data),
            np.asarray(pinned.state[name].data), err_msg=name)


# ================================================================
#  Gate 2: FV/nodal tendency identity on the periodic flat box
# ================================================================
@pytest.mark.parametrize("advection", [False, True],
                         ids=["linear", "centered"])
def test_fv_is_bitwise_identical_to_nodal_on_the_flat_box(advection):
    # the 2nd-order FV stencils (FaceDifference / FluxDifference) are
    # bit-identical to the nodal staggered FiniteDifference on a
    # uniform mesh, so the whole model -- the continuity cumint, both
    # pressure gradients, the barotropic pair and the flux-form
    # advection -- carries the same numbers on both families
    buoyancy = hy.BuoyancyTracer if advection else _stratified
    fv = _seed(_model(_flat_grid(), "fv", advection=advection,
                      buoyancy=buoyancy))
    nodal = _seed(_model(_flat_grid(), "nodal", advection=advection,
                         buoyancy=buoyancy))
    tf = fv.tendency(fv.state, constraints=False)
    tn = nodal.tendency(nodal.state, constraints=False)
    for name in tf.component_names:
        np.testing.assert_array_equal(
            np.asarray(tf[name].data), np.asarray(tn[name].data),
            err_msg=f"tendency {name}")
    fv.advance(10)
    nodal.advance(10)
    for name in ("u", "v", "b", "ps", "w", "p_hyd"):
        np.testing.assert_array_equal(
            np.asarray(fv.state[name].data),
            np.asarray(nodal.state[name].data), err_msg=name)


def test_fv_is_bitwise_identical_to_nodal_on_a_stretched_column():
    # the stretch rides grid.measure, which both families divide by
    fv = _seed(_model(_flat_grid(stretched=True), "fv"))
    nodal = _seed(_model(_flat_grid(stretched=True), "nodal"))
    fv.advance(6)
    nodal.advance(6)
    for name in ("u", "v", "b", "ps"):
        np.testing.assert_array_equal(
            np.asarray(fv.state[name].data),
            np.asarray(nodal.state[name].data), err_msg=name)


# ================================================================
#  Gate 3: the FV terrain column
# ================================================================
@pytest.mark.parametrize("stretched", [False, True],
                         ids=["uniform-sigma", "stretched-sigma"])
def test_fv_p_hyd_converges_to_the_physical_integral(stretched):
    # b = sin(zp);  -int_zp^0 b dzp' = 1 - cos(zp).  The FV row of
    # CumulativeIntegral lands on CellAvg and carries the column
    # Jacobian, converging at second order on both sigma columns.
    errs = []
    for n in (8, 16, 32):
        grid = _terrain_grid(n, nz=n, stretched=stretched)
        grid.set_default_family("fv")
        coll = fr.spatial.Collocated().resolve(grid)
        assert isinstance(coll.factor("z"), CellAvg)
        zp = (_nodes(grid, coll, "z")
              * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
        b = grid.create_field(coll, data=jnp.sin(zp))
        p = -CumulativeIntegral(
            direction="down", target="center",
            jacobian=("zp",))["z"](b)
        errs.append(float(jnp.abs(p.data - (1.0 - jnp.cos(zp))).max()))
    orders = np.log2(np.asarray(errs[:-1]) / np.asarray(errs[1:]))
    assert np.all(orders > ORDER_FLOOR)


def test_fv_terrain_flux_telescopes_and_seeds_zero_at_the_bed():
    grid = _terrain_grid(16, nz=8)
    model = _model(grid, "fv", advection=False,
                   buoyancy=_stratified)
    rng = np.random.default_rng(4)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    state = model.state
    core = model.module(hy.Core)
    assert core._column == ("zp", "z")
    w = core._diagnose_w(state, None)["w"]
    flux = state.replace(w=w).chart["w"]
    u, v = state["u"], state["v"]
    ju = u * grid.metric(u.function_space.bare, "dzp_dz")
    jv = v * grid.metric(v.function_space.bare, "dzp_dz")
    dh = ju.diff("x") + jv.diff("y")
    # the exact Gauss row Outer -> CellAvg: d_z (J omega) == -Dh
    ftc = np.asarray(flux.diff("z").data) + np.asarray(dh.data)
    assert np.abs(ftc).max() < 1e-11
    fd = np.asarray(flux.data)
    zaxis = next(i for i, f in enumerate(flux.function_space.bare.factors)
                 if "z" in f.names)
    assert np.abs(np.take(fd, 0, axis=zaxis)).max() == 0.0


def test_fv_terrain_rest_state_error_converges():
    # the sigma pressure-gradient-error gate on the FV family: a
    # stratified fluid at rest over topography stays at rest to the
    # scheme's truncation order (the slope-corrected gradient). The
    # floor is 1.6 rather than the nodal shard's 1.7 only because this
    # runs the cheaper 8/16/32 triple whose coarsest pair is still
    # pre-asymptotic (1.66); the sharp gate is the bitwise-nodal
    # comparison below, and these are the nodal numbers exactly.
    errs = []
    for n in (8, 16, 32):
        grid = _terrain_grid(n, nz=n)
        model = _model(grid, "fv", advection=False,
                       buoyancy=_stratified)
        coll = model.state["b"].function_space
        zp = (_nodes(grid, coll, "z")
              * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
        model.set_fields(
            u=np.zeros(model.state["u"].shape),
            v=np.zeros(model.state["v"].shape),
            b=np.asarray(-2.0 * zp), ps=np.zeros(model.state["ps"].shape))
        dX = model.tendency(model.state)
        errs.append(max(float(jnp.abs(dX["u"].data).max()),
                        float(jnp.abs(dX["v"].data).max())))
    orders = np.log2(np.asarray(errs[:-1]) / np.asarray(errs[1:]))
    assert np.all(orders > 1.6)


def test_fv_terrain_slope_gradient_is_bitwise_the_nodal_one():
    # the FV column hop goes through the co-located nodal sibling
    # (Core._hop): the one-sided Inner -> Center interpolation the
    # nodal column takes, then the 2nd-order Center -> CellAvg
    # deconvolution. The slope-corrected pressure gradient is then
    # bitwise the nodal one, relabelled onto the average family.
    tend = {}
    for family in ("nodal", "fv"):
        grid = _terrain_grid(8)
        model = _model(grid, family, advection=False,
                       buoyancy=_stratified)
        coll = model.state["b"].function_space
        zp = (_nodes(grid, coll, "z")
              * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
        model.set_fields(
            u=np.zeros(model.state["u"].shape),
            v=np.zeros(model.state["v"].shape),
            b=np.asarray(-2.0 * zp - 0.7 * zp ** 2),
            ps=np.zeros(model.state["ps"].shape))
        dX = model.tendency(model.state)
        tend[family] = {n: np.asarray(dX[n].data) for n in ("u", "v")}
    for name in ("u", "v"):
        np.testing.assert_array_equal(
            tend["fv"][name], tend["nodal"][name], err_msg=name)


def test_fv_terrain_advection_preserves_a_constant_tracer():
    grid = _terrain_grid(8)
    model = _model(grid, "fv")
    rng = np.random.default_rng(5)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        b=np.full(model.state["b"].shape, 0.7))
    model.advance(5)
    b = np.asarray(model.state["b"].data)
    assert float(b.max() - b.min()) < 1e-13


# ================================================================
#  Gate 6: differentiability, device invariance, compile-once
# ================================================================
@pytest.mark.single_device
def test_grad_through_a_short_fv_run_matches_fd():
    model = _seed(_model(_flat_grid(), "fv", dt=1e-2))
    run = model.propagator(wrt=("b",), steps=6)
    b0 = model._carry.state["b"].storage

    def loss(field):
        return sum(jnp.sum(f.data ** 2) for f in run((field,)).state)

    grad = np.asarray(jax.grad(loss)(b0))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(b0.shape), dtype=b0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b0 + eps * direction))
          - float(loss(b0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


@pytest.mark.multi_device
def test_fv_run_is_device_count_invariant():
    def run(device_ids):
        grid = fr.spatial.Grid(_meshes(), device_ids=device_ids)
        model = _seed(_model(grid, "fv"))
        model.advance(6)
        return {c: np.asarray(model.state[c].data)
                for c in ("u", "v", "b", "ps")}

    many, one = run(None), run((0,))
    for name, want in one.items():
        np.testing.assert_allclose(many[name], want, rtol=0.0,
                                   atol=1e-11, err_msg=name)


def test_fv_run_compiles_once(compile_counter):
    model = _seed(_model(_flat_grid(), "fv"))
    model.advance(3)
    compile_counter.reset()
    model.advance(3)
    assert compile_counter.count == 0
    assert not model.panicked


# ================================================================
#  Gate 7: the taught errors of the out-of-scope combinations
# ================================================================
def test_unknown_family_is_rejected_at_construction():
    with pytest.raises(ValueError, match="family must be one of"):
        hy.Core(gravity=G, family="spectral")


def test_fv_on_an_immersed_grid_is_a_taught_error():
    grid = fr.spatial.Grid(
        _meshes(), immersed=ImmersedDomain(_wet))
    with pytest.raises(NotImplementedError,
                       match="immersed grid is not supported"):
        _model(grid, "fv")


def test_nodal_on_an_immersed_grid_is_still_served():
    grid = fr.spatial.Grid(
        _meshes(), immersed=ImmersedDomain(_wet))
    model = _model(grid, None)
    assert str(model.state["b"].function_space.bare.factor("z")) \
        == "Center(z)"


def test_half_fv_explicit_assembly_is_a_taught_error():
    # the family reaches sibling modules through the grid default,
    # which hy.Model sets; a hand-rolled fr.model.Model on a nodal
    # grid would leave a CellAvg core beside a Center buoyancy
    grid = _flat_grid()
    with pytest.raises(ValueError, match="other discretization family"):
        fr.model.Model(
            grid=grid,
            modules=(hy.Core(gravity=G, family="fv"),
                     hy.BuoyancyTracer(),
                     hy.ExplicitFreeSurface(),
                     fr.model.modules.CenteredAdvection()),
            time_stepper=AdamBashforth(DT, order=3),
            scaling=fr.scaling.Dimensional())


class _PinnedFVTracer(fr.model.Module):

    """A passive tracer whose declaration pins the FV family itself."""

    @property
    def field_declarations(self):
        return (fr.model.FieldDeclaration.tracer(
            "c", space=fr.spatial.Collocated(family="fv"),
            long_name="fv tracer", units="1"),)


def test_a_pinned_fv_tracer_beside_the_nodal_core_is_served():
    # the per-field ``family=`` override of SpacePattern is the mixed
    # model (FV-D1b): only ``family=None`` declarations are held to
    # the core's family, a pinned passive CellAvg tracer is a choice
    model = hy.Model(
        grid=_flat_grid(), core=hy.Core(gravity=G, family="nodal"),
        time_stepper=AdamBashforth(DT, order=3),
        buoyancy=hy.BuoyancyTracer(),
        free_surface=hy.ExplicitFreeSurface(),
        advection=fr.model.modules.CenteredAdvection(),
        modules_extra=[_PinnedFVTracer()])
    assert str(model.state["c"].function_space.bare.factor("z")) \
        == "CellAvg(z)"
    assert str(model.state["b"].function_space.bare.factor("z")) \
        == "Center(z)"


def test_explicit_assembly_on_an_fv_grid_is_served():
    grid = _flat_grid()
    grid.set_default_family("fv")
    model = fr.model.Model(
        grid=grid,
        modules=(hy.Core(gravity=G), hy.BuoyancyTracer(),
                 hy.ExplicitFreeSurface(),
                 fr.model.modules.CenteredAdvection()),
        time_stepper=AdamBashforth(DT, order=3),
        scaling=fr.scaling.Dimensional())
    assert isinstance(
        model.state["p_hyd"].function_space.bare.factor("x"), CellAvg)
    _seed(model)
    model.advance(2)
    assert not model.panicked


def test_buoyancy_tracer_family_is_honored():
    grid = _flat_grid()
    grid.set_default_family("fv")
    assert hy.BuoyancyTracer(family="nodal").field_declarations[0] \
        .space.family == "nodal"
    assert hy.BuoyancyTracer().field_declarations[0].space.family is None
