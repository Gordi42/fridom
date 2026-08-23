r"""The z* vertical coordinate: the mapping builder and ZStarGeometry.

Flow-following-coordinates plan, stage Z. Two groups of gates, split
by what they need:

- **Module-level** (this file's first half): the mapping's metrics,
  the module's declarations, its bind-time taught errors, and the
  SELF_UPDATE values (``eta == ps/g``, ``eta_dot == -T*``). These
  exercise ``hydrostatic/modules/zstar.py`` alone and hold whether or
  not the hydrostatic core threads the dynamic mapping parameters.
- **Model-level** (the second half, under its own banner): the
  physics gates — frozen reproduction, constancy, volume, tracer
  content, the nonlinear shallow-water oracle, the linear limit,
  compile-once, autodiff and device-count invariance. They need the
  hydrostatic DIAGNOSE stages / free-surface family to read the
  CURRENT ``eta`` through ``grid.metric(..., params=...)``; until
  that threading lands they measure a model whose interior geometry
  is frozen at ``eta = 0``, and the conservation gates fail by
  construction. Every one of them builds through :func:`zstar_model`.

Self-contained builders (AGENTS oversized-module rule).
"""
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
import fridom.shallowwater2 as sw
from fridom.hydrostatic.params import GRAVITY
from fridom.model.context import StepContext
from fridom.model.errors import MissingFieldError, MissingParameterError
from fridom.model.modules.moving_geometry import MeshVelocityCorrection
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh
MIM = fr.spatial.meshes.MappedIntervalMesh

G = 2.0
N2 = 1.0
DT = 2e-3
N, NZ = 4, 3


# ================================================================
#  Builders
# ================================================================
def depth(x, y):
    """Sloped bottom, 20% of the mean depth."""
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def meshes(n=N, nz=NZ, stretched=False):
    """Doubly-periodic horizontal, bounded base column z in [-1, 0]."""
    if stretched:
        mz = MIM(nz, (-1.0, 0.0),
                 lambda s: -1.0 + s - 0.2 * jnp.sin(jnp.pi * s), name="z")
    else:
        mz = IM(nz, (-1.0, 0.0), periodic=False, name="z")
    return (IM(n, (0.0, 1.0), periodic=True, name="x"),
            IM(n, (0.0, 1.0), periodic=True, name="y"),
            mz)


def zstar_grid(n=N, nz=NZ, bottom=depth, family=None, device_ids=None,
               stretched=False):
    """Return a z* grid ``zp = eta + (H + eta) z``."""
    grid = fr.spatial.Grid(meshes(n, nz, stretched=stretched),
                           mapping=hy.zstar_mapping(bottom),
                           device_ids=device_ids)
    if family is not None:
        grid.set_default_family(family)
    return grid


def sigma_grid(n=N, nz=NZ, bottom=depth, family=None):
    """Return the static terrain twin ``zp = z H(x, y)``."""
    grid = fr.spatial.Grid(
        meshes(n, nz),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": bottom}))
    if family is not None:
        grid.set_default_family(family)
    return grid


def flat_grid(n=N, nz=NZ):
    """Return the unmapped (fixed-domain) twin."""
    return fr.spatial.Grid(meshes(n, nz))


def model(grid, *, buoyancy=None, advection=None, extra=(), dt=DT,
          free_surface=None, gravity=G, coriolis=None):
    """Assemble a hydrostatic model on ``grid``."""
    return hy.Model(
        grid=grid, core=hy.Core(gravity=gravity),
        time_stepper=AdamBashforth(dt, order=3),
        buoyancy=buoyancy, coriolis=coriolis,
        free_surface=free_surface or hy.ExplicitFreeSurface(),
        advection=advection, modules_extra=extra)


def zstar_model(grid, *, ale=True, **kwargs):
    """Assemble the z* pair onto a hydrostatic model."""
    extra = [hy.ZStarGeometry()]
    if ale:
        extra.append(MeshVelocityCorrection())
    return model(grid, extra=tuple(extra), **kwargs)


def smooth(grid, space, fn):
    """Materialize ``fn`` on ``space``, naming only its own axes."""
    var = tuple(n for f in space.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)

    def init(**c):
        return fn(**c) + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return grid.create_field(space, init=init)


def nodes(grid, space, name):
    """Return the evaluation nodes of ``space`` along ``name``."""
    return np.asarray(grid.evaluation_nodes(space.bare, name).data)


def self_update(mdl, gravity=G):
    """Run the bound ZStarGeometry's SELF_UPDATE on the model state."""
    module = mdl.module(hy.ZStarGeometry)
    ctx = StepContext(params={str(GRAVITY): gravity}, clock=0.0,
                      dt=DT, stage_dt=DT)
    return module._update_geometry(mdl.state, ctx)


def bitwise(got, want, name):
    """Frozen reproduction: bitwise on cpu, roundoff-tight elsewhere.

    The static and the z* assembly compile two different HLO
    programs (the z* pipeline threads params= and a zero ALE tendency
    through every metric derivation). On cpu the two lower
    identically — the valuable pin; on gpu XLA is free to reassociate,
    so the contract there is a tolerance (the moving-geometry
    validation precedent).
    """
    got, want = np.asarray(got), np.asarray(want)
    if jax.default_backend() == "cpu":
        assert np.array_equal(got, want), name
    else:
        assert np.allclose(got, want, rtol=0.0, atol=1e-14), name


# ================================================================
#  zstar_mapping -- the declaration
# ================================================================
def test_mapping_declares_the_z_star_metric_vocabulary():
    mapping = zstar_grid().mapping
    assert set(mapping.metric_names) == {
        "dzp_dz", "dzp_dx", "dzp_dy", "dz_dzp", "dzp_dH", "dzp_deta",
        "sqrt_g"}
    assert mapping.param_names == ("H", "eta")
    assert mapping.param_coords == {"H": ("x", "y"),
                                    "eta": ("x", "y")}
    # one single-base analytic column, coupling the horizontal too
    assert mapping.column_corrections == {
        "x": ("zp", "z"), "y": ("zp", "z"), "z": ("zp", "z")}


@pytest.mark.parametrize("bottom", [depth, 1.3],
                         ids=["callable-depth", "constant-depth"])
def test_mapping_metrics_match_the_analytic_map(bottom):
    # J = H + eta, 1/J, and the mesh-velocity sensitivity 1 + z, all
    # against the closed form at the cell centres
    grid = zstar_grid(n=8, nz=4, bottom=bottom)
    cell = fr.spatial.Collocated().resolve(grid)
    x, y, z = (nodes(grid, cell, n) for n in ("x", "y", "z"))
    amp = 0.1
    eta = smooth(grid, fr.spatial.Profile("x", "y").resolve(grid),
                 lambda x, y: amp * jnp.sin(2 * jnp.pi * x)
                 * jnp.cos(2 * jnp.pi * y))
    params = {"eta": eta}
    h = np.asarray(bottom(x, y)) if callable(bottom) else bottom
    e = amp * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    want = np.broadcast_to(h + e, (8, 8, 4))
    for name, ref in (("dzp_dz", want),
                      ("dz_dzp", 1.0 / want),
                      ("dzp_deta", np.broadcast_to(1.0 + z, (8, 8, 4))),
                      ("dzp_dH", np.broadcast_to(z, (8, 8, 4)))):
        got = np.asarray(grid.metric(cell, name, params=params).data)
        np.testing.assert_allclose(got, ref, atol=1e-14, err_msg=name)


def test_mapping_at_zero_eta_is_the_sigma_mapping_bitwise():
    # the structural half of the frozen gate: with the static eta
    # default (zero) every z* metric row reproduces the terrain
    # (sigma) row BITWISE — the map differs only by exact-zero terms
    zs, sg = zstar_grid(n=8, nz=4), sigma_grid(n=8, nz=4)
    for name in ("dzp_dz", "dz_dzp", "dzp_dx", "dzp_dy", "sqrt_g"):
        a = grid_metric(zs, name)
        b = grid_metric(sg, name)
        bitwise(a, b, name)


def grid_metric(grid, name):
    """Return a metric on the collocated cell as a numpy array."""
    cell = fr.spatial.Collocated().resolve(grid)
    return np.asarray(grid.metric(cell, name).data)


def test_mapping_rejects_colliding_and_column_varying_declarations():
    with pytest.raises(TypeError, match="two distinct strings"):
        hy.zstar_mapping(1.0, horizontal=("x", "x"))
    with pytest.raises(ValueError, match="must be distinct"):
        hy.zstar_mapping(1.0, eta="H")
    with pytest.raises(ValueError, match="varies along"):
        hy.zstar_mapping(lambda x, y, z: 1.0 + 0.0 * (x + y + z))


def test_mapping_renames_its_parameter():
    grid = fr.spatial.Grid(meshes(),
                           mapping=hy.zstar_mapping(1.0, eta="surface"))
    assert grid.mapping.param_names == ("H", "surface")
    mdl = model(grid, extra=(hy.ZStarGeometry(eta="surface"),))
    assert {"surface", "surface_dot"} <= {f.name for f in mdl.state}
    mdl.advance(1)
    assert not mdl.panicked


def test_mapping_composes_with_a_stretched_base_column():
    # the stretching rides grid.measure, the chart grid.metric: the
    # graded z* column's Jacobian is still H + eta everywhere
    grid = zstar_grid(n=4, nz=4, bottom=1.0, stretched=True)
    cell = fr.spatial.Collocated().resolve(grid)
    got = np.asarray(grid.metric(cell, "dzp_dz").data)
    np.testing.assert_allclose(got, 1.0, atol=1e-14)


# ================================================================
#  ZStarGeometry -- construction and declarations
# ================================================================
def test_construction_validates_the_names():
    with pytest.raises(TypeError, match="non-empty string"):
        hy.ZStarGeometry(eta="")
    with pytest.raises(TypeError, match="two distinct strings"):
        hy.ZStarGeometry(horizontal=("x",))


def test_declarations_pair_eta_and_its_time_derivative():
    decls = hy.ZStarGeometry().field_declarations
    assert tuple(d.name for d in decls) == ("eta", "eta_dot")
    assert all(d.lifecycle is fr.model.Lifecycle.AUXILIARY
               for d in decls)
    assert all(d.time_dependent for d in decls)
    # the MovingGeometry convention: a mapping parameter carries the
    # FieldDeclaration units sentinel, not a claimed unit
    assert all(d.units == "unknown" for d in decls)
    assert hy.ZStarGeometry().param_names == ("eta",)
    assert hy.ZStarGeometry(eta="s").param_names == ("s",)


def test_declares_the_free_surface_and_velocity_references():
    names = tuple(r.name
                  for r in hy.ZStarGeometry().field_references)
    assert names == ("ps", "u", "v")
    assert all(r.hint for r in hy.ZStarGeometry().field_references)
    (ref,) = hy.ZStarGeometry().parameter_references
    assert str(ref.name) == "hydrostatic.gravity"


def test_stage_is_one_self_update_reading_the_surface_and_velocity():
    (stage,) = hy.ZStarGeometry().stages
    assert stage.kind is fr.model.StageKind.SELF_UPDATE
    assert stage.reads == ("ps", "u", "v", "eta", "eta_dot")
    assert stage.writes == ("eta", "eta_dot")


def test_declares_the_depth_two_stencil_halo():
    mdl = model(zstar_grid(), extra=(hy.ZStarGeometry(),))
    module = mdl.module(hy.ZStarGeometry)
    assert module.extra_halo == HaloSpec({"x": 2, "y": 2, "z": 2})


def test_eta_lives_on_the_barotropic_surface_pressure_space():
    mdl = model(zstar_grid(), extra=(hy.ZStarGeometry(),))
    assert (mdl.state["eta"].function_space
            == mdl.state["ps"].function_space)
    assert (mdl.state["eta_dot"].function_space
            == mdl.state["ps"].function_space)
    # the documented default: zeros (consistent when ps starts at 0)
    np.testing.assert_array_equal(
        np.asarray(mdl.state["eta"].data), 0.0)
    np.testing.assert_array_equal(
        np.asarray(mdl.state["eta_dot"].data), 0.0)


# ================================================================
#  ZStarGeometry -- bind-time taught errors
# ================================================================
def test_bind_requires_a_coordinate_mapping():
    with pytest.raises(ValueError, match="no coordinate mapping"):
        model(flat_grid(), extra=(hy.ZStarGeometry(),))


def test_bind_requires_the_mapping_to_declare_the_parameter():
    with pytest.raises(ValueError, match=r"does not.*declare"):
        model(sigma_grid(), extra=(hy.ZStarGeometry(),))


def test_bind_requires_eta_on_exactly_the_horizontal():
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H, eta: eta + (H + eta) * z},
        params={"H": depth, "eta": lambda x: 0.0 * x})
    grid = fr.spatial.Grid(meshes(), mapping=mapping)
    with pytest.raises(ValueError, match="exactly the horizontal"):
        model(grid, extra=(hy.ZStarGeometry(),))


def test_bind_refuses_an_immersed_grid():
    grid = fr.spatial.Grid(
        meshes(), mapping=hy.zstar_mapping(depth),
        immersed=ImmersedDomain(
            lambda x, y, z: 1.0 + 0.0 * (x + y + z), order=2))
    with pytest.raises(NotImplementedError, match="immersed"):
        hy.ZStarGeometry().bind(SimpleNamespace(grid=grid))


def test_requires_the_surface_pressure_and_the_velocities():
    with pytest.raises(MissingFieldError, match="ps"):
        fr.model.Model(
            grid=zstar_grid(),
            modules=(hy.Core(gravity=G), hy.ZStarGeometry()),
            time_stepper=AdamBashforth(DT, order=3))


def test_requires_the_dimensional_gravity():
    # iteration 1 is dimensional-only: eta = ps/g needs the physical
    # gravitational acceleration, so a nondimensional free surface is
    # a taught missing-parameter error
    with pytest.raises(MissingParameterError,
                       match=r"hydrostatic\.gravity"):
        hy.Model(
            grid=zstar_grid(), core=hy.Core(),
            scaling=fr.scaling.ExternalWave(),
            free_surface=hy.ExplicitFreeSurface(froude_number=1.0),
            time_stepper=AdamBashforth(DT, order=3),
            modules_extra=(hy.ZStarGeometry(),))


# ================================================================
#  ZStarGeometry -- the SELF_UPDATE values
# ================================================================
def test_self_update_writes_eta_and_the_kinematic_tendency():
    # the fully hand-built check: a CONSTANT depth and a CONSTANT ps
    # make J = H + ps/g an exact constant on every face, so
    #   eta_dot = -J int [d_x u + d_y v] dz
    # is a pure numpy backward-difference sum on the C grid.
    mdl = model(zstar_grid(bottom=1.0), extra=(hy.ZStarGeometry(),))
    rng = np.random.default_rng(0)
    u = rng.standard_normal((N, N, NZ))
    v = rng.standard_normal((N, N, NZ))
    ps = 0.3 * np.ones((N, N, 1))
    mdl.set_fields(u=u, v=v, ps=ps)
    out = self_update(mdl)
    np.testing.assert_array_equal(np.asarray(out["eta"].data), ps / G)
    jac = 1.0 + 0.3 / G
    dz, dx = 1.0 / NZ, 1.0 / N
    ref = -jac * dz / dx * (
        (u - np.roll(u, 1, axis=0)).sum(axis=2, keepdims=True)
        + (v - np.roll(v, 1, axis=1)).sum(axis=2, keepdims=True))
    np.testing.assert_allclose(np.asarray(out["eta_dot"].data), ref,
                               atol=1e-13)


def test_self_update_matches_the_transport_divergence_on_a_slope():
    # the same quantity on a sloped bottom with a varying surface,
    # against the COMMUTED spelling (integrate the J-weighted
    # transport first, then difference) — the vertical reduction and
    # the horizontal difference commute exactly in exact arithmetic
    grid = zstar_grid(n=8, nz=4)
    mdl = model(grid, extra=(hy.ZStarGeometry(),))
    rng = np.random.default_rng(1)
    mdl.set_fields(
        u=rng.standard_normal((8, 8, 4)),
        v=rng.standard_normal((8, 8, 4)),
        ps=np.asarray(smooth(
            grid, mdl.state["ps"].function_space,
            lambda x, y: 0.4 * jnp.sin(2 * jnp.pi * x)
            * jnp.cos(2 * jnp.pi * y)).data))
    out = self_update(mdl)
    state = mdl.state
    u, v = state["u"], state["v"]
    params = {"eta": out["eta"]}
    ju = u * grid.metric(u.function_space.bare, "dzp_dz",
                         params=params)
    jv = v * grid.metric(v.function_space.bare, "dzp_dz",
                         params=params)
    ref = (Integral()["z"](ju).diff("x")
           + Integral()["z"](jv).diff("y"))
    np.testing.assert_allclose(np.asarray(out["eta_dot"].data),
                               -np.asarray(ref.data), atol=1e-13)


def test_eta_dot_is_the_surface_pressure_tendency_over_gravity():
    # the GCL's "same discrete operator" claim: d ps/dt = -g T* and
    # eta = ps/g, so eta_dot must equal (d ps/dt)/g to roundoff. At
    # ps == 0 the free surface's own static-default eta IS the
    # current eta, so the check holds independently of the core's
    # dynamic-parameter threading.
    mdl = model(zstar_grid(), extra=(hy.ZStarGeometry(),))
    rng = np.random.default_rng(2)
    mdl.set_fields(u=rng.standard_normal((N, N, NZ)),
                   v=rng.standard_normal((N, N, NZ)),
                   ps=np.zeros((N, N, 1)))
    out = self_update(mdl)
    tendency = mdl.tendency(mdl.state, constraints=False)
    np.testing.assert_allclose(
        np.asarray(out["eta_dot"].data),
        np.asarray(tendency["ps"].data) / G, atol=1e-14)


def test_self_update_tracks_the_surface_through_a_run():
    # the stored eta is the LAST substage's ps/g (the surface the
    # substage's physics saw), so it lags the committed ps by one
    # substage tendency — but tracks it to that order
    mdl = zstar_model(zstar_grid())
    mdl.set_fields(ps=lambda x, y: 0.05 * jnp.sin(2 * jnp.pi * x)
                   + 0.0 * y)
    mdl.advance(4)
    assert not mdl.panicked
    eta = np.asarray(mdl.state["eta"].data)
    ps = np.asarray(mdl.state["ps"].data)
    assert np.abs(eta - ps / G).max() < 10.0 * DT * np.abs(
        np.asarray(mdl.state["eta_dot"].data)).max()


# ================================================================
#  MeshVelocityCorrection -- the z* wiring
# ================================================================
def test_the_ale_correction_skips_the_barotropic_prognostic():
    ale = MeshVelocityCorrection()
    model(zstar_grid(), buoyancy=hy.BuoyancyTracer(),
          extra=(hy.ZStarGeometry(), ale))
    # ps is PROGNOSTIC but column-constant: no column derivative to
    # transport past the moving nodes
    assert set(ale.fields) == {"u", "v", "b"}
    assert ale.driven_params == ("eta",)


# ================================================================
#  MODEL-LEVEL GATES
# ================================================================
# Everything below needs the hydrostatic package to read the CURRENT
# ``eta`` -- the DIAGNOSE stages (``w``, ``p_hyd``, the slope-corrected
# pressure gradient) and the free-surface family's transport
# divergence must query ``grid.metric(..., params=mapping_params(...))``
# instead of the static defaults. Until that threading lands the
# interior geometry of these models is frozen at ``eta = 0`` while the
# surface moves, so the conservation / oracle gates below are expected
# to fail (the frozen and volume gates hold either way, by
# construction).
NB, AMP = 8, 0.2


def stratified():
    """Return the linear background stratification module."""
    return hy.ConstantStratification(n2=N2)


def bump(x, y=None):
    """Return a smooth depth-uniform surface pattern."""
    del y
    return jnp.sin(2 * jnp.pi * x)


def horizontal_sum(field):
    """Return the plain horizontal sum of a barotropic field."""
    return float(jnp.sum(field.data))


def column_content(state, name="b"):
    """Return the discrete tracer content ``sum(J b) dV``."""
    grid = state[name].grid
    params = fr.model.modules.mapping_params(state, grid)
    field = state[name]
    jac = np.asarray(grid.metric(field.function_space, "dzp_dz",
                                 params=params).data)
    cell = 1.0 / (field.data.shape[0] * field.data.shape[1]
                  * field.data.shape[2])
    return float((jac * np.asarray(field.data)).sum()) * cell


# ----------------------------------------------------------------
#  Gate 3: frozen surface reproduces the sigma model
# ----------------------------------------------------------------
def test_frozen_surface_reproduces_the_sigma_tendency_bitwise():
    # u = v = 0 makes T* -- and hence eta_dot -- an exact zero, and
    # ps = 0 makes eta an exact zero, so every z* metric row equals its
    # sigma twin bitwise and the ALE correction is a bitwise zero: the
    # whole tendency must reproduce the static terrain model's, on a
    # genuinely sloped bottom with a non-trivial buoyancy field.
    zs = zstar_model(zstar_grid(n=NB, nz=4), buoyancy=stratified())
    sg = model(sigma_grid(n=NB, nz=4), buoyancy=stratified())
    ver = (np.arange(4) + 0.5) / 4 - 1.0
    hor = (np.arange(NB) + 0.5) / NB
    x, _, z = np.meshgrid(hor, hor, ver, indexing="ij")
    b = 0.01 * np.cos(2 * np.pi * x) * np.cos(np.pi * z)
    for mdl in (zs, sg):
        mdl.set_fields(b=b)
    tz = zs.tendency(zs.state, constraints=False)
    ts = sg.tendency(sg.state, constraints=False)
    for name in ("u", "v", "b", "ps"):
        bitwise(tz[name].data, ts[name].data, name)
    assert float(jnp.abs(tz["u"].data).max()) > 1e-4  # non-trivial


def test_frozen_surface_reproduces_the_sigma_run_bitwise():
    # a genuinely frozen state carried through five AB3 steps: a
    # constant depth, a z-only buoyancy (no horizontal pressure
    # gradient) and u = U(y), v = V(x) (an exactly divergence-free
    # transport, so T* -- and ps, eta, eta_dot -- stay exact zeros).
    # The z* pair must then contribute exactly nothing.
    def flat_depth(x, y):
        return 1.0 + 0.0 * x + 0.0 * y

    zs = zstar_model(zstar_grid(n=NB, nz=4, bottom=1.0),
                     buoyancy=stratified())
    sg = model(sigma_grid(n=NB, nz=4, bottom=flat_depth),
               buoyancy=stratified())
    ver = (np.arange(4) + 0.5) / 4 - 1.0
    hor = (np.arange(NB) + 0.5) / NB
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    fields = {"u": 0.02 * np.sin(2 * np.pi * y),
              "v": 0.02 * np.sin(2 * np.pi * x),
              "b": 0.01 * np.cos(np.pi * z) * np.ones_like(x)}
    for mdl in (zs, sg):
        mdl.set_fields(**fields)
        mdl.advance(5)
    assert not zs.panicked
    for name in ("u", "v", "w", "b", "ps", "p_hyd"):
        bitwise(zs.state[name].data, sg.state[name].data, name)
    np.testing.assert_array_equal(
        np.asarray(zs.state["eta"].data), 0.0)
    np.testing.assert_array_equal(
        np.asarray(zs.state["eta_dot"].data), 0.0)


# ----------------------------------------------------------------
#  Gate 4: constancy under a running free-surface wave
# ----------------------------------------------------------------
def test_uniform_buoyancy_stays_uniform_under_a_surface_wave():
    grid = zstar_grid(n=NB, nz=4, bottom=1.0)
    mdl = zstar_model(grid, buoyancy=hy.BuoyancyTracer(),
                      advection=fr.model.modules.CenteredAdvection())
    shape = mdl.state["b"].data.shape
    hor = (np.arange(NB) + 0.5) / NB
    x, _, _ = np.meshgrid(hor, hor, np.arange(4), indexing="ij")
    mdl.set_fields(b=3.0 * np.ones(shape),
                   ps=G * AMP * np.sin(2 * np.pi * x[:, :, :1]))
    mdl.advance(8)
    assert not mdl.panicked
    b = np.asarray(mdl.state["b"].data)
    # the surface genuinely moved (the check is non-trivial)
    assert float(jnp.abs(mdl.state["eta_dot"].data).max()) > 1e-3
    assert np.abs(b - 3.0).max() < 1e-13


# ----------------------------------------------------------------
#  Gate 5: the column volume int (H + eta) dA is exactly conserved
# ----------------------------------------------------------------
def test_the_column_volume_is_exactly_conserved():
    # H is static, so int (H + eta) dA drifts exactly as int eta dA =
    # int ps dA / g -- which the explicit free surface conserves to
    # roundoff (no 1/H(x, y) division anywhere on the step path).
    grid = zstar_grid(n=NB, nz=4)
    mdl = zstar_model(grid, buoyancy=stratified())
    hor = (np.arange(NB) + 0.5) / NB
    x, _ = np.meshgrid(hor, hor, indexing="ij")
    mdl.set_fields(ps=G * AMP * np.sin(2 * np.pi * x)[:, :, None])
    scale = NB * NB  # int H dA in cell units (mean depth 1)
    totals = []
    for _ in range(5):
        mdl.advance(2)
        totals.append(horizontal_sum(mdl.state["ps"]) / G
                      + scale)
    drift = max(abs(t - totals[0]) for t in totals)
    assert drift < 1e-13 * scale


# ----------------------------------------------------------------
#  Gate 6: the tracer content int J b dV
# ----------------------------------------------------------------
# NOTE (scope): the contract's machine-precision form of this gate --
# ``int J b`` conserved to 1e-12 per step -- needs the CONSERVATIVE
# flux route of ``MeshVelocityCorrection``, which is selected by a
# ``CellAvg`` column factor, i.e. the finite-volume family. The
# hydrostatic package does not run on ``family="fv"`` at all today:
# ``Core/diagnose_w`` raises ``SpaceMismatchError: x: Center(x) vs
# CellAvg(x)`` on a *flat, static* FV grid, so the limitation is
# pre-existing and unrelated to z*. On the nodal family every field
# takes the ADVECTIVE route, whose discrete column chain
# (``physical_diff`` + ``interpolate``) does not telescope, so the
# content drifts at the scheme's truncation order rather than at
# roundoff. What is asserted here is therefore a bound, with the
# measured numbers recorded below; the exact-conservation gate is
# unblocked by FV support in the hydrostatic core.
def _content_drift(free_surface):
    """Return the worst relative ``int J b`` drift over six steps."""
    grid = zstar_grid(n=NB, nz=4, bottom=1.0)
    mdl = zstar_model(grid, buoyancy=hy.BuoyancyTracer(),
                      free_surface=free_surface,
                      advection=fr.model.modules.CenteredAdvection())
    hor = (np.arange(NB) + 0.5) / NB
    ver = (np.arange(4) + 0.5) / 4 - 1.0
    x, _, z = np.meshgrid(hor, hor, ver, indexing="ij")
    mdl.set_fields(
        b=1.0 + 0.3 * np.cos(2 * np.pi * x) * np.cos(np.pi * z),
        ps=G * AMP * np.sin(2 * np.pi * x[:, :, :1]))
    start = column_content(mdl.state)
    worst = 0.0
    for _ in range(6):
        mdl.advance(1)
        worst = max(worst, abs(column_content(mdl.state) - start))
    assert not mdl.panicked
    return worst / abs(start)


def test_tracer_content_drift_stays_at_truncation():
    # measured worst relative drift over six steps (nodal family,
    # advective ALE route): explicit 3.7e-06, implicit 3.7e-06. The
    # implicit variant's number is reported, not asserted -- its
    # ps^{n+1} comes from a solve, so the eta_dot the ALE term read
    # differs from the realized Delta eta / Delta t by O(dt) (the
    # documented owner call, plan 5.1).
    explicit = _content_drift(hy.ExplicitFreeSurface())
    implicit = _content_drift(hy.ImplicitFreeSurface())
    assert explicit < 1e-4
    assert np.isfinite(implicit)


# ----------------------------------------------------------------
#  Gate 7: the nonlinear shallow-water oracle
# ----------------------------------------------------------------
def _barotropic_ic(n, amp):
    """Return a depth-uniform Gaussian surface bump eta(x, y)."""
    hor = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(hor, hor, indexing="ij")
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    return amp * np.exp(-r2 / (2 * 0.12 ** 2))


def _sw_run(n, eta0, dt, steps, *, nonlinear=True):
    """Run the shallow-water oracle (Sadourny, or its linear twin)."""
    grid = fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y")))
    mdl = sw.Model(
        grid=grid, core=sw.Core(gravity=G, depth=1.0),
        advection=sw.SadournyAdvection() if nonlinear else None,
        time_stepper=AdamBashforth(dt, order=3))
    mdl.set_fields(p=G * eta0)
    mdl.advance(steps)
    return (np.asarray(mdl.state["p"].data) / G,
            np.asarray(mdl.state["u"].data))


def _hydrostatic_barotropic_run(grid, eta0, dt, steps, *, zstar,
                                advection=True):
    """Run the barotropic hydrostatic twin (z* or fixed-domain)."""
    scheme = (fr.model.modules.CenteredAdvection() if advection
              else None)
    build = zstar_model if zstar else model
    mdl = build(grid, buoyancy=None, advection=scheme, dt=dt)
    mdl.set_fields(ps=G * eta0[:, :, None])
    mdl.advance(steps)
    assert not mdl.panicked
    return (np.asarray(mdl.state["ps"].data)[:, :, 0] / G,
            np.asarray(mdl.state["u"].data).mean(axis=2))


def _worst(a, b):
    """Worst absolute difference of an (eta, u) pair."""
    return max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max())


def test_barotropic_z_star_matches_the_nonlinear_shallow_water():
    # b == 0, depth-uniform IC, no rotation: the barotropic z* run IS
    # the nonlinear shallow-water system -- int J dz = H + eta and
    # int J u dz = (H + eta) ubar give d_t eta = -div((H + eta) ubar),
    # and the flux-form 3-D advection of a depth-uniform u collapses to
    # (u . grad) u. The two codes are NOT the same scheme (Sadourny's
    # vector-invariant momentum against the hydrostatic flux form), so
    # the agreement is truncation-level, measured against the size of
    # the nonlinearity itself (the oracle minus its own linear twin).
    #
    # Measured on this configuration: the LINEAR hydrostatic barotropic
    # run and the LINEAR shallow-water run agree to 1.4e-17 (bitwise --
    # the two linear discretizations are identical), so the whole
    # discrepancy budget here is nonlinearity, and the nonlinearity
    # signal is 2.3e-02 against an eta scale of 0.25.
    n, nz, dt, steps, amp = 16, 4, 5e-3, 10, 0.4
    eta0 = _barotropic_ic(n, amp)
    oracle = _sw_run(n, eta0, dt, steps)
    linear_oracle = _sw_run(n, eta0, dt, steps, nonlinear=False)
    z_run = _hydrostatic_barotropic_run(
        zstar_grid(n=n, nz=nz, bottom=1.0), eta0, dt, steps, zstar=True)
    fixed = _hydrostatic_barotropic_run(
        flat_grid(n=n, nz=nz), eta0, dt, steps, zstar=False,
        advection=False)
    scale = np.abs(oracle[0]).max()
    signal = _worst(oracle, linear_oracle)
    z_err = _worst(z_run, oracle)
    fixed_err = _worst(fixed, oracle)
    # the comparison is non-trivial: the nonlinearity is a large
    # fraction of the signal at eta/H = 0.4
    assert signal > 0.05 * scale
    # the z* run captures it; the fixed-domain linear run does not
    assert z_err < 0.2 * signal
    assert fixed_err > 4.0 * z_err


# ----------------------------------------------------------------
#  Gate 8: the linear limit
# ----------------------------------------------------------------
def test_small_amplitude_z_star_matches_the_linear_free_surface():
    # at eta/H = 1e-4 the geometry nonlinearity is O(eta/H): the z*
    # run (advection off, so the only nonlinearity is the geometry)
    # and the fixed-domain linear free surface must agree to that
    # order -- and disagree AT it (an exactly zero difference means
    # the moving geometry never reached the interior).
    n, nz, dt, steps, amp = 16, 4, 5e-3, 10, 1e-4
    eta0 = _barotropic_ic(n, amp)
    z_eta, _ = _hydrostatic_barotropic_run(
        zstar_grid(n=n, nz=nz, bottom=1.0), eta0, dt, steps,
        zstar=True, advection=False)
    l_eta, _ = _hydrostatic_barotropic_run(
        flat_grid(n=n, nz=nz), eta0, dt, steps, zstar=False,
        advection=False)
    rel = np.abs(z_eta - l_eta).max() / np.abs(l_eta).max()
    assert rel < 1e-2
    assert rel > 1e-7


# ----------------------------------------------------------------
#  Gate 9: the surface amplitude sweeps through jit
# ----------------------------------------------------------------
def test_sweeping_the_surface_amplitude_compiles_once(compile_counter):
    grid = zstar_grid(n=NB, nz=4, bottom=1.0)
    mdl = zstar_model(grid, buoyancy=stratified())
    hor = (np.arange(NB) + 0.5) / NB
    x, _ = np.meshgrid(hor, hor, indexing="ij")
    pattern = np.sin(2 * np.pi * x)[:, :, None]
    amplitudes = [0.05, 0.1, 0.15]
    fields = [G * a * pattern for a in amplitudes]
    mdl.set_fields(ps=fields[0])
    mdl.advance(2)
    compile_counter.reset()
    for ps in fields:
        mdl.set_fields(ps=ps)
        mdl.advance(2)
    assert compile_counter.count == 0
    assert not mdl.panicked


# ----------------------------------------------------------------
#  Gate 10: differentiability (AGENTS.md policy)
# ----------------------------------------------------------------
def test_propagator_gradient_through_a_z_star_run_matches_fd():
    grid = zstar_grid(n=N, nz=NZ, bottom=1.0)
    mdl = zstar_model(grid, buoyancy=hy.BuoyancyTracer())
    hor = (np.arange(N) + 0.5) / N
    x, _ = np.meshgrid(hor, hor, indexing="ij")
    mdl.set_fields(ps=G * 0.1 * np.sin(2 * np.pi * x)[:, :, None])
    run = mdl.propagator(wrt=("ps",), steps=6)
    ps0 = mdl._carry.state["ps"].storage

    def loss(field):
        final = run((field,))
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(ps0))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(3)
    direction = jnp.asarray(rng.standard_normal(ps0.shape),
                            dtype=ps0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-5
    fd = (float(loss(ps0 + eps * direction))
          - float(loss(ps0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ----------------------------------------------------------------
#  Gate 11: device-count invariance
# ----------------------------------------------------------------
@pytest.mark.multi_device
def test_z_star_run_is_device_count_invariant():
    # to tight rounding, not bitwise: the sharded metric-scaled chains
    # fuse differently between the 1- and 4-device programs
    hor = (np.arange(NB) + 0.5) / NB
    x, _ = np.meshgrid(hor, hor, indexing="ij")
    ps0 = G * AMP * np.sin(2 * np.pi * x)[:, :, None]

    def run(device_ids):
        mdl = zstar_model(
            zstar_grid(n=NB, nz=4, bottom=1.0, device_ids=device_ids),
            buoyancy=stratified())
        mdl.set_fields(ps=ps0)
        mdl.advance(6)
        return {c: np.asarray(mdl.state[c].data)
                for c in ("u", "v", "b", "ps", "eta", "eta_dot")}

    four, one = run(None), run((0,))
    for name, want in one.items():
        np.testing.assert_allclose(four[name], want, rtol=0.0,
                                   atol=1e-11, err_msg=name)
