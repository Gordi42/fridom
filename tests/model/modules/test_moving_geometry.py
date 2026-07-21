"""The moving-geometry modules: schedules, updates, ALE term.

Framework-level modules exercised on nonhydro2 models (the
test_relaxation precedent of borrowing a concrete port): a
terrain-following column ``zp = z * H(x, t)`` with zero physics
(f0 = n2 = 0, no advection) isolates the geometry update and the
mesh-velocity correction, so pointwise tendencies are exact checks.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.declarations import Lifecycle
from fridom.model.model import Model as FrModel
from fridom.model.model import _chunk_body
from fridom.model.module import Module
from fridom.model.modules.moving_geometry import (
    MeshVelocityCorrection,
    MovingGeometry,
    mapping_params,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.reconstruct import LinearReconstruction
from fridom.spatial.spaces.nodal import NodeSet

N = 8
LENGTH = 2 * np.pi
DT = 1e-2
H0 = 0.8
RATE = 0.05


def schedule(x, t):
    """H(x, t) = H0 (1 + 0.1 sin x) + RATE * t."""
    return H0 * (1.0 + 0.1 * jnp.sin(x)) + RATE * t


def make_grid(mapped=True, n=N):
    mx = IntervalMesh(n, (0.0, LENGTH), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, LENGTH), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, 1.0), periodic=False, name="z")
    if not mapped:
        return Grid((mx, my, mz))
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: schedule(x, 0.0)})
    return Grid((mx, my, mz), mapping=mapping)


def make_model(*modules, mapped=True):
    """Zero-physics linear nh model (geometry terms isolated).

    ``family="nodal"`` is explicit: these tests exercise the nodal
    (advective) ALE route. Since the 2026-07-17 ALE-on-FV closure a
    moving-geometry model auto-flips to FV, so pinning nodal keeps
    these on the nodal path; the FV route has its own tests below.
    """
    return nh.Model(
        grid=make_grid(mapped=mapped),
        core=nh.Core(family="nodal"),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        advection=False,
        modules_extra=modules)


def moving():
    return MovingGeometry({"H": schedule})


# ================================================================
#  MovingGeometry: construction-time validation
# ================================================================
def test_schedules_are_validated():
    with pytest.raises(ValueError, match="at least one"):
        MovingGeometry({})
    with pytest.raises(TypeError, match="callables"):
        MovingGeometry({"H": 1.0})
    with pytest.raises(TypeError, match="callables"):
        MovingGeometry({3: schedule})
    with pytest.raises(TypeError, match="time"):
        MovingGeometry({"H": schedule}, time="")
    with pytest.raises(ValueError, match="static parameter"):
        MovingGeometry({"H": lambda x: x})


def test_param_names_and_time_argument():
    module = MovingGeometry({"H": lambda tau: 1.0 + tau},
                            time="tau")
    assert module.param_names == ("H",)


def test_declarations_pair_value_and_dot_fields():
    decls = moving().field_declarations
    assert tuple(d.name for d in decls) == ("H", "H_dot")
    assert all(d.lifecycle is Lifecycle.AUXILIARY for d in decls)


# ================================================================
#  MovingGeometry: bind-time validation (taught errors)
# ================================================================
def test_bind_requires_a_mapping():
    with pytest.raises(ValueError, match="no coordinate mapping"):
        make_model(moving(), mapped=False)


def test_bind_rejects_an_undeclared_parameter():
    with pytest.raises(ValueError, match="names no mapping param"):
        make_model(MovingGeometry({"eta": lambda t: t}))


def test_bind_rejects_excess_schedule_coordinates():
    with pytest.raises(ValueError, match="varies along"):
        make_model(MovingGeometry(
            {"H": lambda x, y, t: 1.0 + 0 * x * y * t}))


# ================================================================
#  MovingGeometry: the geometry update
# ================================================================
def test_defaults_materialize_the_schedule_at_time_zero():
    model = make_model(moving())
    h = np.asarray(model.state["H"].data)
    x = (np.arange(N) + 0.5) * (LENGTH / N)
    np.testing.assert_allclose(
        h, np.asarray(schedule(x, 0.0)).reshape(h.shape),
        atol=1e-15)
    hdot = np.asarray(model.state["H_dot"].data)
    np.testing.assert_allclose(hdot, RATE, atol=1e-15)


def test_update_tracks_the_traced_clock():
    model = make_model(moving())
    model.advance(3)
    # the SELF_UPDATE of step k runs at start-of-step time (k-1) dt
    t_last = 2 * DT
    h = np.asarray(model.state["H"].data)
    x = (np.arange(N) + 0.5) * (LENGTH / N)
    np.testing.assert_allclose(
        h, np.asarray(schedule(x, t_last)).reshape(h.shape),
        atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(model.state["H_dot"].data), RATE, atol=1e-13)


def test_time_only_schedule_lives_on_a_one_dof_profile():
    mx = IntervalMesh(N, (0.0, LENGTH), periodic=True, name="x")
    my = IntervalMesh(N, (0.0, LENGTH), periodic=True, name="y")
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: H0 + 0.0 * x})
    grid = Grid((mx, my, mz), mapping=mapping)
    model = nh.Model(
        grid=grid, core=nh.Core(family="nodal"),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        advection=False,
        modules_extra=(MovingGeometry(
            {"H": lambda t: H0 + RATE * t}),))
    assert model.state["H"].data.size == 1
    np.testing.assert_allclose(
        float(model.state["H"].data.ravel()[0]), H0)
    np.testing.assert_allclose(
        float(model.state["H_dot"].data.ravel()[0]), RATE)


# ================================================================
#  mapping_params: the discovery convention
# ================================================================
def test_mapping_params_discovery():
    flat = make_grid(mapped=False)
    mapped = make_grid()
    h = mapped.create_field(mapped.factors[0].center, name="H")
    assert mapping_params({"H": h}, flat) is None
    assert mapping_params({"b": h}, mapped) is None
    assert mapping_params({"H": h, "b": h}, mapped) == {"H": h}


# ================================================================
#  MeshVelocityCorrection: construction and bind validation
# ================================================================
def test_fields_selection_is_validated():
    with pytest.raises(TypeError, match="non-empty tuple"):
        MeshVelocityCorrection(())
    with pytest.raises(TypeError, match="non-empty tuple"):
        MeshVelocityCorrection(("u", 3))
    assert MeshVelocityCorrection(("b",)).fields == ("b",)
    assert MeshVelocityCorrection().fields is None


def test_ale_bind_requires_a_mapped_column():
    with pytest.raises(ValueError, match="mapped column"):
        make_model(MeshVelocityCorrection(), mapped=False)


def test_ale_bind_requires_moving_geometry():
    with pytest.raises(ValueError, match="MovingGeometry"):
        make_model(MeshVelocityCorrection())


def test_ale_bind_rejects_unknown_and_non_prognostic_fields():
    with pytest.raises(ValueError, match="no module declares"):
        make_model(moving(), MeshVelocityCorrection(("ghost",)))
    with pytest.raises(ValueError, match="PROGNOSTIC"):
        make_model(moving(), MeshVelocityCorrection(("p",)))


def test_ale_defaults_to_every_prognostic_field():
    ale = MeshVelocityCorrection()
    make_model(moving(), ale)
    assert set(ale.fields) == {"u", "v", "w", "b"}
    assert ale.driven_params == ("H",)


def test_ale_rejects_multiple_mapped_columns():
    # two single-base analytic maps (parameter-free defaults keep
    # their coupled coordinate sets disjoint) exceed the stage-C4
    # support, mirroring the C3 pressure solver
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H,
              "yp": lambda y, YN: y * YN},
        params={"H": lambda: H0, "YN": lambda: 0.9})
    mx = IntervalMesh(N, (0.0, LENGTH), periodic=True, name="x")
    my = IntervalMesh(N, (0.0, 1.0), name="y")
    mz = IntervalMesh(N, (0.0, 1.0), name="z")
    grid = Grid((mx, my, mz), mapping=mapping)
    with pytest.raises(NotImplementedError,
                       match="exactly one mapped column"):
        nh.Model(
            grid=grid, core=nh.Core(),
            time_stepper=AdamBashforth(DT, order=3),
            coriolis=nh.FPlaneCoriolis(f0=0.0),
            stratification=nh.ConstantStratification(n2=0.0),
            advection=False,
            modules_extra=(
                MovingGeometry({"H": lambda t: H0 + 0.0 * t}),
                MeshVelocityCorrection()))


# ================================================================
#  MeshVelocityCorrection: the correction term (exact check)
# ================================================================
def test_ale_tendency_is_mdot_times_physical_gradient():
    # b initialized linearly in the computational column, geometry
    # H(x, 0) with H_dot = RATE: the correction is
    #   m_dot * db/dzp = (z * RATE) * (1 / H(x, 0))
    # exactly (linear-in-z fields differentiate and interpolate
    # exactly; the metric factors are pointwise)
    model = make_model(moving(), MeshVelocityCorrection(("b",)))
    hor = (np.arange(N) + 0.5) * (LENGTH / N)
    ver = (np.arange(N) + 0.5) / N
    x, _, z = np.meshgrid(hor, hor, ver, indexing="ij")
    model.set_fields(b=z)
    tendency = model.tendency(model.state, constraints=False)
    expected = z * RATE / np.asarray(schedule(x, 0.0))
    np.testing.assert_allclose(
        np.asarray(tendency["b"].data), expected, atol=1e-12)
    # w is not in the configured field set: its tendency is the
    # buoyancy force alone (b interpolated onto the faces, dsqr=1)
    b_at_w = np.asarray(
        model.state["b"].to(model.state["w"]).data)
    np.testing.assert_allclose(
        np.asarray(tendency["w"].data), b_at_w, atol=1e-14)


def test_ale_corrects_the_wall_staggered_component_too():
    # w lives on the Dirichlet-tagged column faces; the correction
    # resolves the tagged diff row (whose ghost fill is the wall
    # zero — hence a wall-compatible w) and lands back on w's own
    # space. w = z (1 - z) is quadratic, so the staggered two-point
    # difference and the linear interpolation back are exact.
    model = make_model(moving(), MeshVelocityCorrection())
    hor = (np.arange(N) + 0.5) * (LENGTH / N)
    faces = np.arange(1, N) / N  # interior column faces
    x, _, z = np.meshgrid(hor, hor, faces, indexing="ij")
    model.set_fields(w=z * (1.0 - z))
    tendency = model.tendency(model.state, constraints=False)
    expected = (z * RATE) * (1.0 - 2.0 * z) / np.asarray(
        schedule(x, 0.0))
    np.testing.assert_allclose(
        np.asarray(tendency["w"].data), expected, atol=1e-12)


def test_frozen_schedule_yields_a_zero_correction():
    frozen = MovingGeometry(
        {"H": lambda x, t: schedule(x, 0.0) + 0.0 * t})
    model = make_model(frozen, MeshVelocityCorrection())
    ver = (np.arange(N) + 0.5) / N
    _, _, z = np.meshgrid(ver, ver, ver, indexing="ij")
    model.set_fields(b=z)
    tendency = model.tendency(model.state, constraints=False)
    np.testing.assert_allclose(
        np.asarray(tendency["b"].data), 0.0, atol=1e-15)


def test_moving_model_advances_with_and_without_ale():
    # the CS-D4 off switch: both assemblies run to completion
    with_ale = make_model(moving(), MeshVelocityCorrection())
    without = make_model(moving())
    ver = (np.arange(N) + 0.5) / N
    _, _, z = np.meshgrid(ver, ver, ver, indexing="ij")
    with_ale.set_fields(b=0.01 * np.cos(np.pi * z))
    without.set_fields(b=0.01 * np.cos(np.pi * z))
    with_ale.advance(3)
    without.advance(3)
    assert not with_ale.panicked
    assert not without.panicked


# ================================================================
#  MeshVelocityCorrection: the FV (finite-volume) family
# ================================================================
def make_fv_model(*modules, mapped=True, n=N):
    """Zero-physics linear nh model on the FV family (ALE on FV).

    family="fv" is explicit: since the 2026-07-17 ALE-on-FV closure a
    moving-geometry model auto-defaults to FV, and the correction is
    family-aware — the flux form on the ``CellAvg`` column factors of
    ``b`` / ``u`` / ``v``, the advective form on ``w``'s point-valued
    (wall-normal) column factor.
    """
    return nh.Model(
        grid=make_grid(mapped=mapped, n=n),
        core=nh.Core(family="fv"),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        advection=False,
        modules_extra=modules)


def test_fv_ale_routes_per_column_factor():
    # b, u, v carry a CellAvg column (z) factor -> flux route; w is the
    # wall-normal velocity, a point value on its own column axis ->
    # advective route. Routing is resolved at bind on the column factor.
    ale = MeshVelocityCorrection()
    make_fv_model(moving(), ale)
    assert set(ale.fields) == {"u", "v", "w", "b"}
    assert set(ale._flux_fields) == {"u", "v", "b"}
    assert ale._advective_fields == ("w",)


def test_fv_flux_route_is_mdot_times_physical_gradient():
    # the flux form on a CellAvg column reproduces the advective
    # correction m_dot * db/dzp exactly for a field linear in the
    # computational column: reconstruction (the one-sided wall closure
    # included) is exact for linears and the metric factors are
    # pointwise, so b = z -> correction z * RATE / H(x, 0), the same
    # numbers the nodal advective form gives.
    model = make_fv_model(moving(), MeshVelocityCorrection(("b",)))
    hor = (np.arange(N) + 0.5) * (LENGTH / N)
    ver = (np.arange(N) + 0.5) / N
    x, _, z = np.meshgrid(hor, hor, ver, indexing="ij")
    model.set_fields(b=z)
    tendency = model.tendency(model.state, constraints=False)
    expected = z * RATE / np.asarray(schedule(x, 0.0))
    np.testing.assert_allclose(
        np.asarray(tendency["b"].data), expected, atol=1e-12)


def test_fv_w_takes_the_advective_route():
    # w lives on the Dirichlet-tagged column faces; its column factor is
    # a point value, so it keeps the advective form on FV (physical_diff
    # -> CellAvg through the FV flux difference, interpolate back onto
    # the face). w = z (1 - z) is quadratic, so the staggered difference
    # and the linear reconstruction back are exact, matching the nodal
    # numbers: m_dot * dw/dzp = z RATE (1 - 2z) / H.
    model = make_fv_model(moving(), MeshVelocityCorrection(("w",)))
    hor = (np.arange(N) + 0.5) * (LENGTH / N)
    faces = np.arange(1, N) / N
    x, _, z = np.meshgrid(hor, hor, faces, indexing="ij")
    model.set_fields(w=z * (1.0 - z))
    tendency = model.tendency(model.state, constraints=False)
    expected = (z * RATE) * (1.0 - 2.0 * z) / np.asarray(
        schedule(x, 0.0))
    np.testing.assert_allclose(
        np.asarray(tendency["w"].data), expected, atol=1e-12)


def test_fv_constancy_uniform_field_under_motion():
    # the free-stream / GCL property: a uniform field under genuine
    # geometry motion (H_dot != 0) has zero ALE correction to machine
    # precision. The two flux-form terms share the same face mesh
    # velocity, so they cancel on a constant (the reconstruction
    # reproduces constants, walls included). Machine zero, not bitwise:
    # the flux form's c*D(w) and D(c*w) differ at rounding — the
    # bitwise-exact constancy is the frozen-motion (w == 0) case.
    model = make_fv_model(moving(), MeshVelocityCorrection(("b",)))
    model.set_fields(b=lambda x, y, z: 3.0 + 0.0 * z)  # noqa: ARG005
    tendency = model.tendency(model.state, constraints=False)
    np.testing.assert_allclose(
        np.asarray(tendency["b"].data), 0.0, atol=1e-13)


@pytest.mark.parametrize("n", [8, 16])
def test_fv_telescoping_tracer_budget_closes(n):
    # the new invariant closing the gap: summed over the column the
    # flux-form correction telescopes, so the extensive ALE budget
    #   sum(V * corr) + sum(fbar * Vdot) == boundary mesh flux
    # holds to machine precision — the semi-discrete tracer budget
    # closes with zero spatial residual. Machine-exact at BOTH n = 8
    # and n = 16: the invariant is resolution-independent (the exact
    # analogue of F5 advection's exact conservation). b is x-even and
    # nonzero at the moving wall, so the boundary flux does NOT cancel
    # over the periodic transverse: a non-trivial check that the module
    # carries the moving-wall mesh flux (Outer flux_diff), not the
    # Inner variant's zero pad.
    model = make_fv_model(moving(), MeshVelocityCorrection(("b",)), n=n)
    hor = (np.arange(n) + 0.5) * (LENGTH / n)
    ver = (np.arange(n) + 0.5) / n
    x, _, z = np.meshgrid(hor, hor, ver, indexing="ij")
    model.set_fields(b=2.0 + np.cos(np.pi * z) * np.cos(x))
    state = model.state
    grid = state["b"].grid
    params = mapping_params(state, grid)
    b = state["b"]
    corr = model.tendency(state, constraints=False)["b"]
    # independent flux-form reference: reconstruct b onto ALL column
    # faces (walls included), the analytic mesh velocity w = z * H_dot
    # = (dzp/dH) * RATE, and the exact face->cell flux difference
    reconstruct = LinearReconstruction(
        target=NodeSet.OUTER, boundary="one_sided")
    f_face = reconstruct["z"](b)
    w_face = grid.metric(
        f_face.function_space, "dzp_dH", params=params) * RATE
    flux_diff = grid.dispatch.resolve(
        "flux_diff", f_face.function_space.bare.factor("z"))["z"]
    div_fw = flux_diff(f_face * w_face)
    div_w = flux_diff(w_face)
    jac = grid.metric(b.function_space, "dzp_dz", params=params)
    lhs = (float(jnp.sum((jac * corr).integrate().data))
           + float(jnp.sum((b * div_w).integrate().data)))
    boundary = float(jnp.sum(div_fw.integrate().data))
    assert abs(boundary) > 1.0  # the check is non-trivial
    assert abs(lhs - boundary) < 1e-12 * abs(boundary)


def test_fv_frozen_schedule_yields_a_zero_correction():
    # frozen motion (H_dot == 0 through the jvp) -> zero mesh velocity
    # -> bitwise-zero flux-form correction on every field (both routes):
    # G = f_face * 0 = 0, flux_diff(0) = 0, so corr = 0.0 exactly.
    frozen = MovingGeometry(
        {"H": lambda x, t: schedule(x, 0.0) + 0.0 * t})
    model = make_fv_model(frozen, MeshVelocityCorrection())
    ver = (np.arange(N) + 0.5) / N
    _, _, z = np.meshgrid(ver, ver, ver, indexing="ij")
    model.set_fields(b=z, u=z, v=z)
    tendency = model.tendency(model.state, constraints=False)
    for c in ("u", "v", "b"):
        assert np.array_equal(
            np.asarray(tendency[c].data),
            np.zeros_like(tendency[c].data)), c


# ================================================================
#  MeshVelocityCorrection: autodiff regression (FV step path)
# ================================================================
class _MappedCore(Module):

    """Toy FV mapped-column core: one CellAvg tracer, a trivial term.

    Isolates the ALE flux-route step path from the nonhydro pressure
    projection: the mapped-pressure PCG has a PRE-EXISTING reverse-mode
    NaN (a masked-singular VJP flagged in
    ``design/research/jax_grad_run_investigation.md``, present with no
    ALE and no moving geometry — the static mapped model NaNs too), so
    a full ``nh.Model`` run would test that hazard, not this module. The
    ALE flux form itself is linear (reconstruct + flux_diff + a metric
    ``1/J`` product, no sqrt / clip / field division), hence exactly
    differentiable — which this toy run pins.
    """

    field_declarations = (
        fr.model.FieldDeclaration.tracer(
            "b", space=fr.spatial.Collocated(family="fv")),
    )

    @fr.model.term(advances=("b",), linear=True)
    def zero(self, state, _ctx):
        """Return a trivial (zero) tendency so ``b`` is PROGNOSTIC."""
        return {"b": 0.0 * state["b"]}


def test_fv_ale_grad_through_a_run_matches_fd():
    # differentiability policy: jax.grad of a quadratic loss through a
    # short FV moving-geometry run (the pure _chunk_body kernel) w.r.t.
    # the initial buoyancy is finite and matches a central FD. Uses the
    # projection-free toy core (_MappedCore) so the pre-existing
    # mapped-pressure reverse-mode NaN does not mask the ALE result.
    grid = make_grid()
    grid.set_default_family("fv")
    model = FrModel(
        grid=grid,
        modules=(_MappedCore(), MovingGeometry({"H": schedule}),
                 MeshVelocityCorrection(("b",))),
        time_stepper=AdamBashforth(DT, order=3))
    ver = (np.arange(N) + 0.5) / N
    _, _, z = np.meshgrid(ver, ver, ver, indexing="ij")
    model.set_fields(b=0.01 * np.cos(np.pi * z))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    b_leaf = carry.state["b"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is b_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 6, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(b_leaf))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(b_leaf.shape),
                            dtype=b_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b_leaf + eps * direction))
          - float(loss(b_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
