"""The moving-geometry modules: schedules, updates, ALE term.

Framework-level modules exercised on nonhydro2 models (the
test_relaxation precedent of borrowing a concrete port): a
terrain-following column ``zp = z * H(x, t)`` with zero physics
(f0 = n2 = 0, no advection) isolates the geometry update and the
mesh-velocity correction, so pointwise tendencies are exact checks.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.declarations import Lifecycle
from fridom.model.modules.moving_geometry import (
    MeshVelocityCorrection,
    MovingGeometry,
    mapping_params,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
LENGTH = 2 * np.pi
DT = 1e-2
H0 = 0.8
RATE = 0.05


def schedule(x, t):
    """H(x, t) = H0 (1 + 0.1 sin x) + RATE * t."""
    return H0 * (1.0 + 0.1 * jnp.sin(x)) + RATE * t


def make_grid(mapped=True):
    mx = IntervalMesh(N, (0.0, LENGTH), periodic=True, name="x")
    my = IntervalMesh(N, (0.0, LENGTH), periodic=True, name="y")
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    if not mapped:
        return Grid((mx, my, mz))
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: schedule(x, 0.0)})
    return Grid((mx, my, mz), mapping=mapping)


def make_model(*modules, mapped=True):
    """Zero-physics linear nh model (geometry terms isolated).

    family="nodal" is explicit: moving geometry (the C4 machinery
    these tests exercise) is a nodal-only feature — the ALE
    correction is nodal-only — so the 2026-07-17 mapped auto flip
    keeps a moving-geometry model on the nodal path. Pinning here
    states that and keeps these tests on the validated path.
    """
    return nh.Model(
        grid=make_grid(mapped=mapped), dt=DT, advection=False,
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        modules_extra=modules, family="nodal")


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
        grid=grid, dt=DT, advection=False,
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        modules_extra=(MovingGeometry(
            {"H": lambda t: H0 + RATE * t}),),
        family="nodal")  # moving geometry is nodal-only (C4)
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
            grid=grid, dt=DT, advection=False,
            coriolis=nh.FPlaneCoriolis(f0=0.0),
            stratification=nh.ConstantStratification(n2=0.0),
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
