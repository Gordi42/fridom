"""The D4 preset test and the State vocabulary contract."""
import jax
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.shallowwater2.state import MissingComponentError, State

from .conftest import N, make_grid


# ================================================================
#  D4: the preset is a thin factory (identical treedef)
# ================================================================
def test_preset_equals_explicit_assembly_treedef():
    grid = make_grid()
    stepper = fr.model.time_steppers.AdamBashforth(5e-3, order=3)
    preset = sw.Model(
        grid=grid,
        core=sw.Core(froude_number=0.2, depth=1.0),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=0.2),
        time_stepper=stepper)
    explicit = fr.model.Model(
        grid=grid,
        modules=(sw.Core(froude_number=0.2, depth=1.0),
                 sw.modules.FPlaneCoriolis(rossby_number=0.2),
                 sw.modules.SadournyAdvection()),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3),
        scaling=fr.scaling.GravityWave())
    assert (jax.tree_util.tree_structure(preset._carry)
            == jax.tree_util.tree_structure(explicit._carry))


def test_preset_is_a_plain_model_not_a_subclass():
    model = sw.Model(grid=make_grid(),
                     core=sw.Core(gravity=1.0, depth=1.0),
                     time_stepper=fr.model.time_steppers.AdamBashforth(
                         5e-3, order=3))
    assert type(model) is fr.model.Model


# ================================================================
#  The retired preset kwargs teach the new spelling
# ================================================================
@pytest.mark.parametrize(("kwargs", "match"), [
    pytest.param({"csqr": 1.0}, "csqr= is retired", id="csqr"),
    pytest.param({"rossby_number": 0.2},
                 "rossby_number= is retired", id="rossby-number"),
    pytest.param({"coords": ("x", "y")}, "coords= is retired",
                 id="coords"),
])
def test_retired_preset_kwargs_teach_the_new_spelling(kwargs, match):
    with pytest.raises(TypeError, match=match):
        sw.Model(
            grid=make_grid(),
            core=sw.Core(gravity=1.0, depth=1.0),
            time_stepper=fr.model.time_steppers.AdamBashforth(5e-3),
            **kwargs)


def test_time_stepper_is_required():
    # the old preset default stepper is retired: the preset requires
    # an explicit time stepper
    with pytest.raises(TypeError, match="time_stepper"):
        sw.Model(grid=make_grid(),
                 core=sw.Core(gravity=1.0, depth=1.0))


# ================================================================
#  The State vocabulary class (u, v, p)
# ================================================================
def test_core_supplies_the_state_vocabulary():
    model = sw.Model(grid=make_grid(),
                     core=sw.Core(gravity=1.0, depth=1.0),
                     time_stepper=fr.model.time_steppers.AdamBashforth(
                         5e-3, order=3))
    assert isinstance(model.state, State)
    assert model.state.component_names[:3] == ("u", "v", "p")


def test_vocabulary_accessors_return_components():
    model = sw.Model(grid=make_grid(),
                     core=sw.Core(gravity=1.0, depth=1.0),
                     time_stepper=fr.model.time_steppers.AdamBashforth(
                         5e-3, order=3))
    state = model.state
    assert state.u is state["u"]
    assert state.v is state["v"]
    assert state.p is state["p"]


def test_missing_component_accessor_raises_hinted():
    grid = make_grid()
    only_u = State({"u": grid.create_field(
        fr.spatial.Staggered("x").resolve(grid))})
    with pytest.raises(MissingComponentError, match="core"):
        _ = only_u.p


# ================================================================
#  Variable depth: the csqr(y) declaration path
# ================================================================
def csqr_profile(y):
    return 1.0 + 0.5 * np.sin(np.pi * y)


def weighted_fplane(f0=1.0):
    """Build the thickness-weighted f-plane variable depth needs."""
    return sw.modules.FPlaneCoriolis(f0=f0, metric_weight="csqr")


def make_varying(coriolis=None):
    # coriolis=None is the preset default and means NO rotation;
    # dimensional core (csqr = 1.0 * D(y) = the old profile)
    return sw.Model(
        grid=make_grid(periodic_y=False),
        core=sw.Core(gravity=1.0, depth=csqr_profile),
        coriolis=coriolis, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def test_varying_depth_declares_a_profile_and_drops_the_provide():
    # provides-implies-constancy: the callable path materializes the
    # meridional csqr field and provides NO constant depth scalar
    model = make_varying()
    assert sw.params.DEPTH not in model.parameters
    csqr = model.state["csqr"]
    centres = (np.arange(N) + 0.5) / N
    np.testing.assert_allclose(
        np.asarray(csqr.data).ravel(), csqr_profile(centres))


def test_constant_depth_still_provides_the_scalar():
    model = sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=1.0, depth=0.7),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    assert float(model.parameters[sw.params.DEPTH]) == 0.7
    assert float(model.parameters[sw.params.GRAVITY]) == 1.0
    # the constant field stays one-DOF
    assert np.asarray(model.state["csqr"].data).size == 1


def test_the_default_is_no_rotation_at_all():
    # coriolis=None (the argument omitted) installs NO Coriolis
    # module: no f_coriolis field, no rotation term, no coriolis.f0
    # provide. Rotation is opt-in.
    model = sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=1.0, depth=0.7),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    assert "f_coriolis" not in model.state
    assert fr.model.params.CORIOLIS_F0 not in model.parameters
    with pytest.raises(LookupError, match="no live module matches"):
        model.module(FPlaneCoriolis)
    # ... and a named rotation is simply installed as given
    rotating = sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=1.0, depth=0.7),
        coriolis=sw.modules.FPlaneCoriolis(f0=1.5),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    assert float(rotating.parameters[fr.model.params.CORIOLIS_F0]) == 1.5


def test_varying_depth_with_an_unweighted_coriolis_is_taught():
    with pytest.raises(ValueError, match="metric_weight='csqr'"):
        make_varying(coriolis=sw.modules.FPlaneCoriolis(f0=1.0))
    with pytest.raises(ValueError, match="metric_weight='csqr'"):
        make_varying(coriolis=sw.modules.BetaPlaneCoriolis(
            f0=1.0, beta=0.5))


def test_varying_depth_with_a_weighted_coriolis_assembles():
    model = make_varying(coriolis=sw.modules.BetaPlaneCoriolis(
        f0=1.0, beta=0.5, metric_weight="csqr"))
    assert type(model) is fr.model.Model


def test_varying_depth_guard_skips_non_framework_modules():
    # the guard inspects only the framework Coriolis types; a
    # rotation-free custom module slot assembles untouched
    model = make_varying(coriolis=sw.modules.SadournyAdvection())
    assert type(model) is fr.model.Model


def test_varying_depth_model_steps():
    # the tendency terms read the csqr FIELD, so the varying model
    # integrates as-is (advection included via a separate test);
    # the thickness-weighted f-plane the old implicit default
    # installed, now named explicitly
    model = make_varying(coriolis=weighted_fplane())
    rng = np.random.default_rng(5)
    model.set_fields(
        u=0.01 * rng.standard_normal(model.state["u"].shape),
        v=0.01 * rng.standard_normal(model.state["v"].shape),
        p=0.01 * rng.standard_normal(model.state["p"].shape))
    model.advance(3)
    assert not bool(model.state["p"].has_nan())


def test_flat_linear_model_negotiates_the_traced_halo():
    # the core's chart / immersed extra_halo (derived, 1 per axis) is
    # gated on chartedness: the flat gravity term is a plain staggered
    # difference the tracer follows exactly, so a linear flat model
    # negotiates width 1 — an unconditional declaration would pay halo
    # bytes for a chart path a flat model never runs
    model = sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=1.0, depth=1.0),
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    halo = model.grid.decomposition.halo
    assert halo["x"] == 1
    assert halo["y"] == 1


def test_sadourny_model_keeps_its_declared_halo():
    # the advective model is unaffected by the gate: Sadourny
    # declares its own 2 (the corner chain), which masks the core's
    model = sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=1.0, depth=1.0),
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
        advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    halo = model.grid.decomposition.halo
    assert halo["x"] == 2
    assert halo["y"] == 2
