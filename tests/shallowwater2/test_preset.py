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
    preset = sw.Model(grid=grid, csqr=1.0, rossby_number=0.2,
                      coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
                      time_stepper=stepper)
    explicit = fr.model.Model(
        grid=grid,
        modules=(sw.modules.DynamicalCore(csqr=1.0,
                                          rossby_number=0.2),
                 sw.modules.FPlaneCoriolis(f0=1.0),
                 sw.modules.SadournyAdvection()),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    assert (jax.tree_util.tree_structure(preset._carry)
            == jax.tree_util.tree_structure(explicit._carry))


def test_preset_is_a_plain_model_not_a_subclass():
    model = sw.Model(grid=make_grid(),
                     time_stepper=fr.model.time_steppers.AdamBashforth(
                         5e-3, order=3))
    assert type(model) is fr.model.Model


# ================================================================
#  The State vocabulary class (u, v, p)
# ================================================================
def test_core_supplies_the_state_vocabulary():
    model = sw.Model(grid=make_grid(),
                     time_stepper=fr.model.time_steppers.AdamBashforth(
                         5e-3, order=3))
    assert isinstance(model.state, State)
    assert model.state.component_names[:3] == ("u", "v", "p")


def test_vocabulary_accessors_return_components():
    model = sw.Model(grid=make_grid(),
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


def make_varying(coriolis=None):
    return sw.Model(
        grid=make_grid(periodic_y=False), csqr=csqr_profile,
        rossby_number=0.2, coriolis=coriolis, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def test_varying_csqr_declares_a_profile_and_drops_the_provide():
    # provides-implies-constancy: the callable path materializes the
    # meridional csqr field and provides NO shallowwater.csqr scalar
    model = make_varying()
    assert sw.params.CSQR not in model.parameters
    csqr = model.state["csqr"]
    centres = (np.arange(N) + 0.5) / N
    np.testing.assert_allclose(
        np.asarray(csqr.data).ravel(), csqr_profile(centres))


def test_constant_csqr_still_provides_the_scalar():
    model = sw.Model(
        grid=make_grid(), csqr=0.7,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    assert float(model.parameters[sw.params.CSQR]) == 0.7
    # the constant field stays one-DOF
    assert np.asarray(model.state["csqr"].data).size == 1


def test_default_coriolis_is_always_thickness_weighted():
    # the weighted rotation is exactly M-skew for any f and any
    # depth profile and coincides with the unweighted form for
    # constant depth (to rounding), so the preset always uses it
    model = make_varying()
    assert model.module(FPlaneCoriolis).metric_weight == "csqr"
    constant = sw.Model(
        grid=make_grid(), csqr=0.7,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    assert constant.module(FPlaneCoriolis).metric_weight == "csqr"


def test_varying_csqr_with_an_unweighted_coriolis_is_taught():
    with pytest.raises(ValueError, match="metric_weight='csqr'"):
        make_varying(coriolis=sw.modules.FPlaneCoriolis(f0=1.0))
    with pytest.raises(ValueError, match="metric_weight='csqr'"):
        make_varying(coriolis=sw.modules.BetaPlaneCoriolis(
            f0=1.0, beta=0.5))


def test_varying_csqr_with_a_weighted_coriolis_assembles():
    model = make_varying(coriolis=sw.modules.BetaPlaneCoriolis(
        f0=1.0, beta=0.5, metric_weight="csqr"))
    assert type(model) is fr.model.Model


def test_varying_csqr_guard_skips_non_framework_modules():
    # the guard inspects only the framework Coriolis types; a
    # rotation-free custom module slot assembles untouched
    model = make_varying(coriolis=sw.modules.SadournyAdvection())
    assert type(model) is fr.model.Model


def test_varying_csqr_model_steps():
    # the tendency terms read the csqr FIELD, so the varying model
    # integrates as-is (advection included via a separate test)
    model = make_varying()
    rng = np.random.default_rng(5)
    model.set_fields(
        u=0.01 * rng.standard_normal(model.state["u"].shape),
        v=0.01 * rng.standard_normal(model.state["v"].shape),
        p=0.01 * rng.standard_normal(model.state["p"].shape))
    model.advance(3)
    assert not bool(model.state["p"].has_nan())


def test_default_time_stepper_is_adam_bashforth():
    # the preset's cutover default (pass an explicit one for a real
    # run); assembly succeeds and the model advances
    model = sw.Model(grid=make_grid())
    assert isinstance(model._stepper,
                      fr.model.time_steppers.AdamBashforth)
