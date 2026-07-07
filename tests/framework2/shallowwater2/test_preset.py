"""The D4 preset test and the State vocabulary contract."""
import jax
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.shallowwater2.state import MissingComponentError, State

from .conftest import make_grid


# ================================================================
#  D4: the preset is a thin factory (identical treedef)
# ================================================================
def test_preset_equals_explicit_assembly_treedef():
    grid = make_grid()
    stepper = fr.time_steppers.AdamBashforth(5e-3, order=3)
    preset = sw.Model(grid=grid, csqr=1.0, rossby_number=0.2,
                      coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
                      time_stepper=stepper)
    explicit = fr.Model(
        grid=grid,
        modules=(sw.modules.DynamicalCore(csqr=1.0,
                                          rossby_number=0.2),
                 sw.modules.FPlaneCoriolis(f0=1.0),
                 sw.modules.SadournyAdvection()),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))
    assert (jax.tree_util.tree_structure(preset._carry)
            == jax.tree_util.tree_structure(explicit._carry))


def test_preset_is_a_plain_model_not_a_subclass():
    model = sw.Model(grid=make_grid(),
                     time_stepper=fr.time_steppers.AdamBashforth(
                         5e-3, order=3))
    assert type(model) is fr.Model


# ================================================================
#  The State vocabulary class (u, v, p)
# ================================================================
def test_core_supplies_the_state_vocabulary():
    model = sw.Model(grid=make_grid(),
                     time_stepper=fr.time_steppers.AdamBashforth(
                         5e-3, order=3))
    assert isinstance(model.state, State)
    assert model.state.component_names[:3] == ("u", "v", "p")


def test_vocabulary_accessors_return_components():
    model = sw.Model(grid=make_grid(),
                     time_stepper=fr.time_steppers.AdamBashforth(
                         5e-3, order=3))
    state = model.state
    assert state.u is state["u"]
    assert state.v is state["v"]
    assert state.p is state["p"]


def test_missing_component_accessor_raises_hinted():
    grid = make_grid()
    only_u = State({"u": grid.create_field(
        fr.Staggered("x").resolve(grid))})
    with pytest.raises(MissingComponentError, match="core"):
        _ = only_u.p
