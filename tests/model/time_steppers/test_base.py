"""Tests for the time-stepper protocol (time_steppers/base.py).

Covers the ABC surface, the one-shot dt conversion onto the dynamic
leaf, the binding-table provider seam (the stepper joins assembly
step 2 as provider of ``fr.params.TIME_STEP`` bound to its ``dt``
leaf), the stepper-statics fingerprint token, and the host-side
analysis default.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import jaxify
from fridom.model.assembly import ParameterBindingTable
from fridom.model.params import TIME_STEP
from fridom.model.terms import Treatment
from fridom.model.time_steppers.base import (
    StepperState,
    TimeStepper,
)


# ================================================================
#  A minimal concrete stepper (the protocol contract)
# ================================================================
@partial(jaxify, dynamic=("dt",))
class DummyStepper(TimeStepper):
    supported_treatments = frozenset({Treatment.EXPLICIT})

    def init(self, _tendency_template):
        return ()

    def step(self, stepper_state, state, _stages, clock):
        return stepper_state, state, clock.tick(self.dt)


# ================================================================
#  ABC surface
# ================================================================
def test_time_stepper_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        TimeStepper(1.0)


def test_stepper_state_alias_exists():
    assert StepperState is not None


# ================================================================
#  The dt boundary (one-shot conversion onto the dynamic leaf)
# ================================================================
def test_dt_converts_to_a_real_scalar_leaf():
    stepper = DummyStepper(60.0)
    assert isinstance(stepper.dt, jax.Array)
    assert stepper.dt.dtype == jnp.float64  # x64-on suite
    assert stepper.dt.shape == ()
    assert float(stepper.dt) == 60.0


def test_dt_accepts_timedelta64():
    stepper = DummyStepper(np.timedelta64(2, "h"))
    assert float(stepper.dt) == 7200.0


def test_dt_keeps_the_sign():
    # signed dt is the backward-run primitive, not a mode
    assert float(DummyStepper(-30.0).dt) == -30.0


def test_dt_rejects_non_times():
    with pytest.raises(TypeError, match="seconds"):
        DummyStepper("1 hour")


def test_dt_is_the_only_dynamic_leaf():
    stepper = DummyStepper(60.0)
    assert stepper.dynamic_jax_attrs == ("dt",)
    leaves, treedef = jax.tree_util.tree_flatten(stepper)
    assert len(leaves) == 1
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is DummyStepper
    assert float(rebuilt.dt) == 60.0


# ================================================================
#  The binding-table provider seam (assembly step 2)
# ================================================================
def test_stepper_declares_the_time_step_provider_row():
    stepper = DummyStepper(60.0)
    (declaration,) = stepper.parameter_declarations
    assert str(declaration.name) == str(TIME_STEP)
    assert declaration.attr == "dt"
    assert stepper.provided_parameters == {str(TIME_STEP): "dt"}


def test_binding_table_binds_time_step_to_the_dt_leaf():
    # the hard seam: ParameterBindingTable.build duck-types the
    # stepper and yields the TIME_STEP row bound to the dt leaf
    stepper = DummyStepper(60.0)
    table = ParameterBindingTable.build((), stepper)
    entry = table[TIME_STEP]
    assert entry.slot == "stepper"
    assert entry.attr == "dt"
    assert str(TIME_STEP) in table.names


def test_eval_params_reads_the_live_dt_leaf():
    stepper = DummyStepper(60.0)
    table = ParameterBindingTable.build((), stepper)
    params = table.eval_params((), stepper, jnp.asarray(0.0))
    assert float(params[TIME_STEP]) == 60.0
    # the read is live: a sign-flipped stepper reads flipped
    backward = DummyStepper(-60.0)
    params = table.eval_params((), backward, jnp.asarray(0.0))
    assert float(params[TIME_STEP]) == -60.0


def test_host_view_reads_the_dt_leaf():
    stepper = DummyStepper(0.5)
    table = ParameterBindingTable.build((), stepper)
    view = table.host_view((), stepper)
    assert float(view[str(TIME_STEP)]) == 0.5


# ================================================================
#  Fingerprint token (stepper statics; dt excluded)
# ================================================================
def test_fingerprint_token_names_the_class():
    assert DummyStepper(1.0).fingerprint_token() == ("DummyStepper",)


def test_fingerprint_token_excludes_the_dt_leaf():
    # fingerprints hash structure, never leaves (02_rules)
    assert (DummyStepper(1.0).fingerprint_token()
            == DummyStepper(-2.5).fingerprint_token())


# ================================================================
#  Host-side analysis default
# ================================================================
def test_time_discretization_effect_base_raises():
    with pytest.raises(NotImplementedError, match="AdamBashforth"):
        DummyStepper(1.0).time_discretization_effect(
            np.asarray([1.0]))


def test_repr_smoke():
    assert "DummyStepper" in repr(DummyStepper(60.0))
